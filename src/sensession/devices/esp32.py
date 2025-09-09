# bklein 2024
"""
Manages one ESP32 device and offers a simple API to interact with it
"""

import time
import queue
import struct
import threading
from enum import Enum
from pathlib import Path
from dataclasses import dataclass

from loguru import logger

from sensession.config import (
    MacAddr,
    Bandwidth,
    BaseFrameConfig,
    BaseTransmissionConfig,
)
from sensession.util.hash import get_hash
from sensession.util.exceptions import ApiUsageError

try:
    import serial
    import serial.tools.list_ports

    ESP32_AVAILABLE = True
except ImportError:
    ESP32_AVAILABLE = False


class ESPOperationMode(int, Enum):
    """
    ESP operating mode (receiver/transmitter)
    >do not change the enum values<
    """

    RX = int(0x00)
    TX = int(0x01)


class ESP40MHzSecondaryChannel(int, Enum):
    """
    If we operate at 40MHz bandwith, we need to specify whether
    to use a channel above or below the primary wifi channel
    >do not change the enum values<
    """

    ABOVE = int(0x02)
    BELOW = int(0x01)
    NO = int(0x00)


class CSISelection(int, Enum):
    """
    Depending on the configuration, the ESP might return more than one type of CSI data.
    We (fp + bk) have agreed to make a mandatory choice for one type of CSI data.
    >do not change the enum values<
    """

    L_LTF = int(0x00)  # Legacy Long Training Field
    HT_LTF = int(0x01)  # High Throughput Long Training Field


### Configuration
@dataclass
class ESP32Config:
    """
    Device configuration
    """

    # fmt: off
    name       : str                                     # device name (arbitrary, but must be unique)
    short_name : str                                     # short name (mainly used for logging)
    comport    : str                                     # Serial port ESP is attached to. You have to determine this yourself.
                                                         #  In linux you can use "dmesg | grep tty" to list serial devices.
                                                         #  In windows, you can use the device manager to find the device.
                                                         #  Alternatively, use the provided "ESP_CSI_Monitor" application
                                                         #  included in the ESP firmware repo, this lists all available devices
                                                         #  Or, just note down the port the esp-idf uses when flashing the firmware
    baudrate   : int = 3000000                           # only relevant if UART is used, otherwise ignored, but must always be >0 (!)
    mode       : ESPOperationMode = ESPOperationMode.RX  # indicates whether the device is used to transmit wifi packets ('tx') or receive csi data ('rx')
    wifi_bw    : Bandwidth = Bandwidth.TWENTY            # bandwidth to initialize the esp WiFi with. Affects both RX and TX
    csi_type   : CSISelection = CSISelection.HT_LTF      # Choose how to extract CSI
    wifi_40mhz_second_chan: ESP40MHzSecondaryChannel = ESP40MHzSecondaryChannel.NO  # only relevant if 40MHz bandwidth: secondary channel above/below primary
    manu_scale : int = 0 # experimental: csi scaling flag for ESP. NB different values for HE devices (C6)
    # fmt: on

    def device_id(self) -> str:
        """
        Get deterministic device id based on config
        """
        return get_hash(f"{self.name}:{self.comport}")

    def instantiate_device(self):
        """
        Create corresponding device to this config
        """
        return ESP32(self)


### Device API
class ESP32:
    """
    Abstraction for an ESP32 device
    """

    class ESPCommand(int, Enum):
        """
        Commands recognized by the ESP
        >do not change the enum values<
        """

        SET_CHANNEL = int(0x01)
        WHITELIST_ADD_MAC_PAIR = int(0x02)
        WHITELIST_CLEAR = int(0x03)
        PAUSE_ACQUISITION = int(0x04)
        UNPAUSE_ACQUISITION = int(0x05)
        APPLY_DEVICE_CONFIG = int(0x06)
        PAUSE_WIFI_TRANSMIT = int(0x07)
        RESUME_WIFI_TRANSMIT = int(0x08)
        TRANSMIT_CUSTOM_WIFI_FRAME = int(0x09)
        SYNCHRONIZE_TIME_INIT = int(0x0A)
        SYNCHRONIZE_TIME_APPLY = int(0x0B)

    def __init__(self, config: ESP32Config):
        if not ESP32_AVAILABLE:
            raise ModuleNotFoundError(
                "ESP32 imports failed; did you install the optional esp32 dependencies?"
            )

        self.config = config

        self.filepath: Path | None = None
        self.serial_connection: serial.Serial | None = None
        self.connection_established = False
        self.listening_for_csi = False
        self.serial_monitor_thread: threading.Thread | None = None
        self.csi_processing_thread: threading.Thread | None = None

        self.ack_queue: queue.Queue = queue.Queue(maxsize=10000)
        self.csi_queue: queue.Queue = queue.Queue(maxsize=10000)
        self.outer_header_length = 8 + 2
        self.csi_header_length = 8 + 6 + 6 + 2 + 1 + 1 + 1 + 2

        if config.wifi_bw == Bandwidth.FOURTY:
            if config.wifi_40mhz_second_chan == ESP40MHzSecondaryChannel.NO:
                raise RuntimeError(
                    "Must set secondary channel when setting BW to 40 MHz"
                )

    def connect_device(self):
        """
        Open connection and apply operating mode from config. Also synchronize the device clock with the host clock
        """
        self.open_serial_connection()
        self.upload_device_config()
        self.sync_esp_clock()

    def reset(self):
        """
        Restore the device
        """
        self.stop_receiving_csi()
        self.close_serial_connection()
        self.filepath = None

    def get_csi_subcarrier_idxs(self) -> list[int]:
        """
        retrieve ordering of subcarriers in csi data for this device
        guard subcarriers are already removed from the ordering (still to do for 40MHz)

        # Guard subcarriers
        IEEE Std 802.11-2020 19.3.7:
        "In the 20 MHz non-HT format, the signal is transmitted on subcarriers -26 to -1 and 1 to 26, with 0 being the center (DC) carrier."
        "In the 20 MHz     HT format, the signal is transmitted on subcarriers -28 to -1 and 1 to 28."
        "In the case of 40 MHz HT upper format or 40 MHz HT lower format, the upper or lower 20 MHz is divided into 64 subcarriers. The signal is transmitted on subcarriers -60 to -4 in the case of a 40 MHz HT lower format transmission and on subcarriers 4 to 60 in the case of a 40 MHz HT upper format transmission."

        # Subcarrier order
        https://docs.espressif.com/projects/esp-idf/en/stable/esp32/api-guides/wifi.html#wi-fi-channel-state-information
        """
        if self.config.wifi_bw == Bandwidth.TWENTY:
            if self.config.csi_type == CSISelection.L_LTF:
                # 1~26,-26~-1 (20MHz, L-LTF (nonHT or HT))
                order = list(range(1, 27)) + list(range(-26, 0))
            else:
                # 1~28, -28~-1 (20MHz, HT-LTF (HT))
                order = list(range(1, 29)) + list(range(-28, 0))
        elif self.config.wifi_bw == Bandwidth.FOURTY:
            if self.config.csi_type == CSISelection.L_LTF:
                if self.config.wifi_40mhz_second_chan == ESP40MHzSecondaryChannel.ABOVE:
                    # -64~-1 (40MHz, L-LTF (HT), above)
                    # after filtering -> -58~-33, -31~-6
                    order = list(range(-58, -32)) + list(range(-31, -5))
                else:
                    # 0~63 (40MHz, L-LTF (HT), below)
                    # after filtering -> 6~31, 33~58
                    order = list(range(6, 32)) + list(range(33, 59))
            else:
                # 0~63, -64~-1 (40MHz, HT-LTF (HT), above or below)
                # 2~58, -58~-2 actual values -> doesn't conform to the standard, but it's what the ESP gives us...
                order = list(range(2, 59)) + list(range(-58, -1))
        else:
            raise ValueError("ESP32 only support 20 and 40 MHz bandwidths.")

        return order

    def get_subcarrier_mask(self) -> list[int]:
        """
        Retrieve indices of carriers that do provide valid/interesting CSI.
        This mask is used to filter the received csi later on
        Note that these indices reference the subcarriers in their unsorted form, as they are returned from the ESP!
        To be more clear, if the mask contains entry '2', then whichever subcarrier is at index 2 in the array contains useful CSI.
        """
        if self.config.wifi_bw == Bandwidth.TWENTY:
            if self.config.csi_type == CSISelection.L_LTF:
                #  (20MHz, L-LTF (nonHT or HT))
                #   valid sorted indices 6->31, 33->58
                #   mask = list(range(6,32)) + list(range(33,59))
                #   valid unsorted indices 1->26, 38->63
                mask = list(range(1, 27)) + list(range(38, 64))
            else:
                # (20MHz, HT-LTF (HT))
                #   valid sorted indices 4->31, 33->60
                #   mask = list(4,32) + list(33,61)
                #   valid unsorted indices 1->28, 36->63
                mask = list(range(1, 29)) + list(range(36, 64))
        else:
            if self.config.csi_type == CSISelection.L_LTF:
                # (40MHz, L-LTF (HT))
                mask = list(range(6, 32)) + list(range(33, 59))
            else:
                # (40MHz, HT-LTF (HT))
                # doesn't adhere to the standard, but these are the valid indices for ESP32 40MHz (determined by experiment)
                mask = list(range(2, 59)) + list(range(70, 127))

        return mask

    def open_serial_connection(self):
        """
        Creates serial connection to the ESP
        """
        try:
            port = self.config.comport
            baudrate = self.config.baudrate
            logger.debug(f"Creating serial connection to ESP device, port = {port}\n")
            self.serial_connection = serial.Serial(
                port, baudrate, timeout=1, write_timeout=1
            )
            self.serial_connection.setRTS(False)  # to avoid ESP restart on serial close
            self.serial_connection.setDTR(False)  # to avoid ESP restart on serial close
        except serial.SerialException as e:
            logger.error(
                f"ESP32: could not open connection on port {self.config.comport}. "
                + "Make sure the correct port is set in config and not in use."
                + str(e)
            )
            raise

        self.connection_established = True
        self.serial_monitor_thread = threading.Thread(target=self._monitor_serial)
        self.serial_monitor_thread.start()

    def close_serial_connection(self):
        """
        Closes serial connection
        """
        self.connection_established = False
        if self.serial_monitor_thread is not None:
            self.serial_monitor_thread.join()

        if self.serial_connection and self.serial_connection.is_open:
            logger.debug("Closing serial connection to ESP device")
            self.serial_connection.close()

    def start_receiving_csi(self):
        """
        Creates a thread that logs incoming CSI data
        """
        if self.listening_for_csi:
            return

        if not self.connection_established:
            self.connect_device()

        self.listening_for_csi = True
        self.csi_processing_thread = threading.Thread(target=self._process_csi_queue)
        self.csi_processing_thread.start()
        self.unpause_csi_aqcuisition()

    def stop_receiving_csi(self):
        """
        Stop receiving CSI (and let thread terminate)
        """
        # Stopping needs only be done if the tool is running
        if not self.listening_for_csi:
            return

        self.pause_csi_aqcuisition()
        self.listening_for_csi = False
        if self.csi_processing_thread is not None:
            self.csi_processing_thread.join()

    def change_channel(self, channel: int):
        """
        Change channel the ESP32 is listening on
        """
        channel_data = bytearray(1)
        channel_data[0] = channel
        self._send_command(self.ESPCommand.SET_CHANNEL, channel_data)

    def upload_device_config(self):
        """
        Puts the ESP32 in
            Passive mode (to log CSI) or
            ActiveSend mode (to transmit wifi packets)
            and sets all parameters (bw, csi type, ...) according to config
        """
        cmd_data = bytearray(5)
        cmd_data[0] = self.config.mode
        if self.config.wifi_bw == Bandwidth.TWENTY:
            cmd_data[1] = 0x00
        if self.config.wifi_bw == Bandwidth.FOURTY:
            cmd_data[1] = 0x01
        cmd_data[2] = self.config.wifi_40mhz_second_chan
        cmd_data[3] = self.config.csi_type
        cmd_data[4] = self.config.manu_scale

        self._send_command(self.ESPCommand.APPLY_DEVICE_CONFIG, cmd_data)

    def set_filter_rules(
        self, transmitter_address: MacAddr | None, receiver_address: MacAddr | None
    ):
        """
        Installs a filter rule on the device
        """
        self.clear_mac_pair_whitelist()
        if not transmitter_address or not receiver_address:
            return

        src_addr_bytes = bytes(
            int(part, 16) for part in transmitter_address.with_separator(":").split(":")
        )
        dst_addr_bytes = bytes(
            int(part, 16) for part in receiver_address.with_separator(":").split(":")
        )
        self.add_mac_pair_to_whitelist(src_addr_bytes, dst_addr_bytes)

    def add_mac_pair_to_whitelist(self, smac: bytes, dmac: bytes):
        """
        Add a mac to the ESP source mac filter whitelist
        Note: if the list is empty, everything is let through
        """
        command_data = bytearray(len(smac) + len(dmac))
        command_data[: len(smac)] = smac
        command_data[len(dmac) :] = dmac
        self._send_command(self.ESPCommand.WHITELIST_ADD_MAC_PAIR, command_data)

    def clear_mac_pair_whitelist(self):
        """
        Clear the ESP source map filter whitelist
        Note: if the list is empty, everything is let through
        """
        self._send_command(self.ESPCommand.WHITELIST_CLEAR, None)

    def pause_csi_aqcuisition(self):
        """
        Instruct ESP to stop sending CSI data
        """
        self._send_command(self.ESPCommand.PAUSE_ACQUISITION, None)

    def unpause_csi_aqcuisition(self):
        """
        Instruct ESP to start sending CSI data
        """
        self._send_command(self.ESPCommand.UNPAUSE_ACQUISITION, None)

    def sync_esp_clock(self):
        """
        Synchronize the ESP32 clock with this machines clock
        """
        self._send_command(self.ESPCommand.SYNCHRONIZE_TIME_INIT, None)

        time_us = time.time_ns() // 1000
        command_data = bytearray(64)
        command_data[:] = struct.pack("Q", time_us)

        self._send_command(self.ESPCommand.SYNCHRONIZE_TIME_APPLY, command_data)

    def set_filepath(self, filepath: Path):
        """
        Set path for csi log file
        """
        self.filepath = filepath

    def get_config(self) -> ESP32Config:
        """
        Get the internal config
        """
        return self.config

    def transmit_frame_with_tx_config(
        self, frame: BaseFrameConfig, txconf: BaseTransmissionConfig
    ):
        """
        transmits a wifi frame n times according to txconf (not a continous send).
        Device must already be in tx mode!
        (set in config upon creation or call upload_device_config beforehand)
        """
        # device must be a tx device
        if self.config.mode != ESPOperationMode.TX:
            raise RuntimeError("Tried to transmit with non-TX device")
        if not frame.transmitter_address or not frame.receiver_address:
            raise RuntimeError("Frame to transmit did not contain tx/rx MAC")
        args = bytearray(20)
        mac_src = bytes.fromhex(frame.transmitter_address.strip())
        mac_dst = bytes.fromhex(frame.receiver_address.strip())
        args[:6] = mac_dst
        args[6:12] = mac_src
        # pack n_reps and pause_ms into 8 bytes ("i" = 4 byte int)
        packed_ints = struct.pack("ii", txconf.n_reps, txconf.pause_ms)
        args[12:20] = packed_ints
        self._send_command(self.ESPCommand.TRANSMIT_CUSTOM_WIFI_FRAME, args)

    ### private members
    def _send_command(
        self,
        cmd: ESPCommand,
        cmd_data: bytearray | None = None,
        wait_for_ack_timeout: int | None = 2,
    ):
        """
        Sends a command and additional data to the ESP.
        Data must be less than 124 bytes
        """
        if not self.connection_established or self.serial_connection is None:
            raise RuntimeError(
                "Serial connection must be opened before sending commands"
            )

        if not cmd_data:
            cmd_data = bytearray(0)
        cmd_data_len = len(cmd_data)
        assert cmd_data_len < 124, (
            "Packet size is fixed; Command data can't exceed 123 bytes"
        )

        # Construct a 128 byte ESP command packet from
        #   4 bytes preamble (used for detection)
        #   1 byte command
        #   0-123 bytes command data
        #   (zero padded to 128 bytes)
        cmd_preamble = bytes.fromhex("c3c3c3c3")
        data_to_send = bytearray(128)
        data_to_send[:4] = cmd_preamble
        data_to_send[4] = cmd
        data_to_send[5 : 5 + cmd_data_len] = cmd_data

        for n_try in range(3):
            conn_bytes_written = self.serial_connection.write(data_to_send)
            assert conn_bytes_written == len(data_to_send)
            if self._wait_for_cmd_ack(cmd, cmd_data, wait_for_ack_timeout) is False:
                if n_try == 2:
                    raise RuntimeError(
                        "ESP32 failed to acknowledge command 3 times. Check connection."
                    )
            else:
                return

    def _wait_for_cmd_ack(  # pylint: disable=too-many-return-statements
        self,
        cmd: ESPCommand,
        cmd_data: bytearray | None = None,
        timeout: int | None = 2,
    ):
        for _ in range(
            1000
        ):  # -> arbitrary max value to avoid endless loop should we keep getting wrong ACKs
            try:
                ack = self.ack_queue.get(timeout=timeout)
            except queue.Empty:
                return False

            try:
                (ack_cmd,) = struct.unpack(
                    "<B", ack[self.outer_header_length : self.outer_header_length + 1]
                )  # get ack_cmd as LE-uint8 (<B)

                # verify ACK of sent command
                if ack_cmd == cmd:
                    if cmd_data is None:
                        return True
                    # verify ACK of sent data
                    if self.outer_header_length > len(
                        ack
                    ) or self.outer_header_length + len(cmd_data) > len(ack):
                        return False
                    if (
                        ack[
                            self.outer_header_length + 1 : self.outer_header_length
                            + len(cmd_data)
                            + 1
                        ]
                        == cmd_data
                    ):
                        return True
            except struct.error:
                return False

        return False

    def _monitor_serial(self):  # pylint: disable=too-many-branches
        """
        listen to data
        """
        if not self.connection_established or self.serial_connection is None:
            logger.warning("Serial monitoring started but no serial connection opened")
            return

        partial_buffer = b""
        # Preamble of custom packet format sent by the ESP32
        preamble = bytes.fromhex("aaaaaaaaaaaaaaaa")

        while self.connection_established:  # pylint: disable=too-many-nested-blocks
            try:
                bytes_waiting = self.serial_connection.in_waiting
                if bytes_waiting > 0:
                    message = self.serial_connection.read(bytes_waiting)
                    partial_buffer += message

                    while True:
                        if len(partial_buffer) < self.outer_header_length:
                            # not enough data, skip (-> wait for more)
                            break

                        if (preamble_start := partial_buffer.find(preamble)) < 0:
                            # No preamble found. We have at least `outer_header_length` bytes of data.
                            # In the worst case, the entire buffer is garbage but the
                            # final 7 bytes are an incomplete preamble of
                            # the next packet (one byte of the preamble not yet received).
                            # Hence, we discard everything before these last 7.
                            partial_buffer = partial_buffer[-7:]
                            break

                        if preamble_start > 0:
                            # discard any bytes before the preamble
                            partial_buffer = partial_buffer[preamble_start:]

                        if len(partial_buffer) < self.outer_header_length:
                            # if we discarded something before the preamble, we might not have enough
                            # data anymore for a full header (-> wait for more)
                            break

                        # decode the "packet header"
                        # Format string explanation: "<Q h"
                        #   `<`  little-endian order  (lsb first)
                        #   `Q`  `unsigned long long` (8 bytes) -> preamble
                        #   `h`  `signed short`     (2 bytes)   -> packet length
                        header = partial_buffer[: self.outer_header_length]
                        (
                            _,
                            total_length,
                        ) = struct.unpack("<Q h", header)

                        # sign indicates packet type
                        packet_type = "ack" if total_length < 0 else "csi"
                        total_length = abs(total_length)

                        if len(partial_buffer) < total_length:
                            # not all data received yet (-> wait for more)
                            # (only after parsing the header do we know how big the packet actually is.
                            #   so here we check whether we have enough data in the buffer for a full packet)
                            break

                        # at this point, we have either
                        #   a) enough data for >exactly< one packet (possible if little traffic,
                        #       and good enough evidence to assume the packet is of correct form)
                        #   b) enough data for one packet plus 1 to 7 bytes (very unlikely)
                        #   c) enough data for at least one packet plus 8 more bytes.
                        #       If the packet is correctly formatted, these next 8 bytes should contain a preamble
                        if len(partial_buffer) >= total_length + len(preamble):
                            # we were in case c). That means, we can check for correctness by probing the next 8 bytes
                            expected_next_preamble = partial_buffer[
                                total_length : total_length + len(preamble)
                            ]
                            if expected_next_preamble != preamble:
                                # the 8 bytes after the packet are not a preamble, so the first preamble in the buffer
                                # might be a fluke or the packet is malformed. We discard the first preamble, but keep the rest of the buffer.
                                # (in which there might be another preamble, for which we start looking now)
                                partial_buffer = partial_buffer[len(preamble) :]
                                continue

                        # now we have a complete packet
                        packet = partial_buffer[:total_length]

                        if packet_type == "csi":
                            self.csi_queue.put_nowait(packet)
                        elif packet_type == "ack":
                            self.ack_queue.put_nowait(packet)

                        # remove processed packet
                        partial_buffer = partial_buffer[total_length:]

                time.sleep(0.01)
            except IndexError:
                logger.warning("Received malformed ESP32 packet")
            except struct.error:
                logger.warning("Received malformed ESP32 packet")
            except serial.SerialException as e:
                logger.warning(
                    f"ESP32 RX error ({self.config.short_name} on {self.config.comport}): {e}"
                )

    def _process_csi_queue(self):
        """
        Processes all csi packets that get put into the csi queue by the monitoring thread
        """
        if not self.listening_for_csi:
            raise RuntimeError(
                "CSI processing thread started without previous listen call"
            )

        if not self.filepath:
            raise ApiUsageError(
                "Must call `set_filepath` before starting CSI reception on ESP32!"
            )

        with self.filepath.open("w", encoding="UTF-8") as filehandle:
            while self.listening_for_csi:
                try:
                    csi_packet = self.csi_queue.get(timeout=0.1)
                except queue.Empty:
                    continue

                try:
                    # decode the packet
                    # Format string explanation: "<Q 6s 6s H H b B B H"
                    #   `<`  little-endian order  (lsb first)
                    #   `Q`  `unsigned long long` (8 bytes) -> timestamp (us)
                    #   `6s` `string`             (6 bytes) -> MAC
                    #   `H`  `unsigned short`     (2 bytes) -> csi data len, seq
                    #   `b`  `signed char`        (1 byte)  -> rssi
                    #   `B`  `unsigned char       (1 byte)` -> agc/fft
                    (
                        _,
                        _,
                        timestamp,
                        source_mac,
                        destination_mac,
                        rx_seq,
                        rssi,
                        agc_gain,
                        fft_gain,
                        csi_length,
                    ) = struct.unpack(
                        "<Q h Q 6s 6s H b B B H",
                        csi_packet[: self.outer_header_length + self.csi_header_length],
                    )
                except struct.error:
                    logger.warning("Received malformed CSI packet from ESP32")
                    continue

                smac_str = ":".join(f"{b:02x}" for b in source_mac)
                dmac_str = ":".join(f"{b:02x}" for b in destination_mac)
                try:
                    csi_values = list(
                        struct.unpack(
                            f"<{csi_length}b",
                            csi_packet[
                                self.outer_header_length + self.csi_header_length :
                            ],
                        )
                    )
                except struct.error:
                    logger.warning("Received malformed CSI packet from ESP32")
                    continue

                formatted_string = f"{timestamp};{smac_str};{dmac_str};{rssi};{agc_gain};{fft_gain};{rx_seq};{csi_length};{csi_values}\n"
                filehandle.write(formatted_string)

    def _clear_input_buffer(self):
        """
        Tries to get rid of stale data from the ESP
        """
        logger.debug("resetting serial input buffer\n")
        if self.serial_connection:
            self.serial_connection.reset_input_buffer()
