"""
ESP32 sensing proof of concept
"""

import time
from pathlib import Path

import polars as pl
from loguru import logger

from sensession import Channel, MacAddr, Database, Bandwidth
from sensession.config import BaseFrameConfig, BaseTransmissionConfig
from sensession.tools.espion import ESP32Tool
from sensession.devices.esp32 import ESP32, ESP32Config, ESPOperationMode

if __name__ == "__main__":
    # example on linux: (CDC)
    # esp1_cfg = ESP32Config(name="ESP32C6", comport="/dev/ttyACM0")
    # example on linux: (UART)
    # esp1_cfg = ESP32Config(name="ESP32C6", comport="/dev/ttyUSB0", baudrate = 921600)
    # example on windows (CDC or UART)
    # esp1_cfg = ESP32Config(name="ESP32C6", comport="/dev/ttyACM0")
    # example on macos (CDC)
    esp1_cfg = ESP32Config(
        name="ESP1", short_name="ESP1", comport="/dev/cu.usbmodem21201"
    )
    # example on macos (UART)
    espTX_cfg = ESP32Config(
        name="ESPTX",
        short_name="ESP2",
        comport="/dev/tty.usbserial-0001",
        mode=ESPOperationMode.TX,
        baudrate=921600,
    )

    channel = Channel(
        number=6,
        bandwidth=Bandwidth(20),
    )

    # --------------------------------------------------------------------------------

    # Create devices
    esp1 = ESP32(esp1_cfg)
    espTX = ESP32(espTX_cfg)

    esp_tool = ESP32Tool()
    esp1_id = esp_tool.add_device(esp1)
    espTX_id = esp_tool.add_device(espTX)

    # note: everything but the rx and tx addresses is ignored.
    frame = BaseFrameConfig(
        receiver_address=MacAddr("11:11:11:11:11:11"),
        transmitter_address=MacAddr("22:22:22:22:22:22"),
    )

    # Setup devices for rx (via tool)
    esp_tool.setup_capture(
        [esp1_id],
        channel,
        cache_dir=Path.cwd() / ".cache" / "ESP32cache",
        filter_frame=frame,
    )

    # Setup device for tx
    espTX.connect_device()
    espTX.change_channel(channel.number)
    tx_config = BaseTransmissionConfig(n_reps=1000, pause_ms=10)

    # Start capture
    logger.debug("Starting capture with ESP32 device(s)")
    esp_tool.run()
    time.sleep(1)

    # Start transmission
    logger.debug("Starting to transmit n times on ESP32 device")
    espTX.transmit_frame_with_tx_config(frame, tx_config)

    logger.debug("Waiting (10s)")
    time.sleep(10)

    # Stop capture
    time.sleep(1)
    esp_tool.stop()

    # Disconnect TX device
    espTX.close_serial_connection()

    # Collect
    res = esp_tool.reap()

    with Database("data/ESP32test", append=False) as db:
        for r in res:
            db.add_data(r, session_id=("session_1", pl.String))
