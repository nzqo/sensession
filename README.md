# Sensession

Code and data for "Same Signal, Different Story: Demystifying Receiver Effects in Wi-Fi Channel State Information".
Data and computation results can be found [here](https://doi.org/10.4121/48e1331b-91df-4e1a-906b-b0a0e9d16c74).

A completely over-engineered collection of scripts to run CSI capture experiments
with receivers operated by multiple tools. Supports:

- Multiple tools (PicoScenes, Atheros QCA Tool, Nexmon-CSI, ESP32)
- Device-level operation
- Low level shell scripts for common operations (e.g. putting devices into monitor mode)
- WiFi IQ-waveform generation and subsequent USRP transmission
- Frame caching
- Conversion into a shared unified data format (parquet Dataframes)
- Common CSI processing methods
- A helper submodule for long-running experiments (Campaigns)
- Campaign (de-)serialization, allowing to rerun failed parts of campaigns
- Training scripts to replicate our ML experiments on four models under `har-train-scripts`

## Initial Setup

Before being able to use the things in this repo, some things have to be set up.
Install what you aim to use: No need to go through Nexmon installation if you only
plan on using ESP32s, for example.

### CSI Tools

Install the underlying tools. This framework only instruments them. Some of them are
included as submodule. Links to the relevant pages:

- [PicoScenes](https://ps.zpj.io/)
- [PicoScenes Python Toolbox](https://github.com/wifisensing/PicoScenes-Python-Toolbox.git)
- [ath9k and iwl driver modules](https://github.com/nzqo/csi-go.git)
- [espion](https://github.com/benjamin-kl/ESPion.git)
- [sensei](https://github.com/nzqo/sensei.git)
- [usrpulse](https://github.com/nzqo/usrpulse)
- [Nexmon](https://github.com/seemoo-lab/nexmon.git)

Some installation hints:

1. To start, first initialize all submodules

    ```bash
    git submodule update --init --recursive
    ```

#### Picoscenes

1. Don't forget to run `switch5300Firmware csi`
1. Don't forget to build Picoscenes Python Toolbox
1. If you have PicoScenes license, put it in `csi_tools/picoscenes-key.txt`
1. Check `sudo iw dev` for wdev interfaces and delete if present (confuses PicoScenes)
1. For now, you need to manually set the monitor mode using `array_prepare_for_picoscenes`.
With recent updates, the basic monitor script doesn't do it for the tool anymore.
1. Install Picoscenes Toolbox `cd csi_tools/PicoscenesToolbox && pip install -e .`

#### Nexmon

Nexmon is a firmware patching framework for broadcom chips. We use it to build a patched
firmware that allows to extract CSI from e.g. rt-ac86u asus routers. To set up such a
router from scratch first:

1. Start by resetting the router (using the hard reset button)
1. Possibly update firmware
1. Go through the basic config via the web interface (e.g. at `192.168.0.1`)
1. Put the router in AP mode
1. Enable ssh access through the web interface
1. Give the router a static IP (e.g. `192.168.10.31`)
1. Make sure interface on host has compatible IP (with the subnet) assigned. You can
  set it manually in the settings or on the command line, e.g.:

  ```bash
  sudo ip addr add 192.168.10.1/24 dev eno1
  ```

We have moved to a new unreleased version of nexmon. We don't build it manually,
but use a precompiled version shared with us that is found inside `csi_tools/nexmon-csi`.
To set it up:

1. Deploy the `router` directory onto router(s)
1. Log onto routers, reload dhd and start ntp `/jffs/CSING/reload.sh && /jffs/CSING/start_ntp.sh`

General notes for using the routers:

1. Asus routers must be power-cycled whenever you want to reload the kernel module
1. Never store captures in `/jffs`, since exceeding its `36 MB` will brick the device

#### MATLAB

1. Install Matlab together with WLan Toolbox into `matlab_path`
1. Patch Toolbox using `./matlab/patch_wlan_toolbox.sh $matlab_path`
1. Ensure you have a venv active; if not: `python venv .venv && source .venv/bin/activate` 
1. Install python engine `cd $matlab_path/extern/engines/python && pip install .`

#### ESPion

1. Install `esp-idf` framework
1. Make sure to use only the `ESP32-S3` model for now. We found others to yield corrupted data occasionally.
1. Install the appropriate `esp-idf` toolchain
1. Build and flash the `espion/csilogger` firmware
1. Give yourself rights to access the serial ports:

```bash
sudo usermod -a -G tty $USER
sudo usermod -a -G dialout $USER
```

1. You might need to `sudo apt remove brltty`. Odd, but happens.

#### Usrpulse

[Usrpulse](https://github.com/nzqo/usrpulse) is a command server to handle simple
frame transmission on the SDR. Install it on the machine which has the device attached
and make sure its running (and is available within the network).

#### Sensei

Sensei, similarly to sensession, abstracts collection from a few different sources,
but efficiently in rust. We use it to capture data from the iwl5300 and nexmon devices
via the available python binding.

Installation is simple: Make sure you have rust installed, then:

```bash
# make sure you are in your venv, then:
cd ./csi_tools/sensei/py_binding
pip install .
```

### Framework config

1. Populate `sense_config.toml` with global settings (see `EnvironmentConfig` in `/src/sensession/config.py` for options)
1. Set up passwordless SSH connections to remote devices in `~/.ssh/config`, e.g.

    ```config
    Host sdr
      HostName <ip-address>
      PreferredAuthentications publickey
      IdentityFile ~/.ssh/id_rsa
      User <username>
    ```

    Obviously you will need a key and install that on the remote using `ssh-copy-id`.

## Running the experiment

### lib

The tools can be used programmatically. See the `examples` directory
for a few examples or `experiments` for the experiment campaigns.

## General Information

### QCA: Sounding Bit and address

The QCA NIC can not be used to extract CSI when the sounding bit in
the phy preamble of the wifi frame is not set. Therefore, this card
requires a separately crafted frame to be sent with the SDR compared
to the others.

When in monitor mode, the qca will only report CSI from frames with
its actual MAC address set in the MAC header.

### iwl5300: Magic MAC

Probably as part of the firmware, the iwl5300 is not able to extract
CSI in monitor mode from any frames not addressed to the weird magic
MAC address `00:16:ea:12:34:56`. We provide a dedicated frame config
`InterleavedIQFrameGroupConfig` to have alternating frames addressed
to the iwl and the other cards.

## Evaluation

Our evaluation scripts can be found under `./evaluation`. They are to
be run as modules within the environment where sensession is installed.
For example:

```bash
python -m evaluation.tof.compute --help
```
