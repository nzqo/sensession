# Nexmon v2

This repository contains compiled versions of the new nexmon patch provided
by Francesco as well as a few (adapted) scripts for its operation.

## Initial Setup

Copy the directory `router/CSING` into `/jffs/CSING` on the router,
either manually or using `push_to_device.sh`

Upon rebooting the device, run **once**:

```bash
/jffs/CSING/reload.sh
```

## Device Setup

Run

```bash
/jffs/CSING/configure.sh   \
	<interface>            \
	<channel_number>       \
	<channel_bw_mhz>       \
	<source_addr>          \
	<recv_antenna_mask>    \
	<spatial_stream_mask>  \
	<sideband>             \
	<frame_start_byte>
```

Where

```log
<interface>           : The on-device interface to capture CSI with (eth5 or eth6)
<channel_number>      : Channel number to capture on          (default = 1)
<channel_bw_mhz>      : Channel bandwidth in MHz              (default = 20) 
<source_addr>         : A mac source adr to filter            (default = None)
<recv_antenna_mask>   : Bitmask of rx cores to receive        (default = 1 [first antenna only])
<spatial_stream_mask> : Bitmask of nss streams to receive     (default = 1 [first stream only])
<sideband>            : sideband for 2.4 GHz 40 Mhz (u or l)  (default = None)
<frame_start_byte>    : Frame start byte to filter for type   (default = 0x88)
```


## Usage

Afterwards, you can capture with tcpdump and directly stream to the host.
To do so, first open a netcat listener on the host:

```bash
netcat -Nlp"${netcat_port}" > "${capturefile}"
```

And start streaming the data broadcast on the device back to the host:

```bash
# Run from the host, otherwise leave out the ssh part
ssh "${ssh_hostname}" \
    "/jffs/CSING/tcpdump -i${interface} dst port 5500 -A -s0 -U -n -w- | \
    nc ${host_ip} ${netcat_port}"
```

Finally, you may use the `matlab/csireader.m` function to extract CSI
from the resulting `{capturefile}`.
