#!/bin/sh
#----------------------------------------------------------------------------
# Script to create CSI params required for nexmon capture on the ASUS router
# with a bcm4366c0 chipset.
#
# NOTE:
#  - This script is expected to be executed on the router. Best to put it
#    inside /jffs/CSING with all the other scripts.
#----------------------------------------------------------------------------
set -eu

# Helper: run a command, exit and report if it fails
run_cmd() {
  # run the command
  "$@"
  status=$?
  if [ $status -ne 0 ]; then
    echo "Error: command '$*' failed with exit code $status" >&2
    exit $status
  fi
}

# Script arguments
interface=$1                  # interface to configure
channel_number=${2:-1}        # Number of the channel to tune to
channel_bw_mhz=${3:-20}       # Channel bandwidth
source_addr=${4:-}            # Sender source address to filter for
recv_antenna_mask=${5:-1}     # Mask of antenna to receive
spatial_stream_mask=${6:-1}   # Mask of spatial streams to receive
sideband=${7:-""}             # either u or l for 40 MHz 2.4 GHz channel
frame_start_byte=${8:-"0x88"} # QOS frames start with 0x88; only empty if explicitly set ""

if [ -z "$interface" ]; then
  echo "Missing interface, terminating" >&2
  exit 1
fi

echo "Setting up channel on router..."
echo " -- remote interface..: ${interface}"
echo " -- channel number....: ${channel_number}"
echo " -- channel bandwidth.: ${channel_bw_mhz} MHz"
echo " -- country code......: US"
echo " -- misc..............: ap=0, infra=0, PM=0, scansuppress=1, monitor=1, sideband=${sideband}"

# ----------------------------------------------------------------------
# Set up the interface ...
run_cmd wl -i "$interface" down

# Enable 40 MHz if needed
if [ "$interface" = "eth5" ]; then
  run_cmd wl -i eth5 bw_cap 2g 0x3
fi

run_cmd wl -i "$interface" country US
run_cmd wl -i "$interface" chanspec "${channel_number}/${channel_bw_mhz}${sideband}"
run_cmd wl -i "$interface" ap 0
run_cmd wl -i "$interface" infra 0
run_cmd wl -i "$interface" PM 0
run_cmd wl -i "$interface" scansuppress 1
run_cmd wl -i "$interface" monitor 1
run_cmd wl -i "$interface" up

run_cmd ifconfig "$interface" up

# ----------------------------------------------------------------------
# Build filter string only if needed
filter_str=""

[ -n "$frame_start_byte" ] && \
  filter_str="frame[0] == ${frame_start_byte}"

[ -n "$source_addr" ] && {
  if [ -n "$filter_str" ]; then
    filter_str="${filter_str} && addr2 == ${source_addr}"
  else
    filter_str="addr2 == ${source_addr}"
  fi
}

echo "Generating csiparams..."

# ----------------------------------------------------------------------
# Build argument list for makecsiparams, appending -f only if filter_str
set -- \
  -c "${channel_number}/${channel_bw_mhz}${sideband}" \
  -C "${recv_antenna_mask}" \
  -N "${spatial_stream_mask}"

[ -n "$filter_str" ] && \
  set -- "$@" -f "(${filter_str})"

# Run makecsiparams, exit if it fails
if ! csiparams=$(/jffs/CSING/makecsiparams "$@"); then
  echo "Error: makecsiparams failed with exit code $?" >&2
  exit 1
fi

echo "Configuring nexmon CSI extraction ..."
echo " -- csiparams...........: $csiparams"
echo " -- rx_chain_mask.......: $recv_antenna_mask"
echo " -- spatial_stream_mask.: $spatial_stream_mask"
echo " -- filter config.......: $filter_str"

# NOTE: Expansion of csiparams desired; do NOT quote!
# shellcheck disable=SC2086
run_cmd /jffs/CSING/nexutil -I"$interface" -s500 -b ${csiparams}
