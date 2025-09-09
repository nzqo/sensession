#!/bin/bash
#------------------------------------------------------------------------------
# Script to set up the wireless interface on Nexmon device for monitoring the
# proper channel. Will make use of makecsiparams on device
#
# NOTE:
#   - Interface should be eth6 for 5 GHz and eth5 for 2.4 GHz channels
#   - Assumes passwordless access to host named `ssh_hostname` in `~/.ssh/config`
#   - Device must be set up with nexmon tools properly, i.e. contain scripts
#     and apps in /jffs/CSING
#------------------------------------------------------------------------------

# Argument specifying the ssh hostname of the router to connect to
ssh_hostname=$1


# Arguments forwarded to configure.sh on the router
interface=$2                  # interface to configure on the router
channel_number=$3             # Number of the channel to tune to
channel_bw_mhz=$4             # Channel bandwidth
source_addr=$5                # Sender source address to filter for
recv_antenna_mask=${6:-1}     # Mask of antenna to receive, e.g. 0b0100=4 for antenna 3
spatial_stream_mask=${7:-1}   # Mask of spatial streams to receive
sideband=${8:-""}             # either u or l for 40 MHz 2.4GHz channel


# Check whether device is reachable
ssh_status=$(ssh -o ConnectTimeout=5 "${ssh_hostname}" echo ok 2>&1)
if [[ ! $ssh_status == ok ]] ; then
    echo Can not connect to "$ssh_hostname", is access set up properly?
    exit 1
fi


# Call configure on the device
ssh "${ssh_hostname}" "/jffs/CSING/configure.sh \
    $interface \
    $channel_number \
    $channel_bw_mhz \
    $source_addr \
    $recv_antenna_mask \
    $spatial_stream_mask \
    $sideband"
