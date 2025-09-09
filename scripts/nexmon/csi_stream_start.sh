#!/bin/bash

#------------------------------------------------------------------------------
# Start capturing ssh on remote Nexmon device and forward it to a host attached
# via ethernet.
#
# NOTE:
# - You guessed it, ssh_hostname in `~/.ssh/config` must be set up for passwordless
#   access.
#------------------------------------------------------------------------------

ssh_hostname=$1
interface=$2
netcat_port=${3:-5501}
host_ip=${4-"192.168.10.1"}

# Tunnel captured data from port 5500 (the CSI) as pcap format into netcat.
ssh "${ssh_hostname}" "/jffs/CSING/tcpdump -i${interface} dst port 5500 -A -s0 -U -n -w- | nc ${host_ip} ${netcat_port}"
