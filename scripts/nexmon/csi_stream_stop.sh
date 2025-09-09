#!/bin/bash

#------------------------------------------------------------------------------
# Script to kill background processes on the Nexmon remote device, specifically
# tcpdump to capture CSI and netcat to forward captures to the host.
#
# NOTE:
# - You guessed it, 'asus' in `~/.ssh/config` must be set up for passwordless
#   access.
#------------------------------------------------------------------------------

ssh_hostname=$1

ssh "${ssh_hostname}" \
    "[ -z \"\$(pidof tcpdump)\" ] || kill -s SIGINT \$(pidof tcpdump)"

# NOTE: It used to be the case that netcat did not close automatically, causing
# leftover processes. In the current version, this is fixed, and netcat is closed
# when tcpdump is killed.
# kill -s SIGINT \`ps | grep -w nc | grep -v grep | awk '{print \$1}'\`"
