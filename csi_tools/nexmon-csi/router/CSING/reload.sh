#!/bin/sh

SLEEP="usleep 200000"
THISFOLDER="/jffs/CSING"

if [ -f "/tmp/dhd.ko" ]; then
  echo "Using module in /tmp"
  THISFOLDER="/tmp"
elif [ ! -f "$THISFOLDER/dhd.ko" ]; then
  echo "Missing 2.4/5GHz firmware, terminating"
  exit 1
fi

echo "Removing 2.4/5GHz modules"
/sbin/rmmod dhd
$SLEEP

echo "Reinserting 2.4/5GHz module"
/sbin/insmod ${THISFOLDER}/dhd.ko

echo "Removing interfaces from bridge"
/bin/brctl delif br0 eth5
/bin/brctl delif br0 eth6
