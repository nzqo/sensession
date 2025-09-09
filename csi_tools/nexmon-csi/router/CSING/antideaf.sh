#!/bin/sh

IFACE=$1
if [ "$IFACE" = "" ]; then
  echo "Missing interface"
  exit 1
fi

./nexutil -I ${IFACE} -s 502 -l 4 -i -v 0
