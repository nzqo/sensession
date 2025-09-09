#!/bin/bash

#----------------------------------------------------------------------------
# Script to open up a local netcat socket meant to receive data from a remote
# device running a nexmon firmware patch and store it in a file.
#----------------------------------------------------------------------------

netcat_port=${1:-5501}
capturefile=${2:-"./captures/asus.dat"}

# Open local netcat socket.
echo "Opening netcat socket on ${netcat_port}. Storing received results in ${capturefile}."
netcat -Nlp"${netcat_port}" > "${capturefile}"
