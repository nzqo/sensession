#!/bin/bash

# Patches WLan Toolbox installation to allow setting the sounding bit from
# within Matlab scripts.
#
# Generate diffs using:
# diff -u oldfile newfile
#

CURR_SCRIPT_PATH=$(dirname "$0")
MATLAB_PATH=${1-"/usr/local/MATLAB/R2025a"}

sudo patch -u -i "$CURR_SCRIPT_PATH"/patches/wlanHTSIG_patch "$MATLAB_PATH"/toolbox/wlan/wlan/wlanHTSIG.m
sudo patch -u -i "$CURR_SCRIPT_PATH"/patches/wlanHTConfig_patch "$MATLAB_PATH"/toolbox/wlan/wlan/wlanHTConfig.m
