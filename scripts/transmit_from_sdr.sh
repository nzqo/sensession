#!/bin/bash

#------------------------------------------------------------------------------
# Script to transmit waveform from a file containing samples with a USRP device
# n times.
#
# NOTE:
#  - Assumes passwordless access to PC attached to the SDR set up with hostname
#	 'sdr' in the ssh config `~/.ssh/config`
#  - Assumes this remote SDR host has custom built example that allows to send
# 	 a frame exactly n times (see README.md). May be simplified if/once our MR
# 	 is accepted.
#
# Script will sync the frame file to the remote machine and use that. One could
# improve it by providing a proper help message, but ...
#
# Usage:
# transmit_from_file.sh
# 	<path to sample file>
# 	[channel_freq_in_Hz]     (default: 5785000000)
#   [gain_in_db] 	         (default: 25)
#   [rate_in_samples_ps]     (default: 20000000)
#   [n_repeat]               (default: 10000)
#------------------------------------------------------------------------------
sample_file=$1
channel_frequency=${2:-5785000000} 
gain=${3:-25}
sample_rate=${4:-20000000}
n_repeat=${5:-10000}
delay_ms=${6:-10}
serial_id=${7:-""}


tx_command() {
	# NOTE: This is a custom built example that allows to transmit only n times
	# A PR for this is open and we can fall back to the package once merged.
	/home/seemoo/Development/tx_samples_from_file/build/tx_samples_from_file \
		--file "${sample_file}" \
		--freq "${channel_frequency}" \
		--gain "${gain}" \
		--rate "${sample_rate}" \
		--n-repeat "${n_repeat}" \
		--delay "${delay_ms}" \
		--args "${serial_id}"
}

tx_command
