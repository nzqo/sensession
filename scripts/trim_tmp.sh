#!/bin/bash


# 10GB = 10737418240 bytes
max_size=${1:-10737418240}

directory="/tmp"

# NOTE: du throws some errors for /tmp directories it can't read.
# We ignore those by piping errors into /dev/null. What we want is
# on stdout anyway.
SIZE=$(du -sb "${directory}" 2>/dev/null | cut -f 1)    

if [[ $SIZE -gt $max_size ]]; then
    echo "Cleaning ${directory} of frame files"
    rm -f "${directory}/"frame_*
fi
