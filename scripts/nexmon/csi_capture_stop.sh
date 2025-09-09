#!/bin/bash

#-----------------------------------------------------------------------------
# Script to kill netcat instances running on the current device. Used to close
# background processes that accept CSI streams from remote Nexmon devices.
#-----------------------------------------------------------------------------

#ps -ef | grep -w nc | grep -v grep | awk '{print $1}' | xargs -r kill

num_retries=${1:-5}

exit_if_closed() {
    # If the process is already gone, nothing to do; we can exit.
    retval=$(pgrep -x netcat)
    if [[ -z "${retval}" ]]; then
        exit 0
    fi
}

# Give closing the process automatically a bit of time
for (( c=0; c<"${num_retries}"; c++ )); do
    exit_if_closed
    sleep 1
done


# If we arrive here in the control flow, the process failed to
# close automatically -- force it!
echo "Netcat not automatically closed; will force kill."
pkill -SIGINT --full --echo --count netcat
