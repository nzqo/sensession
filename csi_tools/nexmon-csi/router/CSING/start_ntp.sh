#!/bin/sh
host_ip=${1:-"192.168.10.1"}

ntpclient -h "${host_ip}" -i60 -s
