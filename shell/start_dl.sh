#!/bin/bash
source /etc/network_turbo >/dev/null 2>&1
exec /root/miniconda3/bin/python /root/dl_encoders.py
