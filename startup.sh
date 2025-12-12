#!/bin/bash

# Exit on error
set -e

echo "Starting Outbound Service..."

# Infinite loop to auto-restart if script crashes
while true
do
    echo "Running OutboundServiceEntry.py..."
    python ./outboundService/entry.py dev

    echo "Service crashed with exit code $? — restarting in 2 seconds..."
    sleep 2
done
