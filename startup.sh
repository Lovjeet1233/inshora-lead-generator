#!/bin/bash

# Start all services in parallel
python app.py &
python agent.py dev &
python ./outboundService/entry.py dev &

# Keep script running until all background jobs exit
wait
