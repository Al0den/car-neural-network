#!/bin/bash

while true; do
    killall python

    osascript -e 'tell application "Terminal" to close every window'
    
    read -p "Cores: " user_input
    
    osascript -e "tell application \"Terminal\" to do script \"cd Desktop/projects/Car && echo \\\"2\n$user_input\n\\\" | python src/main.py\""
    
    sleep 2

    osascript -e 'tell application "Terminal" to tell window 1 to set size to {1000, 500}'

    sleep 3600
done