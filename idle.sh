#!/bin/bash
read -p "Cores: " user_input

while true; do
    killall python

    osascript -e 'tell application "Terminal" to close every window'
    osascript -e "tell application \"Terminal\" to do script \"cd Desktop/projects/Car && echo \\\"2\n$user_input\n\\\" | python src/main.py\""
    
    sleep 2

    osascript -e 'tell application "Terminal" to tell window 1 to set size to {1200, 500}'

    sleep 1200
done