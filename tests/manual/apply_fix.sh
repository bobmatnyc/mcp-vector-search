#!/bin/bash

# Apply the CSS fix
osascript -e "tell application \"Safari\" to do JavaScript \"document.querySelector('.legend').style.position = 'static'; document.querySelector('.legend').style.marginTop = '12px'; 'Fixed'\" in current tab of window 1"

sleep 1

# Take screenshot
screencapture -x -R 0,0,400,900 "/Users/masa/Projects/mcp-vector-search/.mcp-vector-search/visualization/after-fix-screenshot.png"

echo "Screenshot saved to .mcp-vector-search/visualization/after-fix-screenshot.png"
