#!/bin/bash

# Open Safari and run JavaScript to check control visibility
osascript <<EOF
tell application "Safari"
    activate

    -- Open the URL if no windows are open
    if (count of windows) is 0 then
        make new document with properties {URL:"http://localhost:8082"}
    else
        set URL of current tab of window 1 to "http://localhost:8082"
    end if

    delay 8

    -- Execute JavaScript to check control states
    set js to "
        const results = {
            layoutControls: {},
            edgeFilters: {},
            legend: {},
            loading: {}
        };

        const layoutEl = document.getElementById('layout-controls');
        if (layoutEl) {
            const styles = window.getComputedStyle(layoutEl);
            results.layoutControls = {
                exists: true,
                display: styles.display,
                visibility: styles.visibility,
                opacity: styles.opacity,
                position: styles.position,
                zIndex: styles.zIndex,
                top: styles.top,
                left: styles.left,
                width: styles.width,
                height: styles.height
            };
        }

        const edgeEl = document.getElementById('edge-filters');
        if (edgeEl) {
            const styles = window.getComputedStyle(edgeEl);
            results.edgeFilters = {
                exists: true,
                display: styles.display,
                visibility: styles.visibility,
                opacity: styles.opacity,
                position: styles.position,
                zIndex: styles.zIndex,
                top: styles.top,
                left: styles.left,
                width: styles.width,
                height: styles.height
            };
        }

        const legendEl = document.getElementById('legend');
        if (legendEl) {
            const styles = window.getComputedStyle(legendEl);
            results.legend = {
                exists: true,
                display: styles.display,
                visibility: styles.visibility,
                zIndex: styles.zIndex
            };
        }

        const loadingEl = document.getElementById('loading');
        if (loadingEl) {
            const styles = window.getComputedStyle(loadingEl);
            results.loading = {
                exists: true,
                display: styles.display,
                innerHTML: loadingEl.innerHTML.substring(0, 100)
            };
        }

        JSON.stringify(results, null, 2);
    "

    set result to do JavaScript js in current tab of window 1
    return result
end tell
EOF
