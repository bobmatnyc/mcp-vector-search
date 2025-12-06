tell application "Safari"
    -- Apply CSS fix to test
    do JavaScript "
        const legendEl = document.querySelector('.legend');
        if (legendEl) {
            legendEl.style.position = 'static';
            legendEl.style.marginTop = '12px';
            'Fix applied - legend position changed to static';
        } else {
            'Legend element not found';
        }
    " in current tab of window 1

    delay 1

    -- Verify the fix
    set result to do JavaScript "
        const lc = document.getElementById('layout-controls');
        const ef = document.getElementById('edge-filters');
        const lcRect = lc ? lc.getBoundingClientRect() : null;
        const efRect = ef ? ef.getBoundingClientRect() : null;

        'LayoutControls visible: ' + (lcRect && lcRect.height > 30) + ', ' +
        'EdgeFilters visible: ' + (efRect && efRect.height > 100)
    " in current tab of window 1

    return result
end tell
