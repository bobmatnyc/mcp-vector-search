tell application "Safari"
    set result to do JavaScript "
        const lc = document.getElementById('layout-controls');
        const ls = document.getElementById('layoutSelector');
        'LayoutControls:' + (lc ? getComputedStyle(lc).display : 'null') + ' ' +
        'Selector:' + (ls ? getComputedStyle(ls).display : 'null') + ' ' +
        'SelectorHeight:' + (ls ? getComputedStyle(ls).height : 'null')
    " in current tab of window 1
    return result
end tell
