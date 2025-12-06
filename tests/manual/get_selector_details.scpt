tell application "Safari"
    set result to do JavaScript "
        const ls = document.getElementById('layoutSelector');
        if (!ls) { return 'Selector not found'; }
        const computed = getComputedStyle(ls);
        const rect = ls.getBoundingClientRect();
        JSON.stringify({
            display: computed.display,
            visibility: computed.visibility,
            height: computed.height,
            width: computed.width,
            overflow: computed.overflow,
            rect: {
                width: Math.round(rect.width),
                height: Math.round(rect.height),
                x: Math.round(rect.x),
                y: Math.round(rect.y)
            },
            optionCount: ls.options ? ls.options.length : 0
        }, null, 2);
    " in current tab of window 1
    return result
end tell
