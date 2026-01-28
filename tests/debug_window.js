#!/usr/bin/env node
const { chromium } = require('playwright');

async function runTest() {
    const browser = await chromium.launch({ headless: true });
    const context = await browser.newContext();
    const page = await context.newPage();

    await page.goto('http://localhost:8765', { waitUntil: 'networkidle', timeout: 60000 });

    // Wait a bit for scripts to load
    await page.waitForTimeout(3000);

    // Check what variables are accessible
    const result = await page.evaluate(() => {
        return {
            // Check if variables exist
            hasCurrentVizMode: typeof currentVizMode !== 'undefined',
            hasSetVisualizationMode: typeof setVisualizationMode === 'function',

            // Check window
            windowCurrentVizMode: window.currentVizMode,

            // Try to access directly
            currentVizModeValue: typeof currentVizMode !== 'undefined' ? currentVizMode : 'UNDEFINED',

            // Check if it's in a closure or module
            hasAllNodes: typeof allNodes !== 'undefined'
        };
    });

    console.log('Variable accessibility check:');
    console.log(JSON.stringify(result, null, 2));

    await browser.close();
}

runTest().catch(console.error);
