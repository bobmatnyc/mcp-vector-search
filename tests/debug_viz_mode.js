#!/usr/bin/env node
const { chromium } = require('playwright');

async function runTest() {
    const browser = await chromium.launch({
        headless: true,
        slowMo: 50
    });

    const context = await browser.newContext();
    const page = await context.newPage();

    const consoleLogs = [];
    page.on('console', msg => {
        consoleLogs.push({
            type: msg.type(),
            text: msg.text()
        });
    });

    console.log('='.repeat(60));
    console.log('DEBUG: Checking currentVizMode during navigation');
    console.log('='.repeat(60));

    try {
        await page.goto('http://localhost:8765', { waitUntil: 'networkidle', timeout: 60000 });
        console.log('Page loaded');

        // Wait for data loading
        const maxWait = 60000;
        const startTime = Date.now();
        let dataLoaded = false;

        while (!dataLoaded && (Date.now() - startTime) < maxWait) {
            await page.waitForTimeout(2000);
            const treeCells = await page.locator('.treemap-cell').count();
            const treeNodes = await page.locator('.node circle').count();
            const sunburstArcs = await page.locator('#graph path').count();
            console.log(`Checking content: treemap=${treeCells}, nodes=${treeNodes}, arcs=${sunburstArcs}`);

            if (treeCells > 1 || treeNodes > 5 || sunburstArcs > 5) {
                dataLoaded = true;
                console.log('Data loaded!');
            }
        }

        // Check initial mode
        const initialMode = await page.evaluate(() => window.currentVizMode);
        console.log(`\nInitial currentVizMode: "${initialMode}"`);

        // Switch to treemap
        console.log('\nClicking treemap button...');
        await page.click('button[data-mode="treemap"]');
        await page.waitForTimeout(2000);

        const modeAfterClick = await page.evaluate(() => window.currentVizMode);
        console.log(`currentVizMode after treemap click: "${modeAfterClick}"`);

        // Open a viewer to test navigation
        const cells = await page.locator('.treemap-cell').all();
        console.log(`Found ${cells.length} treemap cells`);

        if (cells.length > 0) {
            // Click to open viewer
            await cells[0].click({ force: true });
            await page.waitForTimeout(500);

            const viewerOpen = await page.locator('#viewer-panel.open').count() > 0;
            console.log(`Viewer is open: ${viewerOpen}`);

            // Check mode when about to call navigateToFileByPath
            consoleLogs.length = 0;

            const testResult = await page.evaluate(() => {
                // Check mode at this exact moment
                const modeNow = window.currentVizMode;
                const check = modeNow === 'treemap' || modeNow === 'sunburst';

                console.log('INSIDE EVALUATE - currentVizMode: "' + modeNow + '"');
                console.log('INSIDE EVALUATE - treemap/sunburst check: ' + check);

                // Find a file path to test with
                const chunk = window.allNodes.find(n => n.file_path && (n.type === 'function' || n.type === 'class'));
                if (chunk) {
                    console.log('CALLING navigateToFileByPath with: ' + chunk.file_path);
                    navigateToFileByPath(chunk.file_path);
                }

                return {
                    mode: modeNow,
                    check: check
                };
            });

            console.log('\nEvaluation result:', testResult);

            // Print console logs from the page
            console.log('\nPage console logs:');
            consoleLogs.forEach(l => {
                if (l.text.includes('INSIDE') || l.text.includes('CALLING') ||
                    l.text.includes('Navigating') || l.text.includes('Zooming') ||
                    l.text.includes('Focusing') || l.text.includes('currentVizMode')) {
                    console.log(`  [${l.type}] ${l.text}`);
                }
            });
        }

    } catch (error) {
        console.error('Error:', error.message);
    } finally {
        await browser.close();
    }
}

runTest().catch(console.error);
