#!/usr/bin/env node
/**
 * Manual test script for file path navigation in treemap/sunburst
 * Run with: node tests/manual_test_file_path_nav.js
 */

const { chromium } = require('playwright');

async function runTest() {
    const browser = await chromium.launch({
        headless: false, // Run headed for debugging
        slowMo: 50
    });

    const context = await browser.newContext();
    const page = await context.newPage();

    // Collect console logs
    const consoleLogs = [];
    page.on('console', msg => {
        consoleLogs.push({
            type: msg.type(),
            text: msg.text()
        });
    });

    console.log('='.repeat(60));
    console.log('FILE PATH NAVIGATION TEST - TREEMAP/SUNBURST');
    console.log('='.repeat(60));

    try {
        console.log('\n1. Navigating to http://localhost:8765...');
        await page.goto('http://localhost:8765', { waitUntil: 'networkidle', timeout: 60000 });
        console.log('   Page loaded successfully');

        // Wait for graph
        await page.waitForSelector('#graph', { timeout: 10000 });
        console.log('   Graph element found');

        // Take initial screenshot
        await page.screenshot({ path: 'test-screenshots/01_initial_load.png', fullPage: true });

        // Wait for loading to complete with multiple strategies
        console.log('\n2. Waiting for data to load (this may take a while for large datasets)...');

        // Wait for data processing to complete by checking for multiple treemap cells or tree nodes
        const maxWait = 180000; // 3 minutes max
        const startTime = Date.now();
        let dataLoaded = false;

        while (!dataLoaded && (Date.now() - startTime) < maxWait) {
            await page.waitForTimeout(3000);

            // Check for loading indicator dialog
            const loadingVisible = await page.locator('.loading-message').isVisible().catch(() => false);
            if (!loadingVisible) {
                // Check if we have actual content
                const treeCells = await page.locator('.treemap-cell').count();
                const treeNodes = await page.locator('.node circle').count();
                const sunburstArcs = await page.locator('#graph path').count();

                console.log(`   Checking content: treemap=${treeCells}, nodes=${treeNodes}, arcs=${sunburstArcs}`);

                if (treeCells > 1 || treeNodes > 5 || sunburstArcs > 5) {
                    dataLoaded = true;
                    console.log('   Data loaded successfully!');
                }
            } else {
                console.log('   Still loading...');
            }
        }

        if (!dataLoaded) {
            console.log('   WARNING: Data may not have fully loaded, continuing anyway...');
        }

        await page.screenshot({ path: 'test-screenshots/01b_after_loading.png', fullPage: true });

        // =====================================================================
        // TEST TREEMAP MODE
        // =====================================================================
        console.log('\n3. TESTING TREEMAP MODE');
        console.log('-'.repeat(40));

        // Switch to treemap
        const treemapBtn = await page.locator('button[data-mode="treemap"]');
        if (await treemapBtn.count() > 0) {
            await treemapBtn.click();
            console.log('   Clicked Treemap button');
            await page.waitForTimeout(2000);

            // Check if treemap is active
            const isActive = await treemapBtn.evaluate(el => el.classList.contains('active'));
            console.log(`   Treemap active: ${isActive}`);

            await page.screenshot({ path: 'test-screenshots/02_treemap_mode.png', fullPage: true });

            // Wait for loading after mode switch
            try {
                await page.waitForSelector('#graph-loading-indicator', { state: 'hidden', timeout: 30000 });
            } catch (e) {}

            // Find treemap cells
            const treemapCells = await page.locator('.treemap-cell').all();
            console.log(`   Found ${treemapCells.length} treemap cells`);

            if (treemapCells.length > 0) {
                // Click on cells to find one with content
                for (let i = 0; i < Math.min(10, treemapCells.length); i++) {
                    try {
                        await treemapCells[i].click({ force: true, timeout: 5000 });
                        await page.waitForTimeout(500);

                        // Check if viewer opened
                        const viewerOpen = await page.locator('#viewer-panel.open').count() > 0;
                        if (viewerOpen) {
                            console.log(`   Viewer opened after clicking cell ${i}`);
                            await page.screenshot({ path: 'test-screenshots/03_viewer_opened.png', fullPage: true });

                            // Look for clickable file path
                            const filePaths = await page.locator('.viewer-info-value.clickable').all();
                            console.log(`   Found ${filePaths.length} clickable elements in viewer`);

                            if (filePaths.length > 0) {
                                const filePathText = await filePaths[0].textContent();
                                console.log(`   File path text: ${filePathText}`);

                                // Clear console logs to see fresh ones
                                consoleLogs.length = 0;

                                // Click on file path
                                console.log('\n   CLICKING ON FILE PATH...');
                                await filePaths[0].click();
                                await page.waitForTimeout(1000);

                                // Check results
                                console.log('\n   RESULTS:');
                                const viewerStillOpen = await page.locator('#viewer-panel.open').count() > 0;
                                console.log(`   - Viewer still open: ${viewerStillOpen}`);

                                // Check for zoom logs
                                const zoomLogs = consoleLogs.filter(l => l.text.includes('Zooming to parent'));
                                const navLogs = consoleLogs.filter(l => l.text.includes('Navigating') || l.text.includes('Focusing'));
                                console.log(`   - Zoom logs: ${zoomLogs.length} messages`);
                                console.log(`   - Navigation logs: ${navLogs.length} messages`);

                                if (zoomLogs.length > 0) {
                                    zoomLogs.forEach(l => console.log(`     > ${l.text}`));
                                }
                                if (navLogs.length > 0) {
                                    navLogs.forEach(l => console.log(`     > ${l.text}`));
                                }

                                await page.screenshot({ path: 'test-screenshots/04_after_file_path_click.png', fullPage: true });

                                break;
                            }
                        }
                    } catch (e) {
                        // Continue to next cell
                    }
                }
            }
        } else {
            console.log('   ERROR: Treemap button not found');
        }

        // =====================================================================
        // TEST SUNBURST MODE
        // =====================================================================
        console.log('\n4. TESTING SUNBURST MODE');
        console.log('-'.repeat(40));

        // Switch to sunburst
        const sunburstBtn = await page.locator('button[data-mode="sunburst"]');
        if (await sunburstBtn.count() > 0) {
            await sunburstBtn.click();
            console.log('   Clicked Sunburst button');
            await page.waitForTimeout(2000);

            // Check if sunburst is active
            const isActive = await sunburstBtn.evaluate(el => el.classList.contains('active'));
            console.log(`   Sunburst active: ${isActive}`);

            await page.screenshot({ path: 'test-screenshots/05_sunburst_mode.png', fullPage: true });

            // Wait for loading after mode switch
            try {
                await page.waitForSelector('#graph-loading-indicator', { state: 'hidden', timeout: 30000 });
            } catch (e) {}

            // Find sunburst arcs (they're path elements inside the SVG)
            const arcs = await page.locator('#graph path').all();
            console.log(`   Found ${arcs.length} path elements`);

            if (arcs.length > 0) {
                // Click on arcs to find one with content
                for (let i = 0; i < Math.min(15, arcs.length); i++) {
                    try {
                        await arcs[i].click({ force: true, timeout: 5000 });
                        await page.waitForTimeout(500);

                        // Check if viewer opened
                        const viewerOpen = await page.locator('#viewer-panel.open').count() > 0;
                        if (viewerOpen) {
                            console.log(`   Viewer opened after clicking arc ${i}`);

                            // Look for clickable file path
                            const filePaths = await page.locator('.viewer-info-value.clickable').all();
                            console.log(`   Found ${filePaths.length} clickable elements in viewer`);

                            if (filePaths.length > 0) {
                                const filePathText = await filePaths[0].textContent();
                                console.log(`   File path text: ${filePathText}`);

                                // Clear console logs
                                consoleLogs.length = 0;

                                // Click on file path
                                console.log('\n   CLICKING ON FILE PATH...');
                                await filePaths[0].click();
                                await page.waitForTimeout(1000);

                                // Check results
                                console.log('\n   RESULTS:');
                                const viewerStillOpen = await page.locator('#viewer-panel.open').count() > 0;
                                console.log(`   - Viewer still open: ${viewerStillOpen}`);

                                // Check for zoom logs
                                const zoomLogs = consoleLogs.filter(l => l.text.includes('Zooming to parent'));
                                const navLogs = consoleLogs.filter(l => l.text.includes('Navigating') || l.text.includes('Focusing'));
                                console.log(`   - Zoom logs: ${zoomLogs.length} messages`);
                                console.log(`   - Navigation logs: ${navLogs.length} messages`);

                                if (zoomLogs.length > 0) {
                                    zoomLogs.forEach(l => console.log(`     > ${l.text}`));
                                }
                                if (navLogs.length > 0) {
                                    navLogs.forEach(l => console.log(`     > ${l.text}`));
                                }

                                await page.screenshot({ path: 'test-screenshots/06_sunburst_after_click.png', fullPage: true });

                                break;
                            }
                        }
                    } catch (e) {
                        // Continue to next arc
                    }
                }
            }
        } else {
            console.log('   ERROR: Sunburst button not found');
        }

        // =====================================================================
        // TEST CLOSE BUTTON
        // =====================================================================
        console.log('\n5. TESTING CLOSE BUTTON');
        console.log('-'.repeat(40));

        // Go back to treemap
        await page.click('button[data-mode="treemap"]');
        await page.waitForTimeout(1000);
        try {
            await page.waitForSelector('#graph-loading-indicator', { state: 'hidden', timeout: 30000 });
        } catch (e) {}

        const cells = await page.locator('.treemap-cell').all();
        if (cells.length > 0) {
            await cells[0].click({ force: true });
            await page.waitForTimeout(500);

            const viewerOpen = await page.locator('#viewer-panel.open').count() > 0;
            console.log(`   Viewer open: ${viewerOpen}`);

            if (viewerOpen) {
                // Click close button
                const closeBtn = await page.locator('.viewer-close-btn');
                if (await closeBtn.count() > 0) {
                    console.log('   Clicking close button...');
                    await closeBtn.click();
                    await page.waitForTimeout(500);

                    const viewerStillOpen = await page.locator('#viewer-panel.open').count() > 0;
                    console.log(`   Viewer still open after close: ${viewerStillOpen}`);

                    if (viewerStillOpen) {
                        console.log('   FAIL: Close button did not close the viewer');
                    } else {
                        console.log('   PASS: Close button works correctly');
                    }

                    await page.screenshot({ path: 'test-screenshots/07_after_close_button.png', fullPage: true });
                }
            }
        }

        // Summary
        console.log('\n' + '='.repeat(60));
        console.log('TEST SUMMARY');
        console.log('='.repeat(60));
        console.log('Screenshots saved in: test-screenshots/');
        console.log('Console logs collected: ' + consoleLogs.length);

        // Print relevant console logs
        const relevantLogs = consoleLogs.filter(l =>
            l.text.includes('Zoom') ||
            l.text.includes('Navigate') ||
            l.text.includes('Focus') ||
            l.text.includes('parent') ||
            l.text.includes('directory')
        );

        if (relevantLogs.length > 0) {
            console.log('\nRelevant console logs:');
            relevantLogs.forEach(l => console.log(`  [${l.type}] ${l.text}`));
        }

    } catch (error) {
        console.error('Test error:', error);
        await page.screenshot({ path: 'test-screenshots/error.png', fullPage: true });
    } finally {
        await browser.close();
    }
}

runTest().catch(console.error);
