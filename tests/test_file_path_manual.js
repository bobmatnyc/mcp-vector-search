// Manual test script to verify file path navigation fix
// Run with: npx playwright test tests/test_file_path_manual.js --reporter=line

const { test, expect } = require('@playwright/test');

test.describe('File Path Navigation - Manual JavaScript Execution', () => {
    test.beforeEach(async ({ page }) => {
        await page.goto('http://localhost:8765');
        // Wait for data to load
        await page.waitForFunction(() => {
            return window.allNodes && window.allNodes.length > 0;
        }, { timeout: 60000 });
        console.log('Data loaded');
    });

    test('verify navigateToFileByPath function exists and works in treemap mode', async ({ page }) => {
        // Capture console logs
        const consoleLogs = [];
        page.on('console', msg => {
            consoleLogs.push({ type: msg.type(), text: msg.text() });
        });

        // Switch to treemap mode using JavaScript
        const result = await page.evaluate(() => {
            // Switch to treemap
            if (typeof setVisualizationMode === 'function') {
                setVisualizationMode('treemap');
            } else {
                // Click button manually
                document.querySelector('button[data-mode="treemap"]')?.click();
            }

            // Wait a moment for mode change
            return new Promise(resolve => {
                setTimeout(() => {
                    resolve({
                        mode: window.currentVizMode,
                        hasNodes: window.allNodes?.length || 0
                    });
                }, 1000);
            });
        });

        console.log('Mode switch result:', result);
        expect(result.mode).toBe('treemap');

        // Find a node with a file_path to test with
        const nodeInfo = await page.evaluate(() => {
            // Find a chunk node with file_path
            const chunkNode = window.allNodes.find(n =>
                (n.type === 'function' || n.type === 'class' || n.type === 'chunk') &&
                n.file_path
            );

            if (!chunkNode) {
                return { found: false };
            }

            return {
                found: true,
                id: chunkNode.id,
                name: chunkNode.name,
                type: chunkNode.type,
                filePath: chunkNode.file_path
            };
        });

        console.log('Found test node:', nodeInfo);

        if (!nodeInfo.found) {
            console.log('No chunk with file_path found - skipping file path navigation test');
            return;
        }

        // Open the viewer by clicking on the chunk
        await page.evaluate((nodeId) => {
            const node = window.allNodes.find(n => n.id === nodeId);
            if (node && typeof showCodeChunkDetail === 'function') {
                showCodeChunkDetail(node);
            }
        }, nodeInfo.id);

        await page.waitForTimeout(500);

        // Check if viewer is open
        const viewerOpen = await page.evaluate(() => {
            const panel = document.getElementById('viewer-panel');
            return panel?.classList.contains('open');
        });
        console.log('Viewer panel is open:', viewerOpen);
        expect(viewerOpen).toBe(true);

        // Check if clickable file path exists
        const hasClickablePath = await page.evaluate(() => {
            return document.querySelector('.viewer-info-value.clickable') !== null;
        });
        console.log('Has clickable file path:', hasClickablePath);
        expect(hasClickablePath).toBe(true);

        // Now test the navigateToFileByPath function directly
        consoleLogs.length = 0; // Clear logs

        const navResult = await page.evaluate((filePath) => {
            // Store initial state
            const initialViewerOpen = document.getElementById('viewer-panel')?.classList.contains('open');
            const initialZoomRoot = window.currentZoomRootId;

            // Call the navigation function
            if (typeof navigateToFileByPath === 'function') {
                navigateToFileByPath(filePath);
            } else {
                return { error: 'navigateToFileByPath function not found' };
            }

            // Wait a moment and check state
            return new Promise(resolve => {
                setTimeout(() => {
                    const finalViewerOpen = document.getElementById('viewer-panel')?.classList.contains('open');
                    const finalZoomRoot = window.currentZoomRootId;

                    resolve({
                        initialViewerOpen,
                        finalViewerOpen,
                        viewerClosed: initialViewerOpen && !finalViewerOpen,
                        initialZoomRoot,
                        finalZoomRoot,
                        zoomChanged: initialZoomRoot !== finalZoomRoot
                    });
                }, 500);
            });
        }, nodeInfo.filePath);

        console.log('Navigation result:', navResult);

        // Print console logs captured during navigation
        const zoomLogs = consoleLogs.filter(l =>
            l.text.includes('Zooming') ||
            l.text.includes('parent directory') ||
            l.text.includes('navigate')
        );
        console.log('Relevant console logs:', zoomLogs);

        // Verify expected behavior
        expect(navResult.viewerClosed).toBe(true);
        console.log('Viewer closed correctly:', navResult.viewerClosed);
    });

    test('verify close button functionality', async ({ page }) => {
        // Switch to treemap mode
        await page.evaluate(() => {
            document.querySelector('button[data-mode="treemap"]')?.click();
        });
        await page.waitForTimeout(1000);

        // Find and click on a chunk to open viewer
        await page.evaluate(() => {
            const chunkNode = window.allNodes.find(n =>
                (n.type === 'function' || n.type === 'class' || n.type === 'chunk')
            );
            if (chunkNode && typeof showCodeChunkDetail === 'function') {
                showCodeChunkDetail(chunkNode);
            }
        });
        await page.waitForTimeout(500);

        // Verify viewer is open
        const viewerOpenBefore = await page.evaluate(() => {
            return document.getElementById('viewer-panel')?.classList.contains('open');
        });
        console.log('Viewer open before close button click:', viewerOpenBefore);
        expect(viewerOpenBefore).toBe(true);

        // Click close button using JavaScript
        await page.evaluate(() => {
            const closeBtn = document.querySelector('.viewer-close-btn');
            if (closeBtn) {
                closeBtn.click();
            }
        });
        await page.waitForTimeout(300);

        // Verify viewer is closed
        const viewerOpenAfter = await page.evaluate(() => {
            return document.getElementById('viewer-panel')?.classList.contains('open');
        });
        console.log('Viewer open after close button click:', viewerOpenAfter);
        expect(viewerOpenAfter).toBe(false);
        console.log('Close button works correctly');
    });

    test('verify sunburst mode file path navigation', async ({ page }) => {
        const consoleLogs = [];
        page.on('console', msg => {
            consoleLogs.push({ type: msg.type(), text: msg.text() });
        });

        // Switch to sunburst mode
        await page.evaluate(() => {
            document.querySelector('button[data-mode="sunburst"]')?.click();
        });
        await page.waitForTimeout(1000);

        // Verify mode
        const mode = await page.evaluate(() => window.currentVizMode);
        console.log('Current mode:', mode);
        expect(mode).toBe('sunburst');

        // Find a node with file_path
        const nodeInfo = await page.evaluate(() => {
            const chunkNode = window.allNodes.find(n =>
                (n.type === 'function' || n.type === 'class' || n.type === 'chunk') &&
                n.file_path
            );
            return chunkNode ? { id: chunkNode.id, filePath: chunkNode.file_path } : null;
        });

        if (!nodeInfo) {
            console.log('No chunk with file_path found - skipping');
            return;
        }

        // Open viewer
        await page.evaluate((nodeId) => {
            const node = window.allNodes.find(n => n.id === nodeId);
            if (node) showCodeChunkDetail(node);
        }, nodeInfo.id);
        await page.waitForTimeout(500);

        // Test navigation
        consoleLogs.length = 0;

        const result = await page.evaluate((filePath) => {
            const initialOpen = document.getElementById('viewer-panel')?.classList.contains('open');
            navigateToFileByPath(filePath);

            return new Promise(resolve => {
                setTimeout(() => {
                    const finalOpen = document.getElementById('viewer-panel')?.classList.contains('open');
                    resolve({
                        viewerClosedCorrectly: initialOpen && !finalOpen
                    });
                }, 500);
            });
        }, nodeInfo.filePath);

        const zoomLogs = consoleLogs.filter(l => l.text.includes('Zooming') || l.text.includes('parent'));
        console.log('Console logs during navigation:', zoomLogs);
        console.log('Result:', result);

        expect(result.viewerClosedCorrectly).toBe(true);
    });
});
