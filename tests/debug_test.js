const { chromium } = require('playwright');

async function runTest() {
    const browser = await chromium.launch({ headless: true });
    const context = await browser.newContext();
    const page = await context.newPage();

    const consoleLogs = [];
    page.on('console', msg => {
        consoleLogs.push(msg.text());
    });

    await page.goto('http://localhost:8765');

    // Wait for data
    await page.waitForFunction(() => window.allNodes && window.allNodes.length > 0, { timeout: 60000 });

    // Switch to treemap
    await page.evaluate(() => {
        document.querySelector('button[data-mode="treemap"]').click();
    });
    await page.waitForTimeout(1000);

    // Check the mode
    const mode1 = await page.evaluate(() => window.currentVizMode);
    console.log('currentVizMode after clicking treemap:', mode1);

    // Find a chunk with file_path
    const testPath = await page.evaluate(() => {
        const chunk = window.allNodes.find(n => n.file_path && (n.type === 'function' || n.type === 'class'));
        return chunk ? chunk.file_path : null;
    });
    console.log('Test file path:', testPath);

    if (!testPath) {
        console.log('No chunk with file_path found');
        await browser.close();
        return;
    }

    // Test the navigation function with debugging
    consoleLogs.length = 0;

    const debugResult = await page.evaluate((path) => {
        const mode = window.currentVizMode;
        const check = mode === 'treemap' || mode === 'sunburst';
        console.log("DEBUG: currentVizMode = '" + mode + "', treemap/sunburst check = " + check);

        // Call the function
        if (typeof navigateToFileByPath === 'function') {
            navigateToFileByPath(path);
        }

        return { mode, check };
    }, testPath);

    console.log('Debug result:', debugResult);
    console.log('Console logs:', consoleLogs.filter(l => l.includes('DEBUG') || l.includes('Zooming') || l.includes('Navigating')));

    await browser.close();
}

runTest().catch(console.error);
