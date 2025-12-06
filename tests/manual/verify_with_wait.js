#!/usr/bin/env node
/**
 * Verification with extended wait for large JSON loading
 */

const { chromium } = require('playwright');

async function verify() {
  const browser = await chromium.launch({ headless: false });
  const context = await browser.newContext({ viewport: { width: 1920, height: 1080 } });
  const page = await context.newPage();

  // Collect console messages
  const consoleLogs = [];
  page.on('console', msg => {
    const text = msg.text();
    const type = msg.type();
    consoleLogs.push(`[${type.toUpperCase()}] ${text}`);
    console.log(`[CONSOLE ${type.toUpperCase()}] ${text}`);
  });

  // Collect page errors
  page.on('pageerror', err => {
    console.log(`[PAGE ERROR] ${err.message}`);
    consoleLogs.push(`[ERROR] ${err.message}`);
  });

  console.log('🚀 Navigating to http://localhost:8082...');
  await page.goto('http://localhost:8082', { waitUntil: 'domcontentloaded', timeout: 60000 });

  console.log('⏳ Waiting 20 seconds for large JSON file to load (54MB)...');

  // Check progress every 2 seconds
  for (let i = 0; i < 10; i++) {
    await page.waitForTimeout(2000);

    const status = await page.evaluate(() => ({
      allNodesLength: window.allNodes ? window.allNodes.length : 0,
      allLinksLength: window.allLinks ? window.allLinks.length : 0,
      cyExists: window.cy !== null && window.cy !== undefined,
      cyNodesLength: window.cy ? window.cy.nodes().length : 0,
      loadingVisible: document.querySelector('#loading-spinner') ?
        window.getComputedStyle(document.querySelector('#loading-spinner')).display !== 'none' : false
    }));

    console.log(`[${i*2}s] Loading status:`, status);

    // If data is loaded, break early
    if (status.allNodesLength > 0) {
      console.log('✅ Data loaded successfully!');
      break;
    }
  }

  // Final check
  await page.waitForTimeout(2000);

  const allNodesLength = await page.evaluate(() => window.allNodes ? window.allNodes.length : 0);
  const allLinksLength = await page.evaluate(() => window.allLinks ? window.allLinks.length : 0);
  const cyNodesLength = await page.evaluate(() => window.cy ? window.cy.nodes().length : 0);

  console.log('\n' + '='.repeat(60));
  console.log('FINAL DATA STATE');
  console.log('='.repeat(60));
  console.log(`allNodes.length: ${allNodesLength}`);
  console.log(`allLinks.length: ${allLinksLength}`);
  console.log(`cy.nodes().length: ${cyNodesLength}`);

  // Take screenshot
  await page.screenshot({
    path: '/Users/masa/Projects/mcp-vector-search/tests/manual/screenshot_after_wait.png',
    fullPage: true
  });
  console.log('📸 Screenshot saved');

  console.log('\n📋 Console Logs:');
  consoleLogs.forEach(log => console.log(log));

  console.log('\n👀 Keeping browser open for 10 seconds...');
  await page.waitForTimeout(10000);

  await browser.close();

  return allNodesLength === 1449;
}

verify().catch(err => {
  console.error('Error:', err);
  process.exit(1);
});
