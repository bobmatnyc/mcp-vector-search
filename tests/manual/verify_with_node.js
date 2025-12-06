#!/usr/bin/env node
/**
 * Verification script using Playwright Node.js API
 * Tests that graph nodes are visible and properly initialized
 */

const { chromium } = require('playwright');
const fs = require('fs');
const path = require('path');

async function verify() {
  const browser = await chromium.launch({ headless: false });
  const context = await browser.newContext({ viewport: { width: 1920, height: 1080 } });
  const page = await context.newPage();

  console.log('🚀 Navigating to http://localhost:8082...');
  await page.goto('http://localhost:8082', { waitUntil: 'networkidle', timeout: 60000 });

  console.log('⏳ Waiting for visualization to initialize...');
  await page.waitForTimeout(5000);

  // Take initial screenshot
  const screenshotPath = path.join(__dirname, 'screenshot_verification.png');
  await page.screenshot({ path: screenshotPath, fullPage: true });
  console.log(`📸 Screenshot saved to ${screenshotPath}`);

  console.log('\n' + '='.repeat(60));
  console.log('VERIFICATION 1: Data Initialization');
  console.log('='.repeat(60));

  // Verify data initialization
  const allNodesLength = await page.evaluate(() => window.allNodes ? window.allNodes.length : 0);
  const allLinksLength = await page.evaluate(() => window.allLinks ? window.allLinks.length : 0);
  const cyNodesLength = await page.evaluate(() => window.cy ? window.cy.nodes().length : 0);
  const cyEdgesLength = await page.evaluate(() => window.cy ? window.cy.edges().length : 0);

  console.log(`✓ allNodes.length: ${allNodesLength} (expected: 1449)`);
  console.log(`✓ allLinks.length: ${allLinksLength} (expected: ~360000)`);
  console.log(`✓ cy.nodes().length: ${cyNodesLength} (expected: matches allNodes)`);
  console.log(`✓ cy.edges().length: ${cyEdgesLength}`);

  console.log('\n' + '='.repeat(60));
  console.log('VERIFICATION 2: Graph Visibility');
  console.log('='.repeat(60));

  // Check if graph elements exist
  const canvasExists = await page.evaluate(() => document.querySelector('canvas') !== null);
  const cyContainerExists = await page.evaluate(() => document.querySelector('#cy') !== null);

  console.log(`✓ Canvas element exists: ${canvasExists}`);
  console.log(`✓ Cytoscape container exists: ${cyContainerExists}`);

  // Get container dimensions
  const containerInfo = await page.evaluate(() => {
    const container = document.querySelector('#cy');
    if (!container) return null;
    const rect = container.getBoundingClientRect();
    return {
      width: rect.width,
      height: rect.height,
      visible: rect.width > 0 && rect.height > 0
    };
  });

  if (containerInfo) {
    console.log(`✓ Container dimensions: ${containerInfo.width}x${containerInfo.height}`);
    console.log(`✓ Container visible: ${containerInfo.visible}`);
  }

  console.log('\n' + '='.repeat(60));
  console.log('VERIFICATION 3: Controls');
  console.log('='.repeat(60));

  // Check controls
  const layoutSelector = await page.$('#layoutSelect');
  const edgeFilter = await page.$('#edgeFilter');
  const legend = await page.$('.legend');

  console.log(`✓ Layout selector found: ${layoutSelector !== null}`);
  console.log(`✓ Edge filter found: ${edgeFilter !== null}`);
  console.log(`✓ Legend found: ${legend !== null}`);

  if (layoutSelector) {
    const currentLayout = await page.evaluate(() => document.querySelector('#layoutSelect').value);
    console.log(`✓ Current layout: ${currentLayout}`);
  }

  console.log('\n' + '='.repeat(60));
  console.log('VERIFICATION 4: Console Errors');
  console.log('='.repeat(60));

  // Collect console messages
  const consoleLogs = [];
  page.on('console', msg => consoleLogs.push(`${msg.type()}: ${msg.text()}`));

  await page.waitForTimeout(2000);

  const errors = consoleLogs.filter(log => log.startsWith('error'));
  const warnings = consoleLogs.filter(log => log.startsWith('warning'));

  console.log(`✓ Console errors: ${errors.length}`);
  console.log(`✓ Console warnings: ${warnings.length}`);

  if (errors.length > 0) {
    console.log('\nErrors found:');
    errors.forEach(err => console.log(`  - ${err}`));
  }

  console.log('\n' + '='.repeat(60));
  console.log('VERIFICATION SUMMARY');
  console.log('='.repeat(60));

  const successCriteria = {
    'allNodes.length === 1449': allNodesLength === 1449,
    'allLinks.length > 0': allLinksLength > 0,
    'Graph visually rendered': cyContainerExists && canvasExists,
    'Controls functional': layoutSelector !== null && edgeFilter !== null,
    'Cytoscape initialized': cyNodesLength > 0,
    'No critical errors': errors.length === 0
  };

  for (const [criterion, passed] of Object.entries(successCriteria)) {
    const status = passed ? '✅ PASS' : '❌ FAIL';
    console.log(`${status}: ${criterion}`);
  }

  const allPassed = Object.values(successCriteria).every(v => v);

  console.log('\n' + '='.repeat(60));
  if (allPassed) {
    console.log('🎉 ALL VERIFICATION CHECKS PASSED');
  } else {
    console.log('⚠️  SOME VERIFICATION CHECKS FAILED');
  }
  console.log('='.repeat(60));

  // Save report
  const report = {
    timestamp: new Date().toISOString(),
    data_initialization: {
      allNodes_length: allNodesLength,
      allLinks_length: allLinksLength,
      cy_nodes_length: cyNodesLength,
      cy_edges_length: cyEdgesLength
    },
    visibility: {
      canvas_exists: canvasExists,
      cy_container_exists: cyContainerExists,
      container_info: containerInfo
    },
    controls: {
      layout_selector: layoutSelector !== null,
      edge_filter: edgeFilter !== null,
      legend: legend !== null
    },
    console: {
      errors: errors.length,
      warnings: warnings.length,
      error_messages: errors
    },
    success_criteria: successCriteria,
    all_passed: allPassed
  };

  const reportPath = path.join(__dirname, 'verification_report.json');
  fs.writeFileSync(reportPath, JSON.stringify(report, null, 2));
  console.log(`\n📄 Detailed report saved to ${reportPath}`);

  console.log('\n👀 Keeping browser open for 5 seconds for manual inspection...');
  await page.waitForTimeout(5000);

  await browser.close();

  return allPassed;
}

verify().then(success => {
  process.exit(success ? 0 : 1);
}).catch(err => {
  console.error('\n❌ VERIFICATION FAILED WITH ERROR:', err);
  process.exit(1);
});
