#!/usr/bin/env node
/**
 * Debug network requests to find 404 error
 */

const { chromium } = require('playwright');

async function debug() {
  const browser = await chromium.launch({ headless: false });
  const context = await browser.newContext({ viewport: { width: 1920, height: 1080 } });
  const page = await context.newPage();

  // Track all requests
  const failedRequests = [];
  page.on('requestfailed', request => {
    const failure = {
      url: request.url(),
      method: request.method(),
      resourceType: request.resourceType(),
      failure: request.failure()
    };
    failedRequests.push(failure);
    console.log(`❌ REQUEST FAILED: ${request.url()}`);
    console.log(`   Failure: ${request.failure()?.errorText || 'unknown'}`);
  });

  page.on('response', response => {
    if (!response.ok()) {
      console.log(`⚠️  HTTP ${response.status()}: ${response.url()}`);
    }
  });

  console.log('🚀 Navigating...');
  await page.goto('http://localhost:8082', { waitUntil: 'networkidle', timeout: 60000 });

  console.log('\n⏳ Waiting 10 seconds...');
  await page.waitForTimeout(10000);

  console.log('\n' + '='.repeat(60));
  console.log('FAILED REQUESTS');
  console.log('='.repeat(60));
  failedRequests.forEach((req, i) => {
    console.log(`${i + 1}. ${req.url}`);
    console.log(`   Type: ${req.resourceType}`);
    console.log(`   Error: ${req.failure?.errorText || 'unknown'}`);
  });

  await browser.close();
}

debug().catch(err => console.error('Error:', err));
