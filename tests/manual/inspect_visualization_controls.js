const { chromium } = require('playwright');

(async () => {
  const browser = await chromium.launch({ headless: false });
  const page = await browser.newPage();

  console.log('=== NAVIGATING TO VISUALIZATION ===');
  await page.goto('http://localhost:8082');

  // Wait for page to load
  await page.waitForTimeout(3000);

  console.log('\n=== CHECKING PAGE TITLE ===');
  const title = await page.title();
  console.log('Page title:', title);

  console.log('\n=== CHECKING CONSOLE ERRORS ===');
  page.on('console', msg => {
    if (msg.type() === 'error') {
      console.log('Console error:', msg.text());
    }
  });

  console.log('\n=== CHECKING LOADING STATE ===');
  const loadingVisible = await page.locator('#loading').isVisible().catch(() => false);
  console.log('Loading element visible:', loadingVisible);

  console.log('\n=== CHECKING LAYOUT CONTROLS ===');
  const layoutControls = page.locator('#layout-controls');
  const layoutExists = await layoutControls.count() > 0;
  console.log('Layout controls exists:', layoutExists);

  if (layoutExists) {
    const layoutVisible = await layoutControls.isVisible().catch(() => false);
    const layoutStyles = await layoutControls.evaluate(el => {
      const computed = window.getComputedStyle(el);
      return {
        display: computed.display,
        visibility: computed.visibility,
        opacity: computed.opacity,
        zIndex: computed.zIndex,
        position: computed.position,
        top: computed.top,
        left: computed.left,
        width: computed.width,
        height: computed.height,
        backgroundColor: computed.backgroundColor
      };
    });
    console.log('Layout controls visible:', layoutVisible);
    console.log('Layout controls computed styles:', JSON.stringify(layoutStyles, null, 2));

    const layoutBoundingBox = await layoutControls.boundingBox();
    console.log('Layout controls bounding box:', layoutBoundingBox);
  }

  console.log('\n=== CHECKING EDGE FILTERS ===');
  const edgeFilters = page.locator('#edge-filters');
  const edgeExists = await edgeFilters.count() > 0;
  console.log('Edge filters exists:', edgeExists);

  if (edgeExists) {
    const edgeVisible = await edgeFilters.isVisible().catch(() => false);
    const edgeStyles = await edgeFilters.evaluate(el => {
      const computed = window.getComputedStyle(el);
      return {
        display: computed.display,
        visibility: computed.visibility,
        opacity: computed.opacity,
        zIndex: computed.zIndex,
        position: computed.position,
        top: computed.top,
        left: computed.left,
        width: computed.width,
        height: computed.height,
        backgroundColor: computed.backgroundColor
      };
    });
    console.log('Edge filters visible:', edgeVisible);
    console.log('Edge filters computed styles:', JSON.stringify(edgeStyles, null, 2));

    const edgeBoundingBox = await edgeFilters.boundingBox();
    console.log('Edge filters bounding box:', edgeBoundingBox);
  }

  console.log('\n=== CHECKING LEGEND ===');
  const legend = page.locator('#legend');
  const legendExists = await legend.count() > 0;
  console.log('Legend exists:', legendExists);

  if (legendExists) {
    const legendVisible = await legend.isVisible().catch(() => false);
    const legendStyles = await legend.evaluate(el => {
      const computed = window.getComputedStyle(el);
      return {
        display: computed.display,
        visibility: computed.visibility,
        opacity: computed.opacity,
        zIndex: computed.zIndex,
        position: computed.position,
        top: computed.top,
        left: computed.left,
        width: computed.width,
        height: computed.height,
        backgroundColor: computed.backgroundColor
      };
    });
    console.log('Legend visible:', legendVisible);
    console.log('Legend computed styles:', JSON.stringify(legendStyles, null, 2));

    const legendBoundingBox = await legend.boundingBox();
    console.log('Legend bounding box:', legendBoundingBox);
  }

  console.log('\n=== CHECKING ALL CONTROL ELEMENTS ===');
  const allControls = await page.locator('#controls > *').all();
  console.log('Number of control elements:', allControls.length);

  for (let i = 0; i < allControls.length; i++) {
    const el = allControls[i];
    const id = await el.getAttribute('id');
    const className = await el.getAttribute('class');
    const visible = await el.isVisible().catch(() => false);
    const boundingBox = await el.boundingBox();
    console.log(`Control ${i}: id="${id}", class="${className}", visible=${visible}, bbox:`, boundingBox);
  }

  console.log('\n=== CHECKING Z-INDEX STACKING ===');
  const zIndexElements = await page.evaluate(() => {
    const elements = Array.from(document.querySelectorAll('#controls *'));
    return elements.map(el => ({
      id: el.id,
      className: el.className,
      tagName: el.tagName,
      zIndex: window.getComputedStyle(el).zIndex,
      position: window.getComputedStyle(el).position
    })).filter(el => el.zIndex !== 'auto');
  });
  console.log('Elements with explicit z-index:', JSON.stringify(zIndexElements, null, 2));

  console.log('\n=== CHECKING OVERLAPPING ELEMENTS ===');
  const overlaps = await page.evaluate(() => {
    const layoutControls = document.getElementById('layout-controls');
    const edgeFilters = document.getElementById('edge-filters');

    if (!layoutControls || !edgeFilters) {
      return { error: 'Controls not found' };
    }

    const layoutRect = layoutControls.getBoundingClientRect();
    const edgeRect = edgeFilters.getBoundingClientRect();

    // Find all elements at the position of layout controls
    const elementsAtLayout = document.elementsFromPoint(
      layoutRect.left + layoutRect.width / 2,
      layoutRect.top + layoutRect.height / 2
    );

    const elementsAtEdge = document.elementsFromPoint(
      edgeRect.left + edgeRect.width / 2,
      edgeRect.top + edgeRect.height / 2
    );

    return {
      layoutRect: {
        top: layoutRect.top,
        left: layoutRect.left,
        width: layoutRect.width,
        height: layoutRect.height
      },
      edgeRect: {
        top: edgeRect.top,
        left: edgeRect.left,
        width: edgeRect.width,
        height: edgeRect.height
      },
      elementsAtLayoutPosition: elementsAtLayout.map(el => ({
        tagName: el.tagName,
        id: el.id,
        className: el.className,
        zIndex: window.getComputedStyle(el).zIndex
      })),
      elementsAtEdgePosition: elementsAtEdge.map(el => ({
        tagName: el.tagName,
        id: el.id,
        className: el.className,
        zIndex: window.getComputedStyle(el).zIndex
      }))
    };
  });
  console.log('Overlapping elements analysis:', JSON.stringify(overlaps, null, 2));

  console.log('\n=== TAKING SCREENSHOT ===');
  await page.screenshot({ path: '/Users/masa/Projects/mcp-vector-search/.mcp-vector-search/visualization/diagnostic-screenshot.png', fullPage: true });
  console.log('Screenshot saved to: .mcp-vector-search/visualization/diagnostic-screenshot.png');

  console.log('\n=== CHECKING NETWORK REQUESTS ===');
  const graphDataLoaded = await page.evaluate(() => {
    return fetch('http://localhost:8082/graph_data.json')
      .then(r => r.ok)
      .catch(() => false);
  });
  console.log('Graph data accessible:', graphDataLoaded);

  console.log('\n=== INSPECTION COMPLETE ===');
  console.log('Browser will remain open for 10 seconds for manual inspection...');
  await page.waitForTimeout(10000);

  await browser.close();
})();
