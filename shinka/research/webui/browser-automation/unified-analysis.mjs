/**
 * Puppeteer-compatible, read-only regressions for the restored analysis surface.
 * Run against an already running `shinka_research ui` and an authorized saved run.
 * No provider, candidate execution, or data mutation occurs.
 * Selectors belong to analysis/viz_tree.html and App's analysis iframe; update here
 * if those semantic controls change. Supply labels/markers from the saved run.
 */
function analysisUrl({ baseUrl, dbPath }, leftTab, selectedId) {
  const child = new URL('/viz_tree.html', baseUrl);
  child.searchParams.set('db_path', dbPath);
  child.searchParams.set('left_tab', leftTab);
  child.searchParams.set('right_tab', 'agent-code');
  if (selectedId) child.searchParams.set('selected_node', selectedId);
  const outer = new URL('/', baseUrl);
  outer.searchParams.set('view', 'runs');
  outer.searchParams.set('analysis', child.pathname + child.search);
  return outer.href;
}

async function analysisFrame(page) {
  await page.waitForSelector('iframe');
  return (await page.$('iframe')).contentFrame();
}

async function selectedCode(frame, id, marker, leftTab) {
  await frame.waitForFunction(({ id, marker, leftTab }) =>
    new URL(location.href).searchParams.get('selected_node') === id &&
    document.querySelector('.left-tab.active')?.dataset.tab === leftTab &&
    document.querySelector('.tab.active')?.dataset.tab === 'agent-code' &&
    document.querySelector('#agent-code')?.textContent.includes(marker), {},
  { id, marker, leftTab });
}

async function selectedTree(frame, id) {
  await frame.waitForFunction(id => {
    const selected = [...document.querySelectorAll('#tree .node.selected')];
    return selected.length === 1 && selected[0].getAttribute('aria-label')?.includes(id) &&
      new URL(location.href).searchParams.get('selected_node') === id;
  }, {}, id);
}

async function selectionReceipt(frame) {
  return frame.evaluate(() => ({
    url: location.href,
    selected: new URL(location.href).searchParams.get('selected_node'),
    leftTab: document.querySelector('.left-tab.active').dataset.tab,
    rightTab: document.querySelector('.tab.active').dataset.tab,
    highlighted: [...document.querySelectorAll('#tree .node.selected')].map(node => node.getAttribute('aria-label')),
  }));
}

/** options: baseUrl, dbPath, promptLabel, programLabel, expectedId, codeMarker. */
export async function verifyPromptProgramSelection(page, options) {
  const { promptLabel, programLabel, expectedId, codeMarker } = options;
  await page.goto(analysisUrl(options, 'meta-prompt-view'), { waitUntil: 'networkidle0' });
  const frame = await analysisFrame(page);
  await frame.waitForFunction(label => [...document.querySelectorAll('.prompt-node')]
    .some(node => node.getAttribute('aria-label') === label), {}, promptLabel);
  await frame.evaluate(label => [...document.querySelectorAll('.prompt-node')]
    .find(node => node.getAttribute('aria-label') === label).focus(), promptLabel);
  await page.keyboard.press('Enter');
  await frame.waitForFunction(label => [...document.querySelectorAll('#meta-prompt-details .program-link')]
    .some(button => button.textContent === label), {}, programLabel);
  await frame.evaluate(label => [...document.querySelectorAll('#meta-prompt-details .program-link')]
    .find(button => button.textContent === label).focus(), programLabel);
  await page.keyboard.press('Enter');
  await selectedCode(frame, expectedId, codeMarker, 'tree-view');
  const viewport = page.viewport();
  try {
    await page.setViewport({ ...viewport, width: viewport.width > 1100 ? 1024 : 1440 });
    await frame.evaluate(() => new Promise(resolve =>
      requestAnimationFrame(() => requestAnimationFrame(resolve))));
    await selectedTree(frame, expectedId);
  } finally {
    await page.setViewport(viewport);
  }
  return selectionReceipt(frame);
}

/** options: baseUrl, dbPath, expectedId, codeMarker, firstPathId, firstPathCodeMarker. */
export async function verifyPathDeepLink(page, options) {
  const { expectedId, codeMarker, firstPathId, firstPathCodeMarker } = options;
  await page.goto(analysisUrl(options, 'best-path-view', expectedId), { waitUntil: 'networkidle0' });
  let frame = await analysisFrame(page);
  await selectedCode(frame, expectedId, codeMarker, 'best-path-view');
  await page.reload({ waitUntil: 'networkidle0' });
  frame = await analysisFrame(page);
  await selectedCode(frame, expectedId, codeMarker, 'best-path-view');
  await frame.waitForSelector('.path-flow-node[role="button"]');
  await frame.focus('.path-flow-node[role="button"]');
  await page.keyboard.press('Enter');
  await selectedCode(frame, firstPathId, firstPathCodeMarker, 'best-path-view');
  await frame.click('.left-tab[data-tab="tree-view"]');
  await selectedTree(frame, firstPathId);
  return selectionReceipt(frame);
}
