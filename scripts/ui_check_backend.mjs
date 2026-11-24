import { chromium } from 'playwright';

const base = process.argv[2] || 'http://127.0.0.1:8010';
const browser = await chromium.launch();
const page = await browser.newPage({ viewport: { width: 1400, height: 1100 } });
const errors = [];
page.on('console', m => {
  if (m.type() === 'error' && !/status of 409/.test(m.text())) errors.push('console: ' + m.text());
});
page.on('pageerror', e => errors.push('pageerror: ' + e.message));
page.on('requestfailed', r => errors.push('requestfailed: ' + r.url() + ' ' + (r.failure()?.errorText || '')));
// A 409 on the group endpoint is the write gate working as designed; the
// dialog below confirms and the resend must succeed, so it is not an error.
page.on('response', r => {
  if (r.status() >= 400 && !r.url().includes('/group')) errors.push('HTTP ' + r.status() + ': ' + r.url());
});

await page.goto(base + '/?run=demo/separable', { waitUntil: 'networkidle' });
await page.waitForSelector('#stateline .pill', { timeout: 15000 });

// 1. backend dropdown lists both backends; run-state shows a backend pill
const backendOptions = await page.$$eval('#nmfBackend option', os => os.map(o => o.value));
const pills = await page.$$eval('#stateline .pill', ps => ps.map(p => p.textContent));
const backendPill = pills.find(t => t.includes('linalg backend'));
console.log('backend options:', backendOptions);
console.log('backend pill:', backendPill);

// 2. automation panel renders one row per template with distinct presets
const autoRows = await page.$$eval('#automationList .tplrow', rows =>
  rows.map(r => ({ preset: r.querySelector('select').value, button: r.querySelector('button').textContent })));
console.log('automation rows:', JSON.stringify(autoRows));
const presets = new Set(autoRows.map(r => r.preset));

// 3. submit a real scipy NMF job from the window and wait for completion
await page.selectOption('#nmfBackend', 'scipy');
await page.fill('#nmfK', '3');
await page.fill('#nmfIter', '120');
await page.click('#nmfRun');
await page.waitForFunction(
  () => document.querySelector('#nmfStatus').textContent.startsWith('done:'),
  { timeout: 90000 },
);
const nmfStatus = await page.textContent('#nmfStatus');
const anomalyText = await page.textContent('#nmfAnomalies');
console.log('NMF status:', nmfStatus);
console.log('anomaly line:', anomalyText.trim());

// the new run should now be loaded with 3 auditionable stems
const tracksAfter = await page.$$eval('#tracks .track', ts => ts.length);
const runMeta = await page.textContent('#runmeta');
console.log('new run tracks:', tracksAfter, '| meta:', runMeta);

// 4. apply an automation curve on the freshly decomposed run
await page.click('#automationList .tplrow:first-child button');
await page.waitForFunction(
  () => /applied/.test(document.querySelector('#automationResult').textContent),
  { timeout: 15000 },
);
const autoResult = await page.textContent('#automationResult');
console.log('automation result:', autoResult);

// 5. grouping confirmation gate on the twins run
await page.selectOption('#runs', 'demo/ambiguous_twins_anchored');
await page.waitForTimeout(1500);
page.once('dialog', d => d.accept()); // accept the operator confirmation dialog
await page.click('#applyGroups');
await page.waitForSelector('#groupAudio audio', { timeout: 30000 });
const groupResult = await page.textContent('#groupResult');
console.log('confirmed group write contains metrics:', /mask Σ error/.test(groupResult));

await page.screenshot({ path: 'artifacts/ui_backend_migration.png', fullPage: true });
console.log('distinct automation presets present:', presets.size >= 2);
console.log('ERRORS', JSON.stringify(errors, null, 2));
await browser.close();

const ok = backendOptions.length === 2 && backendPill && tracksAfter === 3 &&
           presets.size >= 2 && /done:/.test(nmfStatus) &&
           /applied/.test(autoResult) && errors.length === 0;
process.exit(ok ? 0 : 1);
