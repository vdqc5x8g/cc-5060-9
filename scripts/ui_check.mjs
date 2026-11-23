import { chromium } from 'playwright';
const base = process.argv[2] || 'http://127.0.0.1:8010';
const browser = await chromium.launch();
const page = await browser.newPage({ viewport: { width: 1400, height: 1000 } });
const errors = [];
page.on('console', m => { if (m.type() === 'error') errors.push('console: ' + m.text()); });
page.on('pageerror', e => errors.push('pageerror: ' + e.message));
page.on('requestfailed', r => errors.push('requestfailed: ' + r.url() + ' ' + (r.failure()?.errorText || '')));
await page.goto(base + '/', { waitUntil: 'networkidle' });
await page.waitForSelector('#stateline .pill', { timeout: 15000 });
const summary = await page.evaluate(() => ({
  runs: [...document.querySelectorAll('#runs option')].map(o => o.value),
  pills: [...document.querySelectorAll('#stateline .pill')].map(p => p.textContent),
  tracks: document.querySelectorAll('#tracks .track').length,
  candRows: document.querySelectorAll('#cands tr').length,
  evalRows: document.querySelectorAll('#evalBody tr').length,
  figs: document.querySelectorAll('#figs img').length,
  tplCanvas: !!document.querySelector('#templates'),
}));
console.log(JSON.stringify(summary, null, 2));

// exercise grouping
await page.click('#applyGroups');
await page.waitForSelector('#groupResult .warn', { timeout: 30000 });
const groupText = await page.locator('#groupResult').innerText();
console.log('group:', groupText.replace(/\n/g, ' | '));
const groupAudio = await page.locator('#groupAudio audio').count();
console.log('groupAudioPlayers:', groupAudio);
await page.screenshot({ path: 'artifacts/ui_separable.png', fullPage: true });

// exercise render selection (solo)
await page.click('#render');
await page.waitForTimeout(2500);
const renderedSrc = await page.locator('#rendered').getAttribute('src');
console.log('renderedSrc:', renderedSrc);

// switch run
await page.selectOption('#runs', 'unknown/beethoven_op101');
await page.waitForTimeout(2500);
const second = await page.evaluate(() => ({
  meta: document.querySelector('#runmeta').textContent,
  evalText: document.querySelector('#evalBody').textContent.slice(0, 120),
  tracks: document.querySelectorAll('#tracks .track').length,
}));
console.log(JSON.stringify(second, null, 2));
await page.screenshot({ path: 'artifacts/ui_unknown.png', fullPage: true });
console.log('ERRORS', JSON.stringify(errors, null, 2));
await browser.close();
