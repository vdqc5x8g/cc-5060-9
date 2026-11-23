import { chromium } from 'playwright';
const base = process.argv[2] || 'http://127.0.0.1:8010';
const browser = await chromium.launch();
const page = await browser.newPage({ viewport: { width: 1400, height: 1100 } });
const errors = [];
page.on('console', m => { if (m.type() === 'error') errors.push('console: ' + m.text()); });
page.on('pageerror', e => errors.push('pageerror: ' + e.message));
page.on('requestfailed', r => errors.push('requestfailed: ' + r.url()));
await page.goto(base + '/?run=demo/separable_anchored', { waitUntil: 'networkidle' });
await page.waitForSelector('#stateline .pill', { timeout: 15000 });
const pills = await page.evaluate(() => [...document.querySelectorAll('#stateline .pill')].map(p => p.textContent));
console.log('pills:', JSON.stringify(pills));
// group anchored+free: put T0 (anchor band_noise) alone, rest together
await page.evaluate(() => {
  const row = document.querySelector('#groupEditor .tplrow');
  const box = row.querySelectorAll('input[type=checkbox]')[0];
  if (box.checked) box.click();
});
await page.click('#addGroup');
await page.evaluate(() => {
  const rows = [...document.querySelectorAll('#groupEditor .tplrow')];
  rows[1].querySelectorAll('input[type=checkbox]')[0].click();
});
await page.click('#applyGroups');
await page.waitForSelector('#groupResult .warn', { timeout: 30000 });
console.log('group:', (await page.locator('#groupResult').innerText()).replace(/\n/g,' | '));
await page.screenshot({ path: 'artifacts/ui_anchored_grouped.png', fullPage: true });
// unknown k9 load
await page.selectOption('#runs', 'unknown/beethoven_op101_k9');
await page.waitForTimeout(2500);
const unk = await page.evaluate(() => ({
  meta: document.querySelector('#runmeta').textContent,
  pills: [...document.querySelectorAll('#stateline .pill')].map(p=>p.textContent),
  figs: [...document.querySelectorAll('#figs img')].map(i=>i.getAttribute('src')),
  tracks: document.querySelectorAll('#tracks .track').length,
}));
console.log('k9:', JSON.stringify(unk, null, 2));
console.log('ERRORS', JSON.stringify(errors));
await browser.close();
