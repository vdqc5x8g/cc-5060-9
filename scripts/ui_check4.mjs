import { chromium } from 'playwright';
const base = process.argv[2] || 'http://127.0.0.1:8010';
const browser = await chromium.launch();
const page = await browser.newPage({ viewport: { width: 1400, height: 1100 } });
const errors = [];
page.on('console', m => { if (m.type() === 'error') errors.push('console: ' + m.text()); });
page.on('pageerror', e => errors.push('pageerror: ' + e.message));
page.on('requestfailed', r => errors.push('requestfailed: ' + r.url()));
await page.goto(base + '/?run=demo/separable', { waitUntil: 'networkidle' });
await page.waitForSelector('#stateline .pill', { timeout: 15000 });
const pills = await page.evaluate(() => [...document.querySelectorAll('#stateline .pill')].map(p => p.textContent));
console.log('pills:', JSON.stringify(pills));
// group templates 0 and 2, leave 1 separate
// start: one group with all templates. Move T1 into a second group.
await page.evaluate(() => {
  const row = document.querySelector('#groupEditor .tplrow');
  row.querySelectorAll('input[type=checkbox]')[1].click(); // uncheck T1 in group 0
});
await page.click('#addGroup');
await page.evaluate(() => {
  const rows = [...document.querySelectorAll('#groupEditor .tplrow')];
  rows[1].querySelectorAll('input[type=checkbox]')[1].click(); // check T1 in group 1
});
const hint = await page.locator('#groupHint').innerText();
console.log('hint:', hint.replace(/\n/g,' | '));
await page.click('#applyGroups');
await page.waitForSelector('#groupResult .warn', { timeout: 30000 });
console.log('group:', (await page.locator('#groupResult').innerText()).replace(/\n/g,' | '));
const audio = await page.locator('#groupAudio audio').count();
console.log('groupAudioPlayers:', audio);
await page.screenshot({ path: 'artifacts/ui_separable_current.png', fullPage: true });
console.log('ERRORS', JSON.stringify(errors));
await browser.close();
