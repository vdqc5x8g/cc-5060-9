import { chromium } from 'playwright';
const browser = await chromium.launch();
const page = await browser.newPage({ viewport: { width: 1400, height: 1100 } });
const errors = [];
page.on('console', m => { if (m.type() === 'error') errors.push(m.text()); });
page.on('pageerror', e => errors.push('pageerror: ' + e.message));
page.on('response', r => { if (r.status() >= 400) errors.push(r.status() + ' ' + r.url()); });
await page.goto('http://127.0.0.1:8012/?run=demo/separable', { waitUntil: 'networkidle' });
await page.waitForSelector('#stateline .pill');
await page.waitForTimeout(800);
console.log('groups initial:', await page.locator('#groupHint').innerText());
await page.screenshot({ path: 'artifacts/ui_separable.png', fullPage: true });
const boxes = await page.locator('#groupEditor input[type=checkbox]').all();
await boxes[1].check(); await boxes[2].check();   // merge T1,T2 into group0 -> all one group
await page.click('#addGroup');
const boxes2 = await page.locator('#groupEditor input[type=checkbox]').all();
// new group row (second row): boxes 3,4,5. Put T2 there -> duplicate, expect warning
await boxes2[5].check();
console.log('after duplicate:', await page.locator('#groupHint').innerText());
await boxes2[5].uncheck();
await page.click('#applyGroups');
await page.waitForTimeout(1200);
console.log('group result:', await page.locator('#groupResult').innerText());
await page.screenshot({ path: 'artifacts/ui_separable_grouped.png', fullPage: true });
// Now proper 2-group split: uncheck T1,T2 from group0, put them in group1
const b3 = await page.locator('#groupEditor input[type=checkbox]').all();
await b3[1].uncheck(); await b3[2].uncheck(); await b3[4].check(); await b3[5].check();
console.log('hint:', await page.locator('#groupHint').innerText());
const labels = await page.locator('#groupEditor input[type=text]').all();
await labels[0].fill('low'); await labels[1].fill('bell+noise');
await page.click('#applyGroups'); await page.waitForTimeout(1200);
console.log('group result 2:', await page.locator('#groupResult').innerText());
console.log('grouped players:', await page.locator('#groupAudio audio').count());
await page.screenshot({ path: 'artifacts/ui_separable_grouped2.png', fullPage: true });
await page.click('#render'); await page.waitForTimeout(2000);
console.log('rendered:', await page.locator('#rendered').getAttribute('src'));
console.log('errors', JSON.stringify(errors));
await browser.close();
