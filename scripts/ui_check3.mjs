import { chromium } from 'playwright';
const base = process.argv[2] || 'http://127.0.0.1:8010';
const browser = await chromium.launch();
const page = await browser.newPage({ viewport: { width: 1400, height: 1000 } });
const errors = [];
page.on('console', m => { if (m.type() === 'error') errors.push('console: ' + m.text()); });
page.on('pageerror', e => errors.push('pageerror: ' + e.message));
page.on('requestfailed', r => errors.push('requestfailed: ' + r.url() + ' ' + (r.failure()?.errorText || '')));
await page.goto(base + '/?run=demo/separable', { waitUntil: 'networkidle' });
await page.waitForSelector('#stateline .pill', { timeout: 15000 });

// drag-select segment on activation canvas
const box = await page.locator('#activations').boundingBox();
await page.mouse.move(box.x + box.width * 0.30, box.y + box.height * 0.5);
await page.mouse.down();
await page.mouse.move(box.x + box.width * 0.55, box.y + box.height * 0.5, { steps: 8 });
await page.mouse.up();
const segText = await page.locator('#segmentInfo').innerText();
console.log('segmentInfo:', segText.replace(/\n/g, ' | '));

await page.click('#render');
await page.waitForTimeout(2500);
const soloRes = await page.evaluate(async () => {
  const r = await fetch('api/run/demo%2Fseparable/solo', {method:'POST', headers:{'Content-Type':'application/json'},
    body: JSON.stringify({stems:[0], gains_db:{}, start_s:1.0, end_s:2.5})});
  return r.json();
});
console.log('soloSegment:', JSON.stringify(soloRes));

// segment must be silent outside the window; fetch the wav and inspect via decodeAudioData
const stats = await page.evaluate(async () => {
  const buf = await (await fetch('api/run/demo%2Fseparable/audio/solo.wav?t=' + Date.now())).arrayBuffer();
  const ac = new OfflineAudioContext(1, 1, 22050);
  const au = await ac.decodeAudioData(buf);
  const d = au.getChannelData(0);
  let e0 = 0, eIn = 0, eEnd = 0;
  for (let i=0;i<d.length;i++) {
    const t = i/au.sampleRate, v = d[i]*d[i];
    if (t < 0.9) e0 += v;
    if (t > 1.05 && t < 2.45) eIn += v;
    if (t > 2.6) eEnd += v;
  }
  return {sr: au.sampleRate, len: d.length, pre: e0, inside: eIn, post: eEnd};
});
console.log('gateStats:', JSON.stringify(stats));

// grouping still works on ambiguous run with twin anchors
await page.selectOption('#runs', 'demo/ambiguous_twins_anchored');
await page.waitForTimeout(2000);
const amb = await page.evaluate(() => ({
  pills: [...document.querySelectorAll('#stateline .pill')].map(p => p.textContent),
  ambigRows: document.querySelectorAll('#ambig tr').length,
  figs: document.querySelectorAll('#figs img').length,
  figSrcs: [...document.querySelectorAll('#figs img')].map(i => i.getAttribute('src')),
}));
console.log('twins:', JSON.stringify(amb, null, 2));
await page.screenshot({ path: 'artifacts/ui_twins_segment.png', fullPage: true });

// unknown run has no evaluation and no figures errors
await page.selectOption('#runs', 'unknown/beethoven_op101');
await page.waitForTimeout(2500);
const unk = await page.evaluate(() => ({
  meta: document.querySelector('#runmeta').textContent,
  evalText: document.querySelector('#evalBody').textContent.slice(0, 160),
  figs: document.querySelectorAll('#figs img').length,
  tracks: document.querySelectorAll('#tracks .track').length,
}));
console.log('unknown:', JSON.stringify(unk, null, 2));
console.log('ERRORS', JSON.stringify(errors, null, 2));
await browser.close();
