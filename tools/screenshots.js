/* Снимает актуальные скриншоты боевого сайта для README.
   Запуск: docker run ... node screenshots.js   (см. README, раздел «Как это выглядит») */
const { chromium } = require('playwright');

const BASE = process.env.SITE || 'https://shazram.ru';

(async () => {
  const b = await chromium.launch();
  const errs = [];
  const shot = async (p, name, opts = {}) => {
    await p.screenshot({ path: `/out/${name}.png`, ...opts });
    console.log('  ' + name);
  };

  // --- десктоп ---
  const p = await b.newPage({ viewport: { width: 1440, height: 900 } });
  p.on('console', m => { if (m.type() === 'error') errs.push(m.text().slice(0, 120)); });

  await p.goto(`${BASE}/cohort`, { waitUntil: 'networkidle', timeout: 60000 });
  await p.waitForTimeout(900);
  await shot(p, 'start');

  await p.getByRole('button', { name: /демо-данн/ }).click();
  await p.waitForSelector('h1:has-text("внимания требуют")', { timeout: 120000 });
  await p.waitForTimeout(800);
  await shot(p, 'cohort-list');

  await p.locator('button[aria-label^="Студент"]').first().click();
  await p.waitForTimeout(700);
  await shot(p, 'student-card');
  await p.keyboard.press('Escape');
  await p.waitForTimeout(300);

  await p.goto(`${BASE}/survey`, { waitUntil: 'networkidle', timeout: 60000 });
  await p.waitForTimeout(1000);
  await shot(p, 'survey', { fullPage: true });

  // необязательный шаг с оценками
  for (let i = 0; i < 3; i++) {
    await p.getByRole('button', { name: 'Дальше' }).click();
    await p.waitForTimeout(350);
  }
  await shot(p, 'survey-grades', { fullPage: true });
  await p.locator('#q-G1').fill('6');
  await p.locator('#q-G2').fill('5');
  await p.getByRole('button', { name: /Узнать результат/ }).click();
  await p.waitForTimeout(6000);
  await shot(p, 'survey-result', { fullPage: true });

  await p.goto(`${BASE}/model`, { waitUntil: 'networkidle', timeout: 60000 });
  await p.waitForTimeout(5000);
  await shot(p, 'model', { fullPage: true });
  await p.close();

  // --- тёмная тема ---
  const d = await b.newPage({ viewport: { width: 1440, height: 900 } });
  await d.goto(`${BASE}/cohort`, { waitUntil: 'networkidle', timeout: 60000 });
  await d.evaluate(() => { localStorage.setItem('theme', 'dark'); });
  await d.reload({ waitUntil: 'networkidle' });
  await d.getByRole('button', { name: /демо-данн/ }).click();
  await d.waitForSelector('h1:has-text("внимания требуют")', { timeout: 120000 });
  await d.waitForTimeout(800);
  await shot(d, 'dark');
  await d.close();

  // --- мобильный 375px ---
  const m = await b.newPage({ viewport: { width: 375, height: 812 }, isMobile: true });
  await m.goto(`${BASE}/cohort`, { waitUntil: 'networkidle', timeout: 60000 });
  await m.getByRole('button', { name: /демо-данн/ }).click();
  await m.waitForSelector('h1:has-text("внимания требуют")', { timeout: 120000 });
  await m.waitForTimeout(700);
  await shot(m, 'mobile');
  const ox = await m.evaluate(() =>
    document.documentElement.scrollWidth - document.documentElement.clientWidth);
  console.log('MOBILE_OVERFLOW=' + ox);

  console.log('ОШИБКИ КОНСОЛИ: ' + (errs.length ? errs.join(' ;; ') : 'нет'));
  await b.close();
})();
