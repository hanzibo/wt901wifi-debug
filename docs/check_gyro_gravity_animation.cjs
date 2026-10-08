// Dependency-free numerical/event tests, NOT browser visual acceptance.
// Run: node docs/check_gyro_gravity_animation.cjs
const fs = require('node:fs');
const vm = require('node:vm');
const assert = require('node:assert/strict');
const path = require('node:path');
const html = fs.readFileSync(path.join(__dirname, 'GYRO_GRAVITY_ANIMATION.html'), 'utf8');
const draft = fs.readFileSync(path.join(__dirname, 'GYRO_GRAVITY_ANIMATION.md'), 'utf8');
const script = html.match(/<script>\n(\(\(\)=>\{[\s\S]*?)<\/script>/)[1];
const normalize = text => text.replace(/const equationMathML = \/\* @precompute-math \*\/ \[[^\n]*\];/,
  'const equationMathML = /* @precompute-math */ [];');
assert.equal(normalize(script), draft.match(/<script>\n([\s\S]*?)<\/script>/)[1]);
assert(html.includes('<math xmlns="http://www.w3.org/1998/Math/MathML"'));
assert(html.includes('<mfrac>') && html.includes('<msubsup>'));
let draws = 0;
const context = new Proxy({}, {get: (_o, key) => {
  if (key === 'measureText') return text => ({width: text.length * 7});
  return (...args) => {
    for (const arg of args) if (typeof arg === 'number') assert(Number.isFinite(arg));
    draws++;
  };
}});
const ids = [...html.matchAll(/id="(tg[^" ]*|tiltLesson)"/g)].map(m => m[1]);
const elements = Object.fromEntries(ids.map(id => [id, {
  id, value: id === 'tgTilt' ? 28 : id === 'tgYaw' ? 42 : 0,
  checked: ['tgLinkViews', 'tgPerspective'].includes(id),
  style: {}, clientWidth: 480, clientHeight: 420, events: {},
  appendChild() {}, getContext() {return context;}, querySelectorAll() {return [];},
  focus() {}, setPointerCapture() {},
  insertAdjacentHTML(_position, content) {this.innerHTML += content;},
  addEventListener(name, fn) {this.events[name] = fn;},
}]));
let nextFrame;
const sandbox = {
  document: {getElementById: id => elements[id], createElement: () => ({style: {}})},
  window: {devicePixelRatio: 1, addEventListener() {}},
  performance: {now: () => 0}, requestAnimationFrame: fn => {nextFrame = fn;}, Math,
};
vm.runInNewContext(script, sandbox);
const check = () => sandbox.window.tiltLessonCheck();
const event = (id, name, extra = {}) => elements[id].events[name]({
  preventDefault() {}, pointerId: 1, pointerType: 'mouse', button: 0,
  clientX: 100, clientY: 100, ...extra,
});
let maxAlignmentResidual = 0, maxHeadingAxisResidual = 0;
for (const tilt of [0, 1, 28, 65]) for (const yaw of [-120, -42, 0, 42, 120]) {
  for (const stage of [0, 1, 1.5, 2, 3, 3.5, 4, 5]) {
    elements.tgTilt.value = tilt; elements.tgYaw.value = yaw;
    elements.tgTime.value = stage; elements.tgTime.oninput();
    const r = check();
    maxAlignmentResidual = Math.max(maxAlignmentResidual, r.alignmentResidual);
    maxHeadingAxisResidual = Math.max(maxHeadingAxisResidual, r.headingAxisResidual);
    assert(r.alignmentResidual < 1e-12 && r.headingAxisResidual < 1e-12);
    assert(Math.abs(r.tiltAxisDotUp) < 1e-12);
    assert(Math.abs(r.thetaDegrees - tilt) < 1e-10);
    assert(Math.abs(r.recoveredHeadingDegrees - yaw) < 1e-10);
    assert(!elements.tgValues.textContent.includes('NaN'));
    assert(elements.tgEquation.innerHTML.includes('<math'));
    assert(!elements.tgEquation.innerHTML.includes('katex-error'));
  }
}
const before = check();
event('tgWorld', 'pointerdown');
event('tgWorld', 'pointermove', {clientX: 260, clientY: 170});
event('tgWorld', 'pointerup');
assert.notEqual(check().cameras.tgWorld.az, before.cameras.tgWorld.az);
assert.deepEqual(check().cameras.tgWorld, check().cameras.tgBody);
assert.equal(check().alignmentResidual, before.alignmentResidual);
assert.equal(check().thetaDegrees, before.thetaDegrees);
assert.equal(check().stage, before.stage);
assert.notDeepEqual(check().projectedWorldAxes, before.projectedWorldAxes);
elements.tgLinkViews.checked = false; elements.tgLinkViews.onchange();
const bodyBefore = check().cameras.tgBody;
event('tgWorld', 'keydown', {key: 'ArrowRight'});
assert.deepEqual(check().cameras.tgBody, bodyBefore);
for (let i = 0; i < 100; i++) event('tgWorld', 'wheel', {deltaY: -100});
assert.equal(check().cameras.tgWorld.zoom, 1.75);
for (let i = 0; i < 100; i++) event('tgWorld', 'wheel', {deltaY: 100});
assert.equal(check().cameras.tgWorld.zoom, .55);
// Touch pinch, cancellation, and later single-pointer operation.
event('tgBody', 'pointerdown', {pointerId: 2, pointerType: 'touch'});
event('tgBody', 'pointerdown', {pointerId: 3, pointerType: 'touch', clientX: 200});
const zoomBefore = check().cameras.tgBody.zoom;
event('tgBody', 'pointermove', {pointerId: 3, clientX: 250});
assert(check().cameras.tgBody.zoom > zoomBefore);
event('tgBody', 'pointercancel', {pointerId: 3});
const azBefore = check().cameras.tgBody.az;
event('tgBody', 'pointermove', {pointerId: 2, clientX: 120});
assert.notEqual(check().cameras.tgBody.az, azBefore);
event('tgBody', 'lostpointercapture', {pointerId: 2});
elements.tgLinkViews.checked = true; elements.tgLinkViews.onchange();
for (const id of ['tgViewFront', 'tgViewSide', 'tgViewTop', 'tgViewIso']) {
  elements[id].onclick();
  assert(check().projectedWorldAxes.every(p => p.every(Number.isFinite)));
  assert.deepEqual(check().cameras.tgWorld, check().cameras.tgBody);
}
elements.tgPerspective.checked = false; elements.tgPerspective.onchange();
elements.tgViewReset.onclick();
assert.equal(check().cameras.tgWorld.zoom, 1);
elements.tgReset.onclick(); elements.tgPlay.onclick();
for (let t = 0; t <= 30000; t += 100) nextFrame(t);
assert.equal(check().stage, 5);
assert.equal(elements.tgPlay.textContent, '▶ 播放');
elements.tgPrev.onclick(); assert.equal(check().stage, 4);
elements.tgNext.onclick(); assert.equal(check().stage, 5);
elements.tgReset.onclick(); assert.equal(check().stage, 0);
console.log(JSON.stringify({cases: 160, maxAlignmentResidual, maxHeadingAxisResidual,
  orbitZoomPinchKeyboardViewsPlayback: 'PASS', finiteDrawCalls: draws,
  visualReview: 'NOT_PERFORMED'}, null, 2));
