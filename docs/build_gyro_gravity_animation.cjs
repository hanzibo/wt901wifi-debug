// Build the offline explainer, pre-rendering LaTeX to native MathML.
// Usage: node docs/build_gyro_gravity_animation.cjs /path/to/node_modules/katex
// KaTeX is a build-only dependency. The delivered HTML needs no JS math library.
const fs = require('node:fs');
const path = require('node:path');
const {execFileSync} = require('node:child_process');
const katex = require(process.argv[2] || 'katex');
const source = fs.readFileSync(path.join(__dirname, 'GYRO_GRAVITY_ANIMATION.md'), 'utf8');
let count = 0;
function render(tex, display) {
  count++;
  const result = katex.renderToString(tex.trim(), {
    output: 'mathml', displayMode: display, throwOnError: true, strict: 'error', trust: false,
  });
  return `<span class="tg-math-${display ? 'block' : 'inline'}">${result}</span>`;
}
// Compile Markdown math outside fenced code, leaving the animation source intact.
let compiled = source.split(/(```[\s\S]*?```)/g).map(chunk => {
  if (chunk.startsWith('```')) return chunk;
  return chunk.replace(/\$\$([\s\S]*?)\$\$|\$([^$\n]+)\$/g,
    (_match, block, inline) => render(block ?? inline, block !== undefined));
}).join('');
const texBlock = source.match(/const equationTex=\[([\s\S]*?)\n \];/)[1];
const equationHtml = [...texBlock.matchAll(/String\.raw`([^`]*)`/g)].map(match => render(match[1], true));
if (equationHtml.length !== 6) throw new Error('Expected six animation equations');
compiled = compiled.replace('const equationMathML = /* @precompute-math */ [];',
  'const equationMathML = /* @precompute-math */ ' + JSON.stringify(equationHtml) + ';');
const cli = process.env.WT901_HTML_CLI || '/home/hzb/.codex/skills/answer-me-with-html/scripts/am.mjs';
const output = execFileSync(process.execPath, [cli, 'render', '-', '--no-open'], {
  input: compiled, encoding: 'utf8', maxBuffer: 4 * 1024 * 1024,
});
const generated = output.match(/✓ (.+\.html)/)?.[1];
if (!generated) throw new Error(output);
const target = path.join(__dirname, 'GYRO_GRAVITY_ANIMATION.html');
fs.copyFileSync(generated, target);
console.log(output.trim());
console.log(`Built ${count} formulas with KaTeX ${katex.version} into ${target}`);
