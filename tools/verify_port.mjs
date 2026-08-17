/* Verify docs/js/inference.js reproduces the torch backend exactly.
 *
 *   /home/atharv/venv/bin/python3 tools/dump_trace.py > /tmp/trace.json
 *   node tools/verify_port.mjs /tmp/trace.json
 */
import { readFileSync } from "node:fs";
import { AttentionStateInference } from "../docs/js/inference.js";

const trace = JSON.parse(readFileSync(process.argv[2], "utf8"));
const attn = new AttentionStateInference();

let worst = 0;
let worstFrame = -1;
for (let f = 0; f < trace.length; f++) {
  const { o, v, pi } = trace[f];
  const got = attn.forward(o, v[0], v[1]);
  for (let k = 0; k < 5; k++) {
    const err = Math.abs(got[k] - pi[k]);
    if (err > worst) {
      worst = err;
      worstFrame = f;
    }
  }
  const sum = got.reduce((a, b) => a + b, 0);
  if (Math.abs(sum - 1) > 1e-9) {
    console.error(`frame ${f}: pi does not sum to 1 (${sum})`);
    process.exit(1);
  }
}

console.log(`frames        : ${trace.length}`);
console.log(`max |dpi|     : ${worst.toExponential(3)} (frame ${worstFrame})`);
// float32 torch vs float64 JS: differences below ~1e-6 are pure dtype noise.
if (worst > 1e-5) {
  console.error("MISMATCH — the JS port diverges from the torch backend.");
  process.exit(1);
}
console.log("OK — JS inference matches the torch backend.");
