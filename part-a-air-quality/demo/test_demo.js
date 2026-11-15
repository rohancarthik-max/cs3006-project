// Checks that the demo page computes exactly what the Python model computes.
// Run from the project root:  node demo/test_demo.js
// (scripts/04_export_demo.py must have been run first.)
"use strict";
const fs = require("fs");
const path = require("path");
const vm = require("vm");

const dir = __dirname;
const ctx = { window: {} };
vm.createContext(ctx);
vm.runInContext(fs.readFileSync(path.join(dir, "data.js"), "utf8"), ctx);
const data = ctx.window.DEMO_DATA;
const M = require(path.join(dir, "model.js"));
const pyProb = JSON.parse(fs.readFileSync(path.join(dir, "check_probabilities.json"), "utf8"));

let failures = 0;
function check(name, ok, detail) {
  console.log(`${ok ? "PASS" : "FAIL"}  ${name}${detail ? "  (" + detail + ")" : ""}`);
  if (!ok) failures += 1;
}

// 1. Every probability matches Python.
let maxDiff = 0, sideFlips = 0;
data.rows.forEach((r, i) => {
  const p = M.predict(data.model, r[4]);
  maxDiff = Math.max(maxDiff, Math.abs(p - pyProb[i]));
  if (M.modelAlert(data.model, r[4]) !== (pyProb[i] >= data.model.threshold)) sideFlips += 1;
});
check("probabilities match the Python model", maxDiff < 1e-9, `max difference ${maxDiff.toExponential(2)} over ${data.rows.length} rows`);
check("no alert differs from Python", sideFlips === 0, `${sideFlips} rows on the other side of the threshold`);

// 2. The 2025 season reproduces the reported test results (notebook 05).
const s = M.seasonStats(data, 2025);
check("2025 bad days = 157", s.badDays === 157, `got ${s.badDays}`);
check("2025 model: 129 caught, 28 missed, 58 false alarms",
  s.model.caught === 129 && s.model.missed === 28 && s.model.falseAlarms === 58,
  `got ${s.model.caught}/${s.model.missed}/${s.model.falseAlarms}`);
check("2025 rule: 133 caught, 24 missed, 49 false alarms",
  s.rule.caught === 133 && s.rule.missed === 24 && s.rule.falseAlarms === 49,
  `got ${s.rule.caught}/${s.rule.missed}/${s.rule.falseAlarms}`);

// 3. The 2024 validation year reproduces notebook 04.
const v = M.seasonStats(data, 2024);
check("2024 model: 54 caught, 35 false alarms", v.model.caught === 54 && v.model.falseAlarms === 35,
  `got ${v.model.caught}/${v.model.falseAlarms}`);

console.log(failures ? `\n${failures} check(s) FAILED` : "\nAll checks passed.");
process.exit(failures ? 1 : 0);
