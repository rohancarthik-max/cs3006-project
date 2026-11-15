// The saved model's arithmetic, shared by the page (app.js) and the test (test_demo.js).
// Pipeline exported from scikit-learn by scripts/04_export_demo.py:
//   fill missing inputs with the training median -> standardise -> logistic regression.
(function (root) {
  "use strict";

  function filled(m, x) {
    return x.map((v, i) => (v === null || v === undefined || Number.isNaN(v) ? m.median[i] : v));
  }

  // Log-odds contribution of every feature: coef * (value - mean) / scale.
  function contributions(m, x) {
    const v = filled(m, x);
    return v.map((val, i) => ({
      feature: m.features[i], value: val, missing: x[i] === null,
      contrib: (m.coef[i] * (val - m.mean[i])) / m.scale[i],
    }));
  }

  function logOdds(m, x) {
    return contributions(m, x).reduce((s, c) => s + c.contrib, m.intercept);
  }

  function predict(m, x) {
    return 1 / (1 + Math.exp(-logOdds(m, x)));
  }

  // The reference rule as evaluated in stage 7: today's log PM2.5, or the other
  // stations' mean when the station has no reading, compared with a cut-off.
  function ruleScore(m, x) {
    const f = m.features;
    const own = x[f.indexOf("log_pm25")];
    const city = x[f.indexOf("city_log_pm25")];
    return own !== null ? own : city !== null ? city : -1;
  }

  // Probabilities within 1e-9 of the threshold count as equal to it: the threshold was
  // chosen AT a real day's probability, and the browser's floating-point sums can differ
  // from NumPy's in the last digits.
  const EPS = 1e-9;
  function modelAlert(m, x) { return predict(m, x) >= m.threshold - EPS; }
  function ruleAlert(m, x) { return ruleScore(m, x) >= m.rule_log - EPS; }

  // Model and rule results for every labelled row whose TARGET day falls in `year`.
  function seasonStats(data, year, stationIdx) {
    const m = data.model;
    const out = {
      model: { caught: 0, missed: 0, falseAlarms: 0, alerts: 0 },
      rule: { caught: 0, missed: 0, falseAlarms: 0, alerts: 0 },
      badDays: 0, rows: 0,
    };
    for (const r of data.rows) {
      const [di, si, , next, x] = r;
      if (next === null) continue;
      if (stationIdx !== undefined && stationIdx !== null && si !== stationIdx) continue;
      const target = new Date(data.dates[di] + "T00:00:00Z");
      target.setUTCDate(target.getUTCDate() + 1);
      if (target.getUTCFullYear() !== year) continue;
      const bad = next > m.limit_ugm3;
      const a = { model: modelAlert(m, x), rule: ruleAlert(m, x) };
      out.rows += 1;
      if (bad) out.badDays += 1;
      for (const k of ["model", "rule"]) {
        if (a[k]) out[k].alerts += 1;
        if (a[k] && bad) out[k].caught += 1;
        if (!a[k] && bad) out[k].missed += 1;
        if (a[k] && !bad) out[k].falseAlarms += 1;
      }
    }
    for (const k of ["model", "rule"]) {
      const s = out[k];
      s.recall = out.badDays ? s.caught / out.badDays : NaN;
      s.precision = s.alerts ? s.caught / s.alerts : NaN;
    }
    return out;
  }

  const api = { filled, contributions, logOdds, predict, ruleScore, modelAlert, ruleAlert, seasonStats };
  root.PM25Model = api;
  if (typeof module !== "undefined" && module.exports) module.exports = api;
})(typeof window !== "undefined" ? window : globalThis);
