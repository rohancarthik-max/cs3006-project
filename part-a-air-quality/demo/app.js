// Interactive page for the next-day PM2.5 model. All model arithmetic lives in model.js.
(function () {
  "use strict";
  const D = window.DEMO_DATA, M = window.PM25Model, m = D.model;
  const F = (name) => m.features.indexOf(name);
  const $ = (id) => document.getElementById(id);
  const LABEL = {
    st_Manali: "Station: Manali", st_Kodungaiyur: "Station: Kodungaiyur", st_Arumbakkam: "Station: Arumbakkam",
    st_Perungudi: "Station: Perungudi", log_pm25: "PM2.5 today", log_pm25_change: "Change since yesterday",
    log_pm25_roll3: "3-day average PM2.5", log_pm25_roll7: "7-day average PM2.5", pm25_missing: "No reading today",
    city_log_pm25: "Other stations today", month_sin: "Season (sine)", month_cos: "Season (cosine)",
    dewpoint: "Dew point", rh: "Humidity", log_rain: "Rain today", log_rain_3d: "Rain, last 3 days",
    pressure: "Pressure", wind_u: "Wind west→east", wind_v: "Wind south→north", wind_speed: "Wind speed",
    festival_tomorrow: "Festival tomorrow", festival_today: "Festival today",
  };

  // ---- index rows by date
  const byDate = new Map();
  for (const r of D.rows) {
    if (!byDate.has(r[0])) byDate.set(r[0], []);
    byDate.get(r[0]).push(r);
  }
  for (const rs of byDate.values()) rs.sort((a, b) => a[1] - b[1]);
  const dateIdx = new Map(D.dates.map((d, i) => [d, i]));

  const state = { di: dateIdx.get("2025-10-19"), si: 2, override: null };

  // ---- helpers
  const fmt = (v, nd = 0) => (v === null || v === undefined || Number.isNaN(v) ? "–" : v.toFixed(nd));
  const pct = (v) => (Number.isNaN(v) ? "–" : (100 * v).toFixed(0) + "%");
  function addDays(iso, n) {
    const d = new Date(iso + "T00:00:00Z"); d.setUTCDate(d.getUTCDate() + n);
    return d.toISOString().slice(0, 10);
  }
  function nice(iso) {
    const d = new Date(iso + "T00:00:00Z");
    return d.toLocaleDateString("en-GB", { weekday: "short", day: "numeric", month: "short", year: "numeric", timeZone: "UTC" });
  }
  function splitOf(targetIso) {
    const y = +targetIso.slice(0, 4);
    for (const [name, years] of Object.entries(D.split_years)) if (years.includes(y)) return name;
    return "";
  }
  const SPLIT_TEXT = {
    train: ["training year", "The model learned from this year, so these results are in-sample and look better than they would on new data."],
    val: ["validation year (2024)", "The alert threshold was chosen on this year, so these results are slightly optimistic."],
    test: ["test year (2025)", "The model never saw this year: it was evaluated once, after every choice was fixed. These are honest results."],
    monitor: ["monitoring year (2026)", "Unseen data after the test year, kept aside for monitoring."],
  };
  function outcome(alert, next) {
    if (next === null) return ["no reading", "na"];
    const bad = next > m.limit_ugm3;
    if (bad && alert) return ["caught", "caught"];
    if (bad && !alert) return ["missed", "missed"];
    if (!bad && alert) return ["false alarm", "fa"];
    return ["correct: no alert", "ok"];
  }

  // ---- header
  $("modelBadge").textContent =
    `Logistic regression · trained on ${m.trained_on[0]}-${m.trained_on[m.trained_on.length - 1]} · ` +
    `alert if probability ≥ ${m.threshold.toFixed(3)} · ${D.rows.length.toLocaleString()} station-days`;

  // ---- 1. date controls
  const di0 = 0, di1 = D.dates.length - 1;
  $("dateInput").min = D.dates[di0]; $("dateInput").max = D.dates[di1];
  const PRESETS = [
    ["Eve of Diwali 2025", "2025-10-19"], ["Before the biggest miss (2 Dec 2025)", "2025-12-02"],
    ["Eve of Bhogi 2025", "2025-01-12"], ["A monsoon day (15 Jul 2025)", "2025-07-15"],
    ["The day the threshold was set (11 Jan 2024)", "2024-01-11"],
  ];
  for (const [label, iso] of PRESETS) {
    if (!dateIdx.has(iso)) continue;
    const b = document.createElement("button");
    b.type = "button"; b.textContent = label;
    b.addEventListener("click", () => go(dateIdx.get(iso)));
    $("presets").appendChild(b);
  }
  $("prevDay").addEventListener("click", () => go(Math.max(di0, state.di - 1)));
  $("nextDay").addEventListener("click", () => go(Math.min(di1, state.di + 1)));
  $("dateInput").addEventListener("change", (e) => {
    const iso = e.target.value;
    if (dateIdx.has(iso)) return go(dateIdx.get(iso));
    // nearest available day (weather archive and sensor gaps leave a few holes)
    let best = state.di, bestGap = Infinity;
    D.dates.forEach((d, i) => { const g = Math.abs(new Date(d) - new Date(iso)); if (g < bestGap) { bestGap = g; best = i; } });
    go(best);
  });
  function go(di) { state.di = di; state.override = null; render(); }

  // ---- 2. bulletin
  function renderBulletin() {
    const iso = D.dates[state.di], target = addDays(iso, 1), sp = splitOf(target);
    $("dateInput").value = iso;
    const tag = $("splitTag");
    tag.textContent = SPLIT_TEXT[sp] ? SPLIT_TEXT[sp][0] : "";
    tag.className = "split" + (sp === "test" ? " test" : "");
    $("splitNote").textContent = SPLIT_TEXT[sp] ? SPLIT_TEXT[sp][1] : "";
    $("bulletinSub").textContent = `Issued at the end of ${nice(iso)}, for ${nice(target)}.`;
    const tbody = document.querySelector("#bulletin tbody");
    tbody.innerHTML = "";
    const rows = byDate.get(state.di) || [];
    if (!rows.some((r) => r[1] === state.si)) state.si = rows.length ? rows[0][1] : state.si;
    for (const r of rows) {
      const [, si, today, next, x] = r;
      const p = M.predict(m, x), ma = M.modelAlert(m, x), ra = M.ruleAlert(m, x);
      const [oText, oCls] = outcome(ma, next);
      const tr = document.createElement("tr");
      if (si === state.si) tr.className = "sel";
      tr.innerHTML =
        `<th scope="row">${D.stations[si]}</th>` +
        `<td class="num">${today === null ? '<span class="no">no reading</span>' : fmt(today)}</td>` +
        `<td><div class="pbar"><span class="num">${(100 * p).toFixed(0)}%</span><span class="track">` +
        `<span class="fill" style="width:${(100 * p).toFixed(1)}%"></span>` +
        `<span class="thr" style="left:${(100 * m.threshold).toFixed(1)}%" title="alert threshold"></span></span></div></td>` +
        `<td class="${ma ? "yes" : "no"}">${ma ? "ALERT" : "no"}</td>` +
        `<td class="${ra ? "yes" : "no"}">${ra ? "ALERT" : "no"}</td>` +
        `<td class="num">${next === null ? '<span class="no">no reading</span>' : fmt(next) + (next > m.limit_ugm3 ? ' <span class="yes">over</span>' : "")}</td>` +
        `<td><span class="chip ${oCls}">${oText}</span></td>`;
      tr.addEventListener("click", () => { state.si = si; state.override = null; render(); });
      tbody.appendChild(tr);
    }
  }

  // ---- 3. explanation + what-if
  function currentRow() { return (byDate.get(state.di) || []).find((r) => r[1] === state.si); }

  function renderWhy(x) {
    const cs = M.contributions(m, x).map((c, i) => ({ ...c, i }));
    cs.sort((a, b) => Math.abs(b.contrib) - Math.abs(a.contrib));
    const top = cs.slice(0, 8), rest = cs.slice(8).reduce((s, c) => s + c.contrib, 0);
    const items = [...top.map((c) => ({ label: LABEL[c.feature] + (c.missing ? " (filled in)" : ""), v: c.contrib })),
      { label: "All other inputs", v: rest }];
    const W = 560, rowH = 26, H = items.length * rowH + 50, mid = 300, scale = 70;
    const maxAbs = Math.max(1, ...items.map((d) => Math.abs(d.v)));
    const s = Math.min(scale, 250 / maxAbs);
    let svg = `<svg viewBox="0 0 ${W} ${H}" xmlns="http://www.w3.org/2000/svg">`;
    svg += `<line x1="${mid}" y1="4" x2="${mid}" y2="${H - 40}" stroke="var(--muted)" stroke-width="1"/>`;
    items.forEach((d, k) => {
      const y = 8 + k * rowH, w = Math.abs(d.v) * s, x0 = d.v >= 0 ? mid : mid - w;
      svg += `<text x="${mid - 8 - (d.v < 0 ? w : 0)}" y="${y + 13}" text-anchor="end" class="lbl">${d.label}</text>`;
      svg += `<rect x="${x0}" y="${y}" width="${Math.max(w, 1)}" height="16" rx="3" fill="${d.v >= 0 ? "var(--up)" : "var(--down)"}"/>`;
      svg += `<text x="${d.v >= 0 ? mid + w + 4 : mid + 4}" y="${y + 13}">${d.v >= 0 ? "+" : ""}${d.v.toFixed(2)}</text>`;
    });
    const lo = M.logOdds(m, x), p = M.predict(m, x);
    svg += `<text x="4" y="${H - 18}">Average day: log-odds ${m.intercept.toFixed(2)} → ${(100 / (1 + Math.exp(-m.intercept))).toFixed(1)}%</text>`;
    svg += `<text x="4" y="${H - 3}" class="lbl">This day: log-odds ${lo.toFixed(2)} → ${(100 * p).toFixed(1)}% chance tomorrow is over 60</text>`;
    svg += `</svg>`;
    $("whyChart").innerHTML = svg;
  }

  function withOverrides(x0, o) {
    const x = x0.slice();
    if (o.pm !== null && x0[F("log_pm25")] !== null) {
      const oldL = x0[F("log_pm25")], newL = Math.log(o.pm), d = newL - oldL;
      x[F("log_pm25")] = newL;
      if (x0[F("log_pm25_change")] !== null) x[F("log_pm25_change")] = x0[F("log_pm25_change")] + d;
      if (x0[F("log_pm25_roll3")] !== null) x[F("log_pm25_roll3")] = x0[F("log_pm25_roll3")] + d / 3;
      if (x0[F("log_pm25_roll7")] !== null) x[F("log_pm25_roll7")] = x0[F("log_pm25_roll7")] + d / 7;
    }
    if (o.rain !== null) {
      const oldR = Math.expm1(x0[F("log_rain")] ?? 0), old3 = Math.expm1(x0[F("log_rain_3d")] ?? 0);
      x[F("log_rain")] = Math.log1p(o.rain);
      x[F("log_rain_3d")] = Math.log1p(Math.max(0, old3 - oldR + o.rain));
    }
    x[F("festival_tomorrow")] = o.fest ? 1 : 0;
    return x;
  }

  function renderWhatIf(r) {
    const x0 = r[4];
    const base = {
      pm: x0[F("log_pm25")] === null ? null : Math.exp(x0[F("log_pm25")]),
      rain: Math.expm1(x0[F("log_rain")] ?? 0), fest: x0[F("festival_tomorrow")] === 1,
    };
    if (!state.override) state.override = { ...base };
    const o = state.override;
    const pmS = $("pmSlider");
    pmS.disabled = base.pm === null;
    if (base.pm !== null) { pmS.value = Math.round(o.pm); $("pmOut").textContent = Math.round(o.pm); }
    else $("pmOut").textContent = "no reading today";
    $("rainSlider").value = Math.round(o.rain); $("rainOut").textContent = Math.round(o.rain);
    $("festBox").checked = o.fest;
    const x = withOverrides(x0, o);
    const p0 = M.predict(m, x0), p = M.predict(m, x), a = M.modelAlert(m, x);
    const changed = Math.abs(p - p0) > 1e-12;
    $("whatIfResult").innerHTML =
      `<span class="big">${(100 * p).toFixed(1)}%</span> chance tomorrow is over 60 · ` +
      `<span class="${a ? "yes" : "no"}">${a ? "ALERT" : "no alert"}</span><br>` +
      `<span class="sub">${changed ? `Real day: ${(100 * p0).toFixed(1)}%. ` : ""}PM2.5 changes also move the change-since-yesterday and the 3- and 7-day averages (approximately).</span>`;
    renderWhy(x);
  }

  for (const [id, key, conv] of [["pmSlider", "pm", Number], ["rainSlider", "rain", Number]]) {
    $(id).addEventListener("input", (e) => { state.override[key] = conv(e.target.value); renderExplain(); });
  }
  $("festBox").addEventListener("change", (e) => { state.override.fest = e.target.checked; renderExplain(); });
  $("resetWhatIf").addEventListener("click", () => { state.override = null; renderExplain(); });

  function renderExplain() {
    const r = currentRow();
    if (!r) return;
    $("whyStation").textContent = `· ${D.stations[r[1]]}, ${nice(D.dates[r[0]])}`;
    renderWhatIf(r);
  }

  // ---- 4. season
  const years = [...new Set(D.dates.map((d) => +addDays(d, 1).slice(0, 4)))].filter((y) => y >= 2021);
  for (const y of years) $("yearSel").add(new Option(`${y} (${(SPLIT_TEXT[splitOf(y + "-06-01")] || [""])[0]})`, y));
  $("yearSel").value = 2025;
  $("stationSel").add(new Option("All four stations", ""));
  D.stations.forEach((s, i) => $("stationSel").add(new Option(s, i)));
  $("yearSel").addEventListener("change", renderSeason);
  $("stationSel").addEventListener("change", renderSeason);

  function renderSeason() {
    const y = +$("yearSel").value, sv = $("stationSel").value, si = sv === "" ? null : +sv;
    const st = M.seasonStats(D, y, si);
    $("yearNote").textContent = (SPLIT_TEXT[splitOf(y + "-06-01")] || ["", ""])[1] +
      (y === 2025 && si === null ? " These counts are the project's reported test results." : "");
    const tb = document.querySelector("#seasonTable tbody");
    tb.innerHTML = "";
    for (const [name, k] of [["Model (logistic regression)", "model"], ["Rule: today > 55", "rule"]]) {
      const s = st[k];
      tb.insertAdjacentHTML("beforeend",
        `<tr><th scope="row">${name}</th><td class="num">${st.badDays}</td><td class="num">${s.caught}</td>` +
        `<td class="num">${s.missed}</td><td class="num">${s.falseAlarms}</td><td class="num">${pct(s.recall)}</td>` +
        `<td class="num">${pct(s.precision)}</td></tr>`);
    }
    drawTimeline(y, si === null ? state.si : si);
  }

  function drawTimeline(year, si) {
    const pts = [];
    for (const r of D.rows) {
      if (r[1] !== si || r[3] === null) continue;
      const t = addDays(D.dates[r[0]], 1);
      if (+t.slice(0, 4) !== year) continue;
      pts.push({ t, di: r[0], next: r[3], a: M.modelAlert(m, r[4]) });
    }
    const W = 1000, H = 260, L = 44, R = 10, T = 22, B = 26;
    const t0 = Date.UTC(year, 0, 1), t1 = Date.UTC(year + 1, 0, 1);
    const ymax = Math.max(120, Math.min(250, Math.max(0, ...pts.map((p) => p.next)) * 1.05));
    const X = (iso) => L + ((Date.parse(iso + "T00:00:00Z") - t0) / (t1 - t0)) * (W - L - R);
    const Y = (v) => T + (1 - Math.min(v, ymax) / ymax) * (H - T - B);
    let svg = `<svg viewBox="0 0 ${W} ${H}" xmlns="http://www.w3.org/2000/svg">`;
    svg += `<text x="${L}" y="14" class="lbl">${D.stations[si]}, ${year}: PM2.5 on each forecast day (µg/m³)</text>`;
    for (const v of [0, 60, 120, 180, 240].filter((v) => v <= ymax)) {
      svg += `<line x1="${L}" x2="${W - R}" y1="${Y(v)}" y2="${Y(v)}" stroke="var(--line)"/>` +
             `<text x="${L - 6}" y="${Y(v) + 4}" text-anchor="end">${v}</text>`;
    }
    svg += `<line x1="${L}" x2="${W - R}" y1="${Y(60)}" y2="${Y(60)}" stroke="var(--ink2)" stroke-dasharray="5 4"/>`;
    ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"].forEach((mo, k) => {
      const iso = `${year}-${String(k + 1).padStart(2, "0")}-01`;
      svg += `<text x="${X(iso) + 2}" y="${H - 8}">${mo}</text>`;
    });
    let path = "", prev = null;
    for (const p of pts) {
      const gap = prev && Date.parse(p.t) - Date.parse(prev.t) > 86400000 * 1.5;
      path += `${!prev || gap ? "M" : "L"}${X(p.t).toFixed(1)},${Y(p.next).toFixed(1)}`;
      prev = p;
    }
    svg += `<path d="${path}" fill="none" stroke="var(--muted)" stroke-width="1.2"/>`;
    for (const p of pts) {
      const bad = p.next > m.limit_ugm3, cx = X(p.t).toFixed(1), cy = Y(p.next).toFixed(1);
      const tip = `<title>${nice(p.t)}: ${p.next.toFixed(0)} µg/m³ — ${bad ? (p.a ? "caught" : "missed") : "false alarm"} (click to open)</title>`;
      if (bad && p.a) svg += `<circle cx="${cx}" cy="${cy}" r="4.5" fill="var(--blue)" data-di="${p.di}" style="cursor:pointer">${tip}</circle>`;
      else if (bad) svg += `<circle cx="${cx}" cy="${cy}" r="4.5" fill="var(--card)" stroke="var(--alert)" stroke-width="2" data-di="${p.di}" style="cursor:pointer">${tip}</circle>`;
      else if (p.a) svg += `<text x="${cx}" y="${+cy + 4}" text-anchor="middle" class="lbl" font-size="13" data-di="${p.di}" style="cursor:pointer">×${tip}</text>`;
    }
    svg += `</svg>`;
    const el = $("seasonChart");
    el.innerHTML = svg;
    el.querySelectorAll("[data-di]").forEach((n) => n.addEventListener("click", () => {
      state.si = si; go(+n.getAttribute("data-di")); window.scrollTo({ top: 0, behavior: "smooth" });
    }));
  }

  function render() { renderBulletin(); renderExplain(); renderSeason(); }
  render();
})();
