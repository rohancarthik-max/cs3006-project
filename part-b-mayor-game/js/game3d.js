/* game3d.js - the 3D way to play: you walk around Navapuram as the mayor.
 *
 * The game's screens (brief, data tools, policy desk, outcome, reveal, quiz...) are the same as the
 * classic version; here they open as panels over the 3D town when you walk up to the right place:
 *   Town Hall        -> Dr. Asha's brief, then the policy desk where you sign your order
 *   Data Office      -> the data terminal (all the analysis tools)
 *   Newspaper kiosk  -> today's headline
 *   any ward         -> an inspection card for that ward
 * When you sign an order, the season plays out week by week in the 3D town itself. */
(function (M) {
  'use strict';
  var world, overlay, appEl, closeBtn, promptEl, hudLevel, hudObjective, hudWeek, hudMoney, seasonBox;
  var phase = 'start', closable = false, ambientTimer = null, ambientT = 0, visitedData = false;
  var approval = 50, hudApproval; // the town's opinion of you: moves with your true results, decides the election
  var inspected = {}, readPaper = false, cluesSeen = {}, tipsShown = {}; // case-file bookkeeping

  function $(id) { return document.getElementById(id); }
  function S() { return M.ui._state(); }
  function L() { return S().level; }

  // ---------------------------------------------------------------- overlay panel
  function openOverlay(canClose) {
    overlay.classList.add('open');
    closable = !!canClose;
    closeBtn.style.display = closable ? '' : 'none';
    if (!overlay.classList.contains('was')) M.sfx.open();
    overlay.classList.add('was');
    world.setInputEnabled(false);
    promptEl.classList.remove('show');
    world.setOrbit(!S().level);
    world.setPaused(!!S().level); // nothing renders behind a panel during a level
  }
  function closeOverlay() {
    overlay.classList.remove('open'); overlay.classList.remove('was');
    M.sfx.close();
    world.setOrbit(false);
    world.setPaused(false);
    world.setInputEnabled(true);
    M.charts.hideTip();
    hud();
  }
  // ui.js calls this every time it draws a screen
  function onRender() { openOverlay(phase === 'explore' || phase === 'decide'); overlay.scrollTop = 0; }

  function custom(html, canClose) {
    appEl.innerHTML = html;
    openOverlay(canClose !== false);
    appEl.classList.remove('enter'); void appEl.offsetWidth; appEl.classList.add('enter');
  }

  // ---------------------------------------------------------------- HUD
  function objective(text, beacon) { hudObjective.innerHTML = text; world.setBeacon(beacon || null); }
  function hud() {
    var st = S(), lv = st.level;
    var face = approval >= 70 ? '😀' : approval >= 55 ? '🙂' : approval >= 45 ? '😐' : approval >= 30 ? '😟' : '😠';
    hudApproval.innerHTML = face + ' Approval <b>' + Math.round(approval) + '%</b>';
    speech();
    if (!lv) { hudLevel.textContent = 'Navapuram'; hudMoney.textContent = ''; return; }
    hudLevel.textContent = (lv.part ? 'Level 5 · problem ' + lv.part + ' of 3' : 'Level ' + lv.n + ' of 5') + ' · ' + lv.title;
    var total = lv.shared ? M.TERM_BUDGET - st.termSpent : lv.model.budget;
    var pil = (st.pilots || []).reduce(function (a, p) { return a + p.cost; }, 0);
    hudMoney.innerHTML = '₹ <b>' + (total - pil) + '</b> lakh';
  }

  // What the three townspeople say, depending on where you are in the level.
  function speech() {
    var lv = L(), st = S();
    if (!lv) { world.say('asha', 'Welcome to Navapuram, Mayor!'); world.say('editor', 'Extra! New mayor takes office!'); world.say('clerk', 'Good evening, Mayor.'); return; }
    var paper = lv.headline ? lv.headline(st.world) : null;
    world.say('editor', paper ? paper.head : 'No news today.');
    world.say('clerk', lv.model.kind === 'panel' ? 'All 20 wards\' records are on my terminal, Mayor.' : 'The student records are on my terminal, Mayor.');
    world.say('asha', phase === 'brief' ? 'Mayor! Come inside. We need to talk about ' + lv.title.toLowerCase() + '.'
      : phase === 'explore' ? 'Don\'t trust the headline. Check the data first.'
      : phase === 'decide' ? 'Ready? Sign your order at my desk inside.'
      : phase === 'season' ? 'Fingers crossed, Mayor.' : 'Let\'s see what really happened.');
  }

  // ---------------------------------------------------------------- ambient: last year replays in the town
  function startAmbient() {
    stopAmbient();
    var st = S(), lv = st.level;
    if (!lv || lv.model.kind !== 'panel') { hudWeek.textContent = ''; return; }
    var spec = M.ui.townSpec(), h = st.world.history, n = st.world.weeks.length;
    function show() {
      world.update(M.town.frameAt(h, ambientT, '', spec));
      hudWeek.innerHTML = 'Last year · week <b>' + st.world.weeks[ambientT] + '</b>';
      ambientT = (ambientT + 1) % n;
    }
    show();
    ambientTimer = setInterval(show, 1300);
  }
  function stopAmbient() { if (ambientTimer) clearInterval(ambientTimer); ambientTimer = null; }

  // ---------------------------------------------------------------- level flow hooks
  function onLevel(lv) {
    phase = 'brief'; visitedData = false; inspected = {}; readPaper = false; cluesSeen = {};
    world.setPolicy(null); world.setMask(null);
    if (lv.model.kind === 'panel') world.setLevel(M.ui.townSpec()); else world.setLevel(null);
    ambientT = 0; startAmbient();
    world.teleport(0, -33, 0);
    objective('Meet Dr. Asha at the <b>Town Hall</b>.', 'townhall');
    hud();
  }
  function afterBrief() {
    phase = 'explore';
    closeOverlay();
    var lv = L();
    var where = lv.model.kind === 'panel' ? 'the <b>Data Office</b> terminal, walk into wards to inspect them, read the <b>newspaper</b>'
      : 'the student records at the <b>Data Office</b> (or the <b>School</b>), read the <b>newspaper</b>';
    objective('Investigate: ' + where + '. Then sign your order at the <b>Town Hall</b>.', 'data');
    tip('walk', '🚶 Walk with W A S D or the arrow keys (hold Shift to run), or click the ground or the minimap.');
    setTimeout(function () { tip('case', '📒 Your case file (press J) lists clues worth checking.'); }, 3800);
  }
  function afterExplore() {
    phase = 'decide'; visitedData = true;
    closeOverlay();
    objective('Decide. Sign your order at the <b>Town Hall</b> desk (or keep investigating).', 'townhall');
    hud();
  }

  // The season plays out in the 3D town, with and without your plan, week by week.
  function playSeason(res, done) {
    phase = 'season';
    stopAmbient();
    closeOverlay();
    var lv = L(), spec = M.ui.townSpec(), WARM = lv.model.warm || 0;
    var base = res.ev.base.weekly, mine = res.run.weekly, n = mine.length, k = 0, timer = null;
    objective('The season is under way. Watch the town (you can walk around).', null);
    M.sfx.sign(); hud();
    seasonBox.innerHTML = '<div class="season-head"><b>' + lv.title + '</b>: your plan, <i>' + res.policy.label + '</i></div>' +
      '<div class="season-row"><span>Week <b id="sw">0</b> / ' + n + '</span><span class="with">With your plan: <b id="sm">0</b></span><span class="without">Without: <b id="sb">0</b></span></div>' +
      '<div id="schart"></div><button class="btn small-btn" id="sskip">Skip to the result ▸</button>';
    seasonBox.classList.add('show');
    var chart = M.charts.lineChart($('schart'), { x: mine.map(function (_, i) { return i + 1; }), xLabel: 'Week', yLabel: 'Cases', yMin: 0, height: 190,
      yMax: Math.max(M.stats.max(base), M.stats.max(mine)) * 1.08,
      series: [{ name: 'Without', values: base, color: 'var(--s2)', dash: true }, { name: 'With your plan', values: mine, color: 'var(--s1)' }] });
    chart.setProgress(0);
    function step() {
      var i = k;
      world.update(M.town.frameAt(res.run.vars, i + WARM, '', spec));
      var ph = res.pilots.filter(function (p) { return i >= p.phase.from && i < p.phase.to; })[0];
      if (ph) { world.setMask(ph.phase.mask); world.setPolicy(lv.policyVisual[ph.policy], ph.phase.mask); }
      else { world.setMask(null); world.setPolicy(lv.policyVisual[res.choice] || null); }
      k++;
      chart.setProgress(k);
      $('sw').textContent = k; $('sm').textContent = M.charts.fmt(M.stats.sum(mine.slice(0, k))); $('sb').textContent = M.charts.fmt(M.stats.sum(base.slice(0, k)));
      hudWeek.innerHTML = 'This season · week <b>' + k + '</b>';
      if (k >= n) finish();
    }
    function finish() {
      clearInterval(timer); timer = null;
      setTimeout(function () {
        seasonBox.classList.remove('show');
        world.setPolicy(null); world.setMask(null);
        phase = 'result';
        objective('', null);
        done();
      }, 900);
    }
    $('sskip').addEventListener('click', function () { if (!timer) return; clearInterval(timer); timer = null; while (k < n) step(); });
    step();
    timer = setInterval(step, 1400);
  }

  // The randomised pilot happens in the town: coins fly up over every ward and land T or C,
  // then the pilot weeks play in the treated wards only.
  function pilotPlay(info, done) {
    stopAmbient();
    closeOverlay();
    var lv = L(), spec = M.ui.townSpec(), WARM = lv.model.warm || 0;
    objective('Pilot: a coin decides which wards get <b>' + info.label + '</b>. T = gets it, C = comparison.', null);
    world.setPolicy(null); world.setMask(null);
    world.coinToss(info.mask, function (i, heads) { M.sfx.coin(heads); }, function () {
      var k = 0;
      world.setPolicy(lv.policyVisual[info.policy], info.mask);
      objective('Pilot running in the <b>T</b> wards for ' + info.weeks + ' weeks. Watch T against C.', null);
      function step() {
        world.update(M.town.frameAt(info.run.vars, info.phase.from + k + WARM, '', spec));
        hudWeek.innerHTML = 'Pilot week <b>' + (k + 1) + '</b> / ' + info.weeks;
        k++;
        if (k >= info.weeks) {
          clearInterval(tm);
          setTimeout(function () {
            world.setPolicy(null); world.setMask(null);
            objective('Decide. Sign your order at the <b>Town Hall</b> desk.', 'townhall');
            startAmbient();
            done();
          }, 1200);
        }
      }
      step();
      var tm = setInterval(step, 1300);
    });
  }

  // After each problem the town's approval moves with your TRUE result (not with what the paper said).
  function onResult(res) {
    var before = approval;
    approval = Math.max(0, Math.min(100, approval + (res.score.total - 50) / 3 - (res.fellForTrap ? 4 : 0)));
    res.approvalChange = approval - before;
    (res.score.total >= 50 ? M.sfx.good : M.sfx.bad)();
    hud();
  }
  function reportExtra() {
    var won = approval >= 50, pct = Math.round(approval);
    if (won) setTimeout(M.fx.confetti, 600);
    return '<section class="card election"><p class="eyebrow">Election night in Navapuram</p><h1>' + (won ? 'Re-elected!' : 'Voted out') + '</h1>' +
      '<div class="vote"><div class="vote-bar"><span style="width:' + pct + '%"></span></div><p><b>' + pct + '%</b> of Navapuram voted for you.</p></div>' +
      M.town.advisorHTML('<p>' + (won ? 'The town noticed what actually worked, Mayor. Well done.' :
        'The town judged you on results, not headlines, and the results weren\'t there. Next term, test before you trust.') + '</p>', won ? 'happy' : 'worried') +
      '</section>';
  }

  // ---------------------------------------------------------------- case file: clues worth checking (never the answer)
  function clues() {
    var lv = L(); if (!lv) return [];
    var st = S(), fl = st.flags || {}, used = st.toolsUsed || {}, nIns = Object.keys(inspected).length;
    var list = [{ id: 'paper', text: 'Read what the newspaper claims', done: readPaper }];
    if (lv.model.kind === 'panel') list.push({ id: 'wards', text: 'Inspect at least 2 wards in person (' + Math.min(2, nIns) + '/2)', done: nIns >= 2 });
    if (lv.type === 'confounding') list.push({ id: 'scatter', text: 'Plot the suspect against the outcome (Scatter + r)', done: !!used.scatter },
      { id: 'wardavg', text: 'Compare wards on their yearly averages, where the season cancels out', done: !!(fl.wardavg || fl.splitp) });
    if (lv.type === 'reverse') list.push({ id: 'time', text: 'Watch the timing on the Time series', done: !!used.timeseries },
      { id: 'lag', text: 'Check which moved first (What changed first?)', done: !!fl.lag });
    if (lv.type === 'simpson') list.push({ id: 'class', text: 'Look at every student in the Class view', done: !!used['class'] },
      { id: 'split', text: 'Compare like with like: split by prior-marks group', done: !!fl.splitGroup });
    if (lv.type === 'rct') list.push({ id: 'splitp', text: 'Try splitting by a third measure (Split by group)', done: !!fl.splitp },
      { id: 'pilot', text: 'Settle it with a randomised pilot', done: !!fl.pilot });
    return list;
  }
  function renderCaseFile() {
    var list = clues(), box = $('casefile');
    if (!list.length) { box.classList.remove('show'); $('casebtn').style.display = 'none'; return; }
    $('casebtn').style.display = '';
    var done = list.filter(function (c) { return c.done; }).length;
    $('casebtn').innerHTML = '📒 Case file <b>' + done + '/' + list.length + '</b>';
    box.innerHTML = '<h3>Case file: ' + L().title + '</h3><ul>' + list.map(function (c) {
      return '<li class="' + (c.done ? 'done' : '') + '">' + (c.done ? '✔' : '○') + ' ' + c.text + '</li>'; }).join('') + '</ul>' +
      '<p class="muted small">Clues help you decide. They never tell you the answer.</p>';
    list.forEach(function (c) {
      var key = L().id + ':' + c.id;
      if (c.done && !cluesSeen[key]) { cluesSeen[key] = true; if (phase === 'explore' || phase === 'decide') { toast('🔎 Clue found: ' + c.text.replace(/ \(\d\/2\)$/, '')); M.sfx.coin(true); } }
    });
  }
  function toast(text) {
    var el = document.createElement('div'); el.className = 'toast'; el.textContent = text;
    $('toasts').appendChild(el);
    setTimeout(function () { el.classList.add('out'); }, 3200);
    setTimeout(function () { el.remove(); }, 3800);
  }
  function tip(id, text) { if (tipsShown[id]) return; tipsShown[id] = true; toast(text); }

  // ---------------------------------------------------------------- places in town
  function onHotspot(h) {
    var lv = L();
    if (!lv) return;
    if (h.id === 'townhall') {
      if (phase === 'brief') M.ui.brief();
      else if (phase === 'explore' || phase === 'decide') M.ui.desk();
      return;
    }
    if (h.id === 'data' || (h.id === 'school' && lv.model.kind !== 'panel')) {
      if (phase === 'brief') { custom(asha('Come and see me at the Town Hall first, Mayor. I\'ll explain the problem.'), true); return; }
      if (phase === 'explore' || phase === 'decide') { visitedData = true; M.ui.explore(); }
      return;
    }
    if (h.id === 'school') { custom(asha('The Government School: 1,200 students. Its records matter in the coaching problem (Level 3).'), true); return; }
    if (h.id === 'kiosk') {
      readPaper = true;
      var p = lv.headline ? lv.headline(S().world) : null;
      custom('<section class="card"><h2>The Navapuram Times</h2>' + (p ? M.fx.newspaper(p.head, p.sub, p.edition) : '<p>No news today.</p>') +
        '<p class="muted small">The paper is confident. Is it right? Check the data before you act.</p><div class="actions"><button class="btn primary" id="back3d">Back to town</button></div></section>');
      $('back3d').addEventListener('click', closeOverlay);
      return;
    }
    if (h.ward !== undefined) inspectWard(h.ward);
  }
  function asha(text) {
    return '<section class="card">' + M.town.advisorHTML('<p>' + text + '</p>', 'neutral') + '<div class="actions"><button class="btn primary" onclick="Mayor.game3d.close()">Back to town</button></div></section>';
  }
  function inspectWard(i) {
    var lv = L(), st = S();
    if (lv.model.kind !== 'panel') { custom(asha('This problem is about students, not wards. Their records are at the Data Office.'), true); return; }
    var spec = M.ui.townSpec(), h = st.world.history, frame = world.current();
    var rows = [{ label: spec.outcome.label, key: spec.outcome.key, dp: 0 }].concat(spec.layers.map(function (Ly) { return { label: Ly.label, key: Ly.key, dp: Ly.dp }; }));
    var wk = ambientT === 0 ? st.world.weeks.length - 1 : ambientT - 1;
    custom('<section class="card"><p class="eyebrow">Ward inspection</p><h1>Ward ' + (i + 1) + ': ' + world.wardName(i) + '</h1>' +
      '<p class="muted">Last year, week ' + st.world.weeks[wk] + ', as you walked in:</p>' +
      '<div class="stats">' + rows.map(function (r) { return '<div class="stat"><span class="stat-num">' + M.charts.fmt(h[r.key][i][wk], r.dp) + '</span><span class="stat-lbl">' + r.label + '</span></div>'; }).join('') + '</div>' +
      '<h3>This ward over last year</h3><div id="wardchart"></div>' +
      '<p class="muted small">Each line is scaled to its own peak (100%) so you can compare timing.</p>' +
      '<div class="actions"><button class="btn primary" id="back3d">Back to town</button></div></section>', true);
    var cols = ['var(--s1)', 'var(--s3)', 'var(--s4)', 'var(--s2)'];
    M.charts.lineChart($('wardchart'), { x: st.world.weeks, xLabel: 'Week', yLabel: '% of its peak', yMin: 0, yMax: 105, height: 260,
      series: rows.map(function (r, j) { var raw = h[r.key][i]; var pk = M.stats.max(raw) || 1;
        return { name: r.label, values: raw.map(function (x) { return 100 * x / pk; }), tipValues: raw, dp: r.dp, color: cols[j % cols.length] }; }) });
    $('back3d').addEventListener('click', closeOverlay);
    inspected[i] = true;
    M.log.event('ward_inspect', { level: lv.id, ward: i + 1 });
  }

  // ---------------------------------------------------------------- boot
  function boot() {
    if (!M.world3d.supported()) { location.replace('classic.html'); return; }
    overlay = $('overlay'); appEl = $('app'); closeBtn = $('close3d'); promptEl = $('prompt');
    hudLevel = $('hud-level'); hudObjective = $('hud-objective'); hudWeek = $('hud-week'); hudMoney = $('hud-money'); seasonBox = $('season'); hudApproval = $('hud-approval');
    world = M.world3d.create($('stage'));
    world.attachMinimap($('minimap'));
    world.onStep(M.sfx.step);
    world.setOrbit(true);
    setInterval(function () { M.sfx.rain(world.rainLevel()); }, 500);
    window.addEventListener('pointerdown', M.sfx.unlock, { once: true });
    window.addEventListener('keydown', M.sfx.unlock, { once: true });
    $('mute').addEventListener('click', function () { var m = M.sfx.toggle(); $('mute').textContent = m ? '🔇' : '🔊'; $('mute').setAttribute('aria-label', m ? 'Sound off' : 'Sound on'); });
    world.onNear(function (h) {
      if (h && !overlay.classList.contains('open')) { promptEl.innerHTML = '<kbd>E</kbd> ' + h.label; promptEl.classList.add('show'); tip('interact', 'Press E (or click the orange button) to go in.'); }
      else promptEl.classList.remove('show');
      promptEl._h = h;
    });
    world.onHotspot(onHotspot);
    promptEl.addEventListener('click', function () { if (promptEl._h) onHotspot(promptEl._h); });
    closeBtn.addEventListener('click', closeOverlay);
    $('casebtn').addEventListener('click', function () { $('casefile').classList.toggle('show'); });
    window.addEventListener('keydown', function (e) { if (e.key.toLowerCase() === 'j' && !overlay.classList.contains('open')) $('casefile').classList.toggle('show'); });
    setInterval(renderCaseFile, 800);
    window.addEventListener('keydown', function (e) { if (e.key === 'Escape' && closable && overlay.classList.contains('open')) closeOverlay(); });
    M.ui.hooks.onRender = onRender;
    M.ui.hooks.onLevel = onLevel;
    M.ui.hooks.afterBrief = afterBrief;
    M.ui.hooks.afterExplore = afterExplore;
    M.ui.hooks.playSeason = playSeason;
    M.ui.hooks.pilotPlay = pilotPlay;
    M.ui.hooks.onResult = onResult;
    M.ui.hooks.reportExtra = reportExtra;
    M.ui.init(appEl);
    hud();
  }

  M.game3d = { close: closeOverlay, world: function () { return world; } };
  document.addEventListener('DOMContentLoaded', function () {
    try { boot(); } catch (err) {
      document.body.insertAdjacentHTML('beforeend', '<div class="fatal">Something went wrong starting the 3D town: ' + String(err && err.message || err).replace(/</g, '&lt;') +
        '<br><a href="classic.html">Play the classic version</a></div>');
      throw err;
    }
  });
})(globalThis.Mayor = globalThis.Mayor || {});
