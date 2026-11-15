/* sfx.js - tiny sound effects, synthesised with the Web Audio API (no audio files, works offline).
 * Browsers only allow sound after the player has clicked or pressed a key, so the audio context is
 * created lazily on the first effect. M.sfx.toggle() mutes / unmutes. */
(function (M) {
  'use strict';
  var ctx = null, muted = false, master = null, rainGain = null;

  function ac() {
    if (!ctx) {
      try { ctx = new (window.AudioContext || window.webkitAudioContext)(); } catch (e) { return null; }
      master = ctx.createGain(); master.gain.value = muted ? 0 : 0.8; master.connect(ctx.destination);
      startRain();
    }
    if (ctx.state === 'suspended') ctx.resume();
    return ctx;
  }
  function tone(freq, dur, type, vol, slide, delay) {
    var c = ac(); if (!c) return;
    var t0 = c.currentTime + (delay || 0), o = c.createOscillator(), g = c.createGain();
    o.type = type || 'sine'; o.frequency.setValueAtTime(freq, t0);
    if (slide) o.frequency.exponentialRampToValueAtTime(slide, t0 + dur);
    g.gain.setValueAtTime(0.0001, t0); g.gain.exponentialRampToValueAtTime(vol || 0.15, t0 + 0.01); g.gain.exponentialRampToValueAtTime(0.0001, t0 + dur);
    o.connect(g); g.connect(master); o.start(t0); o.stop(t0 + dur + 0.05);
  }
  function noiseBuffer(c, seconds) {
    var b = c.createBuffer(1, c.sampleRate * seconds, c.sampleRate), d = b.getChannelData(0);
    for (var i = 0; i < d.length; i++) d[i] = Math.random() * 2 - 1;
    return b;
  }
  function burst(dur, vol, freq) {
    var c = ac(); if (!c) return;
    var s = c.createBufferSource(); s.buffer = noiseBuffer(c, dur);
    var f = c.createBiquadFilter(); f.type = 'bandpass'; f.frequency.value = freq || 800; f.Q.value = 1.2;
    var g = c.createGain(); g.gain.setValueAtTime(vol || 0.1, c.currentTime); g.gain.exponentialRampToValueAtTime(0.0001, c.currentTime + dur);
    s.connect(f); f.connect(g); g.connect(master); s.start();
  }
  function startRain() {
    var s = ctx.createBufferSource(); s.buffer = noiseBuffer(ctx, 2); s.loop = true;
    var f = ctx.createBiquadFilter(); f.type = 'lowpass'; f.frequency.value = 1400;
    rainGain = ctx.createGain(); rainGain.gain.value = 0;
    s.connect(f); f.connect(rainGain); rainGain.connect(master); s.start();
  }

  M.sfx = {
    step: function () { burst(0.05, 0.05, 500 + Math.random() * 300); },
    open: function () { tone(520, 0.14, 'sine', 0.06, 780); },
    close: function () { tone(700, 0.12, 'sine', 0.05, 460); },
    coin: function (heads) { tone(heads ? 1480 : 1180, 0.12, 'triangle', 0.08); tone(heads ? 1980 : 1500, 0.16, 'triangle', 0.06, null, 0.06); },
    sign: function () { burst(0.18, 0.12, 2400); tone(330, 0.2, 'triangle', 0.06, 440, 0.05); },
    good: function () { [523, 659, 784, 1047].forEach(function (f, i) { tone(f, 0.28, 'triangle', 0.1, null, i * 0.11); }); },
    bad: function () { tone(330, 0.35, 'sawtooth', 0.05, 220); tone(262, 0.45, 'sawtooth', 0.05, 175, 0.25); },
    rain: function (level) { if (!ctx || !rainGain) return; rainGain.gain.setTargetAtTime(Math.max(0, Math.min(1, level)) * 0.18, ctx.currentTime, 0.4); },
    toggle: function () { muted = !muted; if (master) master.gain.value = muted ? 0 : 0.8; return muted; },
    muted: function () { return muted; },
    unlock: function () { ac(); }
  };
})(globalThis.Mayor = globalThis.Mayor || {});
