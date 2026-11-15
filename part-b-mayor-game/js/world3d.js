/* world3d.js - Navapuram in 3D (Three.js), with the mayor you walk around as.
 *
 * Layout: 20 wards in a 5 x 4 grid of blocks, a canal across the middle with bridges, and a civic
 * street along the north edge (Town Hall, Data Office, newspaper kiosk, school). Everything that
 * moves in a ward is driven by the simulated data for the shown week, exactly like the 2D town:
 * floating icons = the outcome (mosquitoes, patients, crashes...), small props = the candidate
 * measures (carts, puddles, tanks, clinics, speed breakers...), rain = rainfall.
 * Decoration (house colours, tree spots) uses its own seeded random stream, never the simulation's.
 *
 * Performance: repeated things (house walls, roofs, palms, props, floating icons) are drawn with
 * InstancedMesh, one draw call per kind instead of one per object. Static objects don't recompute
 * their matrices every frame, shadows are lighter, and nothing renders while a panel covers the town.
 *
 * Collisions: every solid thing gets a box measured from its real 3D shape (plus the mayor's body
 * radius), so walls block. The camera uses the same boxes to stay outside buildings.
 *
 * API: M.world3d.create(container) -> world with setLevel(spec), update(frame), setPolicy(kind, mask),
 *   setMask(mask), setBeacon(id), say(npc, text), coinToss(mask, onLand, done), setOrbit(on),
 *   setPaused(on), onHotspot(fn), onNear(fn), onStep(fn), setInputEnabled(on), teleport(x, z, face),
 *   walkTo(x, z), current(), wardName(i), rainLevel(), attachMinimap(canvas), hotspots */
(function (M) {
  'use strict';
  var T = globalThis.THREE;
  var COLS = 5, BW = 14, BD = 10, ROAD = 4, CANAL = 6, BODY = 0.9;
  var NAMES = ['Gandhi Nagar', 'Market Road', 'Temple Street', 'Lake View', 'Railway Colony', 'Bus Stand', 'Old Town', 'Mill Area',
    'Fisher Lane', 'College Road', 'Park Town', 'Riverside', "Weavers' Colony", 'Hill Side', 'New Extension', 'Salt Pans', 'Bazaar',
    'Canal Bank', 'Airport Road', 'Station Road'];

  function wardCenter(i) {
    var c = i % COLS, r = Math.floor(i / COLS);
    return new T.Vector3((c - 2) * (BW + ROAD), 0, r * (BD + ROAD) + (r >= 2 ? CANAL : 0) - 24);
  }

  // ---------------------------------------------------------------- materials and simple meshes
  var matCache = {};
  function mat(color, opts) {
    var key = color + JSON.stringify(opts || {});
    if (!matCache[key]) matCache[key] = new T.MeshLambertMaterial(Object.assign({ color: color, flatShading: true }, opts || {}));
    return matCache[key];
  }
  function mesh(geo, color, opts, shadow) { var m = new T.Mesh(geo, mat(color, opts)); m.castShadow = !!shadow; m.receiveShadow = true; return m; }
  function box(w, h, d, color, opts, shadow) { return mesh(new T.BoxGeometry(w, h, d), color, opts, shadow !== false); }
  function cyl(rt, rb, h, color, seg, opts, shadow) { return mesh(new T.CylinderGeometry(rt, rb, h, seg || 10), color, opts, shadow !== false); }
  function at(o, x, y, z) { o.position.set(x, y, z); return o; }

  function roundRect(c, x, y, w, h, r) { c.beginPath(); c.moveTo(x + r, y); c.arcTo(x + w, y, x + w, y + h, r); c.arcTo(x + w, y + h, x, y + h, r); c.arcTo(x, y + h, x, y, r); c.arcTo(x, y, x + w, y, r); c.closePath(); }
  function label(text, color, bg) {
    var cv = document.createElement('canvas'); cv.width = 512; cv.height = 96;
    var c = cv.getContext('2d');
    c.fillStyle = bg || 'rgba(10,14,28,0.82)'; roundRect(c, 4, 8, 504, 80, 18); c.fill();
    c.fillStyle = color || '#f5eedc'; c.font = 'bold 40px "Nirmala UI","Segoe UI",sans-serif'; c.textAlign = 'center'; c.textBaseline = 'middle';
    c.fillText(text, 256, 50);
    var tex = new T.CanvasTexture(cv); tex.colorSpace = T.SRGBColorSpace;
    var s = new T.Sprite(new T.SpriteMaterial({ map: tex, transparent: true, depthWrite: false }));
    s.scale.set(7, 1.3, 1);
    return s;
  }

  // ---------------------------------------------------------------- instanced batches
  var ZERO = new T.Matrix4().makeScale(0, 0, 0), tmpM = new T.Matrix4(), tmpM2 = new T.Matrix4(), tmpQ = new T.Quaternion(), tmpE = new T.Euler(), tmpS = new T.Vector3(), tmpP = new T.Vector3();
  function mtx(x, y, z, rx, ry, rz, sx, sy, sz) {
    tmpE.set(rx || 0, ry || 0, rz || 0); tmpQ.setFromEuler(tmpE); tmpP.set(x || 0, y || 0, z || 0);
    tmpS.set(sx === undefined ? 1 : sx, sy === undefined ? (sx === undefined ? 1 : sx) : sy, sz === undefined ? (sx === undefined ? 1 : sx) : sz);
    return new T.Matrix4().compose(tmpP, tmpQ, tmpS);
  }
  /* A batch draws `count` copies of a little model made of `parts` ({ geo, color, opts, local }).
   * set(i, base) places copy i (base = Matrix4, or null to hide it). */
  function Batch(scene, parts, count, shadow) {
    this.parts = parts; this.count = count;
    this.meshes = parts.map(function (p) {
      var m = new T.InstancedMesh(p.geo, p.perColor ? mat('#ffffff', p.opts) : mat(p.color, p.opts), count);
      m.frustumCulled = false; m.castShadow = !!shadow; m.receiveShadow = true;
      for (var i = 0; i < count; i++) m.setMatrixAt(i, ZERO);
      scene.add(m); return m;
    });
  }
  Batch.prototype.set = function (i, base) {
    for (var j = 0; j < this.parts.length; j++) {
      if (!base) this.meshes[j].setMatrixAt(i, ZERO);
      else { tmpM.multiplyMatrices(base, this.parts[j].local || IDENT); this.meshes[j].setMatrixAt(i, tmpM); }
    }
  };
  Batch.prototype.setPart = function (i, j, m) { this.meshes[j].setMatrixAt(i, m || ZERO); };
  Batch.prototype.color = function (i, j, c) { this.meshes[j].setColorAt(i, c); };
  Batch.prototype.flush = function () { this.meshes.forEach(function (m) { m.instanceMatrix.needsUpdate = true; if (m.instanceColor) m.instanceColor.needsUpdate = true; }); };
  Batch.prototype.dispose = function (scene) { this.meshes.forEach(function (m) { scene.remove(m); m.dispose(); }); };
  var IDENT = new T.Matrix4();

  // The models for data props and floating icons, as lists of parts (geometry + colour + local placement).
  function P(geo, color, local, opts) { return { geo: geo, color: color, local: local, opts: opts }; }
  var PROP = {
    cart: function () { return [P(new T.BoxGeometry(0.9, 0.5, 0.6), '#fdf6e3', mtx(0, 0.45, 0)), P(new T.CylinderGeometry(0.02, 0.02, 0.8, 4), '#555555', mtx(0, 1.0, 0)), P(new T.ConeGeometry(0.7, 0.35, 8), '#e87ba4', mtx(0, 1.45, 0))]; },
    bottle: function () { return [P(new T.BoxGeometry(1.0, 0.8, 0.6), '#c94f2a', mtx(0, 0.4, 0)), P(new T.BoxGeometry(1.2, 0.08, 0.8), '#7a2c18', mtx(0, 0.84, 0))]; },
    puddle: function () { return [P(new T.CylinderGeometry(0.8, 0.8, 0.04, 16), '#3b82d6', mtx(0, 0.05, 0))]; },
    tank: function () { return [P(new T.CylinderGeometry(0.45, 0.45, 0.9, 12), '#7f93a8', mtx(0, 0.45, 0)), P(new T.CylinderGeometry(0.42, 0.42, 0.03, 12), '#3987e5', mtx(0, 0.92, 0))]; },
    tyre: function () { return [P(new T.TorusGeometry(0.4, 0.15, 6, 12), '#222222', mtx(0, 0.15, 0, Math.PI / 2))]; },
    clinic: function () { return [P(new T.BoxGeometry(1.2, 1.0, 1.0), '#ffffff', mtx(0, 0.5, 0)), P(new T.BoxGeometry(0.6, 0.18, 0.05), '#d03b3b', mtx(0, 0.7, 0.53)), P(new T.BoxGeometry(0.18, 0.6, 0.05), '#d03b3b', mtx(0, 0.7, 0.53))]; },
    police: function () { return [P(new T.BoxGeometry(1.2, 1.0, 1.0), '#2a5aa8', mtx(0, 0.5, 0)), P(new T.ConeGeometry(0.95, 0.5, 4), '#1c3f78', mtx(0, 1.25, 0, 0, Math.PI / 4))]; },
    pill: function () { return [P(new T.BoxGeometry(0.9, 0.9, 0.7), '#ffffff', mtx(0, 0.45, 0)), P(new T.BoxGeometry(0.92, 0.2, 0.72), '#1baf7a', mtx(0, 0.9, 0))]; },
    lock: function () { return [P(new T.BoxGeometry(0.7, 0.55, 0.3), '#eda100', mtx(0, 0.3, 0)), P(new T.TorusGeometry(0.22, 0.06, 6, 10, Math.PI), '#8a8a8a', mtx(0, 0.58, 0))]; },
    bump: function () { return [P(new T.CylinderGeometry(0.35, 0.35, 1.6, 8, 1, false, 0, Math.PI), '#eda100', mtx(0, 0, 0, 0, Math.PI / 2, Math.PI / 2))]; },
    lamp: function () { return [P(new T.CylinderGeometry(0.06, 0.08, 2.4, 6), '#6b6a66', mtx(0, 1.2, 0)), P(new T.SphereGeometry(0.22, 8, 6), '#ffd66b', mtx(0, 2.45, 0), { emissive: '#ffb83a' })]; },
    bus: function () { return [P(new T.BoxGeometry(1.7, 0.9, 0.8), '#eda100', mtx(0, 0.6, 0)), P(new T.BoxGeometry(1.5, 0.3, 0.82), '#bfe0ff', mtx(0, 0.85, 0))]; },
    meal: function () { return [P(new T.CylinderGeometry(0.5, 0.4, 0.15, 12), '#ffffff', mtx(0, 0.1, 0)), P(new T.SphereGeometry(0.18, 8, 6), '#eda100', mtx(-0.15, 0.25, 0)), P(new T.SphereGeometry(0.16, 8, 6), '#1baf7a', mtx(0.17, 0.24, 0))]; },
    shirt: function () { return [P(new T.BoxGeometry(0.8, 0.9, 0.2), '#3987e5', mtx(0, 0.6, 0)), P(new T.BoxGeometry(1.3, 0.25, 0.2), '#3987e5', mtx(0, 0.95, 0))]; }
  };
  var FLOAT = { // floating outcome icons; wings (part 1 of the mosquito) flap
    mosquito: function () { return [P(new T.SphereGeometry(0.16, 6, 5), '#141414', mtx(0, 0, 0, 0, 0, 0, 1, 1, 2.2)), P(new T.BoxGeometry(1.0, 0.02, 0.2), '#ffffff', null, { transparent: true, opacity: 0.7 })]; },
    patient: function () { return [P(new T.CylinderGeometry(0.18, 0.22, 0.6, 8), '#e66767', mtx(0, 0, 0)), P(new T.SphereGeometry(0.18, 8, 6), '#c68b59', mtx(0, 0.45, 0))]; },
    alarm: function () { return [P(new T.SphereGeometry(0.28, 10, 8), '#ff3b3b', mtx(0, 0, 0), { emissive: '#ff2020' })]; },
    crash: function () { return [P(new T.OctahedronGeometry(0.35), '#ff7a2a', mtx(0, 0, 0), { emissive: '#ff5a10' })]; },
    child: function () { return [P(new T.CylinderGeometry(0.14, 0.16, 0.45, 8), '#3987e5', mtx(0, 0, 0)), P(new T.SphereGeometry(0.15, 8, 6), '#c68b59', mtx(0, 0.33, 0)), P(new T.BoxGeometry(0.2, 0.25, 0.1), '#eb6834', mtx(0, 0.05, -0.17))]; }
  };

  function mayorModel() {
    var g = new T.Group();
    var legL = at(box(0.28, 0.9, 0.3, '#f5f0e6'), -0.18, 0.45, 0), legR = at(box(0.28, 0.9, 0.3, '#f5f0e6'), 0.18, 0.45, 0);
    var body = at(cyl(0.42, 0.5, 1.2, '#ffffff', 10), 0, 1.5, 0);
    var sash = at(box(0.12, 1.25, 0.95, '#f4a63a'), 0, 1.55, 0); sash.rotation.z = 0.55;
    var head = at(mesh(new T.SphereGeometry(0.33, 10, 8), '#c68b59', null, true), 0, 2.38, 0);
    var cap = at(box(0.62, 0.22, 0.42, '#f8f8f2'), 0, 2.66, 0);
    var armL = at(box(0.2, 0.9, 0.22, '#ffffff'), -0.58, 1.55, 0), armR = at(box(0.2, 0.9, 0.22, '#ffffff'), 0.58, 1.55, 0);
    var nose = at(box(0.08, 0.08, 0.12, '#b5774a'), 0, 2.36, 0.33);
    g.add(legL, legR, body, sash, head, cap, armL, armR, nose);
    g.userData = { legL: legL, legR: legR, armL: armL, armR: armR };
    return g;
  }
  function npcModel(coat, hair) {
    var g = new T.Group();
    g.add(at(cyl(0.38, 0.46, 1.5, coat, 10), 0, 0.95, 0));
    g.add(at(mesh(new T.SphereGeometry(0.3, 10, 8), '#c68b59', null, true), 0, 1.95, 0));
    g.add(at(mesh(new T.SphereGeometry(0.31, 10, 8), hair, null, true), 0, 2.05, -0.05));
    return g;
  }

  // ---------------------------------------------------------------- the world
  function create(container) {
    var art = new M.rng.Rng('town-art-3d');
    var renderer = new T.WebGLRenderer({ antialias: true, powerPreference: 'high-performance' });
    renderer.setPixelRatio(Math.min(1.25, window.devicePixelRatio || 1));
    renderer.shadowMap.enabled = true;
    renderer.shadowMap.type = T.PCFShadowMap;
    renderer.outputColorSpace = T.SRGBColorSpace;
    container.appendChild(renderer.domElement);

    var scene = new T.Scene();
    var skyDay = new T.Color('#f2b27a'), skyRain = new T.Color('#5b6577');
    scene.background = new T.Color('#e9a873');
    scene.fog = new T.Fog('#e9a873', 70, 170);
    var camera = new T.PerspectiveCamera(50, 1, 0.2, 400);

    scene.add(new T.HemisphereLight('#ffe9c7', '#3d4a3a', 1.6));
    var sun = new T.DirectionalLight('#ffd2a0', 2.4);
    sun.position.set(-40, 60, 30); sun.castShadow = true;
    sun.shadow.mapSize.set(1024, 1024);
    var sc = sun.shadow.camera; sc.left = -64; sc.right = 64; sc.top = 64; sc.bottom = -64; sc.near = 10; sc.far = 180;
    sun.shadow.bias = -0.0015;
    scene.add(sun);

    // ---- ground, roads, canal, bridges, lane dashes
    var ground = new T.Mesh(new T.PlaneGeometry(260, 260), mat('#5d7a4c'));
    ground.rotation.x = -Math.PI / 2; ground.receiveShadow = true; scene.add(ground);
    var roads = new T.Mesh(new T.PlaneGeometry(104, 92), mat('#4a4740'));
    roads.rotation.x = -Math.PI / 2; roads.position.set(0, 0.02, -10); roads.receiveShadow = true; scene.add(roads);
    var canal = new T.Mesh(new T.BoxGeometry(110, 0.3, CANAL), mat('#2c6fae'));
    canal.position.set(0, 0.03, 0); scene.add(canal);
    var bridgeX = [-27, -9, 9, 27];
    bridgeX.forEach(function (x) { scene.add(at(box(4, 0.4, CANAL + 1, '#9a8566', null, false), x, 0.25, 0)); });
    var dashes = new Batch(scene, [P(new T.BoxGeometry(1.6, 0.03, 0.18), '#d8d2bd')], 26, false);
    for (var di = 0; di < 26; di++) dashes.set(di, mtx(-50 + di * 4, 0.05, -31));
    dashes.flush();

    // ---- collisions: boxes measured from real shapes; h = height (for the camera)
    var colliders = [];
    function solidBox(x0, x1, z0, z1, h) { colliders.push({ x0: x0, x1: x1, z0: z0, z1: z1, h: h || 0 }); }
    var _b = new T.Box3();
    function solidFrom(obj) {
      obj.updateMatrixWorld(true);
      var bb = new T.Box3();
      obj.traverse(function (o) { if (o.isMesh) { _b.setFromObject(o); bb.union(_b); } });
      solidBox(bb.min.x, bb.max.x, bb.min.z, bb.max.z, bb.max.y);
    }
    var cx = -55; // canal is solid except at the bridges (no height: the camera can pass over it)
    bridgeX.concat([55]).forEach(function (bx) { solidBox(cx, bx === 55 ? 55 : bx - 2, -CANAL / 2, CANAL / 2, 0); cx = bx + 2; });

    // ---- the 20 ward blocks: houses and palms are instanced (one draw call per part)
    var HOUSES = [], PALMS = [];
    var wallColors = ['#efe2c6', '#e8d3b0', '#f1d9a8', '#dfe7d4', '#f3c9a8', '#d9d1ee'];
    var roofColors = ['#b8492f', '#a85a36', '#8a5a44', '#c47a2c', '#9a3f3f', '#7a8a3c'];
    var wards = [], wardSigns = [];
    var lots = new Batch(scene, [P(new T.BoxGeometry(BW, 0.25, BD), '#6f9a57')], 20, false);
    for (var i = 0; i < 20; i++) {
      var c0 = wardCenter(i), g = new T.Group();
      g.position.copy(c0);
      lots.set(i, mtx(c0.x, 0.12, c0.z));
      for (var hx = -1; hx <= 1; hx += 1) {
        var px0 = c0.x + hx * 4.2 + (art.uniform() - 0.5), pz0 = c0.z - 2.2;
        if (art.uniform() < 0.82) HOUSES.push({ x: px0, z: pz0, ry: (art.uniform() - 0.5) * 0.3, w: 2.4 + art.uniform() * 1.2, d: 2.2 + art.uniform() * 0.8, h: 1.6 + art.uniform() * 1.4, wall: art.pick(wallColors), roof: art.pick(roofColors) });
        else PALMS.push({ x: c0.x + hx * 4.2, z: pz0, h: 3.4 + art.uniform() * 1.4, lean: (art.uniform() - 0.5) * 0.25 });
      }
      if (art.uniform() < 0.7) PALMS.push({ x: c0.x + (art.uniform() - 0.5) * 10, z: c0.z + 0.8, h: 3.4 + art.uniform() * 1.4, lean: (art.uniform() - 0.5) * 0.25 });
      var sign = label((i + 1) + ' · ' + NAMES[i]); sign.scale.set(5.2, 0.98, 1); sign.position.set(0, 6.6, -1); g.add(sign); wardSigns.push(sign);
      var ring = new T.Mesh(new T.RingGeometry(7.6, 8.4, 40), new T.MeshBasicMaterial({ color: '#2fd09a', transparent: true, opacity: 0.0, side: T.DoubleSide }));
      ring.rotation.x = -Math.PI / 2; ring.position.y = 0.3; ring.scale.set(1, 0.72, 1); g.add(ring);
      scene.add(g);
      wards.push({ group: g, center: c0, ring: ring, tag: null, tagText: '', ban: null, banText: '' });
    }
    lots.flush();
    // houses: walls, roofs, windows, doors
    var houseB = new Batch(scene, [
      { geo: new T.BoxGeometry(1, 1, 1), perColor: true }, { geo: new T.ConeGeometry(1, 1, 4), perColor: true },
      P(new T.BoxGeometry(0.5, 0.45, 0.05), '#ffcf5a', null, { emissive: '#ffb83a', emissiveIntensity: 0.6 }),
      P(new T.BoxGeometry(0.5, 0.9, 0.05), '#6b4636')], HOUSES.length, true);
    var col = new T.Color();
    HOUSES.forEach(function (hs, k) {
      var base = mtx(hs.x, 0.25, hs.z, 0, hs.ry);
      houseB.setPart(k, 0, tmpM2.multiplyMatrices(base, mtx(0, hs.h / 2, 0, 0, 0, 0, hs.w, hs.h, hs.d)).clone());
      var R = Math.max(hs.w, hs.d) * 0.78;
      houseB.setPart(k, 1, tmpM2.multiplyMatrices(base, mtx(0, hs.h + 0.6, 0, 0, Math.PI / 4, 0, R, 1.2, R)).clone());
      houseB.setPart(k, 2, tmpM2.multiplyMatrices(base, mtx(-hs.w / 4, hs.h * 0.6, hs.d / 2 + 0.03)).clone());
      houseB.setPart(k, 3, tmpM2.multiplyMatrices(base, mtx(hs.w / 5, 0.45, hs.d / 2 + 0.03)).clone());
      houseB.color(k, 0, col.set(hs.wall)); houseB.color(k, 1, col.set(hs.roof));
      var hw = Math.max(hs.w, hs.d) / 2 + 0.1; // a little extra for the slight rotation
      solidBox(hs.x - hw, hs.x + hw, hs.z - hs.d / 2 - 0.1, hs.z + hs.d / 2 + 0.1, hs.h + 1.4);
    });
    houseB.flush();
    // palms: trunk + 6 leaves; trunks are solid so you can't walk through them
    function addPalms(list) {
      var trunks = new Batch(scene, [P(new T.CylinderGeometry(0.12, 0.2, 1, 6), '#7a5a3a')], list.length, true);
      var leaves = new Batch(scene, [P(new T.BoxGeometry(0.25, 0.06, 2.0), '#3f8f45')], list.length * 6, true);
      list.forEach(function (p, k) {
        trunks.set(k, mtx(p.x, p.h / 2 + 0.25, p.z, 0, 0, p.lean, 1, p.h, 1));
        for (var l = 0; l < 6; l++) {
          var m = mtx(p.x, p.h + 0.25, p.z, 0, l * Math.PI / 3, 0);
          m.multiply(mtx(0, 0, 0, 0.45, 0, 0)).multiply(mtx(0, 0, 0.9));
          leaves.set(k * 6 + l, m);
        }
        solidBox(p.x - 0.3, p.x + 0.3, p.z - 0.3, p.z + 0.3, 0);
      });
      trunks.flush(); leaves.flush();
    }

    // ---- the civic street (north)
    var civic = new T.Group(); scene.add(civic);
    function building(x, z, w, d, h, wall, roof, name, color) {
      var gg = new T.Group(); gg.position.set(x, 0, z);
      gg.add(at(box(w, h, d, wall), 0, h / 2, 0));
      var r = mesh(new T.ConeGeometry(Math.max(w, d) * 0.72, 3, 4), roof, null, true); r.rotation.y = Math.PI / 4; r.scale.set(1, 1, d / w); r.position.y = h + 1.5; gg.add(r);
      gg.add(at(box(2.2, 2.6, 0.2, '#5a3a24', null, false), 0, 1.3, d / 2 + 0.1));
      for (var k = -1; k <= 1; k += 2) gg.add(at(box(1.2, 1.0, 0.1, '#ffcf5a', { emissive: '#ffb83a', emissiveIntensity: 0.6 }, false), k * w / 3.2, h * 0.62, d / 2 + 0.06));
      civic.add(gg);
      solidBox(x - w / 2, x + w / 2, z - d / 2, z + d / 2 + 0.25, h + 3); // walls and door; the roof's eaves are above head height
      var s = label(name, color || '#ffc94a'); s.position.set(0, h + 4.6, 0); s.scale.set(9, 1.7, 1); gg.add(s);
      return gg;
    }
    // Town Hall: hall with a colonnade, a clock tower and a spire
    var th = new T.Group(); th.position.set(0, 0, -44); civic.add(th);
    th.add(at(box(20, 6, 10, '#f1e6cf'), 0, 3, 0));
    for (var ci = -4; ci <= 4; ci += 2) th.add(at(cyl(0.35, 0.35, 5.6, '#fbf5e6', 8), ci * 2, 2.8, 5.4));
    th.add(at(box(21, 0.8, 1.6, '#e2d3b2'), 0, 6.1, 5.3));
    th.add(at(box(4, 12, 4, '#efe2c6'), 0, 6, -1));
    var spire = mesh(new T.ConeGeometry(3, 4, 4), '#b8492f', null, true); spire.rotation.y = Math.PI / 4; spire.position.set(0, 14, -1); th.add(spire);
    var clockFace = at(cyl(1.1, 1.1, 0.2, '#ffcf5a', 20, { emissive: '#ffb83a', emissiveIntensity: 0.9 }, false), 0, 10, 1.05); clockFace.rotation.x = Math.PI / 2; th.add(clockFace);
    th.add(at(box(3, 3.4, 0.2, '#5a3a24', null, false), 0, 1.7, 5.05));
    solidFrom(th); // includes the colonnade and porch, so you stop at the columns
    var thLabel = label('Town Hall', '#ffc94a'); thLabel.position.set(0, 18.5, 0); thLabel.scale.set(9, 1.7, 1); th.add(thLabel);
    building(-27, -44, 13, 9, 5, '#dfe7d4', '#3f6b8a', 'Data Office', '#7cc4ff');
    building(27, -44, 13, 9, 5, '#f3d9a8', '#8a5a2c', 'Government School', '#b6e3a1');
    var kiosk = new T.Group(); kiosk.position.set(13, 0, -36); civic.add(kiosk);
    kiosk.add(at(box(3, 2.4, 2.2, '#c94f2a'), 0, 1.2, 0)); kiosk.add(at(box(3.6, 0.2, 2.8, '#7a2c18'), 0, 2.5, 0));
    kiosk.add(at(box(2.4, 1.4, 0.05, '#f7f3e8', null, false), 0, 1.4, 1.12));
    solidFrom(kiosk);
    var kl = label('The Navapuram Times', '#1d1b16', 'rgba(247,243,232,0.95)'); kl.position.set(0, 4, 0); kl.scale.set(7, 1.3, 1); kiosk.add(kl);
    [-40, -14, 40, 18].forEach(function (x) { PALMS.push({ x: x, z: -37, h: 3.8, lean: 0.05 }); });
    addPalms(PALMS);

    // ---- NPCs (solid too) and their speech bubbles
    var asha = npcModel('#ffffff', '#2b1d14'); asha.position.set(-3, 0, -36.5); scene.add(asha);
    var editor = npcModel('#3b5998', '#4a3020'); editor.position.set(15.5, 0, -34.5); scene.add(editor);
    var clerk = npcModel('#2a7a6a', '#1d1d1d'); clerk.position.set(-24, 0, -38.5); scene.add(clerk);
    [asha, editor, clerk].forEach(function (n) { solidBox(n.position.x - 0.45, n.position.x + 0.45, n.position.z - 0.45, n.position.z + 0.45, 0); });
    var npcs = { asha: { obj: asha, bubble: null, text: '' }, editor: { obj: editor, bubble: null, text: '' }, clerk: { obj: clerk, bubble: null, text: '' } };
    function bubbleSprite(text) {
      var cv = document.createElement('canvas'); cv.width = 640; cv.height = 220;
      var c = cv.getContext('2d'); c.font = 'bold 34px "Nirmala UI","Segoe UI",sans-serif';
      var words = text.split(' '), lines = [], cur = '';
      words.forEach(function (w) { var t2 = cur ? cur + ' ' + w : w; if (c.measureText(t2).width > 580 && cur) { lines.push(cur); cur = w; } else cur = t2; });
      if (cur) lines.push(cur); lines = lines.slice(0, 4);
      var hh = 30 + lines.length * 42;
      c.fillStyle = 'rgba(255,250,235,0.97)'; roundRect(c, 6, 6, 628, hh, 26); c.fill();
      c.beginPath(); c.moveTo(300, hh + 4); c.lineTo(320, hh + 34); c.lineTo(340, hh + 4); c.fill();
      c.fillStyle = '#231303'; c.textBaseline = 'top';
      lines.forEach(function (ln, k) { c.fillText(ln, 26, 22 + k * 42); });
      var tex = new T.CanvasTexture(cv); tex.colorSpace = T.SRGBColorSpace;
      var s = new T.Sprite(new T.SpriteMaterial({ map: tex, transparent: true, depthWrite: false, depthTest: false }));
      s.renderOrder = 10; s.scale.set(10.5, 3.6, 1); s.center.set(0.5, 0);
      return s;
    }
    function say(id, text) {
      var n = npcs[id]; if (!n || n.text === text) return;
      if (n.bubble) { n.obj.remove(n.bubble); n.bubble.material.map.dispose(); n.bubble.material.dispose(); n.bubble = null; }
      n.text = text;
      if (text) { n.bubble = bubbleSprite(text); n.bubble.position.set(0, 2.6, 0); n.obj.add(n.bubble); }
    }

    // ---- the mayor
    var mayor = mayorModel(); mayor.position.set(0, 0, -33); mayor.rotation.y = Math.PI; scene.add(mayor);

    // ---- objective beacon
    var beacon = new T.Group();
    var beam = new T.Mesh(new T.CylinderGeometry(1.2, 1.2, 30, 16, 1, true), new T.MeshBasicMaterial({ color: '#ffc94a', transparent: true, opacity: 0.18, side: T.DoubleSide, depthWrite: false }));
    beam.position.y = 15; beacon.add(beam);
    var arrow = new T.Mesh(new T.ConeGeometry(0.9, 1.8, 4), new T.MeshBasicMaterial({ color: '#ffc94a' })); arrow.rotation.x = Math.PI; arrow.position.y = 9;
    beacon.add(arrow); beacon.visible = false; scene.add(beacon);

    // ---- rain (one Points object; positions only update while it is raining)
    var RAIN = 1400, rainGeo = new T.BufferGeometry(), rp = new Float32Array(RAIN * 3);
    for (var ri = 0; ri < RAIN; ri++) { rp[ri * 3] = (art.uniform() - 0.5) * 140; rp[ri * 3 + 1] = art.uniform() * 50; rp[ri * 3 + 2] = (art.uniform() - 0.5) * 120 - 10; }
    rainGeo.setAttribute('position', new T.BufferAttribute(rp, 3));
    var rain = new T.Points(rainGeo, new T.PointsMaterial({ color: '#cfe3ff', size: 0.2, transparent: true, opacity: 0.0, depthWrite: false }));
    rain.frustumCulled = false; scene.add(rain);

    // ---- trucks for "crews at work" policies
    var trucks = [];
    for (var tk = 0; tk < 4; tk++) {
      var tr = new T.Group(); tr.add(at(box(2.4, 1.2, 1.1, '#eda100'), -0.4, 0.8, 0)); tr.add(at(box(1.0, 1.0, 1.1, '#c98500'), 1.3, 0.7, 0));
      [-1.2, 1.2].forEach(function (wx) { var wh = cyl(0.32, 0.32, 1.15, '#222222', 8, null, false); wh.rotation.x = Math.PI / 2; wh.position.set(wx, 0.32, 0); tr.add(wh); });
      tr.visible = false; scene.add(tr); trucks.push(tr);
    }

    /* ---- town life: people on the pavements, auto-rickshaws and trucks in their lanes.
     * Nothing walks through anything: people keep to the pavement for their direction (keep left),
     * cross the road when they turn round, and wait (then turn back) if something is in the way;
     * vehicles brake for whatever is in front of them, the mayor included. Each kind is one instanced draw. */
    var H_WALK = [-31, -17, 17, 31], V_WALK = [-45, -27, -9, 9, 27, 45], SIDE = 1.55, LANE = 0.6, WR = 0.36;
    var lifeRng = new M.rng.Rng('town-life-3d'), PEOPLE = 44, walkers = [], lifeBoxes = [];
    var shirt = ['#e66767', '#3987e5', '#2fd09a', '#f2b11c', '#a99bff', '#ff86b3', '#f5f0e6', '#1baf7a'];
    var peopleB = new Batch(scene, [{ geo: new T.CylinderGeometry(0.3, 0.36, 1.3, 7), perColor: true, local: mtx(0, 0.75, 0) }, P(new T.SphereGeometry(0.26, 7, 6), '#c68b59', mtx(0, 1.62, 0))], PEOPLE, true);
    function staticHit(x, z, r) {
      if (x < -51 || x > 51 || z < -51 || z > 33) return true;
      for (var k = 0; k < colliders.length; k++) { var c = colliders[k]; if (x > c.x0 - r && x < c.x1 + r && z > c.z0 - r && z < c.z1 + r) return true; }
      return false;
    }
    function sideFor(wk) { return wk.horiz ? (wk.dir > 0 ? -SIDE : SIDE) : (wk.dir > 0 ? SIDE : -SIDE); }
    function wXZ(wk) { return wk.horiz ? [wk.pos, wk.road + wk.off] : [wk.road + wk.off, wk.pos]; }
    for (var pi = 0; pi < PEOPLE; pi++) {
      for (var tries = 0; tries < 60; tries++) {
        var cand = { horiz: lifeRng.uniform() < 0.55, dir: lifeRng.uniform() < 0.5 ? -1 : 1, speed: 1.8 + lifeRng.uniform() * 1.4, ph: lifeRng.uniform() * 6.28, wait: 0 };
        cand.road = lifeRng.pick(cand.horiz ? H_WALK : V_WALK);
        cand.pos = (lifeRng.uniform() - 0.5) * (cand.horiz ? 98 : 62);
        cand.off = cand.toff = sideFor(cand);
        var cp = wXZ(cand);
        if (staticHit(cp[0], cp[1], WR + 0.1)) continue;
        if (walkers.some(function (o) { var q = wXZ(o); return Math.abs(q[0] - cp[0]) < 1.2 && Math.abs(q[1] - cp[1]) < 1.2; })) continue;
        peopleB.color(walkers.length, 0, new T.Color(lifeRng.pick(shirt)));
        walkers.push(cand); break;
      }
    }
    var outside = walkers.length; // how many are out: fewer when the town is sick
    var RICK_ROADS = [-31, 31, -17, 17, -31, 31], ricks = [];
    var rickB = new Batch(scene, [P(new T.BoxGeometry(2.0, 1.1, 1.1), '#2fa84f', mtx(0, 0.75, 0)), P(new T.BoxGeometry(1.6, 0.25, 1.1), '#141414', mtx(-0.1, 1.45, 0)), P(new T.BoxGeometry(0.5, 0.6, 1.0), '#f2c11c', mtx(0.95, 0.6, 0))], RICK_ROADS.length, true);
    RICK_ROADS.forEach(function (road, k) { ricks.push({ road: road, dir: k < 4 ? (k % 2 ? -1 : 1) : (k % 2 ? 1 : -1), x: -36 + k * 13, speed: 6 + lifeRng.uniform() * 3, v: 0 }); });
    function rickZ(rc) { return rc.road + (rc.dir > 0 ? -LANE : LANE); }
    // trucks drive east in the left lane of roads -17 / 17
    trucks.forEach(function (trk, k) { trk.userData = { road: k % 2 ? 17 : -17, x: [-50, -20, 10, 40][k], speed: 9 + k * 1.5, v: 0 }; });

    // everything that moves, as boxes (rebuilt each frame)
    function buildBoxes() {
      lifeBoxes.length = 0;
      var m = mayor.position; lifeBoxes.push({ o: mayor, x0: m.x - 0.5, x1: m.x + 0.5, z0: m.z - 0.5, z1: m.z + 0.5 });
      for (var k = 0; k < walkers.length && k < outside; k++) { var p = wXZ(walkers[k]); lifeBoxes.push(walkers[k]._b = { o: walkers[k], x0: p[0] - WR, x1: p[0] + WR, z0: p[1] - WR, z1: p[1] + WR }); }
      ricks.forEach(function (rc) { var z = rickZ(rc); lifeBoxes.push(rc._b = danger({ o: rc, x0: rc.x - 1.2, x1: rc.x + 1.2, z0: z - 0.55, z1: z + 0.55 }, rc.dir, rc.v)); });
      trucks.forEach(function (trk) { if (!trk.visible) return; var u = trk.userData, z = u.road - LANE; lifeBoxes.push(u._b = danger({ o: trk, x0: u.x - 1.6, x1: u.x + 1.8, z0: z - 0.58, z1: z + 0.58 }, 1, u.v)); });
    }
    // people treat the road just ahead of a moving vehicle as off-limits (they don't step out in front of it)
    function danger(bx, dir, v) {
      var reach = 0.5 + v * 0.6;
      bx.vehicle = true; bx.dx0 = dir < 0 ? bx.x0 - reach : bx.x0; bx.dx1 = dir > 0 ? bx.x1 + reach : bx.x1;
      return bx;
    }
    function ov(b, x0, x1, z0, z1) { return b.x0 < x1 - 0.02 && b.x1 > x0 + 0.02 && b.z0 < z1 - 0.02 && b.z1 > z0 + 0.02; }
    /* Would moving `self` (box x0..x1, z0..z1) by (dx, dz) run into something? If the two already touch,
     * only a move that takes them apart is allowed, so nothing ever slides through anything. */
    function bumps(self, x0, x1, z0, z1, dx, dz, isWalker) {
      var cx = (x0 + x1) / 2, cz = (z0 + z1) / 2;
      for (var k = 0; k < lifeBoxes.length; k++) {
        var b = lifeBoxes[k]; if (b.o === self) continue;
        var t = (isWalker && b.vehicle) ? { x0: b.dx0, x1: b.dx1, z0: b.z0, z1: b.z1 } : b;
        if (!ov(t, x0 + dx, x1 + dx, z0 + dz, z1 + dz)) continue;
        if (!ov(t, x0, x1, z0, z1)) return true;
        var bx = (b.x0 + b.x1) / 2, bz = (b.z0 + b.z1) / 2;
        if (Math.hypot(cx + dx - bx, cz + dz - bz) < Math.hypot(cx - bx, cz - bz) - 1e-6) return true;
      }
      return false;
    }
    function freeAt(self, x0, x1, z0, z1) {
      for (var k = 0; k < lifeBoxes.length; k++) { var b = lifeBoxes[k]; if (b.o !== self && ov(b, x0, x1, z0, z1)) return false; }
      return true;
    }
    function nearMayor(x, z) { return Math.abs(x - mayor.position.x) < 2.2 && Math.abs(z - mayor.position.z) < 2.2; }
    // trucks drive in from outside the town when a crew policy starts (they never pop up on top of anyone)
    function showTruck(t, k, on) { if (on && !t.visible) { t.userData.x = -58 - k * 14; t.userData.v = t.userData.speed; } t.visible = on; }
    function turnRound(wk) { wk.dir = -wk.dir; wk.toff = sideFor(wk); wk.wait = 0; }
    function updateLife(dt, t) {
      buildBoxes();
      walkers.forEach(function (wk, k) {
        if (k >= outside) { peopleB.set(k, null); return; }
        var p = wXZ(wk), x = p[0], z = p[1], step = wk.speed * dt, rot, dx, dz;
        if (Math.abs(wk.toff - wk.off) > 1e-3) { // crossing to the other pavement
          var s = wk.toff > wk.off ? 1 : -1; dx = wk.horiz ? 0 : s; dz = wk.horiz ? s : 0;
          if (bumps(wk, x - WR, x + WR, z - WR, z + WR, dx * 0.6, dz * 0.6, true)) { wk.wait += dt; if (wk.wait > 1.8) { wk.toff = wk.off >= 0 ? SIDE : -SIDE; wk.wait = 0; } } // give up: back to the nearer pavement
          else { wk.wait = 0; wk.off += Math.max(-step, Math.min(step, wk.toff - wk.off)); }
        } else {
          dx = wk.horiz ? wk.dir : 0; dz = wk.horiz ? 0 : wk.dir;
          if (staticHit(x + dx * (step + 0.05), z + dz * (step + 0.05), WR) || (!wk.horiz && wk.pos + wk.dir * step < -33.5)) turnRound(wk);
          else if (bumps(wk, x - WR, x + WR, z - WR, z + WR, dx * 0.6, dz * 0.6, true)) { wk.wait += dt; if (wk.wait > (nearMayor(x, z) ? 0.5 : 2.5)) turnRound(wk); }
          else { wk.wait = 0; wk.pos += wk.dir * step; }
        }
        rot = Math.atan2(dx, dz);
        p = wXZ(wk);
        var wb = wk._b; wb.x0 = p[0] - WR; wb.x1 = p[0] + WR; wb.z0 = p[1] - WR; wb.z1 = p[1] + WR;
        peopleB.set(k, mtx(p[0], wk.wait ? 0 : Math.abs(Math.sin(t * 7 + wk.ph)) * 0.08, p[1], 0, rot));
      });
      peopleB.flush();
      ricks.forEach(function (rc, k) {
        var z = rickZ(rc), stop = bumps(rc, rc.x - 1.2, rc.x + 1.2, z - 0.55, z + 0.55, rc.dir * (0.8 + rc.v * 0.35), 0);
        rc.v += ((stop ? 0 : rc.speed) - rc.v) * Math.min(1, dt * (stop ? 10 : 1.5));
        rc.x += rc.dir * rc.v * dt;
        if (rc.dir * rc.x > 49) { // end of the road: U-turn into the other lane once it is clear
          rc.x = rc.dir * 49;
          var nz = rc.road + (rc.dir > 0 ? LANE : -LANE);
          if (freeAt(rc, rc.x - 1.2, rc.x + 1.2, nz - 0.55, nz + 0.55)) rc.dir = -rc.dir; else rc.v = 0;
        }
        var rz = rickZ(rc), rb = rc._b; rb.x0 = rc.x - 1.2; rb.x1 = rc.x + 1.2; rb.z0 = rz - 0.55; rb.z1 = rz + 0.55; danger(rb, rc.dir, rc.v);
        rickB.set(k, mtx(rc.x, 0, rz, 0, rc.dir > 0 ? 0 : Math.PI));
      });
      rickB.flush();
      trucks.forEach(function (trk) {
        if (!trk.visible) return;
        var u = trk.userData, z = u.road - LANE, stop = bumps(trk, u.x - 1.6, u.x + 1.8, z - 0.58, z + 0.58, 0.8 + u.v * 0.35, 0);
        u.v += ((stop ? 0 : u.speed) - u.v) * Math.min(1, dt * (stop ? 10 : 1.5));
        u.x += u.v * dt; if (u.x > 60) u.x = -60;
        var tb = u._b; tb.x0 = u.x - 1.6; tb.x1 = u.x + 1.8; danger(tb, 1, u.v);
        trk.position.set(u.x, 0, z);
      });
    }
    // the mayor can't walk into people or vehicles either
    function dynBlocked(x, z) {
      var cx = mayor.position.x, cz = mayor.position.z;
      return bumps(mayor, cx - 0.5, cx + 0.5, cz - 0.5, cz + 0.5, x - cx, z - cz);
    }
    // test helper: how many moving things overlap each other or a solid right now
    var lastPair = null;
    function kindOf(bx) { var o = bx.o; var k = walkers.indexOf(o); if (o === mayor) return 'mayor'; if (k >= 0) return 'walker' + k + (o.horiz ? 'H' : 'V') + ' road ' + o.road + ' dir ' + o.dir + ' off ' + o.off.toFixed(2) + '/' + o.toff.toFixed(2) + ' wait ' + o.wait.toFixed(1) + ' @' + bx.x0.toFixed(1) + ',' + bx.z0.toFixed(1); if (ricks.indexOf(o) >= 0) return 'rick@' + bx.x0.toFixed(1) + ',' + bx.z0.toFixed(1); return 'truck'; }
    function lifeCheck() {
      buildBoxes();
      var pairs = 0, inSolid = 0;
      for (var a = 0; a < lifeBoxes.length; a++) {
        for (var b = a + 1; b < lifeBoxes.length; b++) if (ov(lifeBoxes[a], lifeBoxes[b].x0, lifeBoxes[b].x1, lifeBoxes[b].z0, lifeBoxes[b].z1)) { pairs++; lastPair = [kindOf(lifeBoxes[a]), kindOf(lifeBoxes[b])]; }
      }
      for (var k = 0; k < walkers.length && k < outside; k++) { var p = wXZ(walkers[k]); if (staticHit(p[0], p[1], WR - 0.02)) inSolid++; }
      return { walkers: Math.min(outside, walkers.length), ricks: ricks.length, overlaps: pairs, walkersInSolid: inSolid, lastPair: lastPair };
    }

    // static things never move: skip their per-frame matrix updates
    function freeze(o) { o.updateMatrixWorld(true); o.traverse(function (c) { c.matrixAutoUpdate = false; }); }
    freeze(civic); freeze(ground); freeze(roads); freeze(canal);
    wards.forEach(function (w) { w.group.updateMatrixWorld(true); w.group.matrixAutoUpdate = false; });

    // ---------------------------------------------------------------- hotspots
    var hotspots = [
      { id: 'townhall', label: 'Enter the Town Hall', pos: new T.Vector3(0, 0, -36.8), r: 4.5 },
      { id: 'data', label: 'Use the Data Office terminal', pos: new T.Vector3(-27, 0, -38), r: 4 },
      { id: 'kiosk', label: 'Read the newspaper', pos: new T.Vector3(13, 0, -33.6), r: 3.5 },
      { id: 'school', label: 'Visit the Government School', pos: new T.Vector3(27, 0, -38), r: 4 }
    ];
    wards.forEach(function (w, k) { hotspots.push({ id: 'ward' + k, ward: k, label: 'Inspect Ward ' + (k + 1) + ': ' + NAMES[k], pos: new T.Vector3(w.center.x, 0, w.center.z + 2.5), r: 4.2 }); });

    // ---------------------------------------------------------------- data layers (instanced)
    var spec = null, per = {}, outPer = 1, propBatches = [], floatBatch = null, floatSpots = [], floatCount = [], wingFlap = false;
    var spotRng = new M.rng.Rng('town-props-3d');
    for (var w2 = 0; w2 < 20; w2++) {
      var spots = [];
      for (var f = 0; f < 8; f++) spots.push({ x: (spotRng.uniform() - 0.5) * 11, y: 2.2 + spotRng.uniform() * 2.4, z: (spotRng.uniform() - 0.5) * 6, ph: spotRng.uniform() * 6.28 });
      floatSpots.push(spots); floatCount.push(0);
    }
    function setLevel(s) {
      propBatches.forEach(function (b) { b.batch.dispose(scene); }); propBatches = [];
      if (floatBatch) { floatBatch.dispose(scene); floatBatch = null; }
      spec = s; per = {};
      if (!s) return;
      var S = M.stats, C = M.scm.common, hist = s.history;
      s.layers.forEach(function (Ly) { per[Ly.key] = Ly.exact ? 1 : Math.max(1e-9, S.max(C.flatten(hist[Ly.key])) / (Ly.cap || 3)); });
      outPer = Math.max(1, Math.round(S.max(C.flatten(hist[s.outcome.key])) / 8));
      var slot = 0;
      s.layers.forEach(function (Ly) {
        var cap = Ly.cap || 3, b = new Batch(scene, PROP[Ly.icon](), 20 * cap, false), slots = [];
        for (var k = 0; k < cap; k++) { slots.push({ x: -5.8 + (slot % 9) * 1.45, z: 3.4 - Math.floor(slot / 9) * 1.4 }); slot++; }
        propBatches.push({ key: Ly.key, cap: cap, batch: b, slots: slots });
      });
      var kind = s.outcome.icon || 'mosquito';
      floatBatch = new Batch(scene, FLOAT[kind](), 160, false);
      wingFlap = kind === 'mosquito';
    }
    var current = null, rainLevel = 0;
    function update(frame) {
      current = frame;
      if (!spec || !frame) return;
      propBatches.forEach(function (pb) {
        wards.forEach(function (w, i) {
          var n = Math.min(pb.cap, Math.round(frame.vars[pb.key][i] / per[pb.key]));
          for (var k = 0; k < pb.cap; k++) pb.batch.set(i * pb.cap + k, k < n ? mtx(w.center.x + pb.slots[k].x, 0.25, w.center.z + pb.slots[k].z) : null);
        });
        pb.batch.flush();
      });
      wards.forEach(function (w, i) { floatCount[i] = Math.min(8, Math.round(frame.vars[spec.outcome.key][i] / outPer)); });
      var tot = M.stats.sum(frame.vars[spec.outcome.key]), mxTot = M.stats.max(M.scm.common.weekTotals(spec.history[spec.outcome.key])) || 1;
      outside = Math.round(PEOPLE * (1 - 0.65 * Math.min(1, tot / mxTot)));
      if (spec.rain && spec.history) {
        var mx = M.stats.max(M.scm.common.weekMeans(spec.history[spec.rain])) || 1;
        rainLevel = Math.max(0, Math.min(1, (frame.rain || 0) / mx));
      } else rainLevel = 0;
    }
    function setPolicy(kind, mask) {
      trucks.forEach(function (t, k) { showTruck(t, k, kind === 'trucks'); });
      wards.forEach(function (w, i) {
        var on = (kind === 'ban' || kind === 'closed') && (!mask || mask[i]);
        var want = on ? (kind === 'ban' ? '🚫 BANNED' : '✖ CLOSED') : '';
        if (w.banText === want) return;
        if (w.ban) { w.group.remove(w.ban); w.ban = null; }
        if (want) { w.ban = label(want, '#ffffff', 'rgba(208,59,59,0.95)'); w.ban.scale.set(4, 0.8, 1); w.ban.position.set(-4.2, 4.6, 0); w.group.add(w.ban); w.ban.updateMatrixWorld(true); }
        w.banText = want;
      });
    }
    function setMask(mask) {
      wards.forEach(function (w, i) {
        var v = mask ? mask[i] : null, known = v === true || v === false;
        w.ring.material.opacity = known ? 0.85 : 0;
        w.ring.material.color.set(v === true ? '#2fd09a' : '#9097ab');
        var t = known ? (v ? 'T' : 'C') : '';
        if (t === w.tagText) return;
        if (w.tag) { w.group.remove(w.tag); w.tag = null; }
        if (t) { w.tag = label(t, '#ffffff', v ? 'rgba(47,208,154,0.95)' : 'rgba(110,116,140,0.95)'); w.tag.scale.set(2.2, 1.0, 1); w.tag.position.set(5.6, 4.5, -1); w.group.add(w.tag); w.tag.updateMatrixWorld(true); }
        w.tagText = t;
      });
    }
    var beaconTarget = null;
    function setBeacon(id) {
      beaconTarget = id ? hotspots.filter(function (hh) { return hh.id === id; })[0] : null;
      beacon.visible = !!beaconTarget;
      if (beaconTarget) beacon.position.set(beaconTarget.pos.x, 0, beaconTarget.pos.z);
    }

    // ---------------------------------------------------------------- input
    var keys = {}, inputOn = true, walkTarget = null, yaw = 0, dist = 24, pitch = 0.88;
    var hotCb = null, nearCb = null, near = null, stepCb = null;
    function onKey(e, down) {
      if (!inputOn) return;
      var k = e.key.toLowerCase();
      if (['w', 'a', 's', 'd', 'arrowup', 'arrowdown', 'arrowleft', 'arrowright'].indexOf(k) >= 0) { keys[k] = down; walkTarget = null; e.preventDefault(); }
      if (k === 'shift') keys.shift = down;
      if (down && (k === 'e' || k === 'enter' || k === ' ') && near && hotCb) { e.preventDefault(); hotCb(near); }
      if (down && k === 'q') yaw += 0.3;
      if (down && k === 'r') yaw -= 0.3;
    }
    window.addEventListener('keydown', function (e) { onKey(e, true); });
    window.addEventListener('keyup', function (e) { onKey(e, false); });
    var dragging = false, dragMoved = 0, lastX = 0, lastY = 0, el = renderer.domElement;
    el.addEventListener('pointerdown', function (e) { if (!inputOn) return; dragging = true; dragMoved = 0; lastX = e.clientX; lastY = e.clientY; el.setPointerCapture(e.pointerId); });
    el.addEventListener('pointermove', function (e) {
      if (!dragging) return;
      var dx = e.clientX - lastX, dy = e.clientY - lastY; dragMoved += Math.abs(dx) + Math.abs(dy);
      yaw -= dx * 0.006; pitch = Math.max(0.45, Math.min(1.35, pitch + dy * 0.004));
      lastX = e.clientX; lastY = e.clientY;
    });
    el.addEventListener('pointerup', function (e) { if (!dragging) return; dragging = false; if (dragMoved < 6 && inputOn) clickWalk(e); });
    el.addEventListener('wheel', function (e) { dist = Math.max(10, Math.min(60, dist + e.deltaY * 0.03)); e.preventDefault(); }, { passive: false });
    var ray = new T.Raycaster(), ndc = new T.Vector2(), groundPlane = new T.Plane(new T.Vector3(0, 1, 0), 0);
    function clickWalk(e) {
      var r = el.getBoundingClientRect();
      ndc.set(((e.clientX - r.left) / r.width) * 2 - 1, -((e.clientY - r.top) / r.height) * 2 + 1);
      ray.setFromCamera(ndc, camera);
      var hit = new T.Vector3();
      if (ray.ray.intersectPlane(groundPlane, hit)) walkTarget = hit;
    }
    // the mayor's body is a circle of radius BODY: stop it overlapping any solid box
    function blocked(x, z) {
      if (x < -52 || x > 52 || z < -52 || z > 33) return true;
      for (var k = 0; k < colliders.length; k++) {
        var c = colliders[k];
        if (x > c.x0 - BODY && x < c.x1 + BODY && z > c.z0 - BODY && z < c.z1 + BODY) return true;
      }
      return false;
    }
    /* Camera: first point along the line from the mayor's head to the camera that enters a solid
     * box (slab test). The camera is placed just in front of it, so it never goes inside a building. */
    function firstHit(o, d, len) {
      var best = len;
      for (var k = 0; k < colliders.length; k++) {
        var c = colliders[k]; if (c.h <= 0) continue;
        var tmin = 0, tmax = best, lo, hi, t1, t2;
        lo = [c.x0 - 0.3, 0, c.z0 - 0.3]; hi = [c.x1 + 0.3, c.h + 0.3, c.z1 + 0.3];
        var oo = [o.x, o.y, o.z], dd = [d.x, d.y, d.z], ok = true;
        for (var a = 0; a < 3 && ok; a++) {
          if (Math.abs(dd[a]) < 1e-9) { if (oo[a] < lo[a] || oo[a] > hi[a]) ok = false; continue; }
          t1 = (lo[a] - oo[a]) / dd[a]; t2 = (hi[a] - oo[a]) / dd[a];
          if (t1 > t2) { var tt = t1; t1 = t2; t2 = tt; }
          tmin = Math.max(tmin, t1); tmax = Math.min(tmax, t2);
          if (tmin > tmax) ok = false;
        }
        if (ok && tmin > 0 && tmin < best) best = tmin;
      }
      return best;
    }

    // ---------------------------------------------------------------- loop
    var clock = new T.Clock(), walkPhase = 0, tmpV = new T.Vector3(), stepAcc = 0, frameNo = 0, lastFrameMs = 0;
    var orbit = false, orbitA = 0, paused = false, camDir = new T.Vector3(), camFocus = new T.Vector3(), coins = null, fm = new T.Matrix4();
    function resize() {
      var w = container.clientWidth, h = container.clientHeight;
      renderer.setSize(w, h, false); camera.aspect = w / Math.max(1, h); camera.updateProjectionMatrix();
    }
    window.addEventListener('resize', resize);
    resize();

    function tick() {
      requestAnimationFrame(tick);
      var dt = Math.min(0.05, clock.getDelta()), t = clock.elapsedTime;
      if (paused) return; // a panel covers the town: don't spend anything on it

      // movement relative to the camera
      var fx = 0, fz = 0;
      if (inputOn) {
        if (keys.w || keys.arrowup) fz -= 1; if (keys.s || keys.arrowdown) fz += 1;
        if (keys.a || keys.arrowleft) fx -= 1; if (keys.d || keys.arrowright) fx += 1;
      }
      var mvx = 0, mvz = 0, speed = keys.shift ? 21 : 13;
      if (fx || fz) {
        var len = Math.hypot(fx, fz); fx /= len; fz /= len;
        var sy = Math.sin(yaw), cy = Math.cos(yaw);
        mvx = fx * cy + fz * sy; mvz = -fx * sy + fz * cy;
      } else if (walkTarget) {
        var ddx = walkTarget.x - mayor.position.x, ddz = walkTarget.z - mayor.position.z, dd2 = Math.hypot(ddx, ddz);
        if (dd2 < 0.4) walkTarget = null; else { mvx = ddx / dd2; mvz = ddz / dd2; }
      }
      if (mvx || mvz) {
        var nx = mayor.position.x + mvx * speed * dt, nz = mayor.position.z + mvz * speed * dt, moved = false;
        if (!blocked(nx, mayor.position.z) && !dynBlocked(nx, mayor.position.z)) { mayor.position.x = nx; moved = true; }
        if (!blocked(mayor.position.x, nz) && !dynBlocked(mayor.position.x, nz)) { mayor.position.z = nz; moved = true; }
        if (!moved) walkTarget = null;
        var targetRot = Math.atan2(mvx, mvz), dr = ((targetRot - mayor.rotation.y + Math.PI * 3) % (Math.PI * 2)) - Math.PI;
        mayor.rotation.y += dr * Math.min(1, dt * 12);
        walkPhase += dt * 11;
        stepAcc += dt; if (stepAcc > 0.34) { stepAcc = 0; if (stepCb) stepCb(); }
      } else walkPhase *= 0.85;
      var swing = Math.sin(walkPhase) * 0.6 * (mvx || mvz ? 1 : 0);
      var u = mayor.userData; u.legL.rotation.x = swing; u.legR.rotation.x = -swing; u.armL.rotation.x = -swing; u.armR.rotation.x = swing;
      mayor.position.y = Math.abs(Math.sin(walkPhase)) * 0.08;

      // camera
      if (orbit) {
        orbitA += dt * 0.07;
        tmpV.set(Math.sin(orbitA) * 78, 42, Math.cos(orbitA) * 78 - 10);
        camera.position.lerp(tmpV, Math.min(1, dt * 1.5)); camera.lookAt(0, 0, -10);
      } else {
        camFocus.set(mayor.position.x, 2.2, mayor.position.z);
        tmpV.set(mayor.position.x + Math.sin(yaw) * Math.cos(pitch) * dist, Math.sin(pitch) * dist + 2.2, mayor.position.z + Math.cos(yaw) * Math.cos(pitch) * dist);
        camDir.subVectors(tmpV, camFocus); var full = camDir.length(); camDir.normalize();
        var hitD = firstHit(camFocus, camDir, full);
        if (hitD < full) tmpV.copy(camFocus).addScaledVector(camDir, Math.max(0.6, hitD - 0.4));
        camera.position.lerp(tmpV, Math.min(1, dt * (hitD < full ? 14 : 6)));
        camera.lookAt(camFocus);
      }

      // nearest hotspot
      var best = null, bd = 1e9;
      hotspots.forEach(function (hh) { var d = Math.hypot(hh.pos.x - mayor.position.x, hh.pos.z - mayor.position.z); if (d < hh.r && d < bd) { bd = d; best = hh; } });
      if (best !== near) { near = best; if (nearCb) nearCb(near); }

      // floating outcome icons (instanced): bob and buzz around each ward
      if (floatBatch) {
        for (var wi = 0; wi < 20; wi++) {
          var c0w = wards[wi].center;
          for (var fi = 0; fi < 8; fi++) {
            var idx = wi * 8 + fi;
            if (fi >= floatCount[wi]) { floatBatch.set(idx, null); continue; }
            var sp = floatSpots[wi][fi];
            fm.copy(mtx(c0w.x + sp.x + Math.sin(t * 1.7 + sp.ph) * 0.9, sp.y + Math.sin(t * 2.3 + sp.ph) * 0.35, c0w.z + sp.z + Math.cos(t * 1.3 + sp.ph) * 0.7, 0, t + sp.ph, 0));
            if (wingFlap) {
              floatBatch.setPart(idx, 0, tmpM2.multiplyMatrices(fm, floatBatch.parts[0].local).clone());
              floatBatch.setPart(idx, 1, tmpM2.multiplyMatrices(fm, mtx(0, 0.05, 0, 0, 0, Math.sin(t * 40 + sp.ph) * 0.6)).clone());
            } else floatBatch.set(idx, fm);
          }
        }
        floatBatch.flush();
      }
      updateLife(dt, t);
      // coin toss for pilots
      if (coins) {
        var allDone = true;
        coins.list.forEach(function (cn, k) {
          var p = (t - coins.t0 - k * 0.11) / 0.9;
          if (p < 0) { allDone = false; return; }
          if (p < 1) { allDone = false; cn.position.y = 0.6 + Math.sin(Math.PI * p) * 7; cn.rotation.x = p * Math.PI * 7; return; }
          if (!cn.userData.landed) {
            cn.userData.landed = true; cn.position.y = 0.4; cn.rotation.x = 0;
            cn.material.color.set(coins.mask[k] ? '#ffc94a' : '#9097ab');
            coins.partial[k] = coins.mask[k]; setMask(coins.partial);
            if (coins.onLand) coins.onLand(k, coins.mask[k]);
          }
        });
        if (allDone && !coins.ending) {
          coins.ending = true;
          setTimeout(function () { coins.list.forEach(function (cn) { scene.remove(cn); }); var cb = coins.done; coins = null; if (cb) cb(); }, 700);
        }
      }
      // rain and sky
      rain.material.opacity += ((rainLevel * 0.85) - rain.material.opacity) * Math.min(1, dt * 2);
      rain.visible = rain.material.opacity > 0.02;
      if (rain.visible) {
        var pos = rainGeo.attributes.position.array;
        for (var k2 = 1; k2 < pos.length; k2 += 3) { pos[k2] -= dt * 38; if (pos[k2] < 0) pos[k2] += 50; }
        rainGeo.attributes.position.needsUpdate = true;
      }
      scene.background.lerp(tmpCol.copy(skyDay).lerp(skyRain, rainLevel * 0.8), Math.min(1, dt * 2)); scene.fog.color.copy(scene.background);
      sun.intensity = 2.4 - rainLevel * 1.2;
      beacon.rotation.y += dt; if (beacon.visible) arrow.position.y = 9 + Math.sin(t * 3) * 0.6;
      [asha, editor, clerk].forEach(function (n) { n.rotation.y = Math.atan2(mayor.position.x - n.position.x, mayor.position.z - n.position.z); });
      Object.keys(npcs).forEach(function (id) {
        var n = npcs[id]; if (!n.bubble) return;
        n.bubble.material.opacity = Math.max(0, Math.min(1, (16 - Math.hypot(n.obj.position.x - mayor.position.x, n.obj.position.z - mayor.position.z)) / 5));
      });
      if (frameNo % 4 === 0) wardSigns.forEach(function (sg) { sg.getWorldPosition(tmpV); sg.material.opacity = Math.max(0, Math.min(1, (tmpV.distanceTo(camera.position) - 8) / 10)); });

      var f0 = performance.now();
      renderer.render(scene, camera);
      lastFrameMs = performance.now() - f0;
      if (minimapDraw && frameNo % 5 === 0) minimapDraw();
      frameNo++;
    }
    var tmpCol = new T.Color();

    // ---------------------------------------------------------------- minimap (2D canvas, redrawn every few frames)
    var minimapDraw = null;
    function attachMinimap(cv) {
      var c = cv.getContext('2d');
      cv.style.cursor = 'pointer';
      cv.addEventListener('click', function (e) {
        if (!inputOn) return;
        var r = cv.getBoundingClientRect(), s = r.width / 112;
        walkTarget = new T.Vector3((e.clientX - r.left) / s - 56, 0, (e.clientY - r.top) / s - 56);
      });
      minimapDraw = function () {
        var W = cv.width, H = cv.height, s = W / 112;
        function px(x) { return (x + 56) * s; } function pz(z) { return (z + 56) * s; }
        c.fillStyle = '#4a4740'; c.fillRect(0, 0, W, H);
        c.fillStyle = '#2c6fae'; c.fillRect(0, pz(-CANAL / 2), W, CANAL * s);
        wards.forEach(function (w, k) {
          c.fillStyle = beaconTarget && beaconTarget.ward === k ? '#ffc94a' : '#6f9a57';
          c.fillRect(px(w.center.x - BW / 2), pz(w.center.z - BD / 2), BW * s, BD * s);
        });
        [[0, -44, 20, 10, '#f1e6cf'], [-27, -44, 13, 9, '#dfe7d4'], [27, -44, 13, 9, '#f3d9a8'], [13, -36, 3, 2.2, '#c94f2a']].forEach(function (b) {
          c.fillStyle = b[4]; c.fillRect(px(b[0] - b[2] / 2), pz(b[1] - b[3] / 2), b[2] * s, b[3] * s);
        });
        if (beaconTarget) { c.fillStyle = '#ffc94a'; c.beginPath(); c.arc(px(beaconTarget.pos.x), pz(beaconTarget.pos.z), 4, 0, 7); c.fill(); }
        c.save(); c.translate(px(mayor.position.x), pz(mayor.position.z)); c.rotate(-mayor.rotation.y + Math.PI);
        c.fillStyle = '#ffffff'; c.beginPath(); c.moveTo(0, -6); c.lineTo(4, 4); c.lineTo(-4, 4); c.closePath(); c.fill(); c.restore();
      };
    }

    requestAnimationFrame(tick);

    return {
      setLevel: setLevel, update: update, setPolicy: setPolicy, setMask: setMask, setBeacon: setBeacon, say: say,
      hotspots: hotspots, attachMinimap: attachMinimap,
      setOrbit: function (on) { orbit = on; },
      setPaused: function (on) { paused = on; if (on) renderer.render(scene, camera); },
      rainLevel: function () { return rainLevel; },
      onStep: function (fn) { stepCb = fn; },
      onHotspot: function (fn) { hotCb = fn; }, onNear: function (fn) { nearCb = fn; },
      setInputEnabled: function (on) { inputOn = on; keys = {}; walkTarget = null; },
      teleport: function (x, z, face) { mayor.position.set(x, 0, z); if (face !== undefined) { yaw = face; mayor.rotation.y = face + Math.PI; } },
      walkTo: function (x, z) { walkTarget = new T.Vector3(x, 0, z); },
      coinToss: function (mask, onLand, done) {
        var list = wards.map(function (w) {
          var cn = new T.Mesh(new T.CylinderGeometry(0.9, 0.9, 0.16, 20), new T.MeshLambertMaterial({ color: '#ffd978' }));
          cn.position.set(w.center.x, 0.6, w.center.z); cn.castShadow = true; scene.add(cn); return cn;
        });
        coins = { list: list, mask: mask, partial: mask.map(function () { return null; }), t0: clock.elapsedTime + 0.3, onLand: onLand, done: done };
      },
      current: function () { return current; },
      wardName: function (k) { return NAMES[k]; },
      _blocked: blocked, _pos: function () { return { x: mayor.position.x, z: mayor.position.z }; },
      _life: lifeCheck,
      // test helper: flood-fill the walkable ground from the start; how close can the mayor get to each E spot?
      _reach: function () {
        var S = 0.5, x0 = -52, z0 = -52, nx = Math.round(104 / S) + 1, nz = Math.round(85 / S) + 1, seen = new Uint8Array(nx * nz), q = [];
        var sx = Math.round((0 - x0) / S), sz = Math.round((-33 - z0) / S);
        seen[sz * nx + sx] = 1; q.push(sx, sz);
        var best = hotspots.map(function () { return 1e9; });
        while (q.length) {
          var cz = q.pop(), cx = q.pop(), wx = x0 + cx * S, wz = z0 + cz * S;
          hotspots.forEach(function (h, k) { var d = Math.hypot(h.pos.x - wx, h.pos.z - wz); if (d < best[k]) best[k] = d; });
          [[1, 0], [-1, 0], [0, 1], [0, -1]].forEach(function (o) {
            var ax = cx + o[0], az = cz + o[1]; if (ax < 0 || az < 0 || ax >= nx || az >= nz || seen[az * nx + ax]) return;
            seen[az * nx + ax] = 1; if (!blocked(x0 + ax * S, z0 + az * S)) q.push(ax, az);
          });
        }
        return hotspots.map(function (h, k) { return { id: h.id, closest: +best[k].toFixed(2), radius: h.r, ok: best[k] < h.r - 0.3 }; });
      },
      _interact: function (id) { var h = hotspots.filter(function (hh) { return hh.id === id; })[0]; if (h && hotCb) hotCb(h); },
      _walkers: function () { return walkers.map(function (w, k) { var p = wXZ(w); return { k: k, out: k < outside, horiz: w.horiz, road: w.road, dir: w.dir, off: +w.off.toFixed(2), toff: w.toff, wait: +w.wait.toFixed(2), x: +p[0].toFixed(2), z: +p[1].toFixed(2) }; }); },
      // test helper: run town life for `sec` seconds without drawing; reports the worst moment
      _simLife: function (sec, opts) {
        opts = opts || {}; var worst = { overlaps: 0, walkersInSolid: 0 }, tt = 0, stuck = 0;
        if (opts.trucks) trucks.forEach(function (tr, k) { showTruck(tr, k, true); });
        if (opts.mayorAt) mayor.position.set(opts.mayorAt[0], 0, opts.mayorAt[1]);
        var start = walkers.map(function (w) { return wXZ(w); });
        for (var i = 0; i < sec * 30; i++) {
          tt += 1 / 30; updateLife(1 / 30, tt);
          if (i % 15 === 0) { var c = lifeCheck(); if (c.overlaps > worst.overlaps) worst.overlaps = c.overlaps; if (c.overlaps) (worst.seen = worst.seen || []).length < 8 && worst.seen.push(c.lastPair.join(' x ')); if (c.walkersInSolid > worst.walkersInSolid) worst.walkersInSolid = c.walkersInSolid; }
        }
        walkers.forEach(function (w, k) { var p = wXZ(w); if (Math.hypot(p[0] - start[k][0], p[1] - start[k][1]) < 0.5) stuck++; });
        if (opts.trucks) trucks.forEach(function (tr) { tr.visible = false; });
        worst.stuckWalkers = stuck; worst.rickX = ricks.map(function (r) { return +r.x.toFixed(1); });
        return worst;
      },
      _debug: function () { var n = 0; scene.traverse(function (o) { if (o.isMesh) n++; }); return { calls: renderer.info.render.calls, triangles: renderer.info.render.triangles, meshes: n, colliders: colliders.length, frameMs: lastFrameMs }; }
    };
  }

  M.world3d = { create: create, supported: function () {
    try { var c = document.createElement('canvas'); return !!(globalThis.THREE && (c.getContext('webgl2') || c.getContext('webgl'))); } catch (e) { return false; }
  } };
})(globalThis.Mayor = globalThis.Mayor || {});
