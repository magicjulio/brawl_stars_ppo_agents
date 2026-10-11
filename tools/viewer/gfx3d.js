'use strict';
// 3D view for the replay viewer: the game's own models, textures and lightmap shading (tools/gfx/build.py packs,
// shader logic from the APK's shader/uber.fsh). World = tiles: x right, y = -logic y (up the screen), z up.
const GFX = (() => {
  const S = {textures: {}, materials: {}, models: {}, anims: {}, heroes: {}, themes: {}};
  const loading = {};
  window.BSGFX = d => {
    for (const k of ['textures', 'materials', 'models', 'anims', 'heroes']) Object.assign(S[k], d[k] || {});
    if (d.theme) S.themes[d.theme.name] = d.theme;
    for (const [family, b64] of Object.entries(d.fonts || {}))
      new FontFace(family, bytes(b64)).load().then(f => { document.fonts.add(f); markDirty(); }).catch(() => {});
  };
  const pack = name => loading[name] || (loading[name] = new Promise((res, rej) => {
    const s = document.createElement('script');
    s.src = GFX_BASE + 'gfx/' + name + '.js'; s.onload = res; s.onerror = () => rej(new Error('no pack ' + name));
    document.head.appendChild(s);
  }));
  const bytes = s => { const b = atob(s), u = new Uint8Array(b.length); for (let i = 0; i < b.length; i++) u[i] = b.charCodeAt(i); return u.buffer; };
  const F32 = s => new Float32Array(bytes(s)), U16 = s => new Uint16Array(bytes(s)), U32 = s => new Uint32Array(bytes(s)), U8 = s => new Uint8Array(bytes(s));

  // ---- mat4 (column-major) ----
  const M = {
    id: () => new Float32Array([1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1]),
    mul(a, b, o = new Float32Array(16)) {
      for (let c = 0; c < 4; c++) for (let r = 0; r < 4; r++)
        o[c * 4 + r] = a[r] * b[c * 4] + a[4 + r] * b[c * 4 + 1] + a[8 + r] * b[c * 4 + 2] + a[12 + r] * b[c * 4 + 3];
      return o;
    },
    trs(t, q, s, o = new Float32Array(16)) {
      let [x, y, z, w] = q; const n = Math.hypot(x, y, z, w) || 1; x /= n; y /= n; z /= n; w /= n;
      o[0] = (1 - 2 * (y * y + z * z)) * s[0]; o[1] = 2 * (x * y + z * w) * s[0]; o[2] = 2 * (x * z - y * w) * s[0]; o[3] = 0;
      o[4] = 2 * (x * y - z * w) * s[1]; o[5] = (1 - 2 * (x * x + z * z)) * s[1]; o[6] = 2 * (y * z + x * w) * s[1]; o[7] = 0;
      o[8] = 2 * (x * z + y * w) * s[2]; o[9] = 2 * (y * z - x * w) * s[2]; o[10] = (1 - 2 * (x * x + y * y)) * s[2]; o[11] = 0;
      o[12] = t[0]; o[13] = t[1]; o[14] = t[2]; o[15] = 1; return o;
    },
    persp(fovy, asp, n, f) {
      const t = 1 / Math.tan(fovy / 2), o = new Float32Array(16);
      o[0] = t / asp; o[5] = t; o[10] = (f + n) / (n - f); o[11] = -1; o[14] = 2 * f * n / (n - f); return o;
    },
    lookAt(e, c, up) {
      const z = norm(sub(e, c)), x = norm(cross(up, z)), y = cross(z, x);
      return new Float32Array([x[0], y[0], z[0], 0, x[1], y[1], z[1], 0, x[2], y[2], z[2], 0,
        -dot(x, e), -dot(y, e), -dot(z, e), 1]);
    },
    rotZ(a) { const c = Math.cos(a), s = Math.sin(a), o = M.id(); o[0] = c; o[1] = s; o[4] = -s; o[5] = c; return o; },
    tr(x, y, z) { const o = M.id(); o[12] = x; o[13] = y; o[14] = z; return o; },
    sc(s) { const o = M.id(); o[0] = o[5] = o[10] = s; return o; },
  };
  const sub = (a, b) => [a[0] - b[0], a[1] - b[1], a[2] - b[2]], dot = (a, b) => a[0] * b[0] + a[1] * b[1] + a[2] * b[2];
  const cross = (a, b) => [a[1] * b[2] - a[2] * b[1], a[2] * b[0] - a[0] * b[2], a[0] * b[1] - a[1] * b[0]];
  const norm = a => { const l = Math.hypot(...a) || 1; return [a[0] / l, a[1] / l, a[2] / l]; };
  const Y2Z = new Float32Array([1, 0, 0, 0, 0, 0, 1, 0, 0, -1, 0, 0, 0, 0, 0, 1]);   // Rx(+90): model y-up -> z-up

  // ---- GL ----
  let gl, prog, U = {}, texCache = {}, white, lmFallback;
  const VS = `#version 300 es
in vec3 a_pos; in vec3 a_nrm; in vec2 a_uv;
uniform mat4 u_pv, u_view, u_model;
uniform float u_outline;      // > 0: inverted-hull outline pass, offset along the normal (model units)
uniform mat4 u_lpv;           // light projection-view of the shadow map
out vec2 v_uv; out vec2 v_lm; out vec3 v_wpos; out vec4 v_lpos;
void main() {
  vec4 wp = u_model * vec4(a_pos + a_nrm * u_outline, 1.0);
  v_wpos = wp.xyz; v_lpos = u_lpv * wp;
  vec3 n = normalize(mat3(u_view) * mat3(u_model) * a_nrm);
  v_uv = a_uv; v_lm = n.xy * vec2(0.5, -0.5) + vec2(0.5);
  gl_Position = u_pv * wp;
}`;
  const FS = `#version 300 es
precision mediump float;
in vec2 v_uv; in vec2 v_lm; in vec3 v_wpos; in vec4 v_lpos;
uniform sampler2D u_diff, u_lmd, u_lms;
uniform mediump sampler2DShadow u_shadow;
uniform float u_shadowOn, u_texel;
uniform vec4 u_flat;          // a > 0: flat colour (outline pass)
uniform vec4 u_fade;          // xy centre, z radius, w on: see-through bushes around the own brawler
uniform vec4 u_diffColor, u_specColor, u_colorize, u_tint;
uniform vec3 u_emission;
uniform int u_diffTex, u_lmdOn, u_lmsOn, u_specFromDiff;
uniform float u_alpha;
out vec4 o;
void main() {
  vec4 base = u_diffTex == 1 ? texture(u_diff, v_uv) : u_diffColor;
  vec4 c = base;
  if (u_lmdOn == 1) c.rgb *= texture(u_lmd, v_lm).rgb;
  c *= u_colorize;
  c.rgb += u_emission;
  if (u_lmsOn == 1) c.rgb += texture(u_lms, v_lm).rgb * (u_specFromDiff == 1 ? base.rgb : u_specColor.rgb);
  if (u_shadowOn > 0.0) {       // as the game's uber shader: shadowed colour x 0.75 (3x3 PCF)
    vec3 p = v_lpos.xyz / v_lpos.w * 0.5 + 0.5;
    float lit = 0.0;
    for (int i = -1; i <= 1; i++) for (int j = -1; j <= 1; j++)
      lit += texture(u_shadow, vec3(p.xy + vec2(float(i), float(j)) * u_texel, p.z - 0.0008));
    c.rgb *= mix(0.75, 1.0, lit / 9.0);
  }
  float a = u_alpha;
  if (u_fade.w > 0.0) a *= mix(0.42, 1.0, smoothstep(u_fade.z, u_fade.z + 0.9, distance(v_wpos.xy, u_fade.xy)));
  o = u_flat.a > 0.0 ? u_flat : vec4(c.rgb * u_tint.rgb + u_tint.a * vec3(1.0), a);
}`;

  // ground decals under characters: soft shadow + team ring
  const DVS = `#version 300 es
in vec2 a_c; uniform mat4 u_pv; uniform vec3 u_center; uniform float u_r; out vec2 v_c;
void main() { v_c = a_c; gl_Position = u_pv * vec4(u_center + vec3(a_c * u_r, 0.0), 1.0); }`;
  const DFS = `#version 300 es
precision mediump float;
in vec2 v_c; uniform vec3 u_col; out vec4 o;
void main() {
  float d = length(v_c);
  if (d > 1.0) discard;
  float shadow = 0.38 * (1.0 - smoothstep(0.2, 0.95, d));
  float ring = smoothstep(0.74, 0.8, d) * (1.0 - smoothstep(0.92, 0.99, d));
  o = vec4(u_col * ring, max(shadow * (1.0 - ring), ring * 0.95));
}`;
  let dprog, DU = {}, quad;
  const SVS = `#version 300 es
in vec3 a_pos; uniform mat4 u_lpv, u_model; void main() { gl_Position = u_lpv * u_model * vec4(a_pos, 1.0); }`;
  const SFS = `#version 300 es
precision mediump float; out vec4 o; void main() { o = vec4(1.0); }`;
  const PHONE = matchMedia('(pointer: coarse)').matches;      // touch screens: lighter settings (shadow map, pixel density)
  const SHADOW_N = PHONE ? 1024 : 2048;
  let sprog, SU = {}, shadowFbo, shadowTex, lpv = M.id();
  function initShadow(sh) {
    sprog = gl.createProgram(); gl.attachShader(sprog, sh(gl.VERTEX_SHADER, SVS)); gl.attachShader(sprog, sh(gl.FRAGMENT_SHADER, SFS));
    gl.bindAttribLocation(sprog, 0, 'a_pos'); gl.linkProgram(sprog);
    for (const n of ['u_lpv', 'u_model']) SU[n] = gl.getUniformLocation(sprog, n);
    gl.activeTexture(gl.TEXTURE4);
    shadowTex = gl.createTexture(); gl.bindTexture(gl.TEXTURE_2D, shadowTex);
    gl.texStorage2D(gl.TEXTURE_2D, 1, gl.DEPTH_COMPONENT24, SHADOW_N, SHADOW_N);
    for (const [k, v] of [[gl.TEXTURE_MIN_FILTER, gl.LINEAR], [gl.TEXTURE_MAG_FILTER, gl.LINEAR], [gl.TEXTURE_WRAP_S, gl.CLAMP_TO_EDGE],
      [gl.TEXTURE_WRAP_T, gl.CLAMP_TO_EDGE], [gl.TEXTURE_COMPARE_MODE, gl.COMPARE_REF_TO_TEXTURE], [gl.TEXTURE_COMPARE_FUNC, gl.LEQUAL]])
      gl.texParameteri(gl.TEXTURE_2D, k, v);
    shadowFbo = gl.createFramebuffer(); gl.bindFramebuffer(gl.FRAMEBUFFER, shadowFbo);
    gl.framebufferTexture2D(gl.FRAMEBUFFER, gl.DEPTH_ATTACHMENT, gl.TEXTURE_2D, shadowTex, 0);
    gl.drawBuffers([gl.NONE]); gl.readBuffer(gl.NONE);
    const ok = gl.checkFramebufferStatus(gl.FRAMEBUFFER) === gl.FRAMEBUFFER_COMPLETE;
    gl.bindFramebuffer(gl.FRAMEBUFFER, null);
    return ok;
  }
  let shadowOk = false;
  function lightMatrix(mapW, mapH) {     // orthographic light over the arena; light from the top-right, high up
    const L = norm([-0.3, -0.75, -1]), c = [mapW / 2, -mapH / 2, 0];
    const v = M.lookAt([c[0] - L[0] * 50, c[1] - L[1] * 50, c[2] - L[2] * 50], c, [0, 1, 0]);
    const mn = [1e9, 1e9, 1e9], mx = [-1e9, -1e9, -1e9];
    for (const x of [-4, mapW + 4]) for (const y of [4, -mapH - 4]) for (const z of [-2, 5]) {
      const q = [v[0] * x + v[4] * y + v[8] * z + v[12], v[1] * x + v[5] * y + v[9] * z + v[13], v[2] * x + v[6] * y + v[10] * z + v[14]];
      for (let k = 0; k < 3; k++) { mn[k] = Math.min(mn[k], q[k]); mx[k] = Math.max(mx[k], q[k]); }
    }
    const l = mn[0], r = mx[0], b = mn[1], t = mx[1], n = -mx[2], f = -mn[2], o = new Float32Array(16);
    o[0] = 2 / (r - l); o[5] = 2 / (t - b); o[10] = -2 / (f - n); o[12] = -(r + l) / (r - l); o[13] = -(t + b) / (t - b); o[14] = -(f + n) / (f - n); o[15] = 1;
    return M.mul(o, v);
  }
  function init(canvas) {
    gl = canvas.getContext('webgl2', {antialias: true, alpha: false, preserveDrawingBuffer: true});
    if (!gl) return false;
    const sh = (t, src) => { const s = gl.createShader(t); gl.shaderSource(s, src); gl.compileShader(s);
      if (!gl.getShaderParameter(s, gl.COMPILE_STATUS)) throw new Error(gl.getShaderInfoLog(s)); return s; };
    prog = gl.createProgram(); gl.attachShader(prog, sh(gl.VERTEX_SHADER, VS)); gl.attachShader(prog, sh(gl.FRAGMENT_SHADER, FS));
    gl.bindAttribLocation(prog, 0, 'a_pos'); gl.bindAttribLocation(prog, 1, 'a_nrm'); gl.bindAttribLocation(prog, 2, 'a_uv');
    gl.linkProgram(prog);
    if (!gl.getProgramParameter(prog, gl.LINK_STATUS)) throw new Error(gl.getProgramInfoLog(prog));
    for (const n of ['u_pv', 'u_view', 'u_model', 'u_diff', 'u_lmd', 'u_lms', 'u_diffColor', 'u_specColor', 'u_colorize', 'u_tint',
      'u_emission', 'u_diffTex', 'u_lmdOn', 'u_lmsOn', 'u_specFromDiff', 'u_alpha', 'u_outline', 'u_flat', 'u_fade', 'u_lpv',
      'u_shadow', 'u_shadowOn', 'u_texel']) U[n] = gl.getUniformLocation(prog, n);
    white = solidTex([255, 255, 255, 255]);
    try { shadowOk = initShadow(sh); } catch (e) { console.warn('shadows off:', e); }
    dprog = gl.createProgram(); gl.attachShader(dprog, sh(gl.VERTEX_SHADER, DVS)); gl.attachShader(dprog, sh(gl.FRAGMENT_SHADER, DFS));
    gl.bindAttribLocation(dprog, 0, 'a_c'); gl.linkProgram(dprog);
    for (const n of ['u_pv', 'u_center', 'u_r', 'u_col']) DU[n] = gl.getUniformLocation(dprog, n);
    quad = gl.createVertexArray(); gl.bindVertexArray(quad);
    const qb = gl.createBuffer(); gl.bindBuffer(gl.ARRAY_BUFFER, qb);
    gl.bufferData(gl.ARRAY_BUFFER, new Float32Array([-1, -1, 1, -1, -1, 1, 1, 1]), gl.STATIC_DRAW);
    gl.enableVertexAttribArray(0); gl.vertexAttribPointer(0, 2, gl.FLOAT, false, 0, 0); gl.bindVertexArray(null);
    return true;
  }
  function solidTex(c) {
    gl.activeTexture(gl.TEXTURE4);
    const t = gl.createTexture(); gl.bindTexture(gl.TEXTURE_2D, t);
    gl.texImage2D(gl.TEXTURE_2D, 0, gl.RGBA, 1, 1, 0, gl.RGBA, gl.UNSIGNED_BYTE, new Uint8Array(c)); return t;
  }
  function tex(key) {     // GL texture for a pack texture key (loads asynchronously; white until ready)
    if (!key || !S.textures[key]) return null;
    if (texCache[key]) return texCache[key];
    const t = gl.createTexture(); texCache[key] = t;
    gl.activeTexture(gl.TEXTURE4);      // scratch unit: units 0-3 are bound for drawing (3 = shadow map)
    gl.bindTexture(gl.TEXTURE_2D, t);
    gl.texImage2D(gl.TEXTURE_2D, 0, gl.RGBA, 1, 1, 0, gl.RGBA, gl.UNSIGNED_BYTE, new Uint8Array([200, 200, 200, 255]));
    const img = new Image();
    img.onload = () => {
      gl.activeTexture(gl.TEXTURE4); gl.bindTexture(gl.TEXTURE_2D, t);
      gl.texImage2D(gl.TEXTURE_2D, 0, gl.RGBA, gl.RGBA, gl.UNSIGNED_BYTE, img);
      const lm = /lightmap/.test(key);
      gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_S, lm ? gl.CLAMP_TO_EDGE : gl.REPEAT);
      gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_T, lm ? gl.CLAMP_TO_EDGE : gl.REPEAT);
      gl.generateMipmap(gl.TEXTURE_2D);
      gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MIN_FILTER, gl.LINEAR_MIPMAP_LINEAR);
      gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MAG_FILTER, gl.LINEAR);
      markDirty();
    };
    img.src = S.textures[key];
    return t;
  }
  let dirty = true, redrawQueued = false;
  const API = {onDirty: null};
  function markDirty() {          // an asset arrived after the last frame: ask the page for one redraw
    dirty = true;
    if (API.onDirty && !redrawQueued) { redrawQueued = true; requestAnimationFrame(() => { redrawQueued = false; API.onDirty(); }); }
  }

  // ---- geometry: a prim -> VAO (static), or with dynamic position/normal buffers (skinned) ----
  function makePrim(p, dynamic) {
    const vao = gl.createVertexArray(); gl.bindVertexArray(vao);
    const buf = (loc, data, size, usage) => {
      const b = gl.createBuffer(); gl.bindBuffer(gl.ARRAY_BUFFER, b); gl.bufferData(gl.ARRAY_BUFFER, data, usage);
      gl.enableVertexAttribArray(loc); gl.vertexAttribPointer(loc, size, gl.FLOAT, false, 0, 0); return b;
    };
    const pos = p._pos || F32(p.pos), nrm = p.nrm ? (p._nrm || F32(p.nrm)) : new Float32Array(pos.length);
    const uv = p.uv ? F32(p.uv) : new Float32Array(pos.length / 3 * 2);
    const out = {vao, n: 0, pb: buf(0, pos, 3, dynamic ? gl.DYNAMIC_DRAW : gl.STATIC_DRAW),
      nb: buf(1, nrm, 3, dynamic ? gl.DYNAMIC_DRAW : gl.STATIC_DRAW)};
    buf(2, uv, 2, gl.STATIC_DRAW);
    const idx = p._idx || (p.i32 ? U32(p.index) : U16(p.index));
    const ib = gl.createBuffer(); gl.bindBuffer(gl.ELEMENT_ARRAY_BUFFER, ib); gl.bufferData(gl.ELEMENT_ARRAY_BUFFER, idx, gl.STATIC_DRAW);
    out.n = idx.length; out.type = idx instanceof Uint32Array ? gl.UNSIGNED_INT : gl.UNSIGNED_SHORT;
    gl.bindVertexArray(null);
    return out;
  }

  // node world matrices of a model, with optional animation pose {nodeName: [t, r, s]}
  function nodeWorld(model, pose) {
    const W = [];
    model.nodes.forEach((n, i) => {
      const p = pose && pose[n.name];
      const local = p ? M.trs(p[0], p[1], p[2]) : (n._m || (n._m = M.trs(n.t, n.r, n.s)));
      W[i] = n.parent >= 0 ? M.mul(W[n.parent], local) : local;
    });
    return W;
  }

  // ---- static scene (ground, environment, tiles): baked into world space per material ----
  let scene = null;
  function bakeInto(groups, model, filter, xf) {
    const W = nodeWorld(model);
    for (const m of model.meshes) {
      if (filter && !filter(model, m)) continue;
      const wm = M.mul(xf, W[m.node]);
      for (const p of m.prims) {
        const pos = F32(p.pos), nrm = p.nrm ? F32(p.nrm) : new Float32Array(pos.length), uv = p.uv ? F32(p.uv) : new Float32Array(pos.length / 3 * 2);
        const idx = p.i32 ? U32(p.index) : U16(p.index);
        const g = groups[p.material] || (groups[p.material] = {pos: [], nrm: [], uv: [], idx: [], n: 0});
        const base = g.n;
        for (let i = 0; i < pos.length; i += 3) {
          const x = pos[i], y = pos[i + 1], z = pos[i + 2];
          g.pos.push(wm[0] * x + wm[4] * y + wm[8] * z + wm[12], wm[1] * x + wm[5] * y + wm[9] * z + wm[13], wm[2] * x + wm[6] * y + wm[10] * z + wm[14]);
          const a = nrm[i], b = nrm[i + 1], c = nrm[i + 2];
          const nx = wm[0] * a + wm[4] * b + wm[8] * c, ny = wm[1] * a + wm[5] * b + wm[9] * c, nz = wm[2] * a + wm[6] * b + wm[10] * c, l = Math.hypot(nx, ny, nz) || 1;
          g.nrm.push(nx / l, ny / l, nz / l);
        }
        for (let i = 0; i < uv.length; i++) g.uv.push(uv[i]);
        for (let i = 0; i < idx.length; i++) g.idx.push(idx[i] + base);
        g.n += pos.length / 3;
      }
    }
  }
  function finishGroups(groups) {
    const out = [];
    for (const [mat, g] of Object.entries(groups)) {
      const p = {_pos: new Float32Array(g.pos), _nrm: new Float32Array(g.nrm), uv: null, _idx: new Uint32Array(g.idx)};
      const prim = makePrim({...p, nrm: true, i32: true}, false);
      gl.bindVertexArray(prim.vao);
      const ub = gl.createBuffer(); gl.bindBuffer(gl.ARRAY_BUFFER, ub); gl.bufferData(gl.ARRAY_BUFFER, new Float32Array(g.uv), gl.STATIC_DRAW);
      gl.enableVertexAttribArray(2); gl.vertexAttribPointer(2, 2, gl.FLOAT, false, 0, 0);
      gl.bindVertexArray(null);
      out.push({prim, mat});
    }
    return out;
  }
  const hash = (i, k) => { let h = (i * 374761393 + k * 668265263) | 0; h = (h ^ (h >>> 13)) * 1274126177; return ((h ^ (h >>> 16)) >>> 0) / 4294967296; };

  function buildStatic(themeName, grid, names, w, h) {
    const th = S.themes[themeName]; if (!th) return null;
    const env = {}, tiles = {}, bushes = {};
    if (th.GroundSCW) bakeInto(env, S.models[th.GroundSCW], null, M.id());
    if (th.EnvironmentSCW) bakeInto(env, S.models[th.EnvironmentSCW], null, M.id());
    for (let i = 0; i < grid.length; i++) {
      const t = th.tiles[names[grid[i]]]; if (!t) continue;
      const model = S.models[t.model]; if (!model) continue;
      const tx = i % w + 0.5, ty = -(Math.floor(i / w) + 0.5);
      const step = t.angleStep, ang = step >= 90 ? Math.floor(hash(i, 1) * 4) * step : step === 1 ? hash(i, 2) * 360 : 0;
      const xf = M.mul(M.tr(tx, ty, 0), M.rotZ(ang * Math.PI / 180));
      const want = t.mesh;
      bakeInto(/Forest|Bush/.test(names[grid[i]]) ? bushes : tiles, model, (mo, m) => {
        const nn = mo.nodes[m.node].name;
        if (want) { let k = m.node; while (k >= 0) { if (mo.nodes[k].name === want) return true; k = mo.nodes[k].parent; } return false; }
        return !/destroy|broken|_dead/i.test(nn);
      }, xf);
    }
    return {env: finishGroups(env), tiles: finishGroups(tiles), bushes: finishGroups(bushes)};
  }

  // ---- characters ----
  const heroGL = {};       // model name -> [{mesh, prims:[{p, gl, pos, nrm, j, w}]}]
  function heroInstance(name) {
    const hero = S.heroes[name]; if (!hero) return null;
    const model = S.models[hero.model];
    const meshes = model.meshes.map(m => ({m, prims: m.prims.map(p => {
      const pos = F32(p.pos), nrm = p.nrm ? F32(p.nrm) : new Float32Array(pos.length);
      return {p, pos, nrm, j: p.j ? U8(p.j) : null, w: p.w ? F32(p.w) : null, outPos: new Float32Array(pos.length),
        outNrm: new Float32Array(pos.length), gl: makePrim(p, true)};
    }), inv: m.invBind ? F32(m.invBind) : null, bindShape: m.bindShape ? new Float32Array(m.bindShape) : null}));
    return {hero, model, meshes, anim: 'idle', t0: 0, prevAnim: null};
  }

  function samplePose(animKey, time) {
    const a = S.anims[animKey]; if (!a) return null;
    let f = time * a.fps * (a.speed || 1);
    f = a.loop ? f % a.frames : Math.min(f, a.frames - 1);
    const f0 = Math.floor(f), f1 = a.loop ? (f0 + 1) % a.frames : Math.min(f0 + 1, a.frames - 1), u = f - f0;
    if (!a._t) { a._t = {}; for (const [k, v] of Object.entries(a.tracks)) a._t[k] = [F32(v.t), F32(v.r), F32(v.s)]; }
    const pose = {};
    for (const [k, [T, R, Sx]] of Object.entries(a._t)) {
      const lerp = (A, n, i0, i1) => { const o = []; for (let c = 0; c < n; c++) o.push(A[i0 * n + c] * (1 - u) + A[i1 * n + c] * u); return o; };
      let q0 = [R[f0 * 4], R[f0 * 4 + 1], R[f0 * 4 + 2], R[f0 * 4 + 3]], q1 = [R[f1 * 4], R[f1 * 4 + 1], R[f1 * 4 + 2], R[f1 * 4 + 3]];
      if (q0[0] * q1[0] + q0[1] * q1[1] + q0[2] * q1[2] + q0[3] * q1[3] < 0) q1 = q1.map(x => -x);
      pose[k] = [lerp(T, 3, f0, f1), [0, 1, 2, 3].map(c => q0[c] * (1 - u) + q1[c] * u), lerp(Sx, 3, f0, f1)];
    }
    return pose;
  }

  function skin(inst, pose) {
    const W = nodeWorld(inst.model, pose);
    const J = new Float32Array(16 * 128);
    for (const mm of inst.meshes) {
      const m = mm.m;
      if (mm.inv) {
        const nj = m.joints.length;
        for (let k = 0; k < nj; k++) {
          const jw = W[m.joints[k]] || M.id();
          let t = M.mul(jw, mm.inv.subarray(k * 16, k * 16 + 16));
          if (mm.bindShape) t = M.mul(t, mm.bindShape);
          J.set(t, k * 16);
        }
      }
      const nodeM = W[m.node];
      for (const pr of mm.prims) {
        const P = pr.pos, N = pr.nrm, O = pr.outPos, ON = pr.outNrm, nv = P.length / 3;
        if (mm.inv && pr.j) {
          for (let v = 0; v < nv; v++) {
            let a0 = 0, a1 = 0, a2 = 0, a4 = 0, a5 = 0, a6 = 0, a8 = 0, a9 = 0, a10 = 0, a12 = 0, a13 = 0, a14 = 0;
            for (let k = 0; k < 4; k++) {
              const wt = pr.w[v * 4 + k]; if (wt === 0) continue;
              const o = pr.j[v * 4 + k] * 16;
              a0 += J[o] * wt; a1 += J[o + 1] * wt; a2 += J[o + 2] * wt; a4 += J[o + 4] * wt; a5 += J[o + 5] * wt; a6 += J[o + 6] * wt;
              a8 += J[o + 8] * wt; a9 += J[o + 9] * wt; a10 += J[o + 10] * wt; a12 += J[o + 12] * wt; a13 += J[o + 13] * wt; a14 += J[o + 14] * wt;
            }
            const x = P[v * 3], y = P[v * 3 + 1], z = P[v * 3 + 2];
            O[v * 3] = a0 * x + a4 * y + a8 * z + a12; O[v * 3 + 1] = a1 * x + a5 * y + a9 * z + a13; O[v * 3 + 2] = a2 * x + a6 * y + a10 * z + a14;
            const nx = N[v * 3], ny = N[v * 3 + 1], nz = N[v * 3 + 2];
            ON[v * 3] = a0 * nx + a4 * ny + a8 * nz; ON[v * 3 + 1] = a1 * nx + a5 * ny + a9 * nz; ON[v * 3 + 2] = a2 * nx + a6 * ny + a10 * nz;
          }
        } else {
          const w = nodeM;
          for (let v = 0; v < nv; v++) {
            const x = P[v * 3], y = P[v * 3 + 1], z = P[v * 3 + 2];
            O[v * 3] = w[0] * x + w[4] * y + w[8] * z + w[12]; O[v * 3 + 1] = w[1] * x + w[5] * y + w[9] * z + w[13]; O[v * 3 + 2] = w[2] * x + w[6] * y + w[10] * z + w[14];
            const nx = N[v * 3], ny = N[v * 3 + 1], nz = N[v * 3 + 2];
            ON[v * 3] = w[0] * nx + w[4] * ny + w[8] * nz; ON[v * 3 + 1] = w[1] * nx + w[5] * ny + w[9] * nz; ON[v * 3 + 2] = w[2] * nx + w[6] * ny + w[10] * nz;
          }
        }
        gl.bindBuffer(gl.ARRAY_BUFFER, pr.gl.pb); gl.bufferSubData(gl.ARRAY_BUFFER, 0, O);
        gl.bindBuffer(gl.ARRAY_BUFFER, pr.gl.nb); gl.bufferSubData(gl.ARRAY_BUFFER, 0, ON);
      }
    }
  }

  // ---- drawing ----
  function setMaterial(key, override) {
    const m = S.materials[key] || {};
    const d = m.diffuse || {color: [0.8, 0.8, 0.8, 1]};
    let dt = null;
    if (d.tex === '.') dt = override ? tex(override) : null; else if (d.tex) dt = tex(d.tex);
    gl.uniform1i(U.u_diffTex, dt ? 1 : 0);
    gl.activeTexture(gl.TEXTURE0); gl.bindTexture(gl.TEXTURE_2D, dt || white);
    gl.uniform4fv(U.u_diffColor, d.color || [1, 1, 1, 1]);
    const lmd = tex(m.lightmap_diffuse), lms = tex(m.lightmap_specular);
    gl.uniform1i(U.u_lmdOn, lmd ? 1 : 0); gl.activeTexture(gl.TEXTURE1); gl.bindTexture(gl.TEXTURE_2D, lmd || white);
    gl.uniform1i(U.u_lmsOn, lms ? 1 : 0); gl.activeTexture(gl.TEXTURE2); gl.bindTexture(gl.TEXTURE_2D, lms || white);
    const sp = m.specular || {color: [0, 0, 0, 1]};
    gl.uniform1i(U.u_specFromDiff, sp.tex ? 1 : 0); gl.uniform4fv(U.u_specColor, sp.color || [0, 0, 0, 1]);
    gl.uniform4fv(U.u_colorize, (m.colorize && m.colorize.color) || [1, 1, 1, 1]);
    const em = (m.emission && m.emission.color) || [0, 0, 0, 1];
    gl.uniform3fv(U.u_emission, (m.flags & 16) ? em.slice(0, 3) : [0, 0, 0]);
  }
  function drawPrim(prim) { gl.bindVertexArray(prim.vao); gl.drawElements(gl.TRIANGLES, prim.n, prim.type, 0); }

  // ---- public: camera + frame ----
  const CAM = {fovy: 34 * Math.PI / 180, pitch: 58 * Math.PI / 180};
  let view, proj, pv, W = 1, H = 1, themeName = 'Mine';
  function camera(mode, own, mapW, mapH) {
    const asp = W / H, aimX = mapW / 2;
    const fit = (tx, ty, halfW, halfH) => {      // distance so that the box around the target fits
      const dy = Math.cos(CAM.pitch), dz = Math.sin(CAM.pitch);
      let lo = 2, hi = 200;
      for (let it = 0; it < 40; it++) {
        const d = (lo + hi) / 2, e = [tx, ty - d * dy, d * dz];
        const v = M.lookAt(e, [tx, ty, 0], [0, 0, 1]), p = M.persp(CAM.fovy, asp, 0.5, 400), m = M.mul(p, v);
        let ok = true;
        for (const [x, y] of [[tx - halfW, ty - halfH], [tx + halfW, ty - halfH], [tx - halfW, ty + halfH], [tx + halfW, ty + halfH]]) {
          const cx = m[0] * x + m[4] * y + m[12], cy = m[1] * x + m[5] * y + m[13], cw = m[3] * x + m[7] * y + m[15];
          if (Math.abs(cx / cw) > 1 || Math.abs(cy / cw) > 1) ok = false;
        }
        if (ok) hi = d; else lo = d;
      }
      return hi;
    };
    let tx, ty, d;
    if (mode === 'follow' && own) {
      tx = aimX; ty = Math.max(-mapH + 6, Math.min(-6, own[1]));
      d = fit(tx, ty, mapW / 2 + 0.5, 7);
    } else { tx = aimX; ty = -mapH / 2; d = fit(tx, ty, mapW / 2 + 0.3, mapH / 2 + 0.3); }
    const e = [tx, ty - d * Math.cos(CAM.pitch), d * Math.sin(CAM.pitch)];
    view = M.lookAt(e, [tx, ty, 0], [0, 0, 1]); proj = M.persp(CAM.fovy, asp, 0.5, 400); pv = M.mul(proj, view);
  }
  function invert(m) {          // 4x4 inverse (column-major)
    const a = m, o = new Float32Array(16);
    const b00 = a[0] * a[5] - a[1] * a[4], b01 = a[0] * a[6] - a[2] * a[4], b02 = a[0] * a[7] - a[3] * a[4], b03 = a[1] * a[6] - a[2] * a[5];
    const b04 = a[1] * a[7] - a[3] * a[5], b05 = a[2] * a[7] - a[3] * a[6], b06 = a[8] * a[13] - a[9] * a[12], b07 = a[8] * a[14] - a[10] * a[12];
    const b08 = a[8] * a[15] - a[11] * a[12], b09 = a[9] * a[14] - a[10] * a[13], b10 = a[9] * a[15] - a[11] * a[13], b11 = a[10] * a[15] - a[11] * a[14];
    const det = 1 / (b00 * b11 - b01 * b10 + b02 * b09 + b03 * b08 - b04 * b07 + b05 * b06);
    o[0] = (a[5] * b11 - a[6] * b10 + a[7] * b09) * det; o[1] = (a[2] * b10 - a[1] * b11 - a[3] * b09) * det;
    o[2] = (a[13] * b05 - a[14] * b04 + a[15] * b03) * det; o[3] = (a[10] * b04 - a[9] * b05 - a[11] * b03) * det;
    o[4] = (a[6] * b08 - a[4] * b11 - a[7] * b07) * det; o[5] = (a[0] * b11 - a[2] * b08 + a[3] * b07) * det;
    o[6] = (a[14] * b02 - a[12] * b05 - a[15] * b01) * det; o[7] = (a[8] * b05 - a[10] * b02 + a[11] * b01) * det;
    o[8] = (a[4] * b10 - a[5] * b08 + a[7] * b06) * det; o[9] = (a[1] * b08 - a[0] * b10 - a[3] * b06) * det;
    o[10] = (a[12] * b04 - a[13] * b02 + a[15] * b00) * det; o[11] = (a[9] * b02 - a[8] * b04 - a[11] * b00) * det;
    o[12] = (a[5] * b07 - a[4] * b09 - a[6] * b06) * det; o[13] = (a[0] * b09 - a[1] * b07 + a[2] * b06) * det;
    o[14] = (a[13] * b01 - a[12] * b03 - a[14] * b00) * det; o[15] = (a[8] * b03 - a[9] * b01 + a[10] * b00) * det;
    return o;
  }
  function unproject(sx, sy) {    // CSS pixels -> world point on the ground (z = 0)
    const inv = invert(pv), nx = sx / W * 2 - 1, ny = 1 - sy / H * 2;
    const pt = z => { const x = inv[0] * nx + inv[4] * ny + inv[8] * z + inv[12], y = inv[1] * nx + inv[5] * ny + inv[9] * z + inv[13],
      zz = inv[2] * nx + inv[6] * ny + inv[10] * z + inv[14], w = inv[3] * nx + inv[7] * ny + inv[11] * z + inv[15]; return [x / w, y / w, zz / w]; };
    const a = pt(-1), b = pt(1), t = a[2] / (a[2] - b[2]);
    return [a[0] + (b[0] - a[0]) * t, a[1] + (b[1] - a[1]) * t];
  }
  function project(x, y, z) {   // world -> CSS pixels
    const m = pv, cx = m[0] * x + m[4] * y + m[8] * z + m[12], cy = m[1] * x + m[5] * y + m[9] * z + m[13], cw = m[3] * x + m[7] * y + m[11] * z + m[15];
    return [(cx / cw * 0.5 + 0.5) * W, (1 - (cy / cw * 0.5 + 0.5)) * H, cw];
  }

  const insts = new Map();      // "gid:name" -> character instance (gids restart in every battle, so the brawler's name is part of the key)
  const propGL = {};
  let staticKey = null;
  async function prepare(rep, heroes) {   // heroes: names to load (default: those in the first frames); all in parallel
    if (!heroes) {
      heroes = new Set();
      for (const f of rep.frames.slice(0, 3)) for (const o of f[3]) if (o[1] === 0) heroes.add(rep.names[o[2]]);
    }
    await Promise.all([pack('core'), pack('theme_' + themeName), ...[...heroes].map(h => pack('hero_' + h).catch(() => null))]);
  }

  const DBG = {}, CHAR_SCALE = 0.125, OUTLINE = 0.028;   // model units -> tiles (x characters.csv Scale)
  function frame(st) {
    DBG.drawn = 0;
    // st: {w, h, mapW, mapH, grid, gridKey, names, chars:[{gid, name, x, y, z, team, own, facing, moving, t, hp}], mode, own}
    const cv = gl.canvas;
    W = st.w; H = st.h;
    const dpr = Math.min(window.devicePixelRatio || 1, PHONE ? 2 : 4);
    if (cv.width !== Math.round(W * dpr) || cv.height !== Math.round(H * dpr)) { cv.width = Math.round(W * dpr); cv.height = Math.round(H * dpr); }
    gl.viewport(0, 0, cv.width, cv.height);
    camera(st.mode, st.own, st.mapW, st.mapH);
    if (st.gridKey !== staticKey && S.themes[themeName]) { scene = buildStatic(themeName, st.grid, st.names, st.mapW, st.mapH); staticKey = st.gridKey; }
    gl.clearColor(0.11, 0.09, 0.12, 1); gl.clear(gl.COLOR_BUFFER_BIT | gl.DEPTH_BUFFER_BIT);
    gl.enable(gl.DEPTH_TEST); gl.disable(gl.CULL_FACE);
    gl.useProgram(prog);
    gl.uniform1i(U.u_diff, 0); gl.uniform1i(U.u_lmd, 1); gl.uniform1i(U.u_lms, 2);
    gl.uniformMatrix4fv(U.u_pv, false, pv); gl.uniformMatrix4fv(U.u_view, false, view);
    gl.uniform4fv(U.u_tint, [1, 1, 1, 0]); gl.uniform1f(U.u_alpha, 1);
    gl.uniform1f(U.u_outline, 0); gl.uniform4fv(U.u_flat, [0, 0, 0, 0]); gl.uniform4fv(U.u_fade, [0, 0, 0, 0]);
    const live = new Set(), drawList = [];
    for (const c of st.chars) {
      const ik = c.gid + ':' + c.name;
      let inst = insts.get(ik);
      if (!inst) {
        if (!S.heroes[c.name]) { pack('hero_' + c.name).then(markDirty, () => {}); DBG.missing = c.name; continue; }   // loads lazily
        inst = heroInstance(c.name); insts.set(ik, inst);
      }
      DBG.drawn++;
      live.add(ik);
      const key = inst.hero.anims[c.anim] || (c.anim === 'walk' ? null : inst.hero.anims.idle) || inst.hero.anims.idle;
      const pose = location.hash.includes('noanim') ? null : samplePose(key, c.t);
      if (!location.hash.includes('noskin')) skin(inst, pose);
      const s = CHAR_SCALE * inst.hero.scale;
      const model = M.mul(M.mul(M.tr(c.x, c.y, c.z), M.rotZ(c.facing + Math.PI / 2)), M.mul(M.sc(s), Y2Z));
      drawList.push({c, inst, s, model});
    }
    // shadow map: everything that stands on the arena, seen from the light
    const th0 = S.themes[themeName];
    if (shadowOk && scene) {
      lpv = lightMatrix(st.mapW, st.mapH);
      gl.bindFramebuffer(gl.FRAMEBUFFER, shadowFbo); gl.viewport(0, 0, SHADOW_N, SHADOW_N);
      gl.clear(gl.DEPTH_BUFFER_BIT); gl.useProgram(sprog); gl.uniformMatrix4fv(SU.u_lpv, false, lpv);
      gl.enable(gl.POLYGON_OFFSET_FILL); gl.polygonOffset(2, 4);
      gl.uniformMatrix4fv(SU.u_model, false, M.id());
      for (const g of [scene.tiles, scene.bushes]) for (const {prim} of g) drawPrim(prim);
      for (const p of st.props || []) {
        const key = th0 && th0[p.key]; if (!key || !propGL[key]) continue;
        gl.uniformMatrix4fv(SU.u_model, false, M.tr(p.x, p.y, 0)); for (const {prim} of propGL[key]) drawPrim(prim);
      }
      for (const {inst, model} of drawList) {
        gl.uniformMatrix4fv(SU.u_model, false, model);
        for (const mm of inst.meshes) for (const pr of mm.prims) drawPrim(pr.gl);
      }
      gl.disable(gl.POLYGON_OFFSET_FILL);
      gl.bindFramebuffer(gl.FRAMEBUFFER, null); gl.viewport(0, 0, cv.width, cv.height);
      gl.useProgram(prog);
    }
    gl.uniformMatrix4fv(U.u_lpv, false, lpv); gl.uniform1i(U.u_shadow, 3); gl.uniform1f(U.u_texel, 1 / SHADOW_N);
    gl.activeTexture(gl.TEXTURE3); gl.bindTexture(gl.TEXTURE_2D, shadowOk ? shadowTex : null);
    gl.uniform1f(U.u_shadowOn, shadowOk && scene ? 1 : 0);
    gl.uniformMatrix4fv(U.u_model, false, M.id());
    if (scene) {
      gl.disable(gl.CULL_FACE);
      for (const {prim, mat} of scene.env) { setMaterial(mat); drawPrim(prim); }
      for (const {prim, mat} of scene.tiles) { setMaterial(mat); drawPrim(prim); }
    }
    const th = S.themes[themeName];
    for (const p of st.props || []) {
      const key = th && th[p.key]; if (!key || !S.models[key]) continue;
      if (!propGL[key]) { const g = {}; bakeInto(g, S.models[key], null, M.id()); propGL[key] = finishGroups(g); }
      gl.uniformMatrix4fv(U.u_model, false, M.tr(p.x, p.y, 0));
      for (const {prim, mat} of propGL[key]) { setMaterial(mat); drawPrim(prim); }
    }
    if (st.rings && st.rings.length) {
      gl.useProgram(dprog); gl.uniformMatrix4fv(DU.u_pv, false, pv);
      gl.enable(gl.BLEND); gl.blendFunc(gl.SRC_ALPHA, gl.ONE_MINUS_SRC_ALPHA); gl.depthMask(false);
      gl.bindVertexArray(quad);
      for (const r of st.rings) {
        gl.uniform3fv(DU.u_center, [r.x, r.y, 0.03]); gl.uniform1f(DU.u_r, r.r * CHAR_SCALE / 0.1); gl.uniform3fv(DU.u_col, r.col);
        gl.drawArrays(gl.TRIANGLE_STRIP, 0, 4);
      }
      gl.depthMask(true); gl.disable(gl.BLEND);
      gl.useProgram(prog);
    }
    for (const {c, inst, s, model} of drawList) {
      gl.uniformMatrix4fv(U.u_model, false, model);
      gl.uniform1f(U.u_alpha, 1);
      gl.uniform4fv(U.u_tint, [1, 1, 1, 0]);
      const texKey = inst.hero.tex[c.blue ? 'blue' : 'red'] || inst.hero.tex.blue || inst.hero.tex.red;
      gl.enable(gl.CULL_FACE);
      gl.cullFace(gl.FRONT); gl.uniform1f(U.u_outline, OUTLINE / s); gl.uniform4fv(U.u_flat, [0.06, 0.04, 0.09, 1]);
      for (const mm of inst.meshes) for (const pr of mm.prims) drawPrim(pr.gl);
      gl.cullFace(gl.BACK); gl.uniform1f(U.u_outline, 0); gl.uniform4fv(U.u_flat, [0, 0, 0, 0]);
      for (const mm of inst.meshes) for (const pr of mm.prims) { setMaterial(pr.p.material, texKey); drawPrim(pr.gl); }
      gl.disable(gl.CULL_FACE);
    }
    if (scene && scene.bushes.length) {     // bushes last, see-through around the own brawler (as in the game)
      gl.uniformMatrix4fv(U.u_model, false, M.id());
      gl.enable(gl.BLEND); gl.blendFunc(gl.SRC_ALPHA, gl.ONE_MINUS_SRC_ALPHA);
      gl.uniform4fv(U.u_fade, st.own ? [st.own[0], st.own[1], 1.2, 1] : [0, 0, 0, 0]);
      for (const {prim, mat} of scene.bushes) { setMaterial(mat); drawPrim(prim); }
      gl.uniform4fv(U.u_fade, [0, 0, 0, 0]); gl.disable(gl.BLEND);
    }
    for (const k of [...insts.keys()]) if (!live.has(k) && insts.size > 40) insts.delete(k);
    gl.bindVertexArray(null);
    dirty = false;
  }
  return Object.assign(API, {DBG, unproject, CHAR_SCALE: () => CHAR_SCALE, heroScale: n => (S.heroes[n] || {scale: 1}).scale, init, prepare, frame, project, pack, S, isDirty: () => dirty, setTheme: t => { themeName = t; }});
})();
