"""Reader for the glTF binary (.glb) files of the newer brawlers, converted to the viewer-pack structures of build.py.

    model_entry_glb(pack, path)         -> same dict as build.model_entry: {nodes, meshes, fps}
    anim_entry_glb(path, start, end)    -> same dict as build.anim_entry:  {fps, frames, tracks}
    ../../.venv/bin/python tools/gfx/glb.py <files.glb>    summary of what a file holds

What the files are (all 332 .glb of v36's assets/sc3d were checked): GLB 2.0 = JSON chunk + BIN chunk, written by
Supercell's gltfpack variant "scgltfpack 0.15". Required extension KHR_mesh_quantization, plus the material extension SC_shader.
  geometry  POSITION uint16 x3 (not normalised), NORMAL int8 x3 (normalised), TEXCOORD_0 uint16 x2 = 12 bit (0..4095, not
            normalised, and no KHR_texture_transform is written, so u = q / 4095), JOINTS_0 uint16 x4, WEIGHTS_0 uint8 x4
            (normalised, sum exactly 1), uint16 indices. The mesh node's translation + uniform scale (~0.0015) dequantise the
            positions; glTF ignores that node for skinned meshes, so scgltfpack folds it into the inverse bind matrices. Here it
            is moved back (positions in model units, rigid inverse bind matrices), like the .scw geometry. Files with
            "primitives": [null] (animation containers, intro poses, a few stub *_geo) hold no geometry.
  rig       Maya style: every joint "X" hangs below "X:SSC" (segment scale compensate), which carries the joint's translation
            and sometimes a compensating scale. A .scw rig has one node "X" with all of it, and animations are applied to nodes
            by name (Max: max_geo.scw + max_hermes_*.glb), so the SSC translation is moved into "X"  (X.t = (R S)^-1 t_ssc + t_x,
            exactly equal to the original product); "X:SSC" stays as a scale/rotation-only node, so squash-and-stretch stays exact.
  animation one clip, LINEAR keys at every frame of 30 fps from t = 0, rotation int16-normalised or float.
  materials SC_shader = the .scw MATE of the game's uber shader (constants, blendMode, variables), images are uris "sc3d/<name>.pvr"
            (-> <name>.ktx in sc3d; the hero textures are named by skins.csv) and "sc3d/menu_*_lightmap.png".
  states    surge_geo.glb holds the meshes of all 5 transformation states, named ..._state_<n>_<m>; only state 0 is kept.
Conventions as the .scw characters: Y up, facing +Z, same units. Matrices are 4x4 column-major (WebGL order).
"""
import base64
import json
import math
import pathlib
import re
import struct
import sys
import urllib.parse

import numpy as np

FPS = 30                 # glTF has no frame rate; the animation files are sampled every 1/30 s
BASE_STATE = 0           # surge-style models: keep only meshes tagged for this transformation state (or untagged)
BATTLE_LIGHTMAPS = True  # sc3d/menu_X_lightmap.png (lobby lighting) -> sc3d/X_lightmap.png, the match lightmaps the .scw heroes use
UV_FLIP_V = False        # set after checking the texture row order (see model_entry_glb)

_CTYPE = {5120: np.dtype("i1"), 5121: np.dtype("u1"), 5122: np.dtype("<i2"), 5123: np.dtype("<u2"),
          5125: np.dtype("<u4"), 5126: np.dtype("<f4")}
_NCOMP = {"SCALAR": 1, "VEC2": 2, "VEC3": 3, "VEC4": 4, "MAT2": 4, "MAT3": 9, "MAT4": 16}
_INTMAX = {5120: 127.0, 5121: 255.0, 5122: 32767.0, 5123: 65535.0}


def _build():
    """build.py, whichever way it is loaded (run as a script it is __main__; importing it again would split its caches)"""
    for name in ("build", "__main__"):
        m = sys.modules.get(name)
        if m is not None and hasattr(m, "tex_entry"):
            return m
    sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
    import build
    return build


# ---- small maths ----
def quat_to_mat(q):
    """quaternions [..., 4] (x, y, z, w; normalised here) -> rotation matrices [..., 3, 3]"""
    q = np.asarray(q, np.float64)
    q = q / np.maximum(np.linalg.norm(q, axis=-1, keepdims=True), 1e-12)
    x, y, z, w = q[..., 0], q[..., 1], q[..., 2], q[..., 3]
    R = np.empty(q.shape[:-1] + (3, 3))
    R[..., 0, 0] = 1 - 2 * (y * y + z * z); R[..., 0, 1] = 2 * (x * y - z * w); R[..., 0, 2] = 2 * (x * z + y * w)
    R[..., 1, 0] = 2 * (x * y + z * w); R[..., 1, 1] = 1 - 2 * (x * x + z * z); R[..., 1, 2] = 2 * (y * z - x * w)
    R[..., 2, 0] = 2 * (x * z - y * w); R[..., 2, 1] = 2 * (y * z + x * w); R[..., 2, 2] = 1 - 2 * (x * x + y * y)
    return R


def mat_to_quat(R):
    R = np.asarray(R, np.float64)
    tr = R[0, 0] + R[1, 1] + R[2, 2]
    if tr > 0:
        s = 2 * math.sqrt(tr + 1)
        q = [(R[2, 1] - R[1, 2]) / s, (R[0, 2] - R[2, 0]) / s, (R[1, 0] - R[0, 1]) / s, s / 4]
    elif R[0, 0] > R[1, 1] and R[0, 0] > R[2, 2]:
        s = 2 * math.sqrt(1 + R[0, 0] - R[1, 1] - R[2, 2])
        q = [s / 4, (R[0, 1] + R[1, 0]) / s, (R[0, 2] + R[2, 0]) / s, (R[2, 1] - R[1, 2]) / s]
    elif R[1, 1] > R[2, 2]:
        s = 2 * math.sqrt(1 + R[1, 1] - R[0, 0] - R[2, 2])
        q = [(R[0, 1] + R[1, 0]) / s, s / 4, (R[1, 2] + R[2, 1]) / s, (R[0, 2] - R[2, 0]) / s]
    else:
        s = 2 * math.sqrt(1 + R[2, 2] - R[0, 0] - R[1, 1])
        q = [(R[0, 2] + R[2, 0]) / s, (R[1, 2] + R[2, 1]) / s, s / 4, (R[1, 0] - R[0, 1]) / s]
    q = np.array(q)
    return q / np.linalg.norm(q)


def trs_mat(t, q, s):
    """translation, quaternion (x, y, z, w), scale -> 4x4 matrix (column vectors: M @ [x, y, z, 1])"""
    M = np.eye(4)
    M[:3, :3] = quat_to_mat(q) * np.asarray(s, np.float64)[None, :]
    M[:3, 3] = t
    return M


def decompose(M):
    """4x4 matrix (no shear) -> translation, quaternion, scale"""
    M = np.asarray(M, np.float64)
    c = M[:3, :3]
    s = np.linalg.norm(c, axis=0)
    if np.linalg.det(c) < 0:
        s[0] = -s[0]
    return M[:3, 3].tolist(), mat_to_quat(c / np.where(s == 0, 1, s)[None, :]).tolist(), s.tolist()


def slerp(q0, q1, u):
    """quaternion arrays [F, 4] (u [F]); shortest path; falls back to normalised lerp for nearly equal keys"""
    d = (q0 * q1).sum(-1)
    q1 = np.where((d < 0)[:, None], -q1, q1)
    d = np.abs(d)
    th = np.arccos(np.clip(d, -1, 1))
    sn = np.sin(th)
    ok = sn > 1e-6
    a = np.where(ok, np.sin((1 - u) * th) / np.where(ok, sn, 1), 1 - u)
    b = np.where(ok, np.sin(u * th) / np.where(ok, sn, 1), u)
    q = q0 * a[:, None] + q1 * b[:, None]
    return q / np.maximum(np.linalg.norm(q, axis=-1, keepdims=True), 1e-12)


# ---- the container ----
class Glb:
    def __init__(self, path):
        self.path = pathlib.Path(path)
        d = self.path.read_bytes()
        if d[:4] != b"glTF":
            raise ValueError("not a GLB file")
        version, length = struct.unpack_from("<II", d, 4)
        if version != 2:
            raise ValueError(f"unsupported glTF version {version}")
        o, self.j, bin_ = 12, None, None
        end = min(len(d), length) if length else len(d)
        while o + 8 <= end:
            n, kind = struct.unpack_from("<I4s", d, o)
            chunk = d[o + 8:o + 8 + n]
            if kind == b"JSON":
                self.j = json.loads(chunk.decode("utf-8"))
            elif kind == b"BIN\x00" and bin_ is None:
                bin_ = chunk
            o += 8 + n
        if self.j is None:
            raise ValueError("GLB without a JSON chunk")
        self.bufs = []
        for b in self.j.get("buffers", []):
            uri = b.get("uri")
            if uri is None:
                self.bufs.append(bin_ or b"")
            elif uri.startswith("data:"):
                self.bufs.append(base64.b64decode(uri.split(",", 1)[1]))
            else:
                self.bufs.append((self.path.parent / urllib.parse.unquote(uri)).read_bytes())
        self.nodes_j = self.j.get("nodes", [])

    # -- accessors --
    def view_bytes(self, vi):
        v = self.j["bufferViews"][vi]
        o = v.get("byteOffset", 0)
        return self.bufs[v.get("buffer", 0)][o:o + v["byteLength"]]

    def accessor(self, i):
        """accessor i -> array [count, components]: float32 for floats and normalised integers (dequantised), else the integers"""
        a = self.j["accessors"][i]
        ct = a["componentType"]
        dt, nc, n = _CTYPE[ct], _NCOMP[a["type"]], a["count"]
        if "bufferView" in a:
            v = self.j["bufferViews"][a["bufferView"]]
            buf = self.bufs[v.get("buffer", 0)]
            off = v.get("byteOffset", 0) + a.get("byteOffset", 0)
            stride = v.get("byteStride") or dt.itemsize * nc
            if n and off + stride * (n - 1) + dt.itemsize * nc > len(buf):
                raise ValueError(f"accessor {i} runs past its buffer")
            arr = np.ndarray((n, nc), dt, buf, off, (stride, dt.itemsize)).copy()
        else:
            arr = np.zeros((n, nc), dt)
        sp = a.get("sparse")
        if sp:
            ii, vv = sp["indices"], sp["values"]
            iv, vw = self.j["bufferViews"][ii["bufferView"]], self.j["bufferViews"][vv["bufferView"]]
            idx = np.frombuffer(self.bufs[iv.get("buffer", 0)], _CTYPE[ii["componentType"]], sp["count"],
                                iv.get("byteOffset", 0) + ii.get("byteOffset", 0))
            vals = np.frombuffer(self.bufs[vw.get("buffer", 0)], dt, sp["count"] * nc,
                                 vw.get("byteOffset", 0) + vv.get("byteOffset", 0)).reshape(-1, nc)
            arr[idx.astype(np.int64)] = vals
        if a.get("normalized") and ct in _INTMAX:
            arr = arr.astype(np.float32) / np.float32(_INTMAX[ct])
            if dt.kind == "i":
                arr = np.maximum(arr, -1.0)
        elif dt.kind == "f":
            arr = arr.astype(np.float32)
        return arr

    # -- nodes --
    def names(self):
        out, seen = [], set()
        for i, n in enumerate(self.nodes_j):
            nm = n.get("name") or f"node{i}"
            if nm in seen:
                nm = f"{nm}#{i}"
            seen.add(nm)
            out.append(nm)
        return out

    def parents(self):
        p = [-1] * len(self.nodes_j)
        for i, n in enumerate(self.nodes_j):
            for c in n.get("children") or []:
                p[c] = i
        return p

    def rest(self):
        """local rest pose of every node: (t [N, 3], r [N, 4], s [N, 3]); matrix nodes are decomposed"""
        N = len(self.nodes_j)
        t, r, s = np.zeros((N, 3)), np.tile([0.0, 0, 0, 1], (N, 1)), np.ones((N, 3))
        for i, n in enumerate(self.nodes_j):
            if "matrix" in n:
                t[i], r[i], s[i] = decompose(np.array(n["matrix"], np.float64).reshape(4, 4).T)
            else:
                t[i] = n.get("translation", [0, 0, 0])
                r[i] = n.get("rotation", [0, 0, 0, 1])
                s[i] = n.get("scale", [1, 1, 1])
        return t, r, s

    def ssc_pairs(self):
        """{index of 'X:SSC': index of 'X'} for the Maya segment-scale-compensate pairs (the SSC node has 'X' as its only child)"""
        out = {}
        for i, n in enumerate(self.nodes_j):
            nm, ch = n.get("name") or "", n.get("children") or []
            if nm.endswith(":SSC") and len(ch) == 1 and (self.nodes_j[ch[0]].get("name") or "") == nm[:-4]:
                out[i] = ch[0]
        return out

    # -- images --
    def image_uri(self, tex_index):
        t = self.j["textures"][tex_index]
        im = self.j["images"][t["source"]]
        return im.get("uri")


def _state_ok(*names):
    for nm in names:
        m = re.search(r"_state_(\d+(?:_\d+)*)", nm or "")
        if m:
            return BASE_STATE in {int(x) for x in m.group(1).split("_")}
    return True


# ---- materials ----
def _lightmap_ref(uri):
    ref = uri if uri.startswith("sc3d/") else "sc3d/" + pathlib.PurePosixPath(uri).name
    stem = pathlib.PurePosixPath(ref).stem
    if BATTLE_LIGHTMAPS and stem.startswith("menu_"):
        A = _build().A
        alt = stem[5:]
        if (A / "sc3d" / f"{alt}.ktx").exists() or (A / "sc3d" / f"{alt}.png").exists():
            ref = f"sc3d/{alt}{pathlib.PurePosixPath(ref).suffix}"
    return ref


def _sc_material(g, pack, m, ext):
    """SC_shader material -> the dict build.material_entry makes from a .scw MATE (only the parts the viewer reads matter:
    diffuse/specular/colorize/emission, lightmaps, flags & 16). The hero's texture is '.', filled in from skins.csv."""
    bd = _build()
    v, consts = ext.get("variables", {}), set(ext.get("constants", []))

    def slot(name):                                  # texture image uri of a *Tex2D variable, or None for a constant colour
        x = v.get(name)
        return g.image_uri(x["index"]) if isinstance(x, dict) and "index" in x else None

    def tc(const, tex_var, col_var, default):
        if const not in consts:
            return {"color": list(default)}
        if slot(tex_var) is not None:
            return {"tex": "."}                      # character texture (the hero's own comes from skins.csv)
        return {"color": list(v.get(col_var, default))}

    diffuse = tc("DIFFUSE", "diffuseTex2D", "diffuse", [1, 1, 1, 1])
    specular = tc("SPECULAR", "specularTex2D", "specular", [0, 0, 0, 1])
    colorize = tc("COLORIZE", "colorizeTex2D", "colorize", [1, 1, 1, 1])
    emission = tc("EMISSION", "emissionTex2D", "emission", [0, 0, 0, 1])
    if "EMISSION" in consts and "tex" in emission and "DIFFUSE" not in consts:
        diffuse = {"tex": "."}                       # glowing part: shown as the unlit character texture
    opacity = float(v.get("opacity", 1.0)) if "OPACITY" in consts else 1.0
    lmd = lms = None
    if "LIGHTMAP" in consts:
        if slot("lightmapTex2D"):
            lmd = bd.tex_entry(pack, _lightmap_ref(slot("lightmapTex2D")))
        if "SPECULAR" in consts and slot("lightmapSpecularTex2D"):
            lms = bd.tex_entry(pack, _lightmap_ref(slot("lightmapSpecularTex2D")))
    flags = 0x800
    for c, bit in (("DIFFUSE", 0x2), ("STENCIL", 0x4), ("COLORIZE", 0x8), ("EMISSION", 0x10)):
        if c in consts:
            flags |= bit
    if "color" not in emission:
        flags &= ~0x10                               # an emission texture is drawn as the diffuse, not added as a colour
    if opacity != 1.0:
        flags |= 0x20
    if lmd or lms:
        flags |= 0x40 | (0x80 if lms else 0)
    return {"name": ext.get("name") or m.get("name") or "", "flags": flags,
            "ambient": list(v.get("ambient", [0, 0, 0, 1])) if "AMBIENT" in consts else [0.0, 0.0, 0.0, 1.0],
            "diffuse": diffuse, "specular": specular, "colorize": colorize, "emission": emission,
            "opacity": opacity, "stencil": "." if "STENCIL" in consts else "",
            "lightmap_diffuse": lmd, "lightmap_specular": lms}


def _pbr_material(g, pack, m):
    """plain glTF material (not used by the game's own files, kept for completeness)"""
    bd = _build()
    pbr = m.get("pbrMetallicRoughness", {})
    col = list(pbr.get("baseColorFactor", [1, 1, 1, 1]))
    bt = pbr.get("baseColorTexture")
    diffuse = {"color": col}
    if bt is not None:
        uri = g.image_uri(bt["index"])
        if uri:
            diffuse = {"tex": bd.tex_entry(pack, uri) or "."}
    em = list(m.get("emissiveFactor", [0, 0, 0])) + [1.0]
    return {"name": m.get("name") or "", "flags": 0x802 | (0x10 if any(em[:3]) else 0), "ambient": [0.0, 0.0, 0.0, 1.0],
            "diffuse": diffuse, "specular": {"color": [0, 0, 0, 1]}, "colorize": {"color": [1, 1, 1, 1]},
            "emission": {"color": em}, "opacity": float(col[3]), "stencil": "",
            "lightmap_diffuse": None, "lightmap_specular": None}


def _material_keys(g, pack, path):
    """material index -> key in pack['materials'] (created on first use; keys 'sc3d/x.glb:name' like the .scw ones)"""
    mats = g.j.get("materials", [])
    names = [((m.get("extensions") or {}).get("SC_shader") or {}).get("name") or m.get("name") or f"material{i}"
             for i, m in enumerate(mats)]
    keys = [f"{path}:{nm}" if names.count(nm) == 1 else f"{path}:{nm}#{i}" for i, nm in enumerate(names)]
    return keys, lambda i: _material_entry(g, pack, mats[i], keys[i])


def _material_entry(g, pack, m, key):
    if key not in pack["materials"]:
        ext = (m.get("extensions") or {}).get("SC_shader")
        pack["materials"][key] = _sc_material(g, pack, m, ext) if ext else _pbr_material(g, pack, m)
    return key


# ---- models ----
def _tri_indices(idx, mode):
    if mode == 4:
        return idx[:len(idx) // 3 * 3]
    if mode == 5:                                    # strip
        i = np.arange(len(idx) - 2)
        a = np.where(i % 2 == 0, idx[i], idx[i + 1])
        b = np.where(i % 2 == 0, idx[i + 1], idx[i])
        return np.stack([a, b, idx[i + 2]], 1).ravel()
    if mode == 6:                                    # fan
        i = np.arange(1, len(idx) - 1)
        return np.stack([np.full_like(i, idx[0]), idx[i], idx[i + 1]], 1).ravel()
    return None


def _skin_weights(g, at, nj):
    """JOINTS_0/1 + WEIGHTS_0/1 -> the 4 strongest influences per vertex: uint8 joints [n, 4], float32 weights [n, 4] summing to 1"""
    jj = np.concatenate([g.accessor(at[f"JOINTS_{k}"]).astype(np.int64) for k in range(2) if f"JOINTS_{k}" in at], 1)
    ww = np.concatenate([g.accessor(at[f"WEIGHTS_{k}"]).astype(np.float64) for k in range(2) if f"WEIGHTS_{k}" in at], 1)
    if jj.shape[1] > 4:
        o = np.argsort(-ww, axis=1, kind="stable")[:, :4]
        jj, ww = np.take_along_axis(jj, o, 1), np.take_along_axis(ww, o, 1)
    s = ww.sum(1)
    zero = s <= 1e-9
    ww[zero] = [1, 0, 0, 0]                          # no influence at all (invalid in glTF): follow the first joint
    jj[zero] = 0
    ww /= ww.sum(1, keepdims=True)
    if jj.max(initial=0) >= min(nj, 256):
        raise ValueError("joint index out of range")
    return jj.astype(np.uint8), ww.astype(np.float32)


def model_entry_glb(pack, path):
    """GLB model -> {nodes, meshes, fps} as build.model_entry (meshes keep their vertex arrays; skinned ones are baked in model
    units with rigid inverse bind matrices). path is relative to apk/extracted/assets like the .scw ones."""
    bd = _build()
    b64 = bd.b64
    g = Glb(bd.A / path)
    j = g.j
    N = len(g.nodes_j)
    names, parent = g.names(), g.parents()
    rt, rr, rs = g.rest()
    for ssc, c in g.ssc_pairs().items():             # translation of the compensate node -> the joint node (see module doc)
        A = quat_to_mat(rr[ssc]) * rs[ssc][None, :]
        if abs(np.linalg.det(A)) > 1e-12:
            rt[c] = np.linalg.solve(A, rt[ssc]) + rt[c]
            rt[ssc] = 0
    # parents first (the viewer builds world matrices in array order); glTF files from the game already are
    order, seen = [], set()
    for root in [i for i in range(N) if parent[i] < 0]:
        stack = [root]
        while stack:
            i = stack.pop()
            if i in seen:
                continue
            seen.add(i)
            order.append(i)
            stack.extend(reversed(g.nodes_j[i].get("children") or []))
    order += [i for i in range(N) if i not in seen]
    new = {old: k for k, old in enumerate(order)} if order != list(range(N)) else {i: i for i in range(N)}
    nodes = [{"name": names[i], "parent": new[parent[i]] if parent[i] >= 0 else -1,
              "t": rt[i].tolist(), "r": rr[i].tolist(), "s": rs[i].tolist()} for i in order]

    mat_keys, mat_entry = _material_keys(g, pack, path)
    meshes, skipped = [], 0
    for ni in range(N):
        n = g.nodes_j[ni]
        if "mesh" not in n:
            continue
        mesh = j["meshes"][n["mesh"]]
        prims = [p for p in (mesh.get("primitives") or []) if p]
        if not prims:
            continue
        if not _state_ok(n.get("name"), mesh.get("name")):
            skipped += 1
            continue
        skin = j["skins"][n["skin"]] if "skin" in n else None
        quant = _CTYPE[j["accessors"][prims[0]["attributes"]["POSITION"]]["componentType"]].kind != "f"
        D = trs_mat(*[a[ni] for a in (rt, rr, rs)]) if quant else np.eye(4)   # dequantisation of this mesh node
        if "matrix" in n and quant:
            D = np.array(n["matrix"], np.float64).reshape(4, 4).T
        if skin is None:                              # static mesh: raw positions, the node transform dequantises
            D3, Dinv = np.eye(3), np.eye(4)
            D = np.eye(4)
        else:
            D3, Dinv = D[:3, :3], np.linalg.inv(D)
        out = {"node": new[ni], "name": mesh.get("name") or n.get("name") or "", "prims": []}
        nj = 0
        if skin is not None:
            joints = skin["joints"]
            nj = len(joints)
            ibm = (g.accessor(skin["inverseBindMatrices"]).astype(np.float64) if "inverseBindMatrices" in skin
                   else np.tile(np.eye(4).T.ravel(), (nj, 1)))
            inv = np.stack([(m.reshape(4, 4).T @ Dinv).T.astype(np.float32).ravel() for m in ibm])   # column-major, rigid
            out["joints"] = [new[x] for x in joints]
            out["invBind"] = b64(inv)
            out["bindShape"] = None
        for p in prims:
            at = p["attributes"]
            if "POSITION" not in at:
                continue
            tri = _tri_indices(g.accessor(p["indices"]).ravel().astype(np.int64) if "indices" in p
                               else np.arange(j["accessors"][at["POSITION"]]["count"]), p.get("mode", 4))
            if tri is None:
                print(f"  {path}: primitive mode {p.get('mode')} skipped")
                continue
            pos = g.accessor(at["POSITION"]).astype(np.float64)
            nv = len(pos)
            if tri.size and tri.max() >= nv:
                raise ValueError(f"{path}: index out of range")
            pos = pos @ D[:3, :3].T + D[:3, 3]
            e = {"material": mat_entry(p["material"]) if "material" in p else None, "count": int(nv),
                 "index": b64(tri.astype(np.uint16 if nv < 65536 else np.uint32)), "i32": nv >= 65536,
                 "pos": b64(pos.astype(np.float32))}
            if "NORMAL" in at:
                nrm = g.accessor(at["NORMAL"]).astype(np.float64)
                nrm = nrm @ np.linalg.inv(D3)             # inverse transpose of D3, applied to row vectors
                e["nrm"] = b64((nrm / np.maximum(np.linalg.norm(nrm, axis=1, keepdims=True), 1e-12)).astype(np.float32))
            if "TEXCOORD_0" in at:
                uv = g.accessor(at["TEXCOORD_0"])
                if uv.dtype.kind in "ui":
                    uv = uv.astype(np.float32) / np.float32(4095.0 if uv.max(initial=0) <= 4095 else 65535.0)
                uv = uv.astype(np.float32)
                if UV_FLIP_V:
                    uv[:, 1] = 1 - uv[:, 1]
                e["uv"] = b64(uv)
            if "COLOR_0" in at:
                col = g.accessor(at["COLOR_0"]).astype(np.float32)
                if col.shape[1] == 3:
                    col = np.concatenate([col, np.ones((nv, 1), np.float32)], 1)
                e["col"] = b64(col)
            if skin is not None and "JOINTS_0" in at and "WEIGHTS_0" in at:
                jj, ww = _skin_weights(g, at, nj)
                e["j"], e["w"] = b64(jj), b64(ww)
            out["prims"].append(e)
        if out["prims"]:
            meshes.append(out)
    if skipped:
        print(f"  {path}: {skipped} meshes of other transformation states left out")
    return {"nodes": nodes, "meshes": meshes, "fps": FPS}


# ---- animations ----
def _frame_arg(v):
    """start/end argument: None, '' or a negative number (the csv's -1, as int or string) = open"""
    if v is None or v == "":
        return None
    v = int(float(v))
    return None if v < 0 else v


def _sample(kf, vals, interp, fr, rotation):
    """keyframes at frame positions kf [K] with values [K, n] ([3K, n] for CUBICSPLINE) -> values [F, n] at frames fr [F]"""
    K = len(kf)
    if interp == "CUBICSPLINE":
        tin, v, tout = vals[0::3], vals[1::3], vals[2::3]
    else:
        v = vals
    k1 = np.searchsorted(kf, fr, side="right")
    k0 = np.clip(k1 - 1, 0, K - 1)
    k1 = np.clip(k1, 0, K - 1)
    dt = kf[k1] - kf[k0]
    u = np.where(dt > 0, (fr - kf[k0]) / np.where(dt > 0, dt, 1), 0.0)
    if interp == "STEP":
        return v[k0].astype(np.float64)
    if interp == "CUBICSPLINE":
        h = dt[:, None] / FPS
        u = u[:, None]
        u2, u3 = u * u, u * u * u
        out = ((2 * u3 - 3 * u2 + 1) * v[k0] + (u3 - 2 * u2 + u) * h * tout[k0]
               + (-2 * u3 + 3 * u2) * v[k1] + (u3 - u2) * h * tin[k1])
        return out / np.maximum(np.linalg.norm(out, axis=-1, keepdims=True), 1e-12) if rotation else out
    if rotation:
        return slerp(v[k0].astype(np.float64), v[k1].astype(np.float64), u)
    return v[k0] * (1 - u)[:, None] + v[k1] * u[:, None]


def anim_entry_glb(path, start=None, end=None):
    """GLB animation -> {fps, frames, tracks} as build.anim_entry: per animated node the t (3), r (4), s (3) of every frame of
    [start, end] (file frames at 30 fps from 0; None, '' or -1 = open end), float32 base64. Joint nodes carry their SSC node's
    translation (see module doc), so the tracks fit the .scw rigs too (Max)."""
    bd = _build()
    b64 = bd.b64
    g = Glb(bd.A / path)
    chans = []                                       # (node, 'translation' | 'rotation' | 'scale', frame positions, values, interpolation)
    for an in g.j.get("animations") or []:
        for c in an["channels"]:
            tg = c["target"]
            if tg.get("node") is None or tg.get("path") not in ("translation", "rotation", "scale"):
                continue
            s = an["samplers"][c["sampler"]]
            kf = g.accessor(s["input"]).ravel().astype(np.float64) * FPS
            kf = np.where(np.abs(kf - np.round(kf)) < 1e-3, np.round(kf), kf)        # keys sit on whole frames
            chans.append((tg["node"], tg["path"], kf, g.accessor(s["output"]), s.get("interpolation", "LINEAR")))
    if not chans:
        return None
    lo = int(round(min(c[2][0] for c in chans)))
    hi = int(round(max(c[2][-1] for c in chans)))
    s0, e0 = _frame_arg(start), _frame_arg(end)
    a = lo if s0 is None else max(lo, s0)
    b = hi if e0 is None else min(hi, e0)
    if b < a:
        a, b = lo, hi
    fr = np.arange(a, b + 1, dtype=np.float64)
    F = len(fr)

    got = {}                                         # (node, path) -> [F, n]
    for node, p, kf, vals, interp in chans:
        got[(node, p)] = _sample(kf, vals, interp, fr, p == "rotation")
    names = g.names()
    rt, rr, rs = g.rest()
    animated = {n for n, _ in got}
    ident = {"translation": np.zeros(3), "rotation": np.array([0.0, 0, 0, 1]), "scale": np.ones(3)}

    def full(i):
        t = got.get((i, "translation"))
        r = got.get((i, "rotation"))
        s = got.get((i, "scale"))
        return (t if t is not None else np.tile(rt[i], (F, 1)), r if r is not None else np.tile(rr[i], (F, 1)),
                s if s is not None else np.tile(rs[i], (F, 1)))

    def track(t, r, s):
        return {"t": b64(np.asarray(t, np.float32)), "r": b64(np.asarray(r, np.float32)), "s": b64(np.asarray(s, np.float32))}

    pairs = g.ssc_pairs()
    child_of = set(pairs.values())
    tracks = {}
    for i in range(len(names)):
        if i in pairs:                               # X:SSC + X -> X gets the translation, X:SSC keeps rotation/scale if any
            c = pairs[i]
            if i not in animated and c not in animated:
                continue
            tN, rN, sN = full(i)
            tC, rC, sC = full(c)
            A = quat_to_mat(rN) * sN[:, None, :]     # R diag(s) per frame
            if np.abs(np.linalg.det(A)).min() > 1e-12:
                tC = np.linalg.solve(A, tN[:, :, None])[:, :, 0] + tC
            else:
                tC = tN + tC
            tracks[names[c]] = track(tC, rC, sC)
            if (i, "rotation") in got or (i, "scale") in got or np.abs(rN - ident["rotation"]).max() > 1e-6 \
                    or np.abs(sN - 1).max() > 1e-6:
                tracks[names[i]] = track(np.zeros((F, 3)), rN, sN)
        elif i not in child_of and i in animated:
            tracks[names[i]] = track(*full(i))
    return {"fps": FPS, "frames": int(b - a + 1), "tracks": tracks}


# ---- summary ----
def _summary(path):
    import io
    bd = _build()
    p = pathlib.Path(path)
    rel = str(p.resolve().relative_to(bd.A)) if bd.A in p.resolve().parents else str(p)
    g = Glb(p)
    print(f"{rel}: {len(g.nodes_j)} nodes, {len(g.j.get('meshes', []))} meshes, {len(g.j.get('skins', []))} skins, "
          f"{len(g.j.get('materials', []))} materials, {len(g.j.get('animations', []))} animations, "
          f"extensions {g.j.get('extensionsUsed', [])}")
    pack = bd.new_pack()
    if g.j.get("materials") or any(m.get("primitives") and any(m["primitives"]) for m in g.j.get("meshes", [])):
        m = model_entry_glb(pack, rel)
        npos = 0
        lo, hi = np.full(3, 1e9), np.full(3, -1e9)
        for me in m["meshes"]:
            for pr in me["prims"]:
                npos += pr["count"]
        print(f"  model: {len(m['nodes'])} nodes, {len(m['meshes'])} meshes, {npos} vertices, {len(pack['materials'])} materials")
    if g.j.get("animations"):
        e = anim_entry_glb(rel)
        if e:
            print(f"  anim: {e['frames']} frames at {e['fps']} fps, {len(e['tracks'])} tracks")


if __name__ == "__main__":
    for f in sys.argv[1:]:
        try:
            _summary(f)
        except Exception as ex:  # noqa: BLE001  (report every file)
            print(f"{f}: {type(ex).__name__}: {ex}")
