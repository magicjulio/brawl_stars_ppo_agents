#!/usr/bin/env python3
"""Build the viewer's 3D asset packs from the extracted APK (apk/extracted/assets), with our own parsers only.

    ../../.venv/bin/python tools/gfx/build.py            # -> tools/viewer/gfx/{core,theme_Mine,hero_<Name>}.js

Each pack is a JS file calling BSGFX(<json>) so the viewer can load it with a <script> tag (works from file:// and
http://). Arrays are base64 little-endian; textures are PNG data URLs (ETC1 .ktx decoded, team hue masks applied).
Model conventions (as in the files): tile/environment models are Z-up in tile units, characters are Y-up and face +Z
(the viewer turns them upright and scales them). Matrices are 4x4 column-major (WebGL order).
"""
import base64
import colorsys
import csv
import io
import json
import pathlib
import re
import sys

import numpy as np
from PIL import Image

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
import ktx  # noqa: E402
import scw  # noqa: E402

ROOT = pathlib.Path(__file__).resolve().parents[2]
A = ROOT / "apk/extracted/assets"
OUT = ROOT / "tools/viewer/gfx"


def table(path):
    rows = list(csv.reader(open(A / path, encoding="utf-8")))
    head = rows[0]
    out, last = {}, None
    for r in rows[2:]:
        if r and r[0]:
            last = r[0]
            out[last] = dict(zip(head, r))
    return out


def b64(a):
    return base64.b64encode(np.ascontiguousarray(a).tobytes()).decode()


def png_url(img, quality=None):
    buf = io.BytesIO()
    Image.fromarray(img).save(buf, "PNG", optimize=False, compress_level=6)
    return "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode()


# ---- textures ----
_tex_cache = {}


def texture_image(ref):
    """'sc3d/rock_1.pvr#linear_mipmap_linear' / 'shelly_v2_01.pvr' / 'x.pvr+hue(mask.png,deg)+value(mask.png,v)...'
    -> uint8 RGB(A); the recolour ops (hue in degrees, saturation/value in percent) apply where the mask is white"""
    ref = ref.split("#")[0]
    base, *ops = ref.split("+")
    name = pathlib.Path(base).stem
    path = A / "sc3d" / f"{name}.ktx"
    if not path.exists():
        alt = A / "sc3d" / f"{name}.png"
        if not alt.exists():
            return None
        img = np.asarray(Image.open(alt).convert("RGBA"))
    else:
        img = ktx.load(path)
    for op in ops:
        m = re.match(r"(\w+)\((.+?),([-\d.]+)\)$", op)
        if not m:
            continue
        mask = np.asarray(Image.open(A / "sc3d" / m.group(2)).convert("L").resize(img.shape[1::-1]), np.float32) / 255
        img = recolor(img, m.group(1), float(m.group(3)), mask)
    return img


def recolor(img, op, amount, mask):
    rgb = img[..., :3].astype(np.float32) / 255
    mx, mn = rgb.max(-1), rgb.min(-1)
    d = mx - mn
    r, g, b = rgb[..., 0], rgb[..., 1], rgb[..., 2]
    nz = d > 1e-6
    dd = np.where(nz, d, 1)
    h = np.where(mx == r, ((g - b) / dd) % 6, np.where(mx == g, (b - r) / dd + 2, (r - g) / dd + 4)) / 6
    h = np.where(nz, h, 0)
    s = np.where(mx > 0, d / np.where(mx > 0, mx, 1), 0)
    v = mx
    if op == "hue":
        h = (h + amount / 360) % 1
    elif op == "saturation":
        s = np.clip(s * (1 + amount / 100), 0, 1)
    elif op == "value":
        v = np.clip(v * (1 + amount / 100), 0, 1)
    i = np.floor(h * 6)
    f = h * 6 - i
    p, q, t = v * (1 - s), v * (1 - s * f), v * (1 - s * (1 - f))
    i = i.astype(int) % 6
    out = np.select([(i == k)[..., None] for k in range(6)],
                    [np.stack(x, -1) for x in ((v, t, p), (q, v, p), (p, v, t), (p, q, v), (t, p, v), (v, p, q))])
    res = rgb * (1 - mask[..., None]) + out * mask[..., None]
    o = img.copy()
    o[..., :3] = np.clip(res * 255 + 0.5, 0, 255).astype(np.uint8)
    return o


def tex_entry(pack, ref):
    """adds a texture to the pack once; returns its key"""
    if not ref or ref == ".":
        return None
    key = ref.split("#")[0]
    if key not in pack["textures"]:
        img = _tex_cache.get(key)
        if img is None:
            img = texture_image(ref)
            _tex_cache[key] = img
        if img is None:
            print(f"  missing texture {ref}")
            return None
        pack["textures"][key] = png_url(img)
    return key


# ---- materials ----
def material_entry(pack, m):
    def tc(v):
        return {"tex": tex_entry(pack, v["tex"]) if v["tex"] != "." else "."} if "tex" in v else {"color": v["color"]}
    e = {"name": m["name"], "flags": m["flags"], "ambient": m["ambient"], "diffuse": tc(m["diffuse"]),
         "specular": tc(m["specular"]), "colorize": tc(m["colorize"]), "emission": tc(m["emission"]),
         "opacity": m["opacity"], "stencil": m["stencil"]}
    for k in ("lightmap_diffuse", "lightmap_specular"):
        e[k] = tex_entry(pack, m[k]) if m[k] else None
    return e


# ---- models ----
def mat44(t, r, s):
    x, y, z, w = r
    n = (x * x + y * y + z * z + w * w) ** 0.5 or 1
    x, y, z, w = x / n, y / n, z / n, w / n
    R = np.array([[1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
                  [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
                  [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)]])
    M = np.eye(4)
    M[:3, :3] = R * np.array(s)[None, :]
    M[:3, 3] = t
    return M


def model_entry(pack, path, materials_extra=()):
    """SCW model -> {nodes, meshes}; meshes are de-indexed per primitive (one vertex per distinct index tuple).
    'file.scw:geoA+geoB' (skin_confs Model) keeps only those geometries."""
    path, _, only = path.partition(":")
    only = set(only.split("+")) if only else None
    f = scw.load(A / path)
    mats = {m["name"]: m for m in f["materials"]}
    if f["head"]["materials"]:
        mats.update({m["name"]: m for m in scw.load(A / f["head"]["materials"])["materials"]})
    for extra in materials_extra:
        mats.update({m["name"]: m for m in scw.load(A / extra)["materials"]})
    names = [n["name"] for n in f["nodes"]]
    nodes = []
    for n in f["nodes"]:
        fr = n["frames"][0] if n["frames"] else (0, [0, 0, 0, 1], [0, 0, 0], [1, 1, 1])
        nodes.append({"name": n["name"], "parent": names.index(n["parent"]) if n["parent"] in names else -1,
                      "t": fr[2], "r": fr[1], "s": fr[3]})
    geoms = {g["name"]: g for g in f["geoms"]}
    meshes = []
    for ni, n in enumerate(f["nodes"]):
        for inst in n["instances"]:
            if inst["type"] not in ("GEOM", "CONT") or inst["name"] not in geoms:   # CONT = skinned
                continue
            if only and inst["name"] not in only and n["name"] not in only:
                continue
            g = geoms[inst["name"]]
            binds = dict(inst["binds"])
            src = {}     # index column -> [(semantic, data)]: POSITION and NORMAL can share a column (Piper's gunbrella)
            for k, v in g["sources"].items():
                src.setdefault(v["index"], []).append((k, v["data"]))
            mesh = {"node": ni, "name": g["name"], "prims": []}
            skinned = "skin_weights" in g and g["joints"]
            if skinned:
                mesh["joints"] = [names.index(j) if j in names else -1 for j, _ in g["joints"]]
                mesh["invBind"] = b64(np.stack([m.T.astype(np.float32) for _, m in g["joints"]]))
                mesh["bindShape"] = g["bind_shape"].T.astype(np.float32).ravel().tolist() if "bind_shape" in g else None
            for p in g["prims"]:
                idx = p["indices"]
                uniq, inv = np.unique(idx, axis=0, return_inverse=True)
                inv = inv.ravel()
                out = {}
                for col in range(idx.shape[1]):
                    for kind, data in src.get(col, []):
                        key = {"POSITION": "pos", "VERTEX": "pos", "NORMAL": "nrm", "TEXCOORD": "uv", "COLOR": "col"}.get(kind)
                        if key and key not in out:
                            out[key] = data[uniq[:, col]].astype(np.float32)
                            if key == "pos" and skinned:
                                out["j"] = g["skin_joints"][uniq[:, col]].astype(np.uint8)
                                out["w"] = g["skin_weights"][uniq[:, col]].astype(np.float32)
                mat_name = binds.get(p["material"], p["material"])
                mkey = None
                if mat_name in mats:
                    mkey = f"{path}:{mat_name}" if mat_name not in pack["shared_materials"] else mat_name
                    if mkey not in pack["materials"]:
                        pack["materials"][mkey] = material_entry(pack, mats[mat_name])
                nv = len(uniq)
                tri = inv.astype(np.uint16 if nv < 65536 else np.uint32)
                prim = {"material": mkey, "count": int(nv), "index": b64(tri), "i32": nv >= 65536}
                for k, v in out.items():
                    prim[k] = b64(v)
                mesh["prims"].append(prim)
            meshes.append(mesh)
    return {"nodes": nodes, "meshes": meshes, "fps": f["head"]["fps"]}


def anim_entry(path, start=None, end=None):
    """SCW animation -> per node: frames of t (3), r (4), s (3); frames [start, end] (file frame ids)"""
    f = scw.load(A / path)
    fps = f["head"]["fps"]
    tracks = {}
    lo = hi = None
    for n in f["nodes"]:
        if not n["frames"]:
            continue
        ids = [fr[0] for fr in n["frames"]]
        lo = min(ids) if lo is None else min(lo, min(ids))
        hi = max(ids) if hi is None else max(hi, max(ids))
    if lo is None:
        return None
    a = lo if start in (None, "", -1) else max(lo, int(start))
    b = hi if end in (None, "", -1) else min(hi, int(end))
    if b < a:
        a, b = lo, hi
    for n in f["nodes"]:
        if not n["frames"]:
            continue
        ids = np.array([fr[0] for fr in n["frames"]])
        T = np.array([fr[2] for fr in n["frames"]], np.float32)
        Rq = np.array([fr[1] for fr in n["frames"]], np.float32)
        S = np.array([fr[3] for fr in n["frames"]], np.float32)
        fr_ids = np.arange(a, b + 1)
        k = np.clip(np.searchsorted(ids, fr_ids, side="right") - 1, 0, len(ids) - 1)   # hold last key
        tracks[n["name"]] = {"t": b64(T[k]), "r": b64(Rq[k]), "s": b64(S[k])}
    return {"fps": fps, "frames": int(b - a + 1), "tracks": tracks}


def new_pack():
    return {"textures": {}, "materials": {}, "models": {}, "anims": {}, "heroes": {}, "theme": None,
            "shared_materials": set()}


def write_pack(pack, name):
    pack = dict(pack)
    pack.pop("shared_materials", None)
    OUT.mkdir(parents=True, exist_ok=True)
    p = OUT / f"{name}.js"
    p.write_text("BSGFX(" + json.dumps(pack, separators=(",", ":")) + ");\n")
    print(f"wrote {p.relative_to(ROOT)} ({p.stat().st_size / 1e6:.1f} MB)")


TILE_KEYS = {"Blocking1": "Blocking1", "Blocking2": "Blocking2", "Blocking3": "Blocking3", "Blocking4": "Blocking4",
             "RespawningWall": "RespawningWall", "RespawningForest": "RespawningForest", "Forest": "Forest",
             "Destructable": "DecoDestructible", "Fragile": "BlockingDestroyableWithWeapon", "WaterTile": "Water",
             "Fence": "Fence", "RopeFence": "RopeFence", "Indestructible": "Indestructible"}


def build_theme(theme_name):
    themes = table("csv_logic/location_themes.csv")
    t = themes[theme_name]
    pack = new_pack()
    pack["theme"] = {"name": theme_name, "tiles": {}}
    for col, tile in TILE_KEYS.items():
        f = t.get(f"{col}SCW")
        if not f:
            continue
        key = f.replace("sc3d/", "")
        if key not in pack["models"]:
            try:
                pack["models"][key] = model_entry(pack, f)
            except Exception as e:  # noqa: BLE001
                print(f"  {f}: {e}")
                continue
        pack["theme"]["tiles"][tile] = {"model": key, "mesh": t.get(f"{col}Mesh") or None,
                                        "angleStep": int(t.get(f"{col}AngleStep") or 0)}
    pack["models"]["mine_gem_spawn.scw"] = model_entry(pack, "sc3d/mine_gem_spawn.scw")   # items.csv OrbSpawner SCW
    pack["theme"]["OrbSpawner"] = "mine_gem_spawn.scw"
    for col in ("GroundSCW", "EnvironmentSCW"):
        f = t.get(col)
        if f:
            key = f.replace("sc3d/", "")
            pack["models"][key] = model_entry(pack, f)
            pack["theme"][col] = key
    write_pack(pack, f"theme_{theme_name}")


def build_core():
    pack = new_pack()
    for lm in ("diffuse_lightmap", "specular_lightmap", "hair_diffuse_lightmap", "hair_specular_lightmap",
               "metal_diffuse_lightmap", "metal_specular_lightmap", "skin_lightmap", "skin_specular_lightmap"):
        tex_entry(pack, f"sc3d/{lm}.png")
    pack["fonts"] = {"Lilita One": base64.b64encode((A / "font/LilitaOne-Regular.ttf").read_bytes()).decode()}
    write_pack(pack, "core")


SUMMON_SKIP = ("Tutorial", "Coop", "RaidBoss", "Training", "EventModifier")   # not in PvP modes


def build_heroes(only=None):
    chars = table("csv_logic/characters.csv")
    skins = table("csv_logic/skins.csv")
    confs = table("csv_logic/skin_confs.csv")
    anims = table("csv_client/animations.csv")
    for name, c in chars.items():
        # brawlers, and the summons they place on the map (8-Bit's turret, Jessie's turret, Nita's bear, ...)
        summon = c.get("Type", "").startswith("Minion") and not name.startswith(SUMMON_SKIP)
        if not (c.get("Type") == "Hero" or summon) or not c.get("DefaultSkin") or c.get("Disabled") == "true":
            continue
        if only and name not in only:
            continue
        skin = skins.get(c["DefaultSkin"])
        conf = confs.get(skin["Conf"]) if skin else None
        if not conf or not conf.get("Model"):
            print(f"{name}: no skin conf")
            continue
        pack = new_pack()
        pack["shared_materials"] = set()
        try:
            model = model_entry(pack, "sc3d/" + conf["Model"])
        except Exception as e:  # noqa: BLE001
            print(f"{name}: model {conf['Model']}: {e}")
            continue
        pack["models"][name] = model
        hero = {"model": name, "scale": int(c.get("Scale") or 100) / 100, "anims": {}, "tex": {}}
        for team, col in (("blue", "BlueTexture"), ("red", "RedTexture")):
            ref = skin.get(col)
            if ref:
                hero["tex"][team] = tex_entry(pack, ref)
        for slot, col in (("idle", "IdleAnim"), ("walk", "WalkAnim"), ("attack", "PrimarySkillAnim"),
                          ("recoil", "PrimarySkillRecoilAnim"), ("super", "SecondarySkillAnim"),
                          ("superRecoil", "SecondarySkillRecoilAnim"), ("win", "HappyLoopAnim"), ("lose", "SadLoopAnim")):
            a = anims.get(conf.get(col, ""))
            if not a or not a.get("FileName"):
                continue
            key = f"{a['Name']}"
            if key not in pack["anims"]:
                try:
                    e = anim_entry("sc3d/" + a["FileName"], a.get("StartFrame"), a.get("EndFrame"))
                except Exception as ex:  # noqa: BLE001
                    print(f"  {name} anim {a['FileName']}: {ex}")
                    continue
                if not e:
                    continue
                e["loop"] = a.get("Looping") == "true"
                e["speed"] = int(a["Speed"]) / 100 if a.get("Speed") else 1.0
                pack["anims"][key] = e
            hero["anims"][slot] = key
        pack["heroes"][name] = hero
        write_pack(pack, f"hero_{name}")


if __name__ == "__main__":
    args = sys.argv[1:]
    build_core()
    build_theme("Mine")
    build_heroes(set(args) if args else None)
