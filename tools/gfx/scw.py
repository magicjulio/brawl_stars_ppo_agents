"""Reader for the game's .scw 3D files (magic "SC3D"): models, materials, cameras and animations.

A file is a list of chunks [u32 length, 4-char name, data, u32 crc] (all big-endian), ending with WEND:
  HEAD  version (2 for every file of v36), frame rate, first/last frame, materials file, one more byte
  MATE  a material of the game's uber shader (textures, colours, lightmaps, flags)
  GEOM  a mesh: vertex sources (int16 normalised by 32512, times a per-source scale), optional skin
        (bind shape matrix, joints with inverse bind matrices, 4 weights per vertex), triangle lists per material
  CAME  a camera (fov, aspect, near, far)
  NODE  the scene tree: nodes with parent, attached instances (GEOM/CONT with material binds, CAME) and
        keyframes (rotation quaternion, translation, scale); animation files are HEAD + NODE only.
Every chunk is checked to be consumed exactly (tools/gfx/scw.py <files> parses them and reports).
"""
import struct
import sys

import numpy as np

NSHORT = 32512.0


class Reader:
    def __init__(self, data):
        self.d, self.o = data, 0

    def u8(self):
        v = self.d[self.o]
        self.o += 1
        return v

    def u16(self):
        v = struct.unpack_from(">H", self.d, self.o)[0]
        self.o += 2
        return v

    def u32(self):
        v = struct.unpack_from(">I", self.d, self.o)[0]
        self.o += 4
        return v

    def f32(self):
        v = struct.unpack_from(">f", self.d, self.o)[0]
        self.o += 4
        return v

    def floats(self, n):
        v = np.frombuffer(self.d, ">f4", n, self.o).astype(np.float32)
        self.o += 4 * n
        return v

    def utf(self):
        n = self.u16()
        v = self.d[self.o:self.o + n].decode("utf-8")
        self.o += n
        return v

    def raw(self, n):
        v = self.d[self.o:self.o + n]
        self.o += n
        return v

    def argb(self):
        a, r, g, b = self.raw(4)
        return [r / 255, g / 255, b / 255, a / 255]

    def left(self):
        return len(self.d) - self.o


def chunks(data):
    if data[:4] != b"SC3D":
        raise ValueError("not an SC3D file")
    o = 4
    while o < len(data):
        n = struct.unpack_from(">I", data, o)[0]
        name = data[o + 4:o + 8].decode("latin1")
        yield name, data[o + 8:o + 8 + n]
        o += 12 + n
        if name == "WEND":
            break


def parse_head(r):
    h = {"version": r.u16(), "fps": r.u16(), "first": r.u16(), "last": r.u16()}
    h["materials"] = r.utf()
    if h["version"] >= 2:
        h["extra"] = r.u8()
    return h


def _tex_or_color(r):
    if r.u8():
        return {"tex": r.utf()}
    return {"color": r.argb()}


def parse_mate(r):
    m = {"name": r.utf(), "shader": r.utf(), "b1": r.u8(), "b2": r.u8(), "ambient": r.argb()}
    m["diffuse"] = _tex_or_color(r)
    m["specular"] = _tex_or_color(r)
    m["stencil"] = r.utf()
    m["normal"] = r.utf()
    m["colorize"] = _tex_or_color(r)
    m["emission"] = _tex_or_color(r)
    m["opacity_tex"] = r.utf()
    m["opacity"] = r.f32()
    m["f2"] = r.f32()
    m["lightmap_diffuse"] = r.utf()
    m["lightmap_specular"] = r.utf()
    m["s3"] = r.utf()
    m["flags"] = r.u32()
    if m["flags"] & 0x8000:                 # stencil scale/offset (u_stencilScaleOffset)
        m["stencil_scale_offset"] = r.floats(4).tolist()
    return m


def parse_geom(r):
    g = {"name": r.utf(), "group": r.utf(), "sources": {}}
    for _ in range(r.u8()):
        name = r.utf()
        index, sub, stride = r.u8(), r.u8(), r.u8()
        scale = r.f32()
        count = r.u32()
        v = np.frombuffer(r.raw(2 * stride * count), ">i2").astype(np.float32).reshape(count, stride) / NSHORT * scale
        g["sources"][name] = {"index": index, "sub": sub, "data": v}
    if r.u8():
        g["bind_shape"] = r.floats(16).reshape(4, 4)
    g["joints"] = []
    for _ in range(r.u8()):
        g["joints"].append((r.utf(), r.floats(16).reshape(4, 4)))
    nw = r.u32()
    if nw:
        raw = np.frombuffer(r.raw(12 * nw), np.dtype([("j", "u1", 4), ("w", ">u2", 4)]))
        g["skin_joints"] = raw["j"].copy()
        g["skin_weights"] = raw["w"].astype(np.float32) / 65535.0
    g["prims"] = []
    for _ in range(r.u8()):
        mat = r.utf()
        extra = r.utf()
        ntri = r.u16()
        inputs, isize = r.u8(), r.u8()
        dt = {1: "u1", 2: ">u2", 4: ">u4"}[isize]
        idx = np.frombuffer(r.raw(ntri * 3 * inputs * isize), dt).astype(np.int64).reshape(ntri * 3, inputs)
        g["prims"].append({"material": mat, "extra": extra, "indices": idx})
    return g


def parse_came(r):
    c = {"name": r.utf(), "v1": r.f32(), "xfov": r.f32(), "aspect": r.f32(), "znear": r.f32(), "zfar": r.f32()}
    return c


def parse_node(r):
    nodes = []
    for _ in range(r.u16()):
        n = {"name": r.utf(), "parent": r.utf(), "instances": []}
        for _ in range(r.u16()):
            kind = r.raw(4).decode("latin1")
            inst = {"type": kind, "name": r.utf()}
            if kind in ("GEOM", "CONT"):
                inst["binds"] = [(r.utf(), r.utf()) for _ in range(r.u16())]
            elif kind == "CAME":
                inst["target"] = r.utf()
            else:
                raise ValueError(f"unknown instance type {kind!r}")
            n["instances"].append(inst)
        nf = r.u16()
        frames = []
        if nf:
            flags = r.u8()
            rot, pos, scl = [0, 0, 0, 1], [0, 0, 0], [1, 1, 1]
            for i in range(nf):
                fid = r.u16()
                if flags & 1 or i == 0:
                    rot = [r.u16() for _ in range(4)]
                    rot = [(x - 65536 if x > 32767 else x) / NSHORT for x in rot]
                pos = list(pos)
                scl = list(scl)
                for k in range(3):
                    if flags & (2 << k) or i == 0:
                        pos[k] = r.f32()
                for k in range(3):
                    if flags & (16 << k) or i == 0:
                        scl[k] = r.f32()
                frames.append((fid, rot, pos, scl))
            n["frame_flags"] = flags
        n["frames"] = frames
        nodes.append(n)
    return nodes


PARSERS = {"HEAD": parse_head, "MATE": parse_mate, "GEOM": parse_geom, "CAME": parse_came, "NODE": parse_node}


def load(path_or_bytes):
    data = path_or_bytes if isinstance(path_or_bytes, (bytes, bytearray)) else open(path_or_bytes, "rb").read()
    out = {"head": None, "materials": [], "geoms": [], "cameras": [], "nodes": []}
    for name, chunk in chunks(data):
        if name == "WEND":
            break
        r = Reader(chunk)
        v = PARSERS[name](r)
        if r.left():
            raise ValueError(f"{name}: {r.left()} bytes left over")
        if name == "HEAD":
            out["head"] = v
        elif name == "MATE":
            out["materials"].append(v)
        elif name == "GEOM":
            out["geoms"].append(v)
        elif name == "CAME":
            out["cameras"].append(v)
        else:
            out["nodes"].extend(v)
    return out


if __name__ == "__main__":
    ok = bad = 0
    for p in sys.argv[1:]:
        try:
            load(p)
            ok += 1
        except Exception as e:  # noqa: BLE001  (report every file)
            bad += 1
            print(f"{p}: {type(e).__name__}: {e}")
    print(f"{ok} parsed, {bad} failed")
