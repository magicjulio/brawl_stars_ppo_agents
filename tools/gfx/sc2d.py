"""Reader for the game's 2D .sc files (Supercell's flash-like format): texture atlases, shapes, movie clips.

File: "SC" + u32 version (1) + u32 hash length + hash, then LZMA ("alone" format with a 4-byte size).
Body: counts (shapes, clips, textures, text fields, matrices, colour transforms), 5 bytes, exports (id + name),
then tags [u8 tag, u32 length (little-endian), data]. Textures live in the matching *_tex.sc (same order; the main
file has a 5 byte placeholder per texture).
  textures 1/16/19/24/27/28/29/34 (27-29: 32x32 blocks), shapes 2/18 with draw commands 4/17/22 (textured polygons),
  clips 10/12/35 with sub-tags 11 (frame: element count + label), 31 (scaling grid), 41 (matrix bank), matrices 8
  (/1024) and 36 (/65535) with translation /20, colour transforms 9, tag 42 = new matrix + colour transform bank.
  Clip binds 242/243/244 mark mask shape / masked children / end; blend byte: 3 multiply, 4 screen, 8 add (& 63).

Rendering (SC.render) supports what the HUD, effect and level sprites use: shapes made of textured polygons, movie
clips (one frame, children at frame 0) with affine matrices, colour transforms, blend modes and masks. Textures are
premultiplied RGBA, so everything is composited in premultiplied float space. Text fields are not drawn.
"""
import lzma
import struct

import numpy as np

TEXTURE_TAGS = {1, 16, 19, 24, 27, 28, 29, 34}
SHAPE_TAGS = {2, 18}
CLIP_TAGS = {3, 10, 12, 14, 35}
TEXTFIELD_TAGS = {7, 15, 20, 21, 25, 33, 43, 44}


def decompress(data):
    if data[:2] != b"SC":
        raise ValueError("not an SC file")
    ver = struct.unpack(">I", data[2:6])[0]
    o = 6
    if ver == 4:
        ver = struct.unpack(">I", data[o:o + 4])[0]
        o += 4
    hl = struct.unpack(">I", data[o:o + 4])[0]
    body = data[o + 4 + hl:]
    if body[:4] == b"SCLZ" or body[:4] == b"\x28\xb5\x2f\xfd":
        raise ValueError("unsupported compression")
    hdr = body[:9] + b"\x00\x00\x00\x00"
    return lzma.decompress(hdr + body[9:], format=lzma.FORMAT_ALONE)


class R:
    def __init__(self, d, o=0):
        self.d, self.o = d, o

    def u8(self):
        v = self.d[self.o]
        self.o += 1
        return v

    def u16(self):
        v = struct.unpack_from("<H", self.d, self.o)[0]
        self.o += 2
        return v

    def i16(self):
        v = struct.unpack_from("<h", self.d, self.o)[0]
        self.o += 2
        return v

    def u32(self):
        v = struct.unpack_from("<I", self.d, self.o)[0]
        self.o += 4
        return v

    def i32(self):
        v = struct.unpack_from("<i", self.d, self.o)[0]
        self.o += 4
        return v

    def s8(self):     # u8 length + ascii, 255 = none
        n = self.u8()
        if n == 255:
            return None
        v = self.d[self.o:self.o + n].decode("utf-8", "replace")
        self.o += n
        return v


def _pixels(fmt, w, h, raw):
    if fmt in (0, 1):
        a = np.frombuffer(raw, np.uint8, w * h * 4).reshape(h, w, 4)
    elif fmt == 2:   # RGBA4444
        v = np.frombuffer(raw, "<u2", w * h).reshape(h, w)
        a = np.stack([(v >> 12) & 15, (v >> 8) & 15, (v >> 4) & 15, v & 15], -1).astype(np.uint8) * 17
    elif fmt == 3:   # RGBA5551
        v = np.frombuffer(raw, "<u2", w * h).reshape(h, w)
        a = np.stack([((v >> 11) & 31) * 255 // 31, ((v >> 6) & 31) * 255 // 31, ((v >> 1) & 31) * 255 // 31,
                      (v & 1) * 255], -1).astype(np.uint8)
    elif fmt == 4:   # RGB565
        v = np.frombuffer(raw, "<u2", w * h).reshape(h, w)
        a = np.stack([((v >> 11) & 31) * 255 // 31, ((v >> 5) & 63) * 255 // 63, (v & 31) * 255 // 31,
                      np.full_like(v, 255)], -1).astype(np.uint8)
    elif fmt == 6:   # LA88
        v = np.frombuffer(raw, np.uint8, w * h * 2).reshape(h, w, 2)
        a = np.concatenate([v[..., :1]] * 3 + [v[..., 1:]], -1)
    elif fmt == 10:  # L8
        v = np.frombuffer(raw, np.uint8, w * h).reshape(h, w, 1)
        a = np.concatenate([v] * 3 + [np.full_like(v, 255)], -1)
    else:
        raise ValueError(f"pixel format {fmt}")
    return a


def _unblock(a, w, h):
    """tags 27-29 store 32x32 blocks row by row; a is the flat pixel stream reshaped as [h, w, 4]"""
    flat = a.reshape(-1, 4)
    out = np.zeros((h, w, 4), np.uint8)
    o = 0
    for by in range(0, h, 32):
        for bx in range(0, w, 32):
            bh, bw = min(32, h - by), min(32, w - bx)
            out[by:by + bh, bx:bx + bw] = flat[o:o + bh * bw].reshape(bh, bw, 4)
            o += bh * bw
    return out


BPP = {0: 4, 1: 4, 2: 2, 3: 2, 4: 2, 6: 2, 10: 1}


class SC:
    def __init__(self, path, tex_path=None):
        d = decompress(open(path, "rb").read())
        r = R(d)
        self.n_shapes, self.n_clips, self.n_textures, self.n_text, self.n_mat, self.n_ct = [r.u16() for _ in range(6)]
        r.o += 5
        n = r.u16()
        ids = [r.u16() for _ in range(n)]
        names = [r.s8() for _ in range(n)]
        self.exports = {}
        for i, nm in zip(ids, names):
            self.exports.setdefault(nm, i)
        self.shapes, self.clips, self.textures, self.matrices, self.cts = {}, {}, [], [], []
        self.matrix_banks = [self.matrices]   # tag 42 starts a new bank: own matrices + colour transforms
        self.ct_banks = [self.cts]
        self.textfields = {}
        self._tags(d, r)
        if tex_path:
            td = decompress(open(tex_path, "rb").read())
            self._tags(td, R(td), textures_only=True)

    def _tags(self, d, r, textures_only=False):
        ti = 0   # texture slot: the tex file fills the placeholders (tag length 5) of the main file in order
        while r.o < len(d):
            tag = r.u8()
            length = r.u32()
            start = r.o
            if tag == 0:
                break
            if tag in TEXTURE_TAGS:
                fmt, w, h = r.u8(), r.u16(), r.u16()
                if length > 5:
                    raw = d[r.o:r.o + w * h * BPP[fmt]]
                    a = _pixels(fmt, w, h, raw)
                    if tag in (27, 28, 29):
                        a = _unblock(a, w, h)
                    if textures_only and ti < len(self.textures):
                        self.textures[ti] = a
                    else:
                        self.textures.append(a)
                elif not textures_only:
                    self.textures.append(None)   # pixels come from the _tex file
                ti += 1
            elif textures_only:
                pass
            elif tag in SHAPE_TAGS:
                self._shape(tag, r)
            elif tag in CLIP_TAGS:
                self._clip(tag, r, length)
            elif tag == 8 or tag == 36:
                a, b, c, dd = (r.i32() / (1024 if tag == 8 else 65535) for _ in range(4))
                tx, ty = r.i32() / 20, r.i32() / 20
                self.matrix_banks[-1].append((a, b, c, dd, tx, ty))
            elif tag == 9:
                ra, ga, ba, aa, rm, gm, bm = (r.u8() for _ in range(7))
                self.ct_banks[-1].append({"add": (ra, ga, ba), "mul": (rm / 255, gm / 255, bm / 255, aa / 255)})
            elif tag == 42:
                r.u16()   # matrix / colour transform counts of the new bank
                r.u16()
                self.matrix_banks.append([])
                self.ct_banks.append([])
            elif tag in TEXTFIELD_TAGS:
                tid = r.u16()
                self.textfields[tid] = tag
            r.o = start + length

    def _shape(self, tag, r):
        sid = r.u16()
        r.u16()   # command count
        if tag == 18:
            r.u16()   # total point count
        cmds = []
        while True:
            t = r.u8()
            n = r.u32()
            s = r.o
            if t == 0:
                break
            if t in (4, 17, 22):
                tex = r.u8()
                npts = 4 if t == 4 else r.u8()
                xy = [(r.i32() / 20, r.i32() / 20) for _ in range(npts)]
                uv = [(r.u16(), r.u16()) for _ in range(npts)]
                if t == 22:   # normalised uv
                    uv = [(u / 65535, v / 65535) for u, v in uv]
                cmds.append({"tex": tex, "xy": xy, "uv": uv, "norm": t == 22})
            r.o = s + n
        self.shapes[sid] = cmds

    def _clip(self, tag, r, length):
        start = r.o
        cid = r.u16()
        fps = r.u8()
        nframes = r.u16()
        if tag in (3, 14):
            raise ValueError("old movie clip tag")
        n = r.u32()
        elems = [(r.u16(), r.u16(), r.u16()) for _ in range(n)]
        nb = r.u16()
        binds = [r.u16() for _ in range(nb)]
        if tag in (12, 35):
            blends = [r.u8() for _ in range(nb)]
        else:
            blends = [0] * nb
        bnames = [r.s8() for _ in range(nb)]
        frames, bank = [], 0
        while r.o < start + length:
            t = r.u8()
            ln = r.u32()
            s = r.o
            if t == 0:
                break
            if t == 11:
                cnt = r.u16()
                frames.append({"n": cnt, "label": r.s8()})
            elif t == 41:
                bank = r.u8()
            r.o = s + ln
        self.clips[cid] = {"fps": fps, "elems": elems, "binds": binds, "names": bnames, "blends": blends,
                           "frames": frames, "bank": bank}

    def frame_elems(self, cid, frame=0):
        """(bind index, matrix index, colour transform index) of a clip's frame (clamped); 0xFFFF = none"""
        c = self.clips[cid]
        if not c["frames"]:
            return []
        frame = min(frame, len(c["frames"]) - 1)
        o = sum(f["n"] for f in c["frames"][:frame])
        return c["elems"][o:o + c["frames"][frame]["n"]]

    # ---- rendering ----
    def render(self, name_or_id, scale=1.0, frame=0, pad=2, only=(), box=None, ss=2, frames=None):
        """-> (straight-alpha RGBA uint8 image, (ox, oy)) of a frame of an export name (or any clip/shape id), or
        (None, (0, 0)) when nothing is drawn; (ox, oy) = pixel of the origin inside the image.
        only: path of bind names / indices (an entry may be a tuple of them) keeping just those children, one entry
          per nesting level; the parents' matrices and colour transforms still apply (parts line up with the whole).
        frames: {bind name: frame} for child clips (default 0; None = unnamed binds).
        box: exact (x0, y0, x1, y1) region in export units instead of the content bounds. ss: supersampling."""
        cid = self.exports[name_or_id] if isinstance(name_or_id, str) else name_or_id
        items = []
        self._collect(cid, np.diag([scale * ss, scale * ss, 1.0]), None, 0, (), items, frame, tuple(only), frames or {})
        if box is not None:
            x0, y0, x1, y1 = (int(round(v * scale)) for v in box)
        elif items:
            pts = np.concatenate([it[1] for it in items])
            x0, y0 = np.floor(pts.min(0) / ss) - pad
            x1, y1 = np.ceil(pts.max(0) / ss) + pad
        else:
            return None, (0, 0)
        W, H = int(x1 - x0), int(y1 - y0)
        cv = np.zeros((H * ss, W * ss, 4), np.float32)   # premultiplied
        for it in items:
            self._draw(cv, it, x0 * ss, y0 * ss)
        if ss > 1:
            cv = cv.reshape(H, ss, W, ss, 4).mean((1, 3))
        a = cv[..., 3:]
        rgb = np.where(a > 1e-4, cv[..., :3] / np.maximum(a, 1e-4), 0)
        out = np.concatenate([rgb, a], -1).clip(0, 1)
        return (out * 255 + 0.5).astype(np.uint8), (-x0, -y0)

    def _collect(self, cid, m, ct, blend, masks, out, frame, only, frames):
        """flatten a clip/shape into draw items (tex, xy, uv, colour transform, blend, masks) in painting order"""
        if cid in self.shapes:
            for c in self.shapes[cid]:
                th, tw = self.textures[c["tex"]].shape[:2]
                uv = np.array(c["uv"], np.float64) * ([tw, th] if c["norm"] else 1)
                out.append((c["tex"], np.array(c["xy"], np.float64) @ m[:2, :2].T + m[:2, 2], uv, ct, blend, masks))
            return
        clip = self.clips.get(cid)
        if clip is None:
            return
        bank, cts = self.matrix_banks[clip["bank"]], self.ct_banks[clip["bank"]]
        want = only[0] if only else None
        want = want if isinstance(want, (tuple, list, set)) else (want,)
        mode, mpolys = 0, []   # 242: mask shape follows, 243: masked children follow, 244: end
        for bi, mi, ci in self.frame_elems(cid, frame):
            child = clip["binds"][bi]
            if child in (242, 243, 244):
                mode = {242: 1, 243: 2, 244: 0}[child]
                mpolys = [] if child == 242 else mpolys
                continue
            if only and bi not in want and clip["names"][bi] not in want:
                continue
            cm = m
            if mi != 0xFFFF:
                a, b, c, d, tx, ty = bank[mi]
                cm = m @ [[a, c, tx], [b, d, ty], [0, 0, 1]]
            cc = _ct_mul(ct, cts[ci]) if ci != 0xFFFF else ct
            bl = (clip["blends"][bi] & 63) or blend
            if mode == 1:   # mask geometry: not painted
                tmp = []
                self._collect(child, cm, cc, bl, masks, tmp, frames.get(clip["names"][bi], 0), only[1:], frames)
                mpolys += [t[1] for t in tmp]
            else:
                self._collect(child, cm, cc, bl, masks + ((mpolys,) if mode == 2 else ()), out,
                              frames.get(clip["names"][bi], 0), only[1:], frames)

    def _draw(self, cv, it, x0, y0):
        tex_i, xy, uv, ct, blend, masks = it
        H, W = cv.shape[:2]
        xy = xy - [x0, y0]
        bx0, by0 = np.maximum(np.floor(xy.min(0)).astype(int), 0)
        bx1, by1 = np.minimum(np.ceil(xy.max(0)).astype(int), [W, H])
        if bx1 <= bx0 or by1 <= by0:
            return
        # uv is an affine function of xy (fitted on all points); sample the texture bilinearly at pixel centres
        sol = np.linalg.lstsq(np.c_[xy, np.ones(len(xy))], uv, rcond=None)[0]
        gy, gx = np.mgrid[by0:by1, bx0:bx1] + 0.5
        src = _sample(self.textures[tex_i], gx * sol[0, 0] + gy * sol[1, 0] + sol[2, 0],
                      gx * sol[0, 1] + gy * sol[1, 1] + sol[2, 1])
        cov = _cover([xy], bx0, by0, bx1 - bx0, by1 - by0)
        for g in masks:
            if g:
                cov = cov * _cover([p - [x0, y0] for p in g], bx0, by0, bx1 - bx0, by1 - by0)
        src = src * cov[..., None]
        if ct is not None:
            src = _apply_ct(src, ct)
        _blend(cv[by0:by1, bx0:bx1], src, blend)


def _sample(tex, u, v):
    """bilinear sample (clamped) of a premultiplied uint8 [h, w, 4] texture at texel coordinates -> float32 [..., 4]"""
    h, w = tex.shape[:2]
    u, v = u - 0.5, v - 0.5
    xf, yf = np.floor(u), np.floor(v)
    fx, fy = (u - xf)[..., None].astype(np.float32), (v - yf)[..., None].astype(np.float32)
    x0, y0 = np.clip(xf.astype(int), 0, w - 1), np.clip(yf.astype(int), 0, h - 1)
    x1, y1 = np.clip(xf.astype(int) + 1, 0, w - 1), np.clip(yf.astype(int) + 1, 0, h - 1)
    top = tex[y0, x0] * (1 - fx) + tex[y0, x1] * fx
    bot = tex[y1, x0] * (1 - fx) + tex[y1, x1] * fx
    return (top * (1 - fy) + bot * fy) / 255


def _cover(polys, x0, y0, w, h, k=4):
    """anti-aliased coverage [h, w] (0..1) of polygons (canvas pixels), rasterised at k x resolution"""
    from PIL import Image, ImageDraw
    im = Image.new("L", (w * k, h * k), 0)
    dr = ImageDraw.Draw(im)
    for p in polys:
        dr.polygon([((x - x0) * k - 0.5, (y - y0) * k - 0.5) for x, y in p], fill=255)
    return np.asarray(im.resize((w, h), Image.BOX), np.float32) / 255


def _apply_ct(p, ct):
    """colour transform of a premultiplied image: colour = colour * mul + add (0..255), alpha *= mul alpha"""
    mul, add, a = np.array(ct["mul"], np.float32), np.array(ct["add"], np.float32) / 255, p[..., 3:]
    rgb = (p[..., :3] * mul[:3] + add * a) * mul[3]
    a = a * mul[3]
    return np.concatenate([np.minimum(rgb, a), a], -1)


def _blend(d, s, mode):
    """composite premultiplied s onto d (a view, in place); 8 = add, 3 = multiply, 4 = screen, else normal"""
    sa, da = s[..., 3:], d[..., 3:]
    if mode == 8:   # alpha follows the added light so the result still looks right over a dark background
        rgb = d[..., :3] + s[..., :3]
        d[..., 3:] = np.maximum(da, np.minimum(rgb.max(-1, keepdims=True), 1))
        d[..., :3] = rgb
    elif mode == 3:
        d[..., :3] = s[..., :3] * (1 - da) + d[..., :3] * (1 - sa) + s[..., :3] * d[..., :3]
        d[..., 3:] = sa + da - sa * da
    elif mode == 4:
        d[..., :3] = s[..., :3] + d[..., :3] - s[..., :3] * d[..., :3]
        d[..., 3:] = sa + da - sa * da
    else:
        d[...] = s + d * (1 - sa)


def _ct_mul(parent, child):
    """colour transform of a child inside a parent: the child's applies first"""
    if parent is None:
        return child
    mul = tuple(x * y for x, y in zip(child["mul"], parent["mul"]))
    add = tuple(min(255, a * m + b) for a, m, b in zip(child["add"], parent["mul"], parent["add"]))
    return {"mul": mul, "add": add}
