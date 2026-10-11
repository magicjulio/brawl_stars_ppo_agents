"""KTX 1.1 textures of the game (every .ktx in sc3d/ is ETC1, glInternalFormat 0x8D64) -> RGB arrays.

The decoded image keeps the file's row order (first row = texture row t=0), as the game uploads it.
"""
import struct
import sys

import numpy as np

ETC1 = 0x8D64
MODIFIERS = np.array([[2, 8, -2, -8], [5, 17, -5, -17], [9, 29, -9, -29], [13, 42, -13, -42],
                      [18, 60, -18, -60], [24, 80, -24, -80], [33, 106, -33, -106], [47, 183, -47, -183]], np.int32)


def decode_etc1(data, w, h):
    """ETC1 blocks (big-endian 64-bit words, row-major blocks) -> uint8 [h, w, 3]"""
    bw, bh = (w + 3) // 4, (h + 3) // 4
    blk = np.frombuffer(data, ">u8", bw * bh).astype(np.uint64)
    hi = (blk >> np.uint64(32)).astype(np.int64)
    lo = (blk & np.uint64(0xFFFFFFFF)).astype(np.int64)
    diff = (hi >> 1) & 1
    flip = hi & 1
    t1 = (hi >> 5) & 7
    t2 = (hi >> 2) & 7

    def ext4(v):
        return v * 17

    def ext5(v):
        return (v << 3) | (v >> 2)

    base1 = np.zeros((len(blk), 3), np.int64)
    base2 = np.zeros((len(blk), 3), np.int64)
    for c, sh in enumerate((24, 16, 8)):   # R, G, B fields in the high word
        ind1 = (hi >> (sh + 4)) & 15
        ind2 = (hi >> sh) & 15
        d5 = (hi >> (sh + 3)) & 31
        dd = (hi >> sh) & 7
        dd = np.where(dd >= 4, dd - 8, dd)
        base1[:, c] = np.where(diff == 1, ext5(d5), ext4(ind1))
        base2[:, c] = np.where(diff == 1, ext5((d5 + dd) & 31), ext4(ind2))
    # pixel p = x*4 + y: index msb at bit 16+p, lsb at bit p of the low word
    p = np.arange(16)
    msb = (lo[:, None] >> (16 + p)) & 1
    lsb = (lo[:, None] >> p) & 1
    idx = (msb << 1) | lsb                                   # [n, 16]
    x, y = p // 4, p % 4
    second = np.where(flip[:, None] == 1, y[None, :] >= 2, x[None, :] >= 2)   # [n, 16]
    table = np.where(second, t2[:, None], t1[:, None])
    mod = MODIFIERS[table, idx]                               # [n, 16]
    base = np.where(second[:, :, None], base2[:, None, :], base1[:, None, :])
    px = np.clip(base + mod[:, :, None], 0, 255).astype(np.uint8)   # [n, 16, 3]
    img = np.zeros((bh * 4, bw * 4, 3), np.uint8)
    blocks = px.reshape(bh, bw, 4, 4, 3)                     # [by, bx, x, y, c]
    img[:] = blocks.transpose(0, 3, 1, 2, 4).reshape(bh * 4, bw * 4, 3)
    return img[:h, :w]


def load(path):
    d = open(path, "rb").read()
    if d[:12] != b"\xabKTX 11\xbb\r\n\x1a\n":
        raise ValueError("not a KTX 1.1 file")
    fmt = "<" if d[12:16] == b"\x01\x02\x03\x04" else ">"
    (gl_type, _, gl_format, internal, _, w, h, _, _, _, mips, kvb) = struct.unpack(fmt + "12I", d[16:64])
    o = 64 + kvb
    size = struct.unpack(fmt + "I", d[o:o + 4])[0]
    if internal != ETC1:
        raise ValueError(f"unsupported glInternalFormat 0x{internal:x}")
    return decode_etc1(d[o + 4:o + 4 + size], w, h)


if __name__ == "__main__":
    from PIL import Image
    for p in sys.argv[1:]:
        Image.fromarray(load(p)).save(p.rsplit("/", 1)[-1].replace(".ktx", ".png"))
