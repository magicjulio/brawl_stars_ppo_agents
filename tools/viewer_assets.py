"""Static files of the viewer's 3D mode (tools/viewer/gfx3d.js and the packs in tools/viewer/gfx/, built by
tools/gfx/build.py) for the local servers: pages served over http load them from /gfx/<path>."""
import gzip
import pathlib

VIEWER_DIR = pathlib.Path(__file__).resolve().parent / "viewer"
TYPES = {".js": "text/javascript; charset=utf-8", ".png": "image/png", ".json": "application/json"}


def gfx_file(url_path):
    """'/gfx/<rel>' -> (bytes, content type) of tools/viewer/<rel>, or None"""
    if not url_path.startswith("/gfx/"):
        return None
    f = (VIEWER_DIR / url_path[5:].split("?")[0]).resolve()
    if VIEWER_DIR not in f.parents or not f.is_file():
        return None
    return f.read_bytes(), TYPES.get(f.suffix, "application/octet-stream")


_GZ = {}                                     # path -> (mtime, gzipped bytes, etag)


def send_gfx(handler, url_path):
    """answers a /gfx/ request on a BaseHTTPRequestHandler; False if the path is not one. Gzipped (the packs shrink
    3-4x: a phone over Wi-Fi loads them in seconds) and cached by the browser until a pack is rebuilt (ETag -> 304)."""
    r = gfx_file(url_path)
    if r is None:
        return False
    data, ctype = r
    f = (VIEWER_DIR / url_path[5:].split("?")[0]).resolve()
    mtime = f.stat().st_mtime_ns
    if _GZ.get(f, (None,))[0] != mtime:
        _GZ[f] = (mtime, gzip.compress(data, 6), f'"{mtime:x}-{len(data):x}"')
    _, gz, etag = _GZ[f]
    if handler.headers.get("If-None-Match") == etag:
        handler.send_response(304)
        handler.send_header("ETag", etag)
        handler.send_header("Content-Length", "0")
        handler.end_headers()
        return True
    zipped = "gzip" in handler.headers.get("Accept-Encoding", "")
    body = gz if zipped else data
    handler.send_response(200)
    handler.send_header("Content-Type", ctype)
    handler.send_header("Content-Length", str(len(body)))
    if zipped:
        handler.send_header("Content-Encoding", "gzip")
    handler.send_header("Vary", "Accept-Encoding")
    handler.send_header("ETag", etag)
    handler.send_header("Cache-Control", "no-cache")      # always ask, but a 304 is all it costs
    handler.end_headers()
    handler.wfile.write(body)
    return True
