#!/usr/bin/env python3
"""Run an offline bot battle on the tablet with recording on, pull the replay, and open it in the viewer.

  tools/replay.py                          # Gemgrab1, seed 1
  tools/replay.py loc=Ball1 seed=7         # any driver m2/m3 key=value args pass through
  tools/replay.py --file scratch/replays/x.json   # just (re)build the viewer page for an existing replay

Output: scratch/replays/<loc>_s<seed>.json + .html (self-contained, opens in any browser).
"""
import argparse
import json
import pathlib
import subprocess
import sys
import webbrowser

ROOT = pathlib.Path(__file__).resolve().parent.parent
VIEWER = ROOT / "tools/viewer/viewer.html"
OUT = ROOT / "scratch/replays"
DEV_DIR = "/data/local/tmp/bsh36"


def adb(serial, *args, check=True):
    return subprocess.run(["adb", "-s", serial, *args], check=check, capture_output=True, text=True)


def build_page(replay_json: pathlib.Path) -> pathlib.Path:
    data = replay_json.read_text()
    json.loads(data)  # fail early on a truncated recording
    page = VIEWER.read_text().replace("/*REPLAY_JSON*/", data.replace("</", "<\\/"))
    page = page.replace("/*GFX_FILE_BASE*/", VIEWER.parent.resolve().as_uri() + "/")   # 3D assets (file:// pages)
    out = replay_json.with_suffix(".html")
    out.write_text(page)
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("kv", nargs="*", help="driver args, e.g. loc=Gemgrab1 hero=ShotgunGirl seed=1 ticks=6000")
    ap.add_argument("--serial", default="R52MB0DXE9B")
    ap.add_argument("--file", type=pathlib.Path, help="existing replay .json (skip running the battle)")
    ap.add_argument("--no-open", action="store_true")
    a = ap.parse_args()

    if a.file:
        replay = a.file
    else:
        kv = dict(s.split("=", 1) for s in a.kv)
        kv.setdefault("loc", "Gemgrab1")
        kv.setdefault("seed", "1")
        name = f"{kv['loc']}_s{kv['seed']}"
        dev_rec = f"{DEV_DIR}/rec_{name}.json"
        args = " ".join(f"{k}={v}" for k, v in kv.items())
        cmd = f"{DEV_DIR}/driver {DEV_DIR} {DEV_DIR}/assets m3 every=0 {args} rec={dev_rec}"
        r = adb(a.serial, "shell", f"su -c '{cmd}'", check=False)
        for line in r.stdout.splitlines():
            if any(t in line for t in ("[m2] loc=", "stopped", "speed", "[rec]", "FATAL", "unknown")):
                print(line)
        if "[rec] wrote" not in r.stdout:
            sys.exit("recording failed:\n" + r.stdout[-2000:] + r.stderr)
        OUT.mkdir(parents=True, exist_ok=True)
        replay = OUT / f"{name}.json"
        adb(a.serial, "pull", dev_rec, str(replay))
        adb(a.serial, "shell", f"rm {dev_rec}", check=False)

    page = build_page(replay)
    print(f"viewer: {page}")
    if not a.no_open:
        webbrowser.open(page.resolve().as_uri())


if __name__ == "__main__":
    main()
