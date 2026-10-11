"""Regenerate the viewer's UI sprites: tools/viewer/gfx/ui/<name>.png + index.json (+ heroes.json).

    python tools/gfx/extract_ui.py [out_dir]

Everything is rendered with sc2d.SC.render at scale 1 (game UI units), straight-alpha RGBA PNGs.
index.json: name -> {file, w, h, ox, oy, src}; (ox, oy) = pixel of the export's origin inside the PNG, so pieces cut
out of one export (health bar parts, ammo slots, score bars ...) share that export's origin and line up when drawn
at the same anchor. Text fields are not rendered (draw numbers yourself).
"""
import csv
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))   # also works under python -I
from sc2d import SC   # noqa: E402

from PIL import Image   # noqa: E402

ROOT = HERE.parents[1]
ASSETS = ROOT / "apk/extracted/assets"
OUT = Path(sys.argv[1]) if len(sys.argv) > 1 else ROOT / "tools/viewer/gfx/ui"

ITEMS = []   # (name, sc file, export name or id, render kwargs)


def item(name, sc, target=None, **kw):
    ITEMS.append((name, sc, name if target is None else target, kw))


# --- hero portraits: every Type=Hero row of characters.csv; the game appends _small / _big to IconExportName
rows = list(csv.reader(open(ASSETS / "csv_logic/characters.csv", newline="", encoding="utf-8")))
col = {h: i for i, h in enumerate(rows[0])}
heroes = {}
for r in rows[2:]:
    if len(r) > col["IconExportName"] and r[col["Type"]] == "Hero":
        heroes[r[col["Name"]]] = {"icon": r[col["IconExportName"]], "item": r[col["ItemName"]]}
icons = sorted({h["icon"] for h in heroes.values()})
for ic in icons:
    for suffix in ("_small", "_big"):
        item(ic + suffix, "ui")
item("hero_icon_colt_small2", "ui")

# --- gems: shop/currency icons (green), Gem Grab gem (purple): in-world pickup, rotating icon, HUD score icon
item("icon_resource_gem", "ui")
item("genicon_gem", "ui")
item("blob_pickup", "effects")                 # in-world gem, floats ~24 px above its origin
item("pickup_glow", "effects", frames={None: 50})   # purple ground glow (settled pulse frame)
item("gem_rotate", "effects")
item("hud_gem_icon", "ui", 8891)               # gem next to the score numbers

# --- 3D material textures: exact squares centred on the export origin
item("tex_crystal_1", "effects", box=(-16, -16, 16, 16), pad=0)
item("tex_water_tile_mine", "effects", box=(-32, -32, 32, 32), pad=0)

# --- top HUD (origin = top centre of the screen): score bars for 0..10 gems, countdown glow, timer background
for n in range(11):
    item(f"hud_score_blue_{n}", "ui", "ingame_top_right", only=("score_1", ("score_bar", "points_container")),
         frames={"score_bar": n})
    item(f"hud_score_red_{n}", "ui", "ingame_top_right", only=("score_2", ("score_bar", "points_container")),
         frames={"score_bar": n})
item("hud_score_blue_countdown", "ui", "ingame_top_right", only=("score_1", "bar_countdown"))
item("hud_score_red_countdown", "ui", "ingame_top_right", only=("score_2", "bar_countdown"))
item("hud_timer_bg", "ui", "ingame_top_right", only=("countdown_cont",))

# --- brawler badges (origin = the brawler's feet): own / teammate / enemy (csv_client/health_bars.csv)
VARIANTS = {"own": ("player_info_badge", "player_info_badge_btm"),
            "team": ("yourteam_info_badge", "teammate_info_badge_btm"),
            "enemy": ("enemy_info_badge", "enemy_info_badge_btm")}
for v, (top, btm) in VARIANTS.items():
    item(btm, "ui")                                                   # circle + super ring + emblem
    item(f"ring_{v}", "ui", btm, only=("player_circle",))             # the circle under the brawler
    item(f"ulti_ring_{v}", "ui", btm, only=("ulti_ring",))
    item(f"ulti_ready_{v}", "ui", btm, only=("ulti_ready",))
    item(f"ulti_emblem_{v}", "ui", btm, only=("special",))
    item(f"hpbar_{v}_bg", "ui", top, only=("health_bar", 0), pad=0)   # fill: squeeze x about its left edge
    item(f"hpbar_{v}_fill", "ui", top, only=("health_bar", "bar1"), pad=0)
    item(f"hpbar_{v}_lag", "ui", top, only=("health_bar", "bar2"), pad=0)   # damage-trail bar behind the fill
    item(f"gemcarry_{v}", "ui", top, only=("stars", 3))               # gem icon with the carried count next to it
item("arrow_own", "ui", "player_info_badge", only=("arrow",))         # green arrow above your own brawler

# --- ammo bars of the own badge: reload = 3 slots, reload_2 = 4 slots, reload_3 = 1 slot (bg, full, reload fill)
for grp, n, fill in (("reload", 3, (1, 98)), ("reload_2", 4, (1, 98)), ("reload_3", 1, (1, 99))):
    for k in range(1, n + 1):
        a = f"attack_{k}"
        base = f"ammo{n}_s{k}"
        item(base + "_bg", "ui", "player_info_badge", only=(grp, a, 0), pad=0)
        item(base + "_full", "ui", "player_info_badge", only=(grp, a, (0, 1 if n == 1 else 2)), frames={a: 99}, pad=0)
        item(base + "_fill", "ui", "player_info_badge", only=(grp, a, fill[0]), frames={a: fill[1]}, pad=0)

# --- super button (frame 0 = charging, 1 = ready, 2 = ready + glow) and its charge ring
item("ulti_stick", "ui")
item("ulti_stick_ready", "ui", "ulti_stick", frame=1)
item("ulti_stick_ready_glow", "ui", "ulti_stick", frame=2)
for p in (0, 25, 50, 75, 99):
    item(f"ulti_stick_{p}", "ui", "ulti_stick", only=("bar",), frames={"bar": p, "bar1": p, "bar2": p})
item("hero_stat_icon_super", "ui")             # small skull
item("skull_white", "ui", "skull_white_atlasgenerator_texture_luminance_alpha")

# --- death / respawn effects (frame = settled part of the animation) and rings, shadows under characters
item("deathskull", "effects", frame=20)
item("gen_deathskull_fx", "effects", frame=40)
for c in ("red", "blue", "green"):
    item(f"hero_spawn_{c}", "effects", frame=60)
item("player_shadow", "ui")
item("team_indicator", "ui")
item("second_ability_ring", "ui")
item("shadow_normal_circle", "effects")
item("projectile_shadow", "effects")
item("projectile_shadow_big", "effects")
item("target_ring_feet", "effects", frame=40)
item("target_ring", "effects", frame=40)


def src_of(f, target, kw):
    s = f"{f}.sc:" + (f"#{target}" if isinstance(target, int) else target)
    if kw.get("only"):
        s += "/" + "/".join(str(x) for x in kw["only"])
    if kw.get("frame"):
        s += f"@{kw['frame']}"
    if kw.get("frames"):
        s += " frames=" + json.dumps({str(k): v for k, v in kw["frames"].items()})
    if kw.get("box"):
        s += " box=" + json.dumps(kw["box"])
    return s


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    scs, index = {}, {}
    for name, f, target, kw in ITEMS:
        if f not in scs:
            scs[f] = SC(ASSETS / f"sc/{f}.sc", ASSETS / f"sc/{f}_tex.sc")
        try:
            img, (ox, oy) = scs[f].render(target, **kw)
        except KeyError:
            img = None
        if img is None:
            print("skipped (nothing to draw):", name)
            continue
        Image.fromarray(img).save(OUT / f"{name}.png", optimize=True)
        index[name] = {"file": f"{name}.png", "w": img.shape[1], "h": img.shape[0], "ox": int(ox), "oy": int(oy),
                       "src": src_of(f, target, kw)}
    with open(OUT / "index.json", "w") as fh:
        json.dump(index, fh, indent=1)
    with open(OUT / "heroes.json", "w") as fh:   # characters.csv Name -> icon export (+ _small / _big) and ItemName
        json.dump(heroes, fh, indent=1)
    print(f"{len(index)} sprites -> {OUT}")


if __name__ == "__main__":
    main()
