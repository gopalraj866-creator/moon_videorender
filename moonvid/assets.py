"""Texture + procedural-asset pipeline.

Real NASA-derived textures are downloaded from GitHub raw URLs (public-domain
Blue Marble / planetary maps) and cached in ./assets. Every download has a
fallback so the render never fails because one URL is down.
"""
import os, io, sys, json, urllib.request, urllib.error
import numpy as np, cv2
from PIL import Image

ASSET_DIR = os.environ.get("ASSET_DIR", os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "assets"))
os.makedirs(ASSET_DIR, exist_ok=True)

TJS = "https://raw.githubusercontent.com/mrdoob/three.js/r160/examples/textures/planets/"
WGL = "https://raw.githubusercontent.com/turban/webgl-earth/master/images/"
TXP = "https://raw.githubusercontent.com/jeromeetienne/threex.planets/master/images/"
SSS = "https://www.solarsystemscope.com/textures/download/"
FONT_URL = "https://raw.githubusercontent.com/google/fonts/main/ofl/montserrat/Montserrat%5Bwght%5D.ttf"

# name -> list of candidate URLs (first that works wins)
SOURCES = {
    "earth_day":    [WGL + "2_no_clouds_4k.jpg", TJS + "earth_atmos_2048.jpg"],
    "earth_clouds": [WGL + "fair_clouds_4k.png", TJS + "earth_clouds_1024.png"],
    "earth_water":  [WGL + "water_4k.png", TJS + "earth_specular_2048.jpg"],
    "earth_bump":   [WGL + "elev_bump_4k.jpg"],
    "earth_lights": [TJS + "earth_lights_2048.png"],
    "moon":         [SSS + "8k_moon.jpg", TJS + "moon_1024.jpg", TXP + "moonmap1k.jpg"],
    "mars":         [TXP + "marsmap1k.jpg"],
    "jupiter":      [TXP + "jupitermap.jpg"],
    "saturn":       [TXP + "saturnmap.jpg"],
    "saturn_ring":  [TXP + "saturnringcolor.jpg"],
    "venus":        [TXP + "venusmap.jpg"],
    "mercury":      [TXP + "mercurymap.jpg"],
    "uranus":       [TXP + "uranusmap.jpg"],
    "neptune":      [TXP + "neptunemap.jpg"],
}
MIN_WIDTH = {"moon": 512}   # reject tiny / bogus downloads

CREDITS = """Textures & data
- Earth day map, clouds, ocean mask, elevation: webgl-earth (turban) - derived from NASA Visible Earth / Natural Earth.
- Earth night lights, Moon map (three.js examples): NASA Visible Earth / NASA LRO derived, public domain.
- Planet maps: threex.planets (J. Etienne) - NASA/JPL derived.
- Optional 8k Moon: Solar System Scope (CC BY 4.0) if reachable.
- Font: Montserrat (SIL OFL).
Everything else (stars, Milky Way, craters, city, forest, ocean, sound, music) is procedural.
"""


def _fetch(url, timeout=60):
    req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0 (X11; Linux x86_64) MoonVid/1.0"})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return r.read()


def _local(name):
    return os.path.join(ASSET_DIR, name + ".png")


def download_all(verbose=True):
    """Download every texture once (idempotent). Returns {name: source_url}."""
    meta_path = os.path.join(ASSET_DIR, "sources.json")
    meta = json.load(open(meta_path)) if os.path.exists(meta_path) else {}
    for name, urls in SOURCES.items():
        if os.path.exists(_local(name)) and name in meta:
            continue
        ok = False
        for url in urls:
            try:
                raw = _fetch(url)
                im = Image.open(io.BytesIO(raw))
                im.load()
                if im.width < MIN_WIDTH.get(name, 64):
                    raise ValueError("too small %s" % (im.size,))
                if name in ("earth_clouds",) and im.mode in ("RGBA", "LA", "P"):
                    im = im.convert("RGBA")
                else:
                    im = im.convert("RGB")
                im.save(_local(name), optimize=False)
                meta[name] = url
                ok = True
                if verbose:
                    print("  [tex] %-13s %s  %s" % (name, im.size, url.split('/')[-1]))
                break
            except Exception as e:  # noqa
                if verbose:
                    print("  [tex] %-13s FAILED %s (%s)" % (name, url.split('/')[-1], e), file=sys.stderr)
        if not ok:
            meta[name] = None
    # font
    fp = os.path.join(ASSET_DIR, "Montserrat.ttf")
    if not os.path.exists(fp):
        try:
            open(fp, "wb").write(_fetch(FONT_URL))
        except Exception as e:  # noqa
            print("  [font] download failed:", e, file=sys.stderr)
    json.dump(meta, open(meta_path, "w"), indent=1)
    open(os.path.join(ASSET_DIR, "CREDITS.txt"), "w").write(CREDITS)
    return meta


# ----------------------------------------------------------------------------
def fractal(w, h, oct_start=2, octs=8, seed=0, persistence=0.55):
    """Horizontally-tileable fractal noise in [0,1]."""
    rng = np.random.default_rng(seed)
    out = np.zeros((h, w), np.float32)
    amp, tot = 1.0, 0.0
    for o in range(octs):
        gw = 2 ** (oct_start + o + 1)
        gh = max(gw // 2, 2)
        g = rng.standard_normal((gh, gw)).astype(np.float32)
        g = np.pad(g, ((0, 0), (2, 2)), mode="wrap")
        ww = int(round(w * (gw + 4) / gw))
        up = cv2.resize(g, (ww, h), interpolation=cv2.INTER_CUBIC)
        x0 = int(round(2 * w / gw))
        out += up[:, x0:x0 + w] * amp
        tot += amp
        amp *= persistence
    out /= tot
    lo, hi = np.percentile(out, 1), np.percentile(out, 99)
    return np.clip((out - lo) / (hi - lo + 1e-9), 0, 1)


def make_moon_detail(w=4096, h=2048, seed=3):
    p = os.path.join(ASSET_DIR, "moon_detail.npy")
    if os.path.exists(p):
        return np.load(p)
    rng = np.random.default_rng(seed)
    hmap = (fractal(w, h, 3, 8, seed, 0.6) - 0.5) * 0.35
    n = 5200
    lon = rng.uniform(0, w, n)
    lat = np.arcsin(rng.uniform(-0.97, 0.97, n))
    rad_deg = np.exp(rng.uniform(np.log(0.12), np.log(3.2), n))
    for i in range(n):
        ry = max(rad_deg[i] / 180.0 * h, 1.5)
        rx = ry / max(np.cos(lat[i]), 0.25)
        cy = (0.5 - lat[i] / np.pi) * h
        cx = lon[i]
        R = int(max(rx, ry) * 1.7) + 2
        y0, y1 = int(cy) - R, int(cy) + R + 1
        if y0 < 0 or y1 > h:
            continue
        xs = (np.arange(int(cx) - R, int(cx) + R + 1)) % w
        yy, xx = np.mgrid[y0:y1, 0:len(xs)]
        dx = (xx - R) / rx
        dy = (yy - cy) / ry
        r = np.sqrt(dx * dx + dy * dy)
        depth = 0.55 if ry > 4 else 0.35
        prof = np.where(r < 1, -depth * (1 - r ** 2), 0.28 * np.exp(-(r - 1) * 5.0))
        hmap[y0:y1, xs] += prof.astype(np.float32) * (0.6 if ry > 4 else 0.9)
    # hillshade from upper-left
    gy, gx = np.gradient(hmap)
    shade = np.clip(0.5 + (-gx - gy) * 18.0, 0, 1)
    det = (0.5 + (shade - 0.5) * 1.0).astype(np.float32)
    det = np.clip(det, 0, 1)
    np.save(p, det.astype(np.float16))
    return det


def make_earth_detail(w=4096, h=2048):
    p = os.path.join(ASSET_DIR, "earth_detail.npy")
    if os.path.exists(p):
        return np.load(p)
    d = fractal(w, h, 4, 8, 11, 0.62).astype(np.float16)
    np.save(p, d)
    return d


def galactic_basis():
    gn = np.array([0.18, 0.88, 0.44]); gn /= np.linalg.norm(gn)
    ga = np.cross(gn, [0, 0, 1.0]); ga /= np.linalg.norm(ga)
    gb = np.cross(gn, ga)
    return gn.astype(np.float32), ga.astype(np.float32), gb.astype(np.float32)


def make_stars(n=30000, seed=7):
    p = os.path.join(ASSET_DIR, "stars.npz")
    if os.path.exists(p):
        z = np.load(p)
        return z["dir"], z["inten"], z["col"]
    rng = np.random.default_rng(seed)
    gn = np.array([0.18, 0.88, 0.44]); gn /= np.linalg.norm(gn)
    ga = np.cross(gn, [0, 0, 1.0]); ga /= np.linalg.norm(ga)
    gb = np.cross(gn, ga)
    v = rng.standard_normal((n, 3))
    v /= np.linalg.norm(v, axis=1, keepdims=True)
    # concentrate a share near the galactic plane
    m = rng.random(n) < 0.55
    lat = rng.normal(0, 0.16, m.sum())
    lon = rng.uniform(0, 2 * np.pi, m.sum())
    band = (np.cos(lat)[:, None] * (np.cos(lon)[:, None] * ga + np.sin(lon)[:, None] * gb) + np.sin(lat)[:, None] * gn)
    v[m] = band
    mag = 6.6 - rng.exponential(1.15, n) * 1.0
    mag = np.clip(mag, -0.5, 6.6)
    inten = np.clip(10 ** (-0.4 * (mag - 5.4)) * 0.22, 0.04, 1.6)
    temp = rng.random(n)
    col = np.stack([0.75 + 0.25 * temp, 0.85 + 0.1 * (1 - np.abs(temp - 0.5) * 2) * 1.0, 1.0 - 0.35 * temp], 1)
    np.savez_compressed(p, dir=v.astype(np.float32), inten=inten.astype(np.float32), col=col.astype(np.float32))
    return v.astype(np.float32), inten.astype(np.float32), col.astype(np.float32)


def make_milkyway(w=1024, h=512):
    p = os.path.join(ASSET_DIR, "milkyway.npy")
    if os.path.exists(p):
        return np.load(p)
    n1 = fractal(w, h, 2, 7, 21, 0.6)
    n2 = fractal(w, h, 3, 7, 22, 0.55)
    lat = (0.5 - (np.arange(h) + 0.5) / h) * np.pi
    band = np.exp(-(lat / 0.22) ** 2)[:, None]
    core = np.exp(-(lat / 0.09) ** 2)[:, None]
    mw = band * (0.35 + 0.9 * n1) + core * 0.6 * n1
    dust = np.clip((n2 - 0.52) * 3.0, 0, 1) * band
    mw = np.clip(mw * (1 - 0.8 * dust), 0, None)
    mw = (mw / mw.max()).astype(np.float32)
    np.save(p, mw.astype(np.float16))
    return mw


# ----------------------------------------------------------------------------
def _pyramid(arr, min_w=128):
    pyr = [arr]
    while pyr[-1].shape[1] // 2 >= min_w:
        a = pyr[-1]
        pyr.append(cv2.resize(a, (a.shape[1] // 2, a.shape[0] // 2), interpolation=cv2.INTER_AREA))
    return pyr


def _load_rgb(name):
    p = _local(name)
    if not os.path.exists(p):
        return None
    return np.array(Image.open(p).convert("RGB"))


def _proc_planet(kind, w=1024, h=512):
    """Procedural fallback texture for planets whose download failed."""
    f = fractal(w, h, 2, 7, hash(kind) % 1000, 0.55)
    lat = np.linspace(-1, 1, h)[:, None] * np.ones((1, w))
    if kind in ("jupiter", "saturn"):
        b = 0.5 + 0.5 * np.sin(lat * 14 + f * 3)
        c0, c1 = ((0.85, 0.7, 0.5), (0.6, 0.4, 0.3)) if kind == "jupiter" else ((0.9, 0.8, 0.6), (0.7, 0.6, 0.45))
    elif kind == "mars":
        b = f; c0, c1 = (0.8, 0.45, 0.25), (0.45, 0.22, 0.12)
    elif kind in ("uranus", "neptune"):
        b = 0.5 + 0.1 * np.sin(lat * 6) + 0.2 * f
        c0, c1 = ((0.6, 0.85, 0.9), (0.5, 0.75, 0.85)) if kind == "uranus" else ((0.3, 0.45, 0.9), (0.2, 0.3, 0.75))
    elif kind == "venus":
        b = f; c0, c1 = (0.95, 0.85, 0.6), (0.8, 0.65, 0.4)
    else:
        b = f; c0, c1 = (0.6, 0.58, 0.55), (0.3, 0.29, 0.28)
    c0, c1 = np.array(c0), np.array(c1)
    img = c1[None, None] + (c0 - c1)[None, None] * b[..., None]
    return (np.clip(img, 0, 1) * 255).astype(np.uint8)


class Assets:
    def __init__(self):
        download_all(verbose=False)
        self.pyr = {}
        for name in ["earth_day", "earth_lights", "moon", "mars", "jupiter", "saturn", "venus", "mercury", "uranus", "neptune"]:
            a = _load_rgb(name)
            if a is None:
                a = _proc_planet(name) if name not in ("earth_day", "earth_lights", "moon") else None
            if a is None:
                raise RuntimeError("Required texture missing: %s (check network / ASSET_DIR)" % name)
            self.pyr[name] = _pyramid(np.ascontiguousarray(a))
        # relief shading baked from elevation map
        day = self.pyr["earth_day"][0]
        bump = _load_rgb("earth_bump")
        if bump is not None:
            b = cv2.resize(bump[..., 0], (day.shape[1], day.shape[0]), interpolation=cv2.INTER_CUBIC).astype(np.float32) / 255
            gy, gx = np.gradient(cv2.GaussianBlur(b, (0, 0), 1.2))
            relief = np.clip(0.5 + (-gx - gy) * 9.0, 0, 1)
            relief[b < 0.02] = 0.5
        else:
            relief = np.full(day.shape[:2], 0.5, np.float32)
        self.pyr["earth_relief"] = _pyramid((relief * 255).astype(np.uint8))
        # ocean mask (1 = ocean). Detect polarity from the day map (ocean is blue/dark).
        wat = _load_rgb("earth_water")
        if wat is None:
            wat = np.zeros_like(day)
        wat = cv2.resize(wat[..., 0], (day.shape[1], day.shape[0]), interpolation=cv2.INTER_AREA)
        dayg = cv2.resize(day, (wat.shape[1], wat.shape[0]), interpolation=cv2.INTER_AREA).astype(np.float32)
        blueish = (dayg[..., 2] > dayg[..., 0] * 1.25)
        corr = np.corrcoef(wat.ravel()[::97].astype(np.float32), blueish.ravel()[::97].astype(np.float32))[0, 1]
        if np.isnan(corr):
            corr = 1.0
        if corr < 0:
            wat = 255 - wat
        self.pyr["earth_ocean"] = _pyramid(np.ascontiguousarray(wat))
        # clouds -> single channel density
        cp = _local("earth_clouds")
        if os.path.exists(cp):
            im = Image.open(cp)
            if im.mode == "RGBA" and np.array(im)[..., 3].min() < 250:
                c = np.array(im)[..., 3]
            else:
                c = np.array(im.convert("L"))
        else:
            c = (fractal(2048, 1024, 2, 8, 5) * 255).astype(np.uint8)
        c = cv2.GaussianBlur(c, (0, 0), 0.6)
        self.pyr["earth_clouds"] = _pyramid(np.ascontiguousarray(c))
        # night lights -> luminance-weighted RGB, a touch blurred for glow
        lt = self.pyr["earth_lights"][0]
        lt2 = cv2.GaussianBlur(lt, (0, 0), 1.0)
        self.pyr["earth_lights"] = _pyramid(np.ascontiguousarray(np.maximum(lt, lt2)))
        # detail maps
        self.pyr["moon_detail"] = _pyramid((make_moon_detail().astype(np.float32) * 255).astype(np.uint8))
        self.pyr["earth_detail"] = _pyramid((make_earth_detail().astype(np.float32) * 255).astype(np.uint8))
        # ring (1-D radial profile from texture, alpha from luminance pattern)
        ring = _load_rgb("saturn_ring")
        if ring is not None:
            prof = ring.mean(axis=0 if ring.shape[0] < ring.shape[1] else 1) if ring.ndim == 3 else ring
            row = ring[ring.shape[0] // 2] if ring.shape[1] > ring.shape[0] else ring[:, ring.shape[1] // 2]
            self.ring_col = row.astype(np.float32) / 255
        else:
            r = np.linspace(0, 1, 512)
            v = 0.55 + 0.35 * np.sin(r * 60) * np.sin(r * 17)
            self.ring_col = np.stack([v * 0.9, v * 0.8, v * 0.65], 1).astype(np.float32)
        self.stars = make_stars()
        self.milky = make_milkyway().astype(np.float32)
        # fractal for the Sun's surface
        self.sun_noise = fractal(1024, 512, 3, 7, 41, 0.6)
        self.font_path = os.path.join(ASSET_DIR, "Montserrat.ttf")


_ASSETS = None


def get_assets():
    global _ASSETS
    if _ASSETS is None:
        _ASSETS = Assets()
    return _ASSETS


if __name__ == "__main__":
    print("Downloading textures into", ASSET_DIR)
    m = download_all()
    get_assets()
    print("OK", {k: bool(v) for k, v in m.items()})
