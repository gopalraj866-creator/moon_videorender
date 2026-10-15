"""All shot functions. Each takes a Ctx (c.t global seconds, c.lt local seconds, c.dur) and returns a float image."""
import math
import numpy as np, cv2
from . import engine as E, ground as G
from .engine import F32, smooth, kf, norm, Layer, px
from .assets import get_assets, fractal

RW = lambda: E.RW
RH = lambda: E.RH

SUN = norm([-0.55, 0.28, 0.78])
EARTH_AXIS = E.axis_from_tilt(23.4, 25)
MOON_POS = np.array([8.6, 0.6, -3.2])
MOON_R = 0.273 * 1.0
ORIGIN = np.zeros(3)


class Ctx:
    def __init__(self, t, lt, dur):
        self.t, self.lt, self.dur = t, lt, dur
        self.u = lt / max(dur, 1e-6)


def spin_to(lon_deg, rate=0.0, t=0.0):
    return -math.radians(lon_deg) + rate * t


def space_base(cam, star=1.0, mw=1.0):
    return E.new_frame(cam, star_gain=star, mw_gain=mw)


def earth_moon(img, cam, c, spin, sun=SUN, moon=True, moon_vis=1.0, moon_pos=MOON_POS, lights=1.0, axis=EARTH_AXIS, Rm=MOON_R):
    items = []
    de = np.linalg.norm(ORIGIN - cam.pos)
    dm = np.linalg.norm(moon_pos - cam.pos)
    order = [("e", de)] + ([("m", dm)] if moon else [])
    order.sort(key=lambda x: -x[1])
    for k, _ in order:
        if k == "e":
            E.draw_earth(img, cam, ORIGIN, 1.0, spin, axis, sun, cloud_shift=c.t * 0.0007, lights=lights)
        else:
            E.draw_moon(img, cam, moon_pos, Rm, sun, earth=(ORIGIN, 1.0), vis=moon_vis)


def shake(t, t0, amp=0.04, dur=0.9):
    if t < t0 or t > t0 + dur:
        return np.zeros(3)
    k = (1 - (t - t0) / dur) ** 2
    return np.array([math.sin(t * 90) * amp * k, math.cos(t * 71) * amp * k, math.sin(t * 53) * amp * k * 0.5])


# ============================================================ SCENE 1 / 2
def orbit_wide(c):
    lt = c.lt
    pos = kf(lt, [(0, (-3.2, 1.4, 10.5)), (7, (-0.8, 0.8, 8.6))])
    tgt = kf(lt, [(0, (3.6, 0.0, -0.8)), (7, (5.6, 0.3, -1.6))])
    cam = E.Camera(pos, tgt, kf(lt, [(0, 56), (7, 38)]))
    img = space_base(cam)
    vis = 0.12 if 3.35 < lt < 3.52 else 1.0
    earth_moon(img, cam, c, spin_to(80, 0.05, c.lt), moon_vis=vis)
    return img


def earth_night_close(c, lon=100, flip=1.0):
    lt = c.lt
    cam = E.Camera(kf(lt, [(0, (0.9 * flip, 0.5, 2.9)), (6, (-0.5 * flip, 0.25, 2.5))]), (-0.15 * flip, 0.0, 0.0), 44)
    img = space_base(cam, 1.0)
    sun = norm([1.0 * flip, 0.18, -0.12])
    E.draw_earth(img, cam, ORIGIN, 1.0, spin_to(lon, 0.03, c.lt), EARTH_AXIS, sun, cloud_shift=c.t * 0.0007, lights=1.25)
    return img


def moon_geometry(dist, ang=0.30):
    u0 = norm(MOON_POS - ORIGIN)
    side = norm(np.cross([0, 1, 0], u0))
    d = norm(u0 * math.cos(ang) + side * math.sin(ang) + np.array([0, 0.12, 0]))
    return MOON_POS + d * dist


def moon_push(c, vanished=False):
    lt = c.lt
    if not vanished:
        dist = kf(lt, [(0, 15.0), (7, 4.6)], smooth)
        cam_pos = moon_geometry(dist)
        sh = np.zeros(3)
    else:
        dist = kf(lt, [(0, 4.6), (7, 4.1)], smooth)
        cam_pos = moon_geometry(dist) + shake(c.t, 20.0)
        sh = None
    tgt = MOON_POS + np.array([0, 0.0, 0])
    cam = E.Camera(cam_pos, tgt, 38)
    img = space_base(cam)
    sun = norm(-cam.f * 0.85 + cam.r * 0.35 + cam.u * 0.25)
    earth_moon(img, cam, c, spin_to(75, 0.05, c.lt), sun=sun, moon=not vanished)
    return img


def empty_orbit(c):
    lt = c.lt
    az = kf(lt, [(0, -0.2), (8, 0.35)])
    r = 15.0
    cam = E.Camera((math.sin(az) * r, 5.5, math.cos(az) * r), (0.0, 0, 0), 50)
    img = space_base(cam)
    E.draw_earth(img, cam, ORIGIN, 1.0, spin_to(70, 0.06, c.lt), EARTH_AXIS, SUN, cloud_shift=c.t * 0.0007)
    L = Layer()
    th = np.linspace(0, 2 * np.pi, 260)
    pts = [cam.project(np.array([5.4 * math.cos(a), 0.0, 5.4 * math.sin(a)])) for a in th]
    for i in range(0, len(pts) - 1, 2):
        if pts[i][2] > 0 and pts[i + 1][2] > 0:
            L.line(pts[i][:2], pts[i + 1][:2], (0.5, 0.75, 1.0), 3, 0.75)
    L.over(img, glow=0.8)
    # label near the far-right side of the ring
    px_, py_, _ = cam.project(np.array([5.4 * math.cos(1.1), 0.0, 5.4 * math.sin(1.1)]))
    E.small_text(img, "THE MOON'S ORBIT: EMPTY", px_, py_ - px(46), 34, (0.7, 0.88, 1.0), smooth(lt / 1.5), anchor="c", glow=0.5)
    return img


def earth_sun_orbit(c, moon=False, label=True):
    lt = c.lt
    sunpos = np.array([-70.0, 4.0, -26.0])
    cam = E.Camera(kf(lt, [(0, (4.0, 1.8, 9.0)), (c.dur, (2.4, 1.2, 8.0))]), (-5.0, 0.0, -2.0), 56)
    img = space_base(cam, 1.0)
    sdir = norm([-0.6, 0.15, 0.78])
    E.draw_sun(img, cam, sunpos, 6.0, c.t)
    E.draw_earth(img, cam, ORIGIN, 1.0, spin_to(70, 0.1, c.lt), EARTH_AXIS, sdir, cloud_shift=c.t * 0.0007)
    if moon:
        E.draw_moon(img, cam, MOON_POS * 0.5, MOON_R, sdir, earth=(ORIGIN, 1.0))
    # orbit path: circle around the Sun that passes through Earth
    Lr = Layer()
    centre = sunpos
    rad = np.linalg.norm(ORIGIN - sunpos)
    a0 = math.atan2(ORIGIN[2] - centre[2], ORIGIN[0] - centre[0])
    pts = []
    for i in range(-40, 60):
        a = a0 + i * 0.0035
        pts.append(cam.project(np.array([centre[0] + rad * math.cos(a), 0.0, centre[2] + rad * math.sin(a)])))
    for i in range(len(pts) - 1):
        k = 1 - abs(i - 40) / 60.0
        if pts[i][2] > 0 and pts[i + 1][2] > 0:
            Lr.line(pts[i][:2], pts[i + 1][:2], (0.5, 0.8, 1.0), 2.5, max(0.0, k) * 0.7)
    Lr.over(img, glow=0.6)
    return img


def earth_hero(c, moon=True, drift=1.0):
    lt = c.lt
    az = kf(lt, [(0, 0.35), (c.dur, -0.05)])
    cam = E.Camera(kf(lt, [(0, (-2.0, 1.2, 12.0)), (c.dur, (-0.5, 0.8, 11.0))]), kf(lt, [(0, (3.7, 0, -1.0)), (c.dur, (4.1, 0.1, -1.2))]), kf(lt, [(0, 52), (c.dur, 48)]))
    img = space_base(cam)
    earth_moon(img, cam, c, spin_to(20, 0.05, c.lt), moon=moon)
    return img


# ============================================================ GROUND
def ground_shot(c, kind, hz, moon_keys, yaw=0.0, pan=(0, 0), star=1.0, mw=1.0, sky_speed=0.0, moon_xy=(0.7, 0.24), fov=62.0):
    mk = float(kf(c.lt, moon_keys))
    pan_px = float(kf(c.lt, [(0, pan[0] * E.RW), (c.dur, pan[1] * E.RW)]))
    # sky time-lapse rotation about celestial pole
    pole = norm([0.0, 0.62, -0.78])
    rot = E.rot_axis(pole, sky_speed * c.t) if sky_speed else None
    return G.render({"yaw": yaw}, kind, hz, c.t, mk, star, mw, pan_px=pan_px, sky_rot=rot, moon_xy=moon_xy, lt_for_blink=c.t, fov=fov)


# ============================================================ TIDES (top view)
def _bulge_pts(cx, cy, Rpx, am, as_, thm, ths, k, n=181):
    th = np.linspace(0, 2 * np.pi, n)
    r = Rpx * (1.06 + k * (am * np.cos(2 * (th - thm)) + as_ * np.cos(2 * (th - ths))))
    return np.stack([cx + r * np.cos(th), cy - r * np.sin(th)], 1), th


def tides_top(c, mode="moon"):
    lt = c.lt
    D, R = 5.8, 1.55
    cam = E.Camera((0, 16.5, 0.001), (0.0, 0, -0.3), 52, up=(0, 0, -1))
    ths = math.pi  # Sun on the left
    am, as_ = 1.0, 0.0
    moon_vis = 1.0
    thm = 0.0
    show_sun = False
    label = None
    if mode == "moon":
        thm = kf(lt, [(0, 0.25), (10, 0.9)])
        am, as_ = 1.0, 0.0
    elif mode == "gravity":
        thm = 0.35
    elif mode == "both":
        show_sun = True
        thm = kf(lt, [(0, math.pi), (4.5, math.pi), (10, math.pi * 1.5)])  # new moon (spring) -> first-quarter (neap)
        am, as_ = 1.0, 0.46
    elif mode == "vanish":
        show_sun = True
        thm = math.pi
        am = float(kf(lt, [(0, 1), (3.0, 1), (5.2, 0)]))
        moon_vis = float(kf(lt, [(0, 1), (3.0, 1), (4.0, 0)]))
        as_ = 0.46
    elif mode == "solar":
        show_sun = True
        thm = math.pi
        am, as_, moon_vis = 0.0, 0.46, 0.0
    img = space_base(cam, 0.8, 0.5)
    Rpx = abs(cam.project(np.array([R, 0, 0]))[0] - cam.project(ORIGIN)[0])
    cx, cy, _ = cam.project(ORIGIN)
    pts, th = _bulge_pts(cx, cy, Rpx, am, as_, thm, ths, 0.19)
    sea = Layer()
    sea.polygon([tuple(p) for p in pts], (0.18, 0.5, 1.0), 0.45)
    sea.polyline([tuple(p) for p in pts] + [tuple(pts[0])], (0.55, 0.85, 1.0), 4, 0.95)
    sea.over(img, glow=0.9)
    spin = 0.4 + lt * 0.42
    E.draw_earth(img, cam, ORIGIN, R, spin, np.array([0, 1.0, 0]), norm([-1.0, 0.7, 0.0]), cloud_shift=c.t * 0.0007, atmos=0.5)
    if mode != "solar" and moon_vis > 0:
        mp = np.array([D * math.cos(thm), 0, -D * math.sin(thm)])
        E.draw_moon(img, cam, mp, 0.45, norm([-1.0, 0.7, 0.0]), earth=(ORIGIN, R), vis=moon_vis)
    # orbit ring + sun marker
    L = Layer()
    for i in range(0, 120, 2):
        a0, a1 = i / 120 * 2 * math.pi, (i + 1) / 120 * 2 * math.pi
        p0 = cam.project(np.array([D * math.cos(a0), 0, -D * math.sin(a0)]))[:2]
        p1 = cam.project(np.array([D * math.cos(a1), 0, -D * math.sin(a1)]))[:2]
        L.line(p0, p1, (0.6, 0.7, 0.9), 2, 0.35)
    L.over(img)
    if show_sun or mode == "gravity":
        E.radial_glow(img, px(40), RH() / 2, px(420), (1.0, 0.7, 0.35), 0.8, 2.2, core=0.15)
        E.small_text(img, "TO THE SUN", px(140), RH() / 2 + px(110), 30, (1.0, 0.85, 0.6), 1.0, anchor="c")
    # tide labels at bulge tips
    if am + as_ > 0.01 and mode != "gravity":
        ang_hi = thm if am >= as_ else ths
        for a in (ang_hi, ang_hi + math.pi):
            rr = Rpx * (1.06 + 0.19 * (am * math.cos(2 * (a - thm)) + as_ * math.cos(2 * (a - ths)))) + px(34)
            E.small_text(img, "HIGH TIDE", cx + rr * math.cos(a) + (px(60) if math.cos(a) > 0 else -px(60)), cy - rr * math.sin(a), 28, (0.7, 0.9, 1.0), smooth(lt / 1.2), anchor="c", glow=0.4)
        for a in (ang_hi + math.pi / 2, ang_hi - math.pi / 2):
            rr = Rpx * 1.1 + px(34)
            E.small_text(img, "LOW TIDE", cx + rr * math.cos(a), cy - rr * math.sin(a) + (-px(10) if math.sin(a) > 0 else px(10)), 26, (0.55, 0.7, 0.9), smooth(lt / 1.2) * 0.9, anchor="c")
    # spinning observer marker
    obs = spin * -1.0 + math.pi / 2
    mx, my = cx + Rpx * 1.02 * math.cos(spin + 0.6), cy - Rpx * 1.02 * math.sin(spin + 0.6)
    L2 = Layer()
    L2.ellipse(mx, my, px(9), px(9), (1.0, 0.85, 0.2), 0, 1.0, fill=True, fill_alpha=1.0)
    L2.ellipse(mx, my, px(17), px(17), (1.0, 0.85, 0.2), 3, 0.6)
    L2.over(img, glow=0.8)
    if mode == "gravity":
        rows = ((-0.5, 0.89, 190, "STRONGEST PULL", 1), (0.0, 0.0, 140, "AVERAGE PULL", 1), (0.5, -0.89, 95, "WEAKEST PULL", -1))
        for k, (ry, sx0, ln, lab, side) in enumerate(rows):
            y = cy + ry * Rpx
            x0 = cx + sx0 * Rpx
            L3 = Layer()
            L3.arrow((x0, y), (x0 + px(ln), y), (1.0, 0.8, 0.3), 6, 0.95, 24)
            L3.over(img, glow=0.5)
            a_ = smooth((lt - 1.0 - 0.8 * k) / 0.8)
            if side > 0:
                E.small_text(img, lab, x0 + px(ln) + px(150), y, 26, (1.0, 0.9, 0.6), a_, anchor="c", weight="Bold")
            else:
                E.small_text(img, lab, cx - Rpx - px(150), y, 26, (1.0, 0.9, 0.6), a_, anchor="c", weight="Bold")
    if mode == "both":
        spring = math.cos(2 * (thm - ths))
        lab = "SPRING TIDE: SUN + MOON ALIGN" if spring > 0.55 else ("NEAP TIDE: PULLS AT RIGHT ANGLES" if spring < -0.55 else "")
        if lab:
            E.small_text(img, lab, RW() / 2, px(110), 40, (1.0, 0.95, 0.8) if spring > 0 else (0.75, 0.9, 1.0), 1.0, anchor="c", glow=0.6, weight="ExtraBold")
    return img


# ============================================================ coast (2D)
def _beach_panel(w, h, level, t, label, sub, amp_lo, amp_hi, mark_hi, mark_lo):
    ys, xs = np.mgrid[0:h, 0:w].astype(F32)
    sky = np.stack([0.10 + 0.55 * (ys / h) ** 2.2, 0.12 + 0.30 * (ys / h) ** 1.6, 0.28 + 0.15 * (ys / h)], -1)
    img = sky.astype(F32)
    hor = 0.40 * h
    # sun glow
    img += np.exp(-(((xs - w * 0.25) ** 2 + (ys - hor) ** 2) / (2 * (0.15 * w) ** 2)))[..., None] * np.array([0.7, 0.35, 0.15], F32)
    yw = (amp_lo + (amp_hi - amp_lo) * (1 - level)) * h  # pixel y of the water line (lower number = higher tide)
    yw = float(yw)
    S = h * 0.92 - (xs / w) * h * 0.58
    wob = np.sin(xs * 0.03 + t * 2.0) * 2.0 * E.SC + np.sin(xs * 0.11 - t * 3.1) * 0.8 * E.SC
    water = (ys >= yw + wob) & (ys < S)
    sea = (ys >= hor) & (ys < S)
    depth = np.clip((ys - hor) / (h - hor), 0, 1)
    seacol = np.stack([0.05 + 0.10 * (1 - depth), 0.16 + 0.12 * (1 - depth), 0.28 + 0.12 * (1 - depth)], -1)
    spark = (np.sin(xs * 0.4 + t * 4 + ys * 0.2) * np.sin(xs * 0.13 - t * 2.3) > 0.88)[..., None] * np.array([0.5, 0.4, 0.3], F32)
    # far water body (always) from horizon to waterline region
    far = (ys >= hor) & (ys < yw + wob)
    img = np.where(far[..., None], seacol + spark * 0.5, img)
    img = np.where(water[..., None], seacol * 1.05 + spark, img)
    # sand
    sand_mask = ys >= S
    n = fractal(64, 64, 3, 4, 3)
    nz = cv2.resize(n, (w, h), interpolation=cv2.INTER_LINEAR)
    sand = np.stack([0.50 + 0.14 * nz, 0.40 + 0.12 * nz, 0.28 + 0.10 * nz], -1) * (0.6 + 0.4 * (1 - (ys / h) * 0.3))[..., None]
    wet = np.clip((S - ys + 0) / 1.0, -1, 1)
    below_water = np.clip((ys - yw) / (h * 0.1), 0, 1)
    wetmask = np.exp(-np.clip(S - yw, 0, 1e9) * 0) * 0
    sandc = sand.astype(F32) * (0.7 + 0.3 * np.clip((yw + 14 - ys) / 14, 0, 1))[..., None]
    img = np.where(sand_mask[..., None], sandc, img)
    # foam line
    foam = np.exp(-((ys - (yw + wob)) / (3 * E.SC)) ** 2) * (xs < w) * ((S > yw) & True)
    img += (foam * 0.55)[..., None] * np.array([1, 1, 1], F32) * (ys < S + 6)[..., None]
    # pier posts
    for px_ in (0.52, 0.6, 0.68, 0.76):
        x0 = int(w * px_); top = int(h * 0.30)
        bot = int(S[0, min(x0, w - 1)]) + 6
        img[top:bot, x0:x0 + max(int(8 * E.SC), 1)] = np.array([0.08, 0.06, 0.05], F32)
    img[int(h * 0.28):int(h * 0.31), int(w * 0.5):int(w * 0.80)] = np.array([0.10, 0.075, 0.06], F32)
    # tide marks
    for lv, text in ((mark_hi, "HIGH"), (mark_lo, "LOW")):
        yy = int(lv * h)
        img[yy:yy + max(int(2 * E.SC), 1), int(w * 0.04):int(w * 0.30)] = np.array([1.0, 0.85, 0.3], F32)
    return img


def coast_tides(c):
    W, H = E.RW, E.RH
    lt = c.lt
    hw = W // 2
    T = 6.5
    ph = math.sin(2 * math.pi * lt / T - 0.6)
    lv = 0.5 + 0.5 * ph
    spring = 1.0
    left = _beach_panel(hw, H, lv, c.t, "", "", 0.80, 0.50, 0.50, 0.80)
    right = _beach_panel(hw, H, lv, c.t, "", "", 0.80 - 0.0, 0.70, 0.70, 0.80 * 1.0)
    # right: only a fraction of the range
    right = _beach_panel(hw, H, lv, c.t, "", "", 0.74, 0.66, 0.66, 0.74)
    img = np.concatenate([left, right], axis=1)
    img[:, hw - 2:hw + 2] = 0.9
    # labels
    E.small_text(img, "TODAY: WITH THE MOON", hw / 2, px(88), 40, (1, 1, 1), smooth(lt / 0.8), anchor="c", glow=0.5, weight="ExtraBold")
    E.small_text(img, "NO MOON: SUN-ONLY TIDES", hw + hw / 2, px(88), 40, (1.0, 0.85, 0.6), smooth((lt - 0.4) / 0.8), anchor="c", glow=0.5, weight="ExtraBold")
    lab_l = "HIGH TIDE" if lv > 0.8 else ("LOW TIDE" if lv < 0.2 else "")
    if lab_l:
        E.small_text(img, lab_l, hw / 2, px(150), 34, (0.8, 0.95, 1.0), 1.0, anchor="c")
        E.small_text(img, lab_l, hw + hw / 2, px(150), 34, (0.8, 0.95, 1.0), 1.0, anchor="c")
    E.small_text(img, "TIDE RANGE: LARGE", hw / 2, H - px(60), 30, (1.0, 0.9, 0.5), 0.9, anchor="c")
    E.small_text(img, "TIDE RANGE: MUCH SMALLER", hw + hw / 2, H - px(60), 30, (1.0, 0.9, 0.5), 0.9, anchor="c")
    return img


# ============================================================ Earth close-ups
def earth_day(c, variant="spin"):
    lt = c.lt
    cam = E.Camera(kf(lt, [(0, (1.8, 0.7, 3.4)), (c.dur, (0.6, 0.35, 2.7))]), (0, 0, 0), 42)
    img = space_base(cam)
    sun = norm([0.55, 0.25, 0.8])
    rate = 0.35 if variant == "spin" else 0.9
    E.draw_earth(img, cam, ORIGIN, 1.0, spin_to(70, rate, c.lt), EARTH_AXIS, sun, cloud_shift=c.t * 0.0012)
    if variant == "clock":
        L = Layer()
        L.rect(RW() * 0.58, RH() * 0.31, RW() * 0.94, RH() * 0.52, (0.0, 0.02, 0.05), 0.7, radius=14, outline=(0.4, 0.7, 1.0))
        L.over(img)
        hrs = (lt * 3.0) % 24
        E.small_text(img, "1 DAY = 24 h 00 m 00.000 s", RW() * 0.76, RH() * 0.40, 36, (0.9, 0.95, 1.0), 1.0, anchor="c")
        E.small_text(img, "+0.002 s PER CENTURY", RW() * 0.76, RH() * 0.40 + px(60), 46, (1.0, 0.85, 0.4), smooth((lt - 2) / 1.5), anchor="c", weight="ExtraBold", glow=0.7)
    return img


def earth_limb(c):
    lt = c.lt
    cam = E.Camera(kf(lt, [(0, (0.9, 1.05, 2.4)), (c.dur, (0.2, 1.15, 2.1))]), (0.0, 0.92, 0.0), 50)
    img = space_base(cam)
    sun = norm([-0.2, 0.15, 1.0])
    E.draw_earth(img, cam, ORIGIN, 1.0, spin_to(10, 0.04, c.lt), EARTH_AXIS, sun, cloud_shift=c.t * 0.0007, atmos=1.8)
    return img


# ============================================================ axial tilt
def _axis_overlay(img, cam, C, axis, length, color=(1.0, 0.85, 0.3), alpha=1.0, width=5):
    a = norm(axis)
    p0 = cam.project(np.asarray(C) - a * length)[:2]
    p1 = cam.project(np.asarray(C) + a * length)[:2]
    L = Layer()
    L.line(p0, p1, color, width, alpha)
    L.ellipse(p1[0], p1[1], px(8), px(8), color, 0, alpha, fill=True, fill_alpha=alpha)
    L.over(img, glow=1.2)
    return p1


def tilt_intro(c):
    lt = c.lt
    sunpos = np.array([-20.0, 1.0, -10.0])
    cam = E.Camera(kf(lt, [(0, (1.0, 1.2, 9.0)), (c.dur, (0.4, 1.0, 8.2))]), (-1.6, 0.0, 0.0), 58)
    img = space_base(cam)
    E.draw_sun(img, cam, sunpos, 3.2, c.t)
    ax = E.axis_from_tilt(23.4, 90)
    E.draw_earth(img, cam, (0.5, 0, 0), 1.25, spin_to(30, 0.4, c.lt), ax, norm([-0.6, 0.2, 0.8]), cloud_shift=c.t * 0.001)
    k = smooth((lt - 1.2) / 1.2)
    p1 = _axis_overlay(img, cam, (0.5, 0, 0), ax, 2.1, alpha=k)
    E.small_text(img, "AXIAL TILT: 23.4°", p1[0] + px(230), p1[1] + px(200), 34, (1.0, 0.9, 0.5), k, anchor="c", weight="ExtraBold", glow=0.6)
    return img


def tilt_seasons(c):
    lt = c.lt
    cam = E.Camera((0, 11.5, 14.5), (0, -0.2, 0), 52)
    img = space_base(cam)
    E.draw_sun(img, cam, ORIGIN, 1.5, c.t, glow=0.7)
    ax = E.axis_from_tilt(23.4, 0)
    names = ["JUNE: NORTHERN SUMMER", "SEPTEMBER", "DECEMBER: NORTHERN WINTER", "MARCH"]
    rad = 5.4
    Lr = Layer()
    for i in range(240):
        a0, a1 = i / 240 * 2 * math.pi, (i + 1) / 240 * 2 * math.pi
        if i % 2 == 0:
            p0 = cam.project(np.array([rad * math.cos(a0), 0, rad * math.sin(a0)]))[:2]
            p1 = cam.project(np.array([rad * math.cos(a1), 0, rad * math.sin(a1)]))[:2]
            Lr.line(p0, p1, (0.6, 0.75, 1.0), 2, 0.45)
    Lr.over(img)
    ang0 = lt * 0.05
    order = []
    for i in range(4):
        a = ang0 + i * math.pi / 2 + math.pi / 2
        pos = np.array([rad * math.cos(a), 0, rad * math.sin(a)])
        order.append((np.linalg.norm(pos - cam.pos), i, pos))
    order.sort(reverse=True)
    for _, i, pos in order:
        E.draw_earth(img, cam, pos, 0.62, spin_to(0, 1.2, c.lt), ax, norm(-pos), cloud_shift=c.t * 0.001)
        _axis_overlay(img, cam, pos, ax, 1.1, alpha=0.9, width=3)
    for i in range(4):
        a = ang0 + i * math.pi / 2 + math.pi / 2
        pos = np.array([rad * math.cos(a), 0.0, rad * math.sin(a)])
        sx, sy, _ = cam.project(pos + np.array([0, -1.1, 0]))
        E.small_text(img, names[i], sx, sy + px(30), 22, (1, 1, 1), smooth((lt - 1.0 - 0.4 * i) / 0.8), anchor="c", glow=0.3)
    return img


def tilt_stable(c):
    lt = c.lt
    cam = E.Camera((0, 0.6, 9.0), (0.0, 0, 0), 46)
    img = space_base(cam)
    tilt = 23.3 + 1.15 * math.sin(lt * 1.2)
    ax = E.axis_from_tilt(tilt, 90)
    E.draw_earth(img, cam, (-2.9, 0, 0), 1.45, spin_to(40, 0.4, c.lt), ax, norm([0.5, 0.3, 0.9]), cloud_shift=c.t * 0.001)
    _axis_overlay(img, cam, (-2.9, 0, 0), ax, 2.2)
    # graph panel
    L = Layer()
    gx0, gy0, gx1, gy1 = RW() * 0.50, RH() * 0.30, RW() * 0.93, RH() * 0.72
    L.rect(gx0, gy0, gx1, gy1, (0.02, 0.05, 0.1), 0.6, radius=14, outline=(0.4, 0.6, 0.9))
    def ymap(v):
        return gy1 - (v / 90.0) * (gy1 - gy0)
    L.rect(gx0 + 10, ymap(24.5), gx1 - 10, ymap(22.1), (0.3, 0.8, 1.0), 0.35)
    for v in (0, 30, 60, 90):
        L.line((gx0 + 4, ymap(v)), (gx0 + 12, ymap(v)), (0.7, 0.8, 1.0), 2, 0.8)
    pts = []
    n = int(min(1.0, lt / 9.0) * 220)
    for i in range(n):
        x = gx0 + 14 + i / 220 * (gx1 - gx0 - 28)
        tt = i / 220 * 8 * math.pi
        v = 23.3 + 1.15 * math.sin(tt) + 0.2 * math.sin(tt * 3.1)
        pts.append((x, ymap(v)))
    if len(pts) > 2:
        L.polyline(pts, (1.0, 0.9, 0.4), 4, 1.0)
    L.over(img, glow=0.7)
    for v in (0, 30, 60, 90):
        E.small_text(img, "%d°" % v, gx0 - px(34), ymap(v), 22, (0.8, 0.9, 1), 0.9, anchor="c")
    E.small_text(img, "EARTH'S TILT OVER ~40,000 YEARS (WITH THE MOON)", (gx0 + gx1) / 2, gy0 - px(26), 24, (0.8, 0.9, 1.0), 1.0, anchor="c")
    E.small_text(img, "ONLY 22.1° – 24.5°", (gx0 + gx1) / 2, gy1 + px(40), 36, (0.5, 0.9, 1.0), smooth((lt - 3) / 1.0), anchor="c", weight="ExtraBold", glow=0.6)
    return img


def _wild_tilt(tt):
    v = 40 + 28 * math.sin(tt * 0.55 + 0.6) + 14 * math.sin(tt * 1.37 + 2.0) + 8 * math.sin(tt * 2.9 + 4.0)
    return float(np.clip(v, 2, 84))


def tilt_compare(c):
    lt = c.lt
    cam = E.Camera((0, 0.6, 12.0), (0, 0, 0), 62)
    img = space_base(cam)
    sun = norm([0.0, 0.2, 1.0])
    tl = 23.3 + 1.1 * math.sin(lt * 1.3)
    tr = _wild_tilt(lt)
    axl, axr = E.axis_from_tilt(tl, 90), E.axis_from_tilt(tr, 90)
    xl, xr = -3.9, 3.9
    E.draw_earth(img, cam, (xl, 0.5, 0), 1.65, spin_to(40, 0.4, c.lt), axl, sun, cloud_shift=c.t * 0.001)
    E.draw_earth(img, cam, (xr, 0.5, 0), 1.65, spin_to(40, 0.4, c.lt), axr, sun, cloud_shift=c.t * 0.001)
    _axis_overlay(img, cam, (xl, 0.5, 0), axl, 2.5, (0.4, 0.9, 1.0))
    _axis_overlay(img, cam, (xr, 0.5, 0), axr, 2.5, (1.0, 0.55, 0.3))
    L = Layer()
    L.line((RW() / 2, px(90)), (RW() / 2, RH() - px(90)), (0.8, 0.9, 1.0), 2, 0.4)
    # histories
    for side, (cx0, col, fn) in enumerate(((RW() * 0.1, (0.4, 0.9, 1.0), lambda s: 23.3 + 1.1 * math.sin(s * 1.3)), (RW() * 0.6, (1.0, 0.55, 0.3), _wild_tilt))):
        gx0, gx1 = cx0, cx0 + RW() * 0.3
        gy0, gy1 = RH() * 0.78, RH() * 0.93
        pts = []
        for i in range(160):
            s = lt - (160 - i) * 0.06
            if s < 0:
                continue
            v = fn(s)
            pts.append((gx0 + i / 160 * (gx1 - gx0), gy1 - v / 90 * (gy1 - gy0)))
        L.rect(gx0, gy0, gx1, gy1, (0.02, 0.04, 0.08), 0.55, radius=8)
        if len(pts) > 2:
            L.polyline(pts, col, 3, 1.0)
    L.over(img, glow=0.5)
    E.small_text(img, "WITH THE MOON", RW() * 0.25, px(100), 40, (0.5, 0.9, 1.0), 1.0, anchor="c", weight="ExtraBold", glow=0.6)
    E.small_text(img, "WITHOUT THE MOON", RW() * 0.75, px(100), 40, (1.0, 0.65, 0.4), smooth((lt - 0.5) / 0.8), anchor="c", weight="ExtraBold", glow=0.6)
    E.small_text(img, "TILT: %.1f°" % tl, RW() * 0.25, RH() * 0.74, 38, (0.8, 0.95, 1.0), 1.0, anchor="c")
    E.small_text(img, "TILT: %.1f°" % tr, RW() * 0.75, RH() * 0.74, 38, (1.0, 0.85, 0.7), 1.0, anchor="c")
    E.small_text(img, "ILLUSTRATIVE: MODELS DIFFER ON HOW FAR THE TILT WOULD SWING", RW() / 2, RH() - px(30), 22, (0.75, 0.8, 0.9), 0.85, anchor="c")
    return img


def tilt_extremes(c):
    lt = c.lt
    cam = E.Camera((0, 0.2, 12.5), (0, 0, 0), 64)
    img = space_base(cam)
    sun = norm([0.0, 0.0, 1.0])
    items = [(-4.6, 0.0, "NO SEASONS", "TILT 0°", (0.6, 0.85, 1.0)), (0.0, 23.4, "TODAY", "TILT 23.4°", (1.0, 1.0, 0.8)), (4.6, 85.0, "EXTREME SEASONS", "TILT 85°", (1.0, 0.6, 0.35))]
    for i, (x, tilt, name, sub, col) in enumerate(items):
        ax = E.axis_from_tilt(tilt, 90)
        wob = 0.0
        E.draw_earth(img, cam, (x, 0.2, 0), 1.55, spin_to(40, 0.4, c.lt), ax, sun, cloud_shift=c.t * 0.001)
        _axis_overlay(img, cam, (x, 0.2, 0), ax, 2.3, col, alpha=smooth((lt - 0.3 * i) / 0.8))
        sx, sy, _ = cam.project(np.array([x, -2.5, 0]))
        E.small_text(img, name, sx, sy, 36, col, smooth((lt - 0.6 - 0.8 * i) / 0.8), anchor="c", weight="ExtraBold", glow=0.5)
        E.small_text(img, sub, sx, sy + px(46), 26, (0.85, 0.9, 1.0), smooth((lt - 0.6 - 0.8 * i) / 0.8), anchor="c")
    return img


def mars_wobble(c):
    lt = c.lt
    cam = E.Camera((0, 0.4, 10.5), (0, 0, 0), 42)
    img = space_base(cam)
    tilt = 30 + 28 * math.sin(lt * 0.8) 
    ax = E.axis_from_tilt(tilt, 90)
    E.draw_planet(img, cam, "mars", (-1.6, 0, 0), 1.5, spin_to(0, 0.25, c.lt), ax, norm([0.4, 0.25, 0.9]), atmos=(0.9, 0.55, 0.35), gain=1.2)
    _axis_overlay(img, cam, (-1.6, 0, 0), ax, 2.4, (1.0, 0.75, 0.4))
    # the two tiny moons
    for i, (rr, sp, rad) in enumerate(((2.9, 1.3, 0.09), (4.0, 0.55, 0.06))):
        a = c.t * sp + i * 2
        pos = np.array([-1.6 + rr * math.cos(a), 0.15 * math.sin(a), rr * 0.3 * math.sin(a) - 0.0])
        E.draw_moon(img, cam, pos, rad, norm([0.4, 0.25, 0.9]), earth=((-1.6, 0, 0), 1.0))
    L = Layer()
    gx0, gy0, gx1, gy1 = RW() * 0.58, RH() * 0.28, RW() * 0.94, RH() * 0.68
    L.rect(gx0, gy0, gx1, gy1, (0.02, 0.04, 0.08), 0.55, radius=12, outline=(0.7, 0.5, 0.3))
    pts = []
    for i in range(200):
        s = lt - (200 - i) * 0.05
        if s < 0:
            continue
        v = 30 + 28 * math.sin(s * 0.8)
        pts.append((gx0 + 14 + i / 200 * (gx1 - gx0 - 28), gy1 - 10 - v / 90 * (gy1 - gy0 - 20)))
    if len(pts) > 2:
        L.polyline(pts, (1.0, 0.7, 0.35), 4, 1.0)
    L.over(img, glow=0.6)
    E.small_text(img, "MARS' TILT SWINGS BETWEEN ~0° AND ~60°", (gx0 + gx1) / 2, gy0 - px(26), 26, (1.0, 0.85, 0.6), 1.0, anchor="c")
    E.small_text(img, "OVER MILLIONS OF YEARS (TINY MOONS, NO STABILIZER)", (gx0 + gx1) / 2, gy1 + px(34), 22, (0.9, 0.85, 0.8), 0.95, anchor="c")
    return img


def timescale(c):
    lt = c.lt
    cam = E.Camera((0.0, 0.0, 10.0), (0.0, 0.0, 0.0), 44)
    img = space_base(cam, 0.8)
    E.draw_earth(img, cam, (-5.0, 2.6, 0), 1.0, spin_to(40, 0.3, c.lt), EARTH_AXIS, SUN, cloud_shift=c.t * 0.001)
    img *= 0.55
    L = Layer()
    x0, x1, y = RW() * 0.08, RW() * 0.92, RH() * 0.60
    L.line((x0, y), (x1, y), (0.8, 0.9, 1.0), 4, 0.9)
    marks = [(0.0, "TODAY"), (0.25, "10,000 YEARS\nALL OF CIVILIZATION"), (0.55, "1 MILLION YEARS"), (0.9, "10 MILLION YEARS\nTILT DRIFT UNFOLDS")]
    prog = smooth(lt / 6.0)
    L.line((x0, y), (x0 + (x1 - x0) * prog, y), (1.0, 0.85, 0.3), 8, 1.0)
    for pos, lab in marks:
        xx = x0 + (x1 - x0) * pos
        L.line((xx, y - px(18)), (xx, y + px(18)), (1, 1, 1), 3, 0.9)
    L.over(img, glow=0.8)
    for pos, lab in marks:
        xx = x0 + (x1 - x0) * pos
        a = smooth((prog - pos) * 8 + 0.2)
        for j, line in enumerate(lab.split("\n")):
            E.small_text(img, line, xx, y + px(58 + 34 * j), 26 if j == 0 else 22, (0.95, 0.97, 1.0) if j == 0 else (0.7, 0.8, 0.95), a, anchor="c", weight="Bold")
    return img


# ============================================================ eclipses / phases
def lunar_eclipse(c):
    lt = c.lt
    cam = E.Camera((9.0, 1.0, -3.0), (0.0, 0.0, -3.0), 66)
    img = space_base(cam)
    sun = np.array([0.0, 0.0, 1.0])
    mx = kf(lt, [(0, 3.4), (12, -3.4)], lambda x: x)
    Mp = np.array([mx, 0.0, -6.0])
    E.draw_earth(img, cam, ORIGIN, 1.0, spin_to(0, 0.1, c.lt), EARTH_AXIS, sun, cloud_shift=c.t * 0.0007)
    E.draw_moon(img, cam, Mp, 0.75, sun, earth=(ORIGIN, 1.0), eclipse=(ORIGIN, 1.0, 0.22), ambient=0.004)
    # shadow cone hint
    L = Layer()
    p0 = cam.project(np.array([0, 1.0, 0]))[:2]; p1 = cam.project(np.array([0, 0.7, -8]))[:2]
    q0 = cam.project(np.array([0, -1.0, 0]))[:2]; q1 = cam.project(np.array([0, -0.7, -8]))[:2]
    L.polygon([p0, p1, q1, q0], (0.0, 0.0, 0.0), 0.25)
    L.over(img)
    return img


def solar_eclipse(c):
    W, H = E.RW, E.RH
    lt = c.lt
    cx, cy = W * 0.5, H * 0.42
    Rs = H * 0.17
    prog = lt / c.dur
    dx = kf(lt, [(0, 2.6), (c.dur * 0.42, 0.0), (c.dur * 0.62, 0.0), (c.dur, -2.6)], lambda x: x) * Rs
    ys_, xs_ = np.mgrid[0:H, 0:W].astype(F32)
    r_s = np.sqrt((xs_ - cx) ** 2 + (ys_ - cy) ** 2)
    mxx = cx + dx * 1.0
    r_m = np.sqrt((xs_ - mxx) ** 2 + (ys_ - (cy + dx * 0.08)) ** 2)
    sun_mask = np.clip((Rs - r_s) + 0.5, 0, 1)
    moon_mask = np.clip((Rs * 1.012 - r_m) + 0.5, 0, 1)
    cover = float(np.clip(1.0 - (abs(dx) / (2 * Rs)), 0, 1))
    # visible fraction of sun
    vis = max(0.0, min(1.0, abs(dx) / Rs * 0.5 + 0.0)) if True else 1
    visible = sun_mask * (1 - moon_mask)
    vis_frac = float(visible.sum() / max(sun_mask.sum(), 1))
    limb = 0.45 + 0.55 * np.sqrt(np.clip(1 - (r_s / Rs) ** 2, 0, 1))
    sky_b = 0.02 + 0.45 * vis_frac ** 0.8
    sky = np.zeros((H, W, 3), F32)
    sky[...] = np.array([0.30, 0.50, 0.95], F32) * sky_b * (0.5 + 0.5 * (1 - ys_ / H))[..., None] * 0.9
    sky += np.exp(-r_s / (Rs * 2.2))[..., None] * np.array([1.0, 0.8, 0.5], F32) * 0.45 * vis_frac
    img = sky
    # corona: only visible when sun mostly covered
    ang = np.arctan2(ys_ - cy, xs_ - cx)
    streak = 1 + 0.55 * np.cos(6 * ang + 0.7) * np.exp(-((r_s - Rs) / (Rs * 2.0)))
    corona = (Rs / np.maximum(r_s, Rs)) ** 3.2 * streak * (r_s > Rs * 0.99)
    cvis = (1 - vis_frac) ** 2
    img += corona[..., None] * np.array([0.9, 0.9, 1.0], F32) * 1.3 * cvis
    # sun disc
    img += (visible * limb * 6.0)[..., None] * np.array([1.0, 0.92, 0.75], F32) * (0.35 + 0.65)
    # moon silhouette
    img = img * (1 - moon_mask[..., None] * 0.985) + moon_mask[..., None] * np.array([0.012, 0.012, 0.018], F32) * (r_m < Rs * 1.012)[..., None]
    # diamond ring near contact
    ring_k = max(0.0, 1.0 - abs(vis_frac - 0.012) / 0.03) if vis_frac < 0.06 else 0.0
    if ring_k > 0:
        sgn = 1 if dx > 0 else -1
        E.radial_glow(img, cx + sgn * Rs * 0.0 + (-sgn) * Rs * 0.2, cy - Rs * 0.95, px(260), (1.0, 0.95, 0.85), 1.6 * ring_k, 2.0, core=0.05)
    # dark landscape strip
    hz = int(H * 0.88)
    ridge = hz + 14 * np.sin(np.arange(W) / 90.0) + 8 * np.sin(np.arange(W) / 23.0)
    mask = (ys_ >= ridge[None, :])
    img = np.where(mask[..., None], np.array([0.004, 0.005, 0.008], F32), img)
    return img


def phases(c):
    lt = c.lt
    W, H = E.RW, E.RH
    cam = E.Camera((0, 0, 60), (0, -2.0, 0), 34)
    img = E.new_frame(cam, 0.8, 0.0)
    names = ["NEW", "WAXING CRESCENT", "FIRST QUARTER", "WAXING GIBBOUS", "FULL", "WANING GIBBOUS", "LAST QUARTER", "WANING CRESCENT"]
    cur = (lt * 0.6) % 8.0
    # big Moon
    a = cur * math.pi / 4
    sun_big = np.array([math.sin(a), 0.05, -math.cos(a)])
    E.draw_moon(img, cam, (-5.5, 1.4, 0), 4.0, sun_big, earth=((-5.5, 1.4, 60), 1.0), vis=1.0, ambient=0.006)
    for i in range(8):
        ai = i * math.pi / 4
        sun_i = np.array([math.sin(ai), 0.05, -math.cos(ai)])
        x = (i - 3.5) * 3.8
        E.draw_moon(img, cam, (x, -8.0, 0), 1.3, sun_i, earth=((x, -8.0, 60), 1.0), vis=1.0, ambient=0.006)
    L = Layer()
    for i in range(8):
        x = (i - 3.5) * 3.8
        sx, sy, _ = cam.project(np.array([x, -6.9, 0]))
        hot = abs(((cur - i + 4) % 8) - 4)
        if hot < 0.6:
            L.ellipse(sx, sy, px(118), px(118), (0.5, 0.85, 1.0), 4, 1.0)
    L.over(img, glow=0.8)
    for i in range(8):
        x = (i - 3.5) * 3.8
        sx, sy, _ = cam.project(np.array([x, -6.9 - 2.1, 0]))
        hot = abs(((cur - i + 4) % 8) - 4) < 0.6
        E.small_text(img, names[i], sx, sy + px(34), 22, (0.5, 0.85, 1.0) if hot else (0.65, 0.7, 0.8), 1.0, anchor="c")
    day = (cur / 8) * 29.5
    E.small_text(img, "DAY %.0f OF 29.5" % day, W * 0.74, H * 0.36, 40, (0.9, 0.95, 1.0), 1.0, anchor="c", weight="ExtraBold", glow=0.5)
    E.small_text(img, names[int(round(cur)) % 8], W * 0.74, H * 0.36 + px(54), 28, (0.5, 0.85, 1.0), 1.0, anchor="c")
    return img


def calendars(c):
    lt = c.lt
    W, H = E.RW, E.RH
    cam = E.Camera((0, 0, 60), (0, 0, 0), 34)
    img = E.new_frame(cam, 0.7, 0.4)
    # big dim moon at left, slowly turning phase
    a = lt * 0.5
    E.draw_moon(img, cam, (-12.0, 0.5, 0), 4.2, np.array([math.sin(a), 0.1, -math.cos(a)]), earth=((-12.0, 0, 60), 1.0), vis=0.9, ambient=0.01)
    cards = [("ISLAMIC CALENDAR", "Every month begins with the new crescent. Eid follows the Moon.", (0.5, 0.9, 0.7)),
             ("HINDU FESTIVALS", "Diwali falls on a new-moon night. Holi is celebrated at the full Moon.", (1.0, 0.75, 0.35)),
             ("CHINESE NEW YEAR", "The festival starts on a new Moon in late winter.", (1.0, 0.5, 0.5)),
             ("JEWISH CALENDAR", "Months follow the Moon. Passover begins at a full Moon.", (0.6, 0.75, 1.0))]
    L = Layer()
    gx0, gy0 = W * 0.36, H * 0.14
    cw, ch = W * 0.28, H * 0.34
    gaps = W * 0.02
    pos = []
    for i, (t_, d_, col) in enumerate(cards):
        r, cc = divmod(i, 2)
        x0 = gx0 + cc * (cw + gaps); y0 = gy0 + r * (ch + gaps)
        a_ = smooth((lt - 0.4 - 0.9 * i) / 0.7)
        off = (1 - a_) * px(30)
        L.rect(x0, y0 + off, x0 + cw, y0 + ch + off, (0.03, 0.06, 0.12), 0.72 * a_, radius=14, outline=col, width=2)
        pos.append((x0, y0 + off, a_, t_, d_, col))
    L.over(img, glow=0.3)
    for (x0, y0, a_, t_, d_, col) in pos:
        E.small_text(img, t_, x0 + px(30), y0 + px(48), 30, col, a_, anchor="tl", weight="ExtraBold", max_w=480)
        E.small_text(img, d_, x0 + px(30), y0 + px(120), 27, (0.92, 0.95, 1.0), a_, anchor="tl", weight="SemiBold", max_w=440)
    return img


def coral_spawn(c):
    W, H = E.RW, E.RH
    lt = c.lt
    ys_ = (np.arange(H, dtype=F32)[:, None] / H)
    img = np.zeros((H, W, 3), F32)
    img += (np.array([0.0, 0.10, 0.20], F32) * (1 - ys_)[..., None] + np.array([0.0, 0.02, 0.06], F32) * ys_[..., None])
    xs_ = np.arange(W, dtype=F32)[None, :] / W
    # moonlight shafts from the surface
    for k in range(7):
        x0 = 0.1 + 0.13 * k + 0.02 * math.sin(lt * 0.4 + k)
        shaft = np.exp(-((xs_ - x0 - (1 - ys_) * 0.10) / 0.025) ** 2) * (1 - ys_) ** 1.2
        img += shaft[..., None] * np.array([0.10, 0.22, 0.34], F32) * 0.6
    # moon glow at top
    E.radial_glow(img, W * 0.7, -H * 0.05, px(700), (0.5, 0.7, 1.0), 0.5, 2.0)
    # reef silhouette
    xs = np.arange(W)
    ridge = H * 0.86 + 22 * np.sin(xs / 70.0) + 14 * np.sin(xs / 23.0 + 1.0) + 10 * np.sin(xs / 9.0)
    m = (np.arange(H)[:, None] >= ridge[None, :])
    img = np.where(m[..., None], np.array([0.01, 0.025, 0.04], F32), img)
    # spawn particles
    rng = np.random.default_rng(5)
    N = 650
    x0 = rng.random(N) * W
    y0 = H * 0.86 + rng.random(N) * 4
    ph = rng.random(N) * 6
    spd = 30 + 60 * rng.random(N)
    start = rng.random(N) * 5.0
    age = np.clip(lt - start, 0, None)
    y = y0 - age * spd * E.SC
    x = x0 + np.sin(age * 0.8 + ph) * 25 * E.SC
    ok = (age > 0) & (y > 0)
    layer = np.zeros_like(img)
    xi, yi = x[ok].astype(int), y[ok].astype(int)
    inside = (xi >= 0) & (xi < W) & (yi >= 0) & (yi < H)
    col = np.stack([np.ones(N), 0.55 + 0.3 * rng.random(N), 0.6 + 0.3 * rng.random(N)], 1)[ok][inside]
    np.add.at(layer, (yi[inside], xi[inside]), col * 0.8)
    layer = cv2.GaussianBlur(layer, (0, 0), 2.2 * E.SC) * 5
    img += layer
    return img


def orbit_top(c, mode="recede"):
    lt = c.lt
    W, H = E.RW, E.RH
    cam = E.Camera((0, 18.0, 0.001), (0, 0, 0), 50, up=(0, 0, -1))
    img = space_base(cam, 0.8, 0.4)
    D0 = 6.4
    if mode == "recede":
        D = D0 + 0.55 * smooth(lt / c.dur)
    else:
        D = D0
    ang = lt * 0.5
    Mp = np.array([D * math.cos(ang), 0, -D * math.sin(ang)])
    sun = norm([-1.0, 0.6, 0.0])
    E.draw_earth(img, cam, ORIGIN, 1.6, 0.4 + lt * 0.5, np.array([0, 1.0, 0]), sun, cloud_shift=c.t * 0.0007, atmos=0.6)
    vis = 1.0
    if mode == "vanish":
        vis = float(kf(lt, [(0, 1), (7.0, 1), (7.05, 0)]))
        if lt > 7.0:
            ang = 7.0 * 0.5
            Mp = np.array([D * math.cos(ang), 0, -D * math.sin(ang)])
    if vis > 0:
        E.draw_moon(img, cam, Mp, 0.44, sun, earth=(ORIGIN, 1.6), vis=vis)
    L = Layer()
    for i in range(0, 160, 2):
        a0, a1 = i / 160 * 2 * math.pi, (i + 1) / 160 * 2 * math.pi
        p0 = cam.project(np.array([D * math.cos(a0), 0, -D * math.sin(a0)]))[:2]
        p1 = cam.project(np.array([D * math.cos(a1), 0, -D * math.sin(a1)]))[:2]
        L.line(p0, p1, (0.6, 0.75, 1.0), 2.5, 0.55)
    L.over(img, glow=0.4)
    if mode == "recede":
        E.small_text(img, "+3.8 cm PER YEAR", W * 0.5, H * 0.14, 54, (1.0, 0.9, 0.5), smooth(lt / 1.2), anchor="c", weight="ExtraBold", glow=0.8)
        E.small_text(img, "THE MOON IS SLOWLY DRIFTING AWAY", W * 0.5, H * 0.14 + px(62), 28, (0.85, 0.92, 1.0), smooth((lt - 0.6) / 1.2), anchor="c")
    else:
        day = (lt * 0.5 / (2 * math.pi)) * 27.3
        if lt < 7.0:
            E.small_text(img, "ONE ORBIT: 27.3 DAYS   |   DAY %d" % int(day), W * 0.5, H * 0.12, 36, (0.85, 0.92, 1.0), 1.0, anchor="c")
        else:
            E.small_text(img, "THE ORBIT STAYS EMPTY", W * 0.5, H * 0.12, 46, (1.0, 0.85, 0.5), smooth((lt - 7.3) / 1.0), anchor="c", weight="ExtraBold", glow=0.7)
    return img


# ============================================================ Theia
def theia(c):
    lt = c.lt
    W, H = E.RW, E.RH
    cam = E.Camera((0, 3.5, 17.0), (0.5, 0.0, 0.0), 40)
    img = space_base(cam, 0.9)
    sun = norm([-0.7, 0.4, 0.6])
    tc = 5.2  # collision time
    ax = EARTH_AXIS
    glow_k = float(kf(lt, [(0, 0.0), (tc - 0.2, 0.0), (tc + 0.2, 1.0), (tc + 2.5, 0.35), (c.dur, 0.2)]))
    # young Earth: molten tint
    mol = float(kf(lt, [(0, 0.0), (tc, 0.0), (tc + 0.4, 1.0), (c.dur, 0.5)]))
    tint = np.array([1.0, 1.0, 1.0]) * (1 - mol) + np.array([1.6, 0.55, 0.2]) * mol
    E.draw_earth(img, cam, (0, 0, 0), 2.2, 0.8 + 0.2 * c.t, ax, sun, cloud_shift=0, lights=0.0, tint=tint, atmos=1.0)
    # Theia (Mars-sized)
    if lt < tc + 0.25:
        k = smooth(lt / tc)
        pos = np.array([14.0 * (1 - k), 2.4 * (1 - k), -4.5 * (1 - k)]) + np.array([3.0 * k * 0, 0, 0])
        pos = pos + np.array([2.6 * k, 0.0, 0.0])
        E.draw_planet(img, cam, "mars", pos, 1.25, c.t * 0.5, ax, sun, atmos=None, gain=1.3)
    # debris
    if lt > tc:
        rng = np.random.default_rng(9)
        N = 4200
        a = rng.random(N) * 2 * math.pi
        sp = (0.5 + rng.random(N) * 1.5)
        z = (rng.random(N) - 0.5) * 0.35
        age = lt - tc
        formed = float(smooth((lt - (tc + 5)) / 5))   # debris falls into a ring then clumps into the Moon
        rad = 2.4 + sp * age * 1.2 * (1 - formed) + 4.8 * formed + rng.random(N) * 0.5 * (1 - formed)
        rad = np.minimum(rad, 11.0)
        ang = a + age * (0.9 / (0.4 + rad * 0.2))
        P = np.stack([rad * np.cos(ang), z * rad * 0.2 + (rng.random(N) - 0.5) * 0.3 * (1 - formed), rad * np.sin(ang)], 1)
        sx, sy, zz = cam.project(P)
        ok = (zz > 0) & (sx > 0) & (sx < W - 1) & (sy > 0) & (sy < H - 1)
        layer = np.zeros_like(img)
        heat = np.clip(1 - age / 9.0, 0, 1)
        colr = np.stack([np.ones(N), 0.35 + 0.5 * heat * rng.random(N), 0.1 + 0.4 * heat * rng.random(N) * heat], 1)
        np.add.at(layer, (sy[ok].astype(int), sx[ok].astype(int)), colr[ok] * (0.9 * (1 - 0.7 * formed)))
        layer = cv2.GaussianBlur(layer, (0, 0), 1.3 * E.SC) * 5
        img += layer
        # the Moon coalescing
        if formed > 0:
            Mp = np.array([6.6 * math.cos(c.t * 0.35), 0.0, 6.6 * math.sin(c.t * 0.35)])
            E.draw_moon(img, cam, Mp, 0.2 + 0.9 * formed, sun, earth=(ORIGIN, 2.2), vis=formed)
    if glow_k > 0:
        sx, sy, _ = cam.project(np.array([0.5, 0, 0]))
        E.radial_glow(img, sx, sy, px(900), (1.0, 0.7, 0.35), 1.1 * glow_k, 2.2, core=0.25)
    return img


# ============================================================ Solar system
PLANETS = [
    # name, orbit r, radius, angle, tex, ring, atmos
    ("Mercury", 62.0, 0.9, 1.5, "mercury", None, None),
    ("Venus", 96.0, 1.4, -0.8, "venus", None, (1.0, 0.85, 0.5)),
    ("Earth", 135.0, 1.6, 0.0, "earth", None, None),
    ("Mars", 178.0, 1.1, 0.6, "mars", None, (0.9, 0.5, 0.3)),
    ("Jupiter", 262.0, 11.0, 2.4, "jupiter", None, None),
    ("Saturn", 350.0, 9.0, 3.6, "saturn", (1.25, 2.3), None),
    ("Uranus", 425.0, 5.0, 4.5, "uranus", None, (0.5, 0.8, 0.9)),
    ("Neptune", 490.0, 4.8, 5.5, "neptune", None, (0.3, 0.5, 1.0)),
]
SUN_R = 26.0


def _ppos(i):
    n, r, R, a, *_ = PLANETS[i]
    return np.array([r * math.cos(a), 0.0, r * math.sin(a)])


def _draw_system(img, cam, c, boost=1.0, only=None, labels=False, orbits=False, t=0.0):
    E.draw_sun(img, cam, ORIGIN, SUN_R, c.t)
    items = []
    for i, (n, r, R, a, tex, ring, atm) in enumerate(PLANETS):
        pos = _ppos(i)
        items.append((np.linalg.norm(pos - cam.pos), i))
    items.sort(reverse=True)
    for _, i in items:
        n, r, R, a, tex, ring, atm = PLANETS[i]
        pos = _ppos(i)
        sdir = norm(-pos)
        Rb = R * boost
        if tex == "earth":
            E.draw_earth(img, cam, pos, Rb, spin_to(70, 0.3, c.lt), EARTH_AXIS, sdir, cloud_shift=c.t * 0.001)
            if boost < 3:
                E.draw_moon(img, cam, pos + np.array([3.4, 0.3, 1.0]) * (0.7 + 0.3 * min(boost, 3)) * 0.0 + np.array([2.6, 0.2, 0.6]) * max(boost * 0.8, 1.0) * 0 , 0.3 * Rb * 0.9, sdir, earth=(pos, Rb)) if False else None
        else:
            E.draw_planet(img, cam, tex, pos, Rb, c.t * 0.2, E.axis_from_tilt(26 if tex != "uranus" else 82, 0), sdir, atmos=atm, ring=ring, gain=0.95)
        sx, sy, z = cam.project(pos)
        if z > 0:
            rpx = cam.fx * Rb / max(np.linalg.norm(pos - cam.pos), 1e-3)
            if rpx < 5:
                E.radial_glow(img, sx, sy, px(9), (1.0, 0.95, 0.85), 0.8, 2.0, core=0.25)
    if orbits:
        L = Layer()
        for i, (n, r, R, a, *_r) in enumerate(PLANETS):
            pts = [cam.project(np.array([r * math.cos(b), 0.0, r * math.sin(b)])) for b in np.linspace(0, 2 * math.pi, 220)]
            for j in range(len(pts) - 1):
                if pts[j][2] > 0 and pts[j + 1][2] > 0:
                    L.line(pts[j][:2], pts[j + 1][:2], (0.6, 0.75, 1.0), 1.6, 0.30)
        L.over(img)
    if labels:
        for i, (n, r, R, a, *_r) in enumerate(PLANETS):
            sx, sy, z = cam.project(_ppos(i))
            if z > 0:
                E.small_text(img, n.upper(), sx, sy - px(28), 26, (0.85, 0.92, 1.0) if n != "Earth" else (0.5, 0.95, 1.0), smooth((t - 0.5 - 0.35 * i) / 0.6), anchor="c", glow=0.3)


def system_pullaway(c):
    lt = c.lt
    ep = _ppos(2)
    sunward = norm(-ep)
    pos = kf(lt, [(0, ep + sunward * 4.5 + np.array([0, 1.2, 1.8])), (4, ep + sunward * 12 + np.array([0, 9, 14])), (14, np.array([185.0, 290.0, 520.0]))], lambda x: x * x * (3 - 2 * x))
    tgt = kf(lt, [(0, ep), (5, ep), (14, np.array([40.0, 0, 0]))])
    cam = E.Camera(pos, tgt, kf(lt, [(0, 38), (14, 52)]))
    img = space_base(cam)
    boost = float(kf(lt, [(0, 1), (3, 1.0), (9, 4.5), (14, 6.5)]))
    _draw_system(img, cam, c, boost=boost, orbits=lt > 8, labels=False)
    return img


def system_planet(c, which):
    lt = c.lt
    i = [p[0] for p in PLANETS].index(which)
    pp = _ppos(i)
    R = PLANETS[i][2]
    sunward = norm(-pp)
    side = norm(np.cross([0, 1, 0], sunward))
    ang = kf(lt, [(0, 0.5), (c.dur, -0.15)])
    d = kf(lt, [(0, 14.0), (c.dur, 6.0 if which == 'Saturn' else 3.8)], smooth) * R
    off = norm(sunward * math.cos(ang) + side * math.sin(ang))
    cam = E.Camera(pp + off * d + np.array([0, 0.35 * R, 0]), pp, 38)
    img = space_base(cam)
    _draw_system(img, cam, c, boost=1.0)
    return img


def system_wide(c):
    lt = c.lt
    ang = kf(lt, [(0, 0.2), (c.dur, -0.12)])
    r = 1250.0
    cam = E.Camera((r * math.sin(ang) * 0.3, r * 0.62, r * 0.78 * math.cos(ang)), (0, 0, 0), 46)
    img = space_base(cam)
    _draw_system(img, cam, c, boost=kf(lt, [(0, 6.5), (c.dur, 6.5)]), orbits=True, labels=True, t=lt)
    return img
