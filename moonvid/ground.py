"""Ground-level night scenes (city / forest / ocean / desert) with real sky + Moon sphere.

Static silhouette layers are generated once (seeded -> identical in every parallel CI job),
then lit per-frame by 'moon_k' (1 = full Moon, 0 = Moon gone).
"""
import math, functools
import numpy as np, cv2
from . import engine as E
from .engine import F32, smooth
from .assets import fractal

S2 = 2  # supersample for silhouette masks


def _mask_to_f(m, W, H):
    return cv2.resize(m, (W, H), interpolation=cv2.INTER_AREA).astype(F32) / 255.0


def _composite(body, alpha, emit, mk, color):
    body *= (1 - mk[..., None]); body += mk[..., None] * np.array(color, F32)
    alpha[:] = np.maximum(alpha, mk)
    emit *= (1 - mk[..., None])


def _ridge(W, rng, amp, base, nterms=7, rough=0.55):
    x = np.arange(W) / W
    y = np.zeros(W)
    a = amp
    for k in range(nterms):
        f = 2 ** k * rng.uniform(0.7, 1.4)
        y += a * np.sin(2 * np.pi * (f * x + rng.random()))
        a *= rough
    return base + y


def _fill_under(curve_y, W, H):
    m = np.zeros((H * S2, W * S2), np.uint8)
    xs = np.repeat(np.arange(W * S2), 1)
    cy = np.interp(xs / S2, np.arange(W), curve_y) * S2
    pts = np.stack([xs, cy], 1)
    pts = np.concatenate([pts, [[W * S2, H * S2], [0, H * S2]]]).astype(np.int32)
    cv2.fillPoly(m, [pts], 255, lineType=cv2.LINE_AA)
    return _mask_to_f(m, W, H)


def _pine(m, x, yb, h, rng):
    w = h * rng.uniform(0.26, 0.36)
    tiers = 6
    top = yb - h
    for j in range(tiers):
        ay = top + j * h * 0.15
        by = ay + h * 0.34
        hw = w * (0.30 + 0.17 * j)
        pts = np.array([[x * S2, ay * S2], [(x - hw) * S2, by * S2], [(x + hw) * S2, by * S2]], np.int32)
        cv2.fillPoly(m, [pts], 255, lineType=cv2.LINE_AA)
    cv2.rectangle(m, (int((x - h * 0.015) * S2), int((yb - h * 0.08) * S2)), (int((x + h * 0.015) * S2), int((yb + h * 0.02) * S2)), 255, -1)


# ----------------------------------------------------------------------------
def _build_city(W, H, hz, seed):
    rng = np.random.default_rng(seed)
    yw = int(hz * H)
    body = np.zeros((H, W, 3), F32); alpha = np.zeros((H, W), F32); emit = np.zeros((H, W, 3), F32)
    blink = []
    # (max height frac of H, width range, color, window probability, window size)
    layers = [(0.30, (18, 46), (0.20, 0.24, 0.32), 0.10, 3),
              (0.36, (26, 66), (0.12, 0.15, 0.21), 0.22, 4),
              (0.46, (40, 96), (0.06, 0.075, 0.11), 0.34, 5)]
    for li, (hmax, wr, col, pwin, ws) in enumerate(layers):
        m = np.zeros((H * S2, W * S2), np.uint8)
        x = -rng.integers(0, 30)
        rects = []
        while x < W:
            bw = int(rng.integers(*wr))
            tall = rng.random() < 0.10
            bh = int((rng.uniform(0.18, 1.0) ** 1.4) * hmax * H * (1.35 if tall else 0.8))
            bh = min(bh, int(hmax * 1.4 * H))
            y0 = yw - bh
            cv2.rectangle(m, (x * S2, y0 * S2), ((x + bw) * S2, (yw + 4) * S2), 255, -1)
            if rng.random() < 0.28:  # stepped crown
                cw = int(bw * 0.6)
                cv2.rectangle(m, ((x + (bw - cw) // 2) * S2, (y0 - int(bh * 0.08)) * S2), ((x + (bw + cw) // 2) * S2, y0 * S2), 255, -1)
            if tall and bw > 30 * 1.0:
                ax = x + bw // 2
                ah = int(bh * 0.18)
                cv2.line(m, (ax * S2, y0 * S2), (ax * S2, (y0 - ah) * S2), 255, 2 * S2)
                if li == 2:
                    blink.append((ax, y0 - ah, rng.random() * 6.28))
            rects.append((x, y0, bw, bh))
            x += bw + int(rng.integers(0, 6))
        mk = _mask_to_f(m, W, H)
        _composite(body, alpha, emit, mk, col)
        # windows
        for (bx, by, bw, bh) in rects:
            gx, gy = ws * 2, ws * 2
            for wy in range(by + 6, yw - 4, gy):
                for wx in range(bx + 4, bx + bw - ws - 2, gx):
                    if rng.random() < pwin:
                        c = rng.choice(3, p=[0.62, 0.26, 0.12])
                        colw = [(1.0, 0.72, 0.38), (1.0, 0.95, 0.8), (0.55, 0.75, 1.0)][c]
                        k = rng.uniform(0.35, 1.0) * (0.5 + 0.18 * li)
                        emit[wy:wy + ws, wx:wx + ws] += np.array(colw, F32) * k
        emit *= 1.0
    # shoreline lights
    for x in range(8, W, 14):
        if rng.random() < 0.55:
            emit[yw - 3:yw - 1, x:x + 2] += np.array([1.0, 0.78, 0.45], F32) * rng.uniform(0.5, 1.0)
    return dict(body=body, alpha=alpha, emit=emit, blink=blink, yw=yw, water=True, dome=(1.0, 0.55, 0.25))


def _build_forest(W, H, hz, seed):
    rng = np.random.default_rng(seed + 1)
    yw = int(hz * H)
    body = np.zeros((H, W, 3), F32); alpha = np.zeros((H, W), F32); emit = np.zeros((H, W, 3), F32)
    # distant ridges
    for base, amp, col in ((0.50, 0.035, (0.11, 0.14, 0.21)), (0.58, 0.03, (0.08, 0.10, 0.16))):
        cy = _ridge(W, rng, amp * H, base * H * (yw / (0.7 * H)), 6, 0.5)
        _composite(body, alpha, emit, _fill_under(cy, W, H), col)
    # tree layers far -> near
    for (hmin, hmax, dens, col) in ((0.10, 0.16, 70, (0.12, 0.15, 0.22)), (0.16, 0.26, 40, (0.08, 0.10, 0.15)), (0.28, 0.42, 22, (0.05, 0.065, 0.10))):
        m = np.zeros((H * S2, W * S2), np.uint8)
        for _ in range(dens):
            x = rng.uniform(-20, W + 20); h = rng.uniform(hmin, hmax) * H
            _pine(m, x, yw + rng.uniform(-4, 6), h, rng)
        _composite(body, alpha, emit, _mask_to_f(m, W, H), col)
    # framing foreground trees
    m = np.zeros((H * S2, W * S2), np.uint8)
    for x, h in ((W * 0.04, 0.95), (W * 0.15, 0.62), (W * 0.93, 0.9), (W * 0.82, 0.58), (W * 0.99, 1.1)):
        _pine(m, x, yw + 18, h * H, rng)
    _composite(body, alpha, emit, _mask_to_f(m, W, H), (0.03, 0.04, 0.06))
    # grass/shore strip
    cy = np.full(W, yw + 2.0) + 4 * np.sin(np.arange(W) / 37.0)
    _composite(body, alpha, emit, _fill_under(cy, W, H) * (np.arange(H)[:, None] < yw + 14), (0.01, 0.014, 0.02))
    # fireflies: positions
    ff = [(rng.uniform(0, W), rng.uniform(yw - 0.18 * H, yw + 6), rng.random() * 6.28) for _ in range(34)]
    return dict(body=body, alpha=alpha, emit=emit, blink=[], yw=yw, water=True, ff=ff, dome=None)


def _build_ocean(W, H, hz, seed):
    rng = np.random.default_rng(seed + 2)
    yw = int(hz * H)
    body = np.zeros((H, W, 3), F32); alpha = np.zeros((H, W), F32); emit = np.zeros((H, W, 3), F32)
    # low haze on horizon, small headland on the left with lighthouse
    cy = _ridge(W, rng, 0.012 * H, yw - 0.012 * H, 5, 0.55)
    cy = np.where(np.arange(W) < W * 0.28, np.minimum(cy, yw - 0.07 * H * np.clip(1 - np.arange(W) / (W * 0.28), 0, 1) ** 0.8), yw + 3)
    cy = np.where(np.arange(W) > W * 0.28, yw + 3, cy)
    _composite(body, alpha, emit, _fill_under(cy, W, H) * (np.arange(H)[:, None] < yw + 10), (0.012, 0.016, 0.024))
    # lighthouse
    m = np.zeros((H * S2, W * S2), np.uint8)
    lx = int(W * 0.13); ly = int(yw - 0.075 * H)
    pts = np.array([[(lx - 9) * S2, ly * S2], [(lx + 9) * S2, ly * S2], [(lx + 5) * S2, (ly - 0.16 * H) * S2], [(lx - 5) * S2, (ly - 0.16 * H) * S2]], np.int32)
    cv2.fillPoly(m, [pts], 255, lineType=cv2.LINE_AA)
    cv2.rectangle(m, ((lx - 8) * S2, int((ly - 0.18 * H) * S2)), ((lx + 8) * S2, int((ly - 0.16 * H) * S2)), 255, -1)
    _composite(body, alpha, emit, _mask_to_f(m, W, H), (0.01, 0.012, 0.02))
    emit[int(ly - 0.172 * H):int(ly - 0.162 * H), lx - 4:lx + 4] += np.array([1.0, 0.9, 0.6], F32) * 1.5
    # distant ships (small lit rectangles near the horizon)
    for sx, sw in ((0.62, 16), (0.71, 9), (0.83, 12)):
        x = int(W * sx)
        emit[yw - 4:yw - 2, x:x + sw // 2] += np.array([1.0, 0.85, 0.55], F32) * 0.9
        emit[yw - 6:yw - 4, x + sw // 4:x + sw // 4 + 2] += np.array([1.0, 0.3, 0.2], F32) * 0.7
        m2 = np.zeros((H, W), np.float32); m2[yw - 5:yw, x - 2:x + sw] = 1
        _composite(body, alpha, emit, m2, (0.01, 0.012, 0.02))
    return dict(body=body, alpha=alpha, emit=emit, blink=[], yw=yw, water=True, lighthouse=(lx, ly - 0.17 * H), dome=None, ocean=True)


def _build_desert(W, H, hz, seed):
    rng = np.random.default_rng(seed + 3)
    yw = int(hz * H)
    body = np.zeros((H, W, 3), F32); alpha = np.zeros((H, W), F32); emit = np.zeros((H, W, 3), F32)
    cy = _ridge(W, rng, 0.05 * H, yw - 0.07 * H, 6, 0.55)
    _composite(body, alpha, emit, _fill_under(cy, W, H), (0.10, 0.11, 0.16))
    for base, amp, col in ((0.035, 0.018, (0.24, 0.23, 0.24)), (0.0, 0.02, (0.17, 0.16, 0.17)), (-0.03, 0.025, (0.11, 0.10, 0.11))):
        cy = _ridge(W, rng, amp * H, yw + base * H, 4, 0.5)
        _composite(body, alpha, emit, _fill_under(cy, W, H), col)
    m = np.zeros((H * S2, W * S2), np.uint8)
    for x, h in ((0.14, 0.2), (0.3, 0.14), (0.78, 0.26), (0.9, 0.17)):
        cx = int(W * x * S2); base = int((yw + 0.05 * H) * S2); hh = int(h * H * S2); t = int(0.012 * H * S2)
        cv2.rectangle(m, (cx - t, base - hh), (cx + t, base), 255, -1)
        cv2.circle(m, (cx, base - hh), t, 255, -1)
        for sgn, hy in ((-1, 0.55), (1, 0.42)):
            ay = base - int(hh * hy)
            cv2.rectangle(m, (cx + sgn * 3 * t - t // 2, ay - int(hh * 0.28)), (cx + sgn * 3 * t + t // 2, ay), 255, -1)
            cv2.rectangle(m, (min(cx, cx + sgn * 3 * t) - t // 2, ay - t // 2), (max(cx, cx + sgn * 3 * t) + t // 2, ay + t // 2), 255, -1)
            cv2.circle(m, (cx + sgn * 3 * t, ay - int(hh * 0.28)), t // 2, 255, -1)
    _composite(body, alpha, emit, _mask_to_f(m, W, H), (0.03, 0.035, 0.05))
    return dict(body=body, alpha=alpha, emit=emit, blink=[], yw=yw, water=False, dome=None)


BUILDERS = dict(city=_build_city, forest=_build_forest, ocean=_build_ocean, desert=_build_desert)


@functools.lru_cache(maxsize=6)
def get_layers(kind, hz, W, H):
    return BUILDERS[kind](W, H, hz, seed=7)


@functools.lru_cache(maxsize=2)
def _noise(W, H):
    n1 = fractal(512, 256, 3, 6, 5, 0.6)
    n2 = fractal(512, 256, 4, 6, 6, 0.6)
    return (cv2.resize(n1, (W, H), interpolation=cv2.INTER_CUBIC) - 0.5).astype(F32), (cv2.resize(n2, (W, H), interpolation=cv2.INTER_CUBIC) - 0.5).astype(F32)


def _water(img, yw, t, moon_k, calm=1.0, tint=(0.04, 0.09, 0.17), streak=False):
    H, W = img.shape[:2]
    hw = H - yw
    if hw <= 4:
        return
    n1, n2 = _noise(W, hw)
    sx = int(t * 14) % W; sy = int(t * 5) % hw
    a = np.roll(n1, sx, axis=1); b = np.roll(np.roll(n2, -int(t * 9) % W, axis=1), sy, axis=0)
    ys = np.arange(hw, dtype=F32)[:, None]
    depth = (ys + 1) / hw
    dx = (a * 14 + b * 7) * (0.25 + 1.6 * depth) * calm * E.SC
    dy = (b * 6) * depth * E.SC
    xs = np.arange(W, dtype=F32)[None, :] + dx
    sy_src = (yw - 1 - ys * (0.92 + 0.3 * depth)) + dy
    sy_src = np.clip(sy_src, 0, yw - 1)
    refl = cv2.remap(img, xs.astype(F32), sy_src.astype(F32), cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT)
    refl = cv2.GaussianBlur(refl, (1, 0), 0, sigmaY=max(3.0 * E.SC, 0.5))
    refl = refl * (0.62 - 0.22 * depth)[..., None]
    if streak:
        hi = np.maximum(refl - 0.25, 0)
        refl = refl + cv2.GaussianBlur(hi, (1, 0), 0, sigmaY=max(26.0 * E.SC, 1)) * 2.2 * (0.5 + np.clip(a * 4 + 0.5, 0, 1.5))[..., None]
    # fresnel-like sky reflection brightens toward horizon, then base water colour
    base = np.array(tint, F32)[None, None, :] * (0.12 + moon_k) * (1 - 0.5 * depth)[..., None]
    sparkle = (np.clip(a + b - 0.12, 0, 1) ** 1.5) * 3.0
    out = refl + base
    # darken edge with horizon haze
    img[yw:] = out


def render(c_cam_args, kind, hz, t, moon_k, star_gain, mw_gain, pan_px=0.0, sky_rot=None, moon_xy=(0.72, 0.24),
           lt_for_blink=0.0, fov=62.0, lighthouse=True):
    """Return float32 frame of the ground scene. moon_az_el are (yaw, pitch) offsets of Moon vs camera axis in radians."""
    W, H = E.RW, E.RH
    L = get_layers(kind, round(hz, 3), W, H)
    fx = (W / 2) / math.tan(math.radians(fov) / 2)
    pitch = math.atan((hz * H - H / 2) / fx)
    yaw = c_cam_args.get("yaw", 0.0) + pan_px / fx
    f = np.array([math.sin(yaw) * math.cos(pitch), math.sin(pitch), -math.cos(yaw) * math.cos(pitch)])
    cam = E.Camera((0, 0, 0), f, fov)
    sg = star_gain * (1 - 0.72 * moon_k)
    img = E.new_frame(cam, star_gain=sg, mw_gain=mw_gain * (1 - 0.8 * moon_k), rot=sky_rot)
    # sky tone: airglow + moon scatter
    ys = np.arange(H, dtype=F32)[:, None]
    hy = hz * H
    gl = np.exp(-np.clip(hy - ys, 0, None) / (0.22 * H))[..., None]
    img += gl * np.array([0.012, 0.022, 0.034], F32) * (0.6 + 0.8 * 1)
    img += np.ones((H, W, 1), F32) * np.array([0.03, 0.075, 0.16], F32) * moon_k * (0.6 + 1.1 * gl)
    # Moon
    d = E.norm(cam.f + ((moon_xy[0] * W - W / 2) / cam.fx) * cam.r - ((moon_xy[1] * H - H / 2) / cam.fx) * cam.u)
    D = 100.0
    Mpos = d * D
    if moon_k > 0.005:
        sx, sy, z = cam.project(Mpos)
        E.radial_glow(img, sx, sy, E.px(620), (0.40, 0.58, 1.0), 0.30 * moon_k, 2.0)
        E.radial_glow(img, sx, sy, E.px(130), (0.9, 0.95, 1.0), 0.35 * moon_k, 2.4)
        sun = -f + np.array([0.12, 0.08, 0.0])
        E.draw_moon(img, cam, Mpos, D * 0.0125, sun, earth=((0, 0, 0), 1.0), vis=moon_k * 1.15, ambient=0.0)
    # city light dome
    if L.get("dome") is not None:
        dc = np.array(L["dome"], F32)
        img += np.exp(-np.clip(hy - ys, 0, None) / (0.16 * H))[..., None] * dc * 0.16 * (ys < hy)[..., None]
    # lighthouse beam
    if L.get("lighthouse") and lighthouse:
        lx, ly = L["lighthouse"]
        ang = lt_for_blink * 0.9
        reach = W * 0.8
        dx = math.cos(ang) * reach
        beam_w = 0.02
        ys2, xs2 = np.mgrid[0:int(hy), 0:W].astype(F32)
        vx, vy = xs2 - lx, (ys2 - ly)
        fwd = vx * math.cos(ang) + 0.0
        side = np.abs(vy * 1.0 + 0.0 - 0.0)
        # a wide horizontal beam whose length oscillates (rotating beam seen from the side)
        mag = max(math.cos(ang), 0.0) ** 6
        if mag > 0.01:
            sgn = 1.0
            bx = np.clip(vx / (W * 0.7), 0, 1)
            beam = np.exp(-(vy / (4 + 22 * bx * 1.0)) ** 2) * bx ** 0.4 * (1 - bx) * mag * (vx > 0)
            img[:int(hy)] += beam[..., None] * np.array([1.0, 0.95, 0.75], F32) * 0.35
    # ground layers
    body, alpha, emit = L["body"], L["alpha"], L["emit"]
    if pan_px:
        M = np.float32([[1, 0, -pan_px], [0, 1, 0]])
        body = cv2.warpAffine(body, M, (W, H), borderMode=cv2.BORDER_REPLICATE)
        alpha = cv2.warpAffine(alpha, M, (W, H), borderMode=cv2.BORDER_REPLICATE)
        emit = cv2.warpAffine(emit, M, (W, H), borderMode=cv2.BORDER_REPLICATE)
    moon_light = np.array([0.55, 0.70, 1.0], F32) * (2.3 * moon_k)
    amb = np.array([0.012, 0.016, 0.026], F32) + np.array([0.004, 0.006, 0.01], F32) * (1 if L.get("dome") is None else 0)
    lit = body * (moon_light + amb)[None, None, :]
    # rim of moonlight along the top of silhouettes
    col = lit + emit * 1.0
    a = alpha[..., None]
    img = img * (1 - a) + col * a
    # fireflies / blinkers
    for (bx, by, ph) in L.get("blink", []):
        k = max(0.0, math.sin(lt_for_blink * 2.2 + ph)) ** 6
        E.radial_glow(img, bx - pan_px, by, E.px(22), (1.0, 0.15, 0.1), 0.9 * k, 2.0, core=0.12)
    for (fx_, fy_, ph) in L.get("ff", []):
        k = 0.5 + 0.5 * math.sin(lt_for_blink * 1.7 + ph)
        x = fx_ + 25 * math.sin(lt_for_blink * 0.5 + ph) - pan_px
        y = fy_ + 15 * math.sin(lt_for_blink * 0.7 + ph * 2)
        E.radial_glow(img, x, y, E.px(14), (0.8, 1.0, 0.4), 0.7 * k, 2.0, core=0.15)
    if L.get("water"):
        _water(img, L["yw"], t, moon_k, calm=1.0 if kind != "ocean" else 1.5, streak=(kind == "ocean"))
        if kind == "ocean":
            # light shimmer where the Moon path would be
            pass
    return img
