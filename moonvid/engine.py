"""Tiny numpy/OpenCV 3D engine: textured spheres, atmosphere, stars, bloom, text."""
import os, math, functools
import numpy as np, cv2
from PIL import Image, ImageDraw, ImageFont, ImageFilter
from .assets import get_assets, galactic_basis

F32 = np.float32
BASE_W, BASE_H, FPS = 1920, 1080, 30
RW, RH, SC = BASE_W, BASE_H, 1.0


def set_scale(s):
    global RW, RH, SC
    RW, RH, SC = int(round(BASE_W * s)), int(round(BASE_H * s)), float(s)
    _vignette.cache_clear()


def size():
    return RW, RH


def px(v):
    """Scale a 1080p pixel size to the current render size."""
    return v * SC


# ------------------------------------------------------------------ helpers
def norm(v):
    v = np.asarray(v, np.float64)
    return v / (np.linalg.norm(v, axis=-1, keepdims=True) + 1e-12)


def smooth(x):
    x = np.clip(x, 0.0, 1.0)
    return x * x * (3 - 2 * x)


def smoother(x):
    x = np.clip(x, 0.0, 1.0)
    return x * x * x * (x * (x * 6 - 15) + 10)


def lerp(a, b, t):
    return a + (b - a) * t


def kf(t, keys, ease=smooth):
    """Piecewise keyframes [(time, value)...] with smooth easing; values may be vectors."""
    if t <= keys[0][0]:
        return np.array(keys[0][1], np.float64) if np.ndim(keys[0][1]) else keys[0][1]
    for (t0, v0), (t1, v1) in zip(keys[:-1], keys[1:]):
        if t <= t1:
            k = ease((t - t0) / max(t1 - t0, 1e-9))
            return np.array(v0, np.float64) * (1 - k) + np.array(v1, np.float64) * k if np.ndim(v0) else v0 + (v1 - v0) * k
    return np.array(keys[-1][1], np.float64) if np.ndim(keys[-1][1]) else keys[-1][1]


def rot_y(a):
    c, s = math.cos(a), math.sin(a)
    return np.array([[c, 0, s], [0, 1, 0], [-s, 0, c]], np.float64)


def rot_axis(axis, a):
    axis = norm(axis); x, y, z = axis; c, s = math.cos(a), math.sin(a); C = 1 - c
    return np.array([[c + x * x * C, x * y * C - z * s, x * z * C + y * s],
                     [y * x * C + z * s, c + y * y * C, y * z * C - x * s],
                     [z * x * C - y * s, z * y * C + x * s, c + z * z * C]], np.float64)


def y_to(axis):
    axis = norm(axis); y = np.array([0, 1.0, 0])
    k = np.cross(y, axis); s = np.linalg.norm(k)
    if s < 1e-9:
        return np.eye(3) if axis[1] > 0 else rot_axis([1, 0, 0], math.pi)
    return rot_axis(k, math.atan2(s, float(y @ axis)))


def body_rot(axis, spin):
    return rot_axis(axis, spin) @ y_to(axis)


def axis_from_tilt(tilt_deg, az_deg=0.0):
    t, a = math.radians(tilt_deg), math.radians(az_deg)
    return np.array([math.sin(t) * math.cos(a), math.cos(t), math.sin(t) * math.sin(a)])


# ------------------------------------------------------------------- camera
class Camera:
    def __init__(self, pos, target, fov=40.0, up=(0, 1, 0), roll=0.0):
        self.pos = np.asarray(pos, np.float64)
        f = norm(np.asarray(target, np.float64) - self.pos)
        r = norm(np.cross(f, up)); u = np.cross(r, f)
        if roll:
            c, s = math.cos(roll), math.sin(roll)
            r, u = r * c + u * s, u * c - r * s
        self.f, self.r, self.u = f, r, u
        self.fov = fov
        self.fx = (RW / 2) / math.tan(math.radians(fov) / 2)

    def dirs(self, x0, y0, x1, y1, step=1):
        xs = (np.arange(x0, x1, step, dtype=F32) + 0.5 - RW / 2) / F32(self.fx)
        ys = (np.arange(y0, y1, step, dtype=F32) + 0.5 - RH / 2) / F32(self.fx)
        d = (self.f.astype(F32)[None, None, :] + xs[None, :, None] * self.r.astype(F32)[None, None, :]
             - ys[:, None, None] * self.u.astype(F32)[None, None, :])
        d /= np.linalg.norm(d, axis=-1, keepdims=True)
        return d

    def project(self, p):
        v = np.asarray(p, np.float64) - self.pos
        z = v @ self.f
        z_ = np.where(np.abs(z) < 1e-6, 1e-6, z) if np.ndim(z) else (z if abs(z) > 1e-6 else 1e-6)
        return RW / 2 + self.fx * (v @ self.r) / z_, RH / 2 - self.fx * (v @ self.u) / z_, z

    def project_dir(self, d):
        z = d @ self.f
        z_ = np.where(np.abs(z) < 1e-6, 1e-6, z)
        return RW / 2 + self.fx * (d @ self.r) / z_, RH / 2 - self.fx * (d @ self.u) / z_, z


# ----------------------------------------------------------------- backdrop
def new_frame(cam, star_gain=1.0, mw_gain=1.0, tint=(1, 1, 1), rot=None):
    """Black space + Milky Way + crisp stars."""
    A = get_assets()
    img = np.zeros((RH, RW, 3), F32)
    if mw_gain > 0:
        w4, h4 = RW // 4, RH // 4
        d = cam.dirs(0, 0, RW, RH, step=4)[:h4, :w4]
        gn, ga, gb = galactic_basis()
        if rot is not None:
            d = d @ rot.astype(F32)
        lat = np.arcsin(np.clip(d @ gn, -1, 1)); lon = np.arctan2(d @ gb, d @ ga)
        mx = ((lon / (2 * np.pi) + 0.5) * A.milky.shape[1] - 0.5).astype(F32)
        my = ((0.5 - lat / np.pi) * A.milky.shape[0] - 0.5).astype(F32)
        m = cv2.remap(A.milky, mx, my, cv2.INTER_LINEAR, borderMode=cv2.BORDER_WRAP)
        m = cv2.resize(m, (RW, RH), interpolation=cv2.INTER_LINEAR)[..., None]
        img += m * F32(0.16 * mw_gain) * np.array([0.62, 0.72, 1.0], F32)
        img += (m ** 2) * F32(0.10 * mw_gain) * np.array([1.0, 0.85, 0.6], F32)
    if star_gain > 0:
        dirs, inten, col = A.stars
        if rot is not None:
            dirs = (dirs @ rot.T.astype(F32))
        sx, sy, z = cam.project_dir(dirs.astype(np.float64))
        ok = (z > 0.05) & (sx > 1) & (sx < RW - 2) & (sy > 1) & (sy < RH - 2)
        sx, sy, I, c = sx[ok], sy[ok], inten[ok] * star_gain, col[ok]
        ix, iy = np.floor(sx).astype(int), np.floor(sy).astype(int)
        fx, fy = (sx - ix).astype(F32), (sy - iy).astype(F32)
        layer = np.zeros_like(img)
        for dx, dy, w in ((0, 0, (1 - fx) * (1 - fy)), (1, 0, fx * (1 - fy)), (0, 1, (1 - fx) * fy), (1, 1, fx * fy)):
            np.add.at(layer, (iy + dy, ix + dx), c * (I * w)[:, None])
        layer = cv2.GaussianBlur(layer, (0, 0), 0.55 * max(SC, 0.6)) * 2.6
        img += layer
    if tint != (1, 1, 1):
        img *= np.array(tint, F32)
    return img


# --------------------------------------------------------------- sphere core
class Geom:
    pass


def sphere_geom(cam, C, R, pad=1.0, max_pad_px=None):
    C = np.asarray(C, np.float64)
    v = C - cam.pos
    z = float(v @ cam.f)
    dist = float(np.linalg.norm(v))
    if dist <= R * 1.001 or z <= 1e-3:
        return None
    sx = RW / 2 + cam.fx * float(v @ cam.r) / z
    sy = RH / 2 - cam.fx * float(v @ cam.u) / z
    rpx_nom = cam.fx * R / math.sqrt(max(dist * dist - R * R, 1e-9))
    ang = math.asin(min(R * pad / dist, 0.999))
    rpx = cam.fx * math.tan(ang)
    rho = math.hypot(sx - RW / 2, sy - RH / 2)
    rpx = rpx * (1 + (rho / cam.fx) ** 2) * 1.08 + 3
    x0, x1 = int(max(0, math.floor(sx - rpx))), int(min(RW, math.ceil(sx + rpx)))
    y0, y1 = int(max(0, math.floor(sy - rpx))), int(min(RH, math.ceil(sy + rpx)))
    if x1 <= x0 or y1 <= y0:
        return None
    g = Geom()
    g.x0, g.y0, g.x1, g.y1 = x0, y0, x1, y1
    d = cam.dirs(x0, y0, x1, y1)
    oc = (cam.pos - C).astype(F32)
    b = d @ oc
    c0 = F32(oc @ oc - R * R)
    disc = b * b - c0
    hp2 = F32(oc @ oc) - b * b
    h = np.sqrt(np.maximum(hp2, 0))
    pxw = dist / cam.fx
    g.cov = np.clip((F32(R) - h) / F32(pxw) + 0.5, 0, 1) * (b < 0)
    t = -b - np.sqrt(np.maximum(disc, 0))
    p = oc[None, None, :] + t[..., None] * d
    n = p / F32(R)
    n /= np.linalg.norm(n, axis=-1, keepdims=True) + F32(1e-9)
    g.d, g.b, g.h, g.t, g.n = d, b, h, t, n
    g.v = -d
    g.nv = np.clip(np.einsum("ijk,ijk->ij", n, g.v), 0, 1)
    g.rpx, g.sx, g.sy, g.dist, g.C, g.R, g.pxw, g.oc = rpx_nom, sx, sy, dist, C, R, pxw, oc
    g.cam = cam
    return g


def uv_from_local(nl):
    lat = np.arcsin(np.clip(nl[..., 1], -1, 1))
    lon = np.arctan2(nl[..., 0], nl[..., 2])
    return (lon / (2 * np.pi) + 0.5).astype(F32), (0.5 - lat / np.pi).astype(F32)


def to_local(n, Rb):
    return (n.reshape(-1, 3) @ Rb.astype(F32)).reshape(n.shape)


def lookup(pyr, u, v, rpx, bias=0.0, uscale=1.0, vscale=1.0, interp=cv2.INTER_LINEAR):
    W0 = pyr[0].shape[1]
    lvl = int(np.clip(math.floor(math.log2(max(W0 * uscale / (2 * math.pi * max(rpx, 1.0)), 1e-6)) + bias), 0, len(pyr) - 1))
    tex = pyr[lvl]
    h, w = tex.shape[:2]
    mx = (u * F32(w * uscale) - 0.5).astype(F32)
    my = (v * F32(h * vscale) - 0.5).astype(F32)
    return cv2.remap(tex, mx, my, interp, borderMode=cv2.BORDER_WRAP)


def f01(a):
    return a.astype(F32) * F32(1 / 255.0)


def sstep(a, b, x):
    t = np.clip((x - a) / (b - a), 0, 1)
    return t * t * (3 - 2 * t)


def blend(img, g, color, cov=None):
    cov = g.cov if cov is None else cov
    reg = img[g.y0:g.y1, g.x0:g.x1]
    reg *= (1 - cov[..., None])
    reg += color * cov[..., None]


def add(img, g, color):
    img[g.y0:g.y1, g.x0:g.x1] += color


def atmosphere_glow(img, g, sun, color=(0.30, 0.55, 1.0), thick=0.075, strength=1.0):
    """Outer limb glow (additive) around a sphere, lit by sun direction."""
    R = g.R
    h = g.h
    s = (h - R) / (R * thick)
    outside = (g.b < 0) & (s > 0) & (s < 1)
    q = g.oc[None, None, :] - g.b[..., None] * g.d           # closest-approach vector
    qn = q / (np.linalg.norm(q, axis=-1, keepdims=True) + 1e-6)
    lit = sstep(-0.25, 0.55, np.einsum("ijk,k->ij", qn, sun.astype(F32)))
    inten = np.where(outside, np.clip(1 - s, 0, 1) ** 2.4, 0) * lit * F32(strength)
    reg = img[g.y0:g.y1, g.x0:g.x1]
    reg += inten[..., None] * np.array(color, F32) * F32(1.1)


def radial_glow(img, cx, cy, radius, color, strength=1.0, power=2.0, core=0.0):
    x0, x1 = int(max(0, cx - radius)), int(min(RW, cx + radius))
    y0, y1 = int(max(0, cy - radius)), int(min(RH, cy + radius))
    if x1 <= x0 or y1 <= y0:
        return
    ys, xs = np.mgrid[y0:y1, x0:x1].astype(F32)
    r = np.sqrt((xs + 0.5 - cx) ** 2 + (ys + 0.5 - cy) ** 2) / F32(radius)
    a = np.clip(1 - r, 0, 1) ** power * strength
    if core > 0:
        a += np.exp(-(r / core) ** 2) * strength
    img[y0:y1, x0:x1] += a[..., None] * np.array(color, F32)


# ---------------------------------------------------------------- the bodies
def draw_earth(img, cam, C, R, spin, axis, sun, cloud_shift=0.0, lights=1.0, atmos=1.0,
               clouds=1.0, detail=1.0, tint=None):
    A = get_assets()
    g = sphere_geom(cam, C, R, pad=1.12)
    if g is None:
        return None
    sun = norm(sun).astype(F32)
    Rb = body_rot(axis, spin)
    nl = to_local(g.n, Rb)
    u, v = uv_from_local(nl)
    rp = g.rpx
    day = f01(lookup(A.pyr["earth_day"], u, v, rp))
    ocean = f01(lookup(A.pyr["earth_ocean"], u, v, rp))
    relief = f01(lookup(A.pyr["earth_relief"], u, v, rp))
    # close-up micro detail on land
    if rp > 450 and detail > 0:
        k = float(np.clip((rp - 450) / 900, 0, 1)) * detail
        dn = f01(lookup(A.pyr["earth_detail"], u, v, rp, uscale=6, vscale=6))
        dn2 = f01(lookup(A.pyr["earth_detail"], u, v, rp, uscale=23, vscale=23))
        mod = 1 + k * (0.55 * (dn - 0.5) + 0.35 * (dn2 - 0.5))
        day *= mod[..., None]
    day *= (0.82 + 0.5 * (relief - 0.5) * (1 - ocean))[..., None]
    ndl = np.einsum("ijk,k->ij", g.n, sun)
    nv = g.nv
    diff = sstep(-0.06, 0.55, ndl) ** 0.85
    surf = day * diff[..., None] * F32(1.05)
    # ocean sun-glint
    rv = 2 * ndl[..., None] * g.n - sun[None, None, :]
    spec = np.clip(np.einsum("ijk,ijk->ij", rv, g.v), 0, 1)
    glint = (spec ** 200) * ocean * 0.9 + (spec ** 28) * ocean * 0.09
    surf += (glint * (ndl > 0))[..., None] * np.array([1.0, 0.95, 0.85], F32)
    # twilight band
    tw = np.exp(-((ndl - 0.02) / 0.10) ** 2)
    surf += (tw * 0.10 * nv)[..., None] * np.array([1.0, 0.45, 0.2], F32) * (day.mean(axis=-1, keepdims=True) + 0.2)
    # clouds
    if clouds > 0:
        cu = (u + cloud_shift) % 1.0
        cl = f01(lookup(A.pyr["earth_clouds"], cu, v, rp)) * clouds
        # shadow: sample slightly shifted along sun
        sh = f01(lookup(A.pyr["earth_clouds"], (cu + 0.0016 * np.sign(sun[0] + 1e-6)) % 1.0, np.clip(v - 0.0012 * np.sign(sun[1] + 1e-6), 0, 1), rp))
        surf *= (1 - 0.38 * sh * clouds)[..., None]
        ccol = (0.97 * diff + 0.02)[..., None] * np.array([1.0, 1.0, 1.0], F32) + (tw * 0.12)[..., None] * np.array([1.0, 0.5, 0.3], F32)
        surf = surf * (1 - 0.92 * cl[..., None]) + ccol * (0.92 * cl[..., None])
    else:
        cl = np.zeros_like(ndl)
    # night lights
    if lights > 0:
        lt = f01(lookup(A.pyr["earth_lights"], u, v, rp))
        night = (1 - sstep(-0.12, 0.10, ndl))
        flick = 1.0
        lc = lt * np.array([1.0, 0.82, 0.52], F32) * 2.2
        surf += lc * (night * (1 - 0.75 * cl) * lights * flick)[..., None]
    # ambient night-side deep blue
    surf += (1 - diff)[..., None] * np.array([0.004, 0.008, 0.02], F32) * (0.5 + nv[..., None])
    # atmosphere rim
    fres = (1 - nv) ** 3.0
    rim_lit = sstep(-0.35, 0.6, ndl)
    surf += (fres * (0.10 + 1.0 * rim_lit) * atmos)[..., None] * np.array([0.25, 0.5, 1.0], F32) * F32(0.95)
    if tint is not None:
        surf *= np.array(tint, F32)
    blend(img, g, surf)
    if atmos > 0:
        atmosphere_glow(img, g, sun, strength=atmos)
    return g


def moon_basis(Mpos, Epos):
    zb = norm(np.asarray(Epos, np.float64) - np.asarray(Mpos, np.float64))
    xb = norm(np.cross([0, 1.0, 0], zb)); yb = np.cross(zb, xb)
    return np.stack([xb, yb, zb], 1)


def draw_moon(img, cam, C, R, sun, earth=None, vis=1.0, ambient=0.012, eclipse=None, shine=1.0, tint=None):
    """earth=(center,radius) used for tidal locking + eclipse shadow."""
    A = get_assets()
    g = sphere_geom(cam, C, R, pad=1.0)
    if g is None:
        return None
    sun = norm(sun).astype(F32)
    Ec = np.array(earth[0], np.float64) if earth else np.array([0, 0, 0.0])
    Rb = moon_basis(C, Ec)
    nl = to_local(g.n, Rb)
    u, v = uv_from_local(nl)
    rp = g.rpx
    alb = f01(lookup(A.pyr["moon"], u, v, rp))
    if alb.ndim == 3:
        lum = alb.mean(axis=-1)
    det = f01(lookup(A.pyr["moon_detail"], u, v, rp))
    mod = 1 + 1.0 * (det - 0.5)
    if rp > 200:
        k = float(np.clip((rp - 200) / 500, 0, 1))
        det2 = f01(lookup(A.pyr["moon_detail"], u, v, rp, uscale=5, vscale=5))
        det3 = f01(lookup(A.pyr["moon_detail"], u, v, rp, uscale=17, vscale=17))
        mod = mod + k * (0.8 * (det2 - 0.5) + 0.5 * (det3 - 0.5))
    alb = np.clip(alb * mod[..., None], 0, 1.6)
    alb = alb * np.array([1.02, 1.0, 0.95], F32)
    ndl = np.einsum("ijk,k->ij", g.n, sun)
    nd = np.clip(ndl, 0, 1)
    lam = nd / (nd + g.nv + 0.06) * 2.05
    lam *= sstep(-0.02, 0.06, ndl)
    shadow = np.ones_like(lam)
    redtint = np.ones((1, 1, 3), F32)
    if eclipse is not None:
        Ce, Re, k_pen = eclipse
        p = (C[None, None, :] + (g.n * R)).astype(F32)
        e = p - np.asarray(Ce, F32)[None, None, :]
        along = -np.einsum("ijk,k->ij", e, sun)
        perp = np.linalg.norm(e + along[..., None] * sun[None, None, :], axis=-1)
        um = Re - 0.045 * np.maximum(along, 0)
        pen = Re + k_pen * np.maximum(along, 0)
        inpen = np.clip((pen - perp) / np.maximum(pen - um, 1e-3), 0, 1) * (along > 0)
        shadow = 1 - 0.97 * smooth_np(inpen)
        redk = smooth_np(np.clip((um + 0.25 * R - perp) / (0.5 * R + 1e-3), 0, 1)) * (along > 0)
        redtint = (1 - redk)[..., None] * np.array([1, 1, 1], F32) + redk[..., None] * np.array([1.0, 0.33, 0.12], F32)
        # red-orange residual light inside umbra
        lam = lam * shadow + 0.55 * redk * (1.0 - 0.0) * np.clip(nv_ if False else g.nv * 0 + 1, 0, 1)
    col = alb * (lam[..., None] * F32(1.15 * shine)) * redtint
    col += alb * ambient * (1.0)
    col *= F32(vis)
    if tint is not None:
        col *= np.array(tint, F32)
    blend(img, g, col, cov=g.cov * F32(1.0))
    return g


def smooth_np(x):
    x = np.clip(x, 0, 1)
    return x * x * (3 - 2 * x)


def draw_planet(img, cam, name, C, R, spin, axis, sun, ambient=0.02, atmos=None, ring=None, gamma=1.0, gain=1.0):
    """Generic textured planet with optional ring=(r_in, r_out) multiples of R."""
    A = get_assets()
    pad = (ring[1] * 1.05) if ring else (1.12 if atmos else 1.0)
    g = sphere_geom(cam, C, R, pad=pad)
    if g is None:
        return None
    sun = norm(sun).astype(F32)
    Rb = body_rot(axis, spin)
    nl = to_local(g.n, Rb)
    u, v = uv_from_local(nl)
    tex = f01(lookup(A.pyr[name], u, v, g.rpx))
    ndl = np.einsum("ijk,k->ij", g.n, sun)
    diff = sstep(-0.04, 0.5, ndl) ** 0.9
    col = tex ** gamma * diff[..., None] * F32(gain) + tex * ambient
    if atmos:
        fres = (1 - g.nv) ** 3.0
        col += (fres * (0.1 + sstep(-0.3, 0.6, ndl)))[..., None] * np.array(atmos, F32) * 0.8
    cov = g.cov
    if ring:
        # ray-plane intersection with the equatorial plane
        ax = norm(axis).astype(F32)
        oc = (cam.pos - np.asarray(C)).astype(F32)
        denom = np.einsum("ijk,k->ij", g.d, ax)
        tpl = -(oc @ ax) / np.where(np.abs(denom) < 1e-6, 1e-6, denom)
        hit = oc[None, None, :] + tpl[..., None] * g.d
        rho = np.linalg.norm(hit, axis=-1) / F32(R)
        inring = (rho > ring[0]) & (rho < ring[1]) & (tpl > 0)
        rr = np.clip((rho - ring[0]) / (ring[1] - ring[0]), 0, 1)
        ridx = (rr * (len(A.ring_col) - 1)).astype(int)
        rcol = A.ring_col[ridx]
        ralpha = np.clip(rcol.mean(axis=-1) * 1.5, 0, 0.95) * inring
        # planet occludes ring if planet is nearer along the ray
        planet_front = (g.cov > 0.5) & (g.t < tpl)
        # planet shadow on ring
        along = -np.einsum("ijk,k->ij", hit, sun)
        perp = np.linalg.norm(hit + along[..., None] * sun[None, None, :], axis=-1)
        shadow = np.where((along > 0) & (perp < R), 0.12, 1.0)
        lit = 0.25 + 0.9 * np.clip(np.abs(np.einsum("k,k->", ax, sun)) * 0.0 + 0.8, 0, 1)
        rc = rcol * (shadow * lit)[..., None] * F32(1.25)
        ra = np.where(planet_front, 0.0, ralpha)
        reg = img[g.y0:g.y1, g.x0:g.x1]
        order_ring_front = inring & (~planet_front) & ((tpl < g.t) | (g.cov < 0.5))
        # composite planet first then ring where ring is in front (or planet absent)
        blend(img, g, col)
        reg = img[g.y0:g.y1, g.x0:g.x1]
        a = (ra * order_ring_front)[..., None]
        reg *= (1 - a)
        reg += rc * a
    else:
        blend(img, g, col)
    if atmos:
        atmosphere_glow(img, g, sun, color=atmos, strength=0.6)
    return g


def draw_sun(img, cam, C, R, t=0.0, glow=1.0):
    A = get_assets()
    C = np.asarray(C, np.float64)
    sx, sy, z = cam.project(C)
    if z <= 0:
        return None
    dist = float(np.linalg.norm(C - cam.pos))
    rpx = cam.fx * R / math.sqrt(max(dist * dist - R * R, 1e-9))
    # big soft glows (screen space)
    radial_glow(img, sx, sy, rpx * 9 + 40, (1.0, 0.72, 0.38), 0.55 * glow, 2.6, core=0.0)
    radial_glow(img, sx, sy, rpx * 3.2 + 20, (1.0, 0.85, 0.6), 0.9 * glow, 1.6)
    g = sphere_geom(cam, C, R, pad=1.0)
    if g is None:
        return None
    nl = g.n
    u, v = uv_from_local(to_local(g.n, rot_y(t * 0.02)))
    nz = f01(lookup([(A.sun_noise * 255).astype(np.uint8)], u, v, g.rpx, uscale=1))
    mu = g.nv
    limb = 0.35 + 0.65 * mu ** 0.55
    col = np.array([1.0, 0.62, 0.22], F32)[None, None, :] * (1.5 + 0.9 * (nz[..., None] - 0.5)) * limb[..., None]
    col = col + np.array([0.3, 0.25, 0.1], F32) * mu[..., None]
    blend(img, g, col)
    return g


# ------------------------------------------------------------------- post FX
@functools.lru_cache(maxsize=4)
def _vignette():
    ys, xs = np.mgrid[0:RH, 0:RW].astype(F32)
    r = np.sqrt(((xs - RW / 2) / (RW / 2)) ** 2 + ((ys - RH / 2) / (RH / 2)) ** 2) / 1.41
    return (1 - 0.55 * np.clip(r, 0, 1.2) ** 2.4)[..., None]


def post(img, frame_no=0, bloom=0.55, vignette=1.0, grain=0.012, exposure=1.0, thresh=0.62):
    w4, h4 = max(RW // 4, 1), max(RH // 4, 1)
    small = cv2.resize(img, (w4, h4), interpolation=cv2.INTER_AREA)
    br = np.maximum(small - F32(thresh), 0)
    b = cv2.GaussianBlur(br, (0, 0), 3 * SC) * 0.55 + cv2.GaussianBlur(br, (0, 0), 10 * SC) * 0.6
    b3 = cv2.resize(cv2.GaussianBlur(cv2.resize(br, (max(w4 // 4, 1), max(h4 // 4, 1)), interpolation=cv2.INTER_AREA), (0, 0), 5 * SC),
                    (w4, h4), interpolation=cv2.INTER_LINEAR)
    b = (b + b3 * 0.6)
    img = img * F32(exposure) + cv2.resize(b, (RW, RH), interpolation=cv2.INTER_LINEAR) * F32(bloom)
    if vignette:
        img = img * (1 - vignette * (1 - _vignette()))
    # soft shoulder
    hi = img > 0.8
    img = np.where(hi, 0.8 + 0.2 * np.tanh((img - 0.8) / 0.2), img)
    if grain:
        rng = np.random.default_rng(frame_no + 12345)
        gn = rng.standard_normal((RH // 2 + 1, RW // 2 + 1, 1)).astype(F32)
        gn = cv2.resize(gn, (RW, RH), interpolation=cv2.INTER_LINEAR)[..., None] if gn.ndim == 3 else gn
        img = img + gn * F32(grain) * (0.4 + 0.6 * (1 - np.clip(img.mean(axis=-1, keepdims=True), 0, 1)))
    return img


def to_u8(img):
    return np.clip(img * 255.0 + 0.5, 0, 255).astype(np.uint8)


# ------------------------------------------------------------- text / layers
_FONT_CACHE = {}


def get_font(size, weight="Bold"):
    key = (int(round(size)), weight)
    if key in _FONT_CACHE:
        return _FONT_CACHE[key]
    A = get_assets()
    f = None
    try:
        if os.path.exists(A.font_path):
            f = ImageFont.truetype(A.font_path, int(round(size)))
            try:
                f.set_variation_by_name(weight)
            except Exception:
                pass
    except Exception:
        f = None
    if f is None:
        for cand in ("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", "/usr/share/fonts/truetype/liberation/LiberationSans-Bold.ttf"):
            if os.path.exists(cand):
                f = ImageFont.truetype(cand, int(round(size)))
                break
    if f is None:
        f = ImageFont.load_default()
    _FONT_CACHE[key] = f
    return f


@functools.lru_cache(maxsize=256)
def _text_mask(text, size, weight, spacing, max_w):
    """Return (mask float32 HxW 0..1) for text, wrapped to max_w px. Lines centered."""
    f = get_font(size, weight)
    words = text.split(" ")
    lines, cur = [], ""
    def width(s):
        return sum(f.getlength(ch) + spacing for ch in s)
    for w in words:
        trial = (cur + " " + w).strip()
        if width(trial) <= max_w or not cur:
            cur = trial
        else:
            lines.append(cur); cur = w
    lines.append(cur)
    lh = int(size * 1.22)
    pad = int(size * 0.5)
    W_ = int(max(width(l) for l in lines)) + pad * 2
    H_ = lh * len(lines) + pad * 2
    im = Image.new("L", (W_, H_), 0)
    d = ImageDraw.Draw(im)
    for i, l in enumerate(lines):
        x = pad + (W_ - pad * 2 - width(l)) / 2
        y = pad + i * lh
        for ch in l:
            d.text((x, y), ch, font=f, fill=255)
            x += f.getlength(ch) + spacing
    return np.asarray(im, F32) / 255.0


def blit_mask(img, mask, cx, cy, color, alpha=1.0, glow=0.0, glow_color=None, anchor="c"):
    h, w = mask.shape
    if anchor == "c":
        x0, y0 = int(cx - w / 2), int(cy - h / 2)
    else:  # top-left
        x0, y0 = int(cx), int(cy)
    xs0, ys0 = max(x0, 0), max(y0, 0)
    xs1, ys1 = min(x0 + w, RW), min(y0 + h, RH)
    if xs1 <= xs0 or ys1 <= ys0 or alpha <= 0:
        return
    m = mask[ys0 - y0:ys1 - y0, xs0 - x0:xs1 - x0]
    reg = img[ys0:ys1, xs0:xs1]
    if glow > 0:
        gl = cv2.GaussianBlur(mask, (0, 0), max(px(10), 1))[ys0 - y0:ys1 - y0, xs0 - x0:xs1 - x0]
        gc = np.array(glow_color if glow_color is not None else color, F32)
        reg += gl[..., None] * gc * F32(glow * alpha)
    reg *= (1 - (m * alpha)[..., None])
    reg += m[..., None] * np.array(color, F32) * F32(alpha)


def title_text(img, text, t, t_in, t_out, y=0.5, size=92, color=(1, 1, 1), glow=0.9, weight="ExtraBold",
               spacing=0.06, fade=0.6, drift=18, glow_color=(0.35, 0.7, 1.0)):
    """Big cinematic title with fade/drift. t, t_in, t_out are in the same time base (seconds)."""
    if t < t_in or t > t_out:
        return
    a = smooth((t - t_in) / fade) * smooth((t_out - t) / fade)
    if a <= 0:
        return
    s = px(size)
    mask = _text_mask(text, int(s), weight, s * spacing, int(RW * 0.86))
    k = smooth((t - t_in) / (fade * 2.0))
    cy = RH * y + px(drift) * (1 - k)
    blit_mask(img, mask, RW / 2, cy, color, a, glow=glow, glow_color=glow_color)


def small_text(img, text, x, y, size=34, color=(1, 1, 1), alpha=1.0, weight="SemiBold", spacing=0.02, max_w=900, anchor="tl", glow=0.0):
    s = px(size)
    mask = _text_mask(text, int(s), weight, s * spacing, int(px(max_w)))
    if anchor == "c":
        blit_mask(img, mask, x, y, color, alpha, glow=glow)
    else:
        blit_mask(img, mask, x - s * 0.5, y - s * 0.5, color, alpha, anchor="tl", glow=glow)


def fact_box(img, lines, t, t_in, t_out, pos="bl", accent=(0.35, 0.75, 1.0), size=34, width=760, fade=0.5, head=None):
    """Lower-third 'did you know' box."""
    if t < t_in or t > t_out:
        return
    a = smooth((t - t_in) / fade) * smooth((t_out - t) / fade)
    if a <= 0:
        return
    s = px(size)
    text = lines if isinstance(lines, str) else " ".join(lines)
    mask = _text_mask(text, int(s), "SemiBold", s * 0.01, int(px(width)))
    h, w = mask.shape
    hd = None
    if head:
        hd = _text_mask(head, int(s * 0.62), "ExtraBold", s * 0.12, int(px(width)))
    pad = px(26)
    bw = w + pad * 2 - s; bh = h + pad * 2 - s * 0.6 + (hd.shape[0] - s * 0.4 if hd is not None else 0)
    x0 = px(80) - (1 - smooth((t - t_in) / (fade * 1.5))) * px(40)
    y1 = RH - px(120)
    y0 = y1 - bh
    xa, ya, xb, yb = int(x0), int(y0), int(x0 + bw), int(y1)
    xa, xb = max(xa, 0), min(xb, RW)
    reg = img[ya:yb, xa:xb]
    reg *= (1 - 0.55 * a)
    reg += np.array([0.02, 0.05, 0.1], F32) * (0.55 * a)
    img[ya:yb, xa:min(xa + int(px(6)), RW)] = np.array(accent, F32) * a + img[ya:yb, xa:min(xa + int(px(6)), RW)] * (1 - a)
    ty = y0 + pad - s * 0.15
    if hd is not None:
        blit_mask(img, hd, x0 + pad + px(10) - s * 0.3, ty - s * 0.2, accent, a, anchor="tl")
        ty += hd.shape[0] - s * 0.5
    blit_mask(img, mask, x0 + pad + px(10) - s * 0.5, ty - s * 0.1, (0.95, 0.97, 1.0), a, anchor="tl")


def caption(img, text, alpha=1.0, size=36):
    if not text or alpha <= 0:
        return
    s = px(size)
    mask = _text_mask(text, int(s), "SemiBold", s * 0.01, int(RW * 0.78))
    h, w = mask.shape
    cx, cy = RW / 2, RH - px(70) - h / 2 + s * 0.2
    x0, x1 = int(cx - w / 2 + s * 0.2), int(cx + w / 2 - s * 0.2)
    y0, y1 = int(cy - h / 2 + s * 0.3), int(cy + h / 2 - s * 0.1)
    x0, y0 = max(x0, 0), max(y0, 0); x1, y1 = min(x1, RW), min(y1, RH)
    reg = img[y0:y1, x0:x1]
    reg *= (1 - 0.5 * alpha)
    blit_mask(img, mask, cx, cy, (1, 1, 1), alpha)


class Layer:
    """Anti-aliased vector overlay (lines, arcs, ellipses, polygons) using PIL at 2x."""

    def __init__(self, ss=2):
        self.ss = ss
        self.im = Image.new("RGBA", (RW * ss, RH * ss), (0, 0, 0, 0))
        self.d = ImageDraw.Draw(self.im)

    def _c(self, p):
        return (float(p[0]) * self.ss, float(p[1]) * self.ss)

    def line(self, p0, p1, color, width=3, alpha=1.0):
        c = tuple(int(255 * x) for x in color) + (int(255 * alpha),)
        self.d.line([self._c(p0), self._c(p1)], fill=c, width=max(int(width * SC * self.ss), 1))

    def polyline(self, pts, color, width=3, alpha=1.0):
        c = tuple(int(255 * x) for x in color) + (int(255 * alpha),)
        self.d.line([self._c(p) for p in pts], fill=c, width=max(int(width * SC * self.ss), 1), joint="curve")

    def ellipse(self, cx, cy, rx, ry, color, width=3, alpha=1.0, fill=False, fill_alpha=0.25):
        box = [(cx - rx) * self.ss, (cy - ry) * self.ss, (cx + rx) * self.ss, (cy + ry) * self.ss]
        if fill:
            self.d.ellipse(box, fill=tuple(int(255 * x) for x in color) + (int(255 * fill_alpha),))
        if width > 0:
            self.d.ellipse(box, outline=tuple(int(255 * x) for x in color) + (int(255 * alpha),), width=max(int(width * SC * self.ss), 1))

    def polygon(self, pts, color, alpha=1.0):
        self.d.polygon([self._c(p) for p in pts], fill=tuple(int(255 * x) for x in color) + (int(255 * alpha),))

    def rect(self, x0, y0, x1, y1, color, alpha=1.0, radius=0, outline=None, width=2):
        box = [x0 * self.ss, y0 * self.ss, x1 * self.ss, y1 * self.ss]
        fill = tuple(int(255 * x) for x in color) + (int(255 * alpha),) if color is not None else None
        out = tuple(int(255 * x) for x in outline) + (255,) if outline is not None else None
        if radius:
            self.d.rounded_rectangle(box, radius=radius * self.ss, fill=fill, outline=out, width=int(width * self.ss))
        else:
            self.d.rectangle(box, fill=fill, outline=out, width=int(width * self.ss))

    def arrow(self, p0, p1, color, width=4, alpha=1.0, head=18):
        self.line(p0, p1, color, width, alpha)
        a = math.atan2(p1[1] - p0[1], p1[0] - p0[0])
        hs = head * SC
        pts = [p1, (p1[0] - hs * math.cos(a - 0.45), p1[1] - hs * math.sin(a - 0.45)),
               (p1[0] - hs * math.cos(a + 0.45), p1[1] - hs * math.sin(a + 0.45))]
        self.polygon(pts, color, alpha)

    def over(self, img, glow=0.0, opacity=1.0):
        arr = np.asarray(self.im, np.uint8)
        if self.ss > 1:
            a = arr[..., 3:4].astype(F32) / 255
            rgb = arr[..., :3].astype(F32) / 255 * a
            rgb = cv2.resize(rgb, (RW, RH), interpolation=cv2.INTER_AREA)
            a = cv2.resize(a[..., 0], (RW, RH), interpolation=cv2.INTER_AREA)[..., None]
        else:
            a = arr[..., 3:4].astype(F32) / 255
            rgb = arr[..., :3].astype(F32) / 255 * a
        if glow > 0:
            gl = cv2.GaussianBlur(cv2.resize(rgb, (RW // 2, RH // 2), interpolation=cv2.INTER_AREA), (0, 0), px(6))
            img += cv2.resize(gl, (RW, RH), interpolation=cv2.INTER_LINEAR) * F32(glow * opacity)
        img *= (1 - a * F32(opacity))
        img += rgb * F32(opacity)
        return img
