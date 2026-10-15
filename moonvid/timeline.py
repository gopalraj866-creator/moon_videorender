"""Master timeline: 0:00 - 8:00, seven scenes, ~60 shots. Shared by the video renderer and the audio builder."""
import json, os, math
import numpy as np
from . import engine as E, shots as S
from .engine import F32, smooth, Layer, px

FPS = 30
DURATION = 480.0
TOTAL_FRAMES = int(DURATION * FPS)


class Shot:
    def __init__(self, t0, t1, fn, titles=(), facts=(), xf=0.5, cut=False, post=None, sfx=()):
        self.t0, self.t1, self.fn = t0, t1, fn
        self.titles, self.facts = list(titles), list(facts)
        self.xf = 0.0 if cut else xf
        self.post = post or {}
        self.sfx = list(sfx)
        self.dur = t1 - t0

    def render(self, t):
        return self.fn(S.Ctx(t, t - self.t0, self.dur))


def T(text, a, b, y=0.5, size=88, hit=True, color=(1, 1, 1), glow_color=(0.35, 0.7, 1.0)):
    return dict(text=text, a=a, b=b, y=y, size=size, hit=hit, color=color, glow_color=glow_color)


def F(text, a, b, head="DID YOU KNOW?"):
    return dict(text=text, a=a, b=b, head=head)


g = S.ground_shot
SHOTS = [
    # ------------------------------------------------ SCENE 1 (0:00 - 0:20) the mystery
    Shot(0, 7, S.orbit_wide, titles=[T("WHAT IF THE MOON VANISHED TONIGHT?", 0.5, 6.6, y=0.22, size=76)], cut=True),
    Shot(7, 13, S.earth_night_close, facts=[F("The Moon is 384,400 km away.", 8.0, 12.4)]),
    Shot(13, 20, S.moon_push, xf=0.6),
    # ------------------------------------------------ SCENE 2 (0:20 - 0:50) the disappearance
    Shot(20, 27, lambda c: S.moon_push(c, True), titles=[T("THE MOON IS GONE.", 21.4, 26.4, y=0.2, size=96)], cut=True),
    Shot(27, 35, lambda c: g(c, "forest", 0.86, [(0, 0.0)], star=1.25, mw=1.5, sky_speed=0.012, moon_xy=(0.7, 0.2)), cut=True),
    Shot(35, 43, S.empty_orbit),
    Shot(43, 50, lambda c: S.earth_sun_orbit(c), facts=[F("The Moon has just 1.2% of Earth's mass.", 44.0, 49.0)]),
    # ------------------------------------------------ SCENE 3 (0:50 - 1:40) a darker night
    Shot(50, 58, lambda c: g(c, "city", 0.66, [(0, 1), (3.0, 1), (6.5, 0)], pan=(0, 0.015)),
         titles=[T("NO MORE MOONLIGHT", 51.2, 57.4, y=0.16, size=84)], cut=True),
    Shot(58, 66, lambda c: g(c, "forest", 0.72, [(0, 1), (2.5, 1), (5.5, 0)], pan=(0.0, -0.02)), xf=0.6),
    Shot(66, 74, lambda c: g(c, "ocean", 0.5, [(0, 1), (2.5, 1), (5.5, 0)], star=1.1), xf=0.6),
    Shot(74, 83, lambda c: S.earth_night_close(c, lon=15, flip=-1.0),
         facts=[F("Full Moon: ~0.1 lux. Starlight alone: ~0.001 lux.", 75.0, 82.0, "MOONLIGHT vs STARLIGHT")]),
    Shot(83, 92, lambda c: g(c, "desert", 0.86, [(0, 0.0)], star=1.3, mw=1.8, sky_speed=0.05),
         facts=[F("Without moonlight, the Milky Way would be easier to see.", 84.0, 91.0)]),
    Shot(92, 100, lambda c: g(c, "city", 0.66, [(0, 0.0)], pan=(0.02, -0.01)),
         facts=[F("Cities keep glowing. Wilderness goes truly dark.", 93.0, 99.0)]),
    # ------------------------------------------------ SCENE 4 (1:40 - 3:00) tides
    Shot(100, 108, lambda c: S.tides_top(c, "moon"), titles=[T("TIDES WOULD CHANGE", 100.8, 107.0, y=0.12, size=80)], cut=True),
    Shot(108, 120, lambda c: S.tides_top(c, "gravity")),
    Shot(120, 130, lambda c: S.tides_top(c, "both"), facts=[F("Solar tides are ~46% as strong as lunar tides.", 121.0, 129.0)]),
    Shot(130, 140, lambda c: S.tides_top(c, "vanish"), titles=[T("LUNAR TIDES: GONE", 134.0, 139.0, y=0.9, size=64)]),
    Shot(140, 152, S.coast_tides),
    Shot(152, 162, lambda c: S.earth_day(c, "spin"), facts=[F("Tide pools, salt marshes and mangroves depend on the tides.", 153.0, 161.0, "COASTAL LIFE")]),
    Shot(162, 172, lambda c: S.earth_day(c, "clock")),
    Shot(172, 180, lambda c: S.tides_top(c, "solar"), titles=[T("SOLAR TIDES WOULD STAY", 173.0, 179.0, y=0.12, size=68)]),
    # ------------------------------------------------ SCENE 5 (3:00 - 4:30) tilt
    Shot(180, 190, S.tilt_intro, titles=[T("COULD EARTH'S CLIMATE CHANGE?", 180.6, 189.0, y=0.1, size=66)], cut=True),
    Shot(190, 200, S.tilt_seasons, facts=[F("Tilt, not distance from the Sun, causes the seasons.", 191.0, 199.0, "SEASONS")]),
    Shot(200, 212, S.tilt_stable),
    Shot(212, 232, S.tilt_compare),
    Shot(232, 245, S.tilt_extremes),
    Shot(245, 258, S.mars_wobble),
    Shot(258, 270, S.timescale, titles=[T("NOT AN INSTANT CATASTROPHE", 259.0, 268.0, y=0.25, size=68)]),
    # ------------------------------------------------ SCENE 6 (4:30 - 6:00) what else changes
    Shot(270, 282, S.lunar_eclipse, titles=[T("NO MORE LUNAR ECLIPSES", 271.0, 280.0, y=0.14, size=72)], cut=True),
    Shot(282, 294, S.solar_eclipse, titles=[T("NO MORE SOLAR ECLIPSES", 283.0, 292.0, y=0.84, size=64)]),
    Shot(294, 308, S.phases, titles=[T("NO MORE MOON PHASES", 295.0, 303.0, y=0.07, size=64, hit=False)]),
    Shot(308, 322, S.calendars, titles=[T("THE OLDEST CALENDAR", 309.0, 316.0, y=0.08, size=60, hit=False)]),
    Shot(322, 336, S.coral_spawn, facts=[F("Great Barrier Reef corals spawn in sync after a full Moon.", 323.0, 335.0, "ECOSYSTEMS")]),
    Shot(336, 348, lambda c: S.orbit_top(c, "recede")),
    Shot(348, 360, lambda c: S.orbit_top(c, "vanish")),
    # ------------------------------------------------ SCENE 7 (6:00 - 8:00) the final question
    Shot(360, 372, lambda c: S.earth_hero(c, moon=False), titles=[T("WOULD EARTH STILL SURVIVE?", 361.0, 371.0, y=0.14, size=76)], cut=True),
    Shot(372, 386, lambda c: S.earth_sun_orbit(c), titles=[T("EARTH WOULD STAY IN ORBIT", 373.0, 384.0, y=0.12, size=64, hit=False)]),
    Shot(386, 398, S.earth_limb, facts=[F("Earth's gravity holds our atmosphere, not the Moon's.", 387.0, 397.0)]),
    Shot(398, 414, S.theia, titles=[T("HOW THE MOON WAS BORN", 399.0, 406.0, y=0.12, size=68)],
         facts=[F("~4.5 billion years ago a Mars-sized body, Theia, hit the young Earth.", 407.0, 413.0, "GIANT IMPACT")]),
    Shot(414, 426, lambda c: S.earth_hero(c, moon=True), facts=[F("Our Moon is unusually large for a rocky planet's companion.", 415.0, 425.0)]),
    Shot(426, 440, S.system_pullaway, xf=0.8),
    Shot(440, 448, lambda c: S.system_planet(c, "Jupiter"), facts=[F("Jupiter and Saturn have scores of moons.", 441.0, 447.0)]),
    Shot(448, 456, lambda c: S.system_planet(c, "Saturn")),
    Shot(456, 468, S.system_wide),
    Shot(468, 480, lambda c: S.earth_hero(c, moon=True),
         titles=[T("THE MOON IS MORE THAN A LIGHT IN THE SKY", 469.0, 478.5, y=0.2, size=62, hit=False)], xf=1.0),
]

# Call-to-action overlays drawn over several shots (global time based)
CTA_COMMENT = (30.6, 42.0)      # "what would change first?"
CTA_END = (459.0, 467.5)        # comment + subscribe


def find_shot(t):
    for i, s in enumerate(SHOTS):
        if s.t0 <= t < s.t1:
            return i
    return len(SHOTS) - 1


def _cta_cards(img, t, a, b, header, labels, y=0.22):
    if t < a or t > b:
        return
    k = smooth((t - a) / 0.7) * smooth((b - t) / 0.7)
    W, H = E.RW, E.RH
    E.title_text(img, header, t, a, b, y=y, size=56, glow=0.8, weight="ExtraBold", glow_color=(1.0, 0.8, 0.4))
    L = Layer()
    n = len(labels)
    cw, ch = W * 0.2, H * 0.14
    gap = W * 0.035
    x0 = (W - (n * cw + (n - 1) * gap)) / 2
    y0 = H * (y + 0.17)
    for i, (lab, col) in enumerate(labels):
        kk = smooth((t - a - 0.7 - 0.25 * i) / 0.5)
        L.rect(x0 + i * (cw + gap), y0 + (1 - kk) * px(20), x0 + i * (cw + gap) + cw, y0 + ch + (1 - kk) * px(20), (0.03, 0.06, 0.12), 0.75 * kk * k, radius=18, outline=col, width=3)
    L.over(img, glow=0.3)
    for i, (lab, col) in enumerate(labels):
        kk = smooth((t - a - 0.7 - 0.25 * i) / 0.5) * k
        E.small_text(img, lab, x0 + i * (cw + gap) + cw / 2, y0 + ch / 2, 36, col, kk, anchor="c", weight="ExtraBold", glow=0.4)
    E.small_text(img, "WRITE YOUR ANSWER IN THE COMMENTS", W / 2, y0 + ch + px(70), 30, (0.9, 0.95, 1.0), smooth((t - a - 1.8) / 0.6) * k, anchor="c")


def _captions(t, caps):
    for c in caps:
        if c["start"] <= t <= c["end"]:
            return c["text"], smooth((t - c["start"]) / 0.15) * smooth((c["end"] - t) / 0.15)
    return None, 0.0


_CAPS = None


def load_captions(path):
    global _CAPS
    if path and os.path.exists(path):
        _CAPS = json.load(open(path))
    else:
        _CAPS = []
    return _CAPS


def render_frame(n, captions=True):
    t = n / FPS
    i = find_shot(t)
    sh = SHOTS[i]
    img = sh.render(t)
    if sh.xf > 0 and i > 0 and t < sh.t0 + sh.xf:
        b = smooth((t - sh.t0) / sh.xf)
        prev = SHOTS[i - 1]
        img = prev.render(t) * (1 - b) + img * b
    # overlays (text) are global-time based so they survive crossfades
    for s2 in SHOTS[max(0, i - 1):i + 2]:
        for ti in s2.titles:
            E.title_text(img, ti["text"], t, ti["a"], ti["b"], y=ti["y"], size=ti["size"], color=ti["color"], glow_color=ti["glow_color"])
        for fa in s2.facts:
            E.fact_box(img, fa["text"], t, fa["a"], fa["b"], head=fa["head"])
    _cta_cards(img, t, CTA_COMMENT[0], CTA_COMMENT[1], "WHAT WOULD CHANGE FIRST?",
               [("THE TIDES", (0.45, 0.8, 1.0)), ("THE NIGHT SKY", (0.8, 0.7, 1.0)), ("EARTH'S CLIMATE", (1.0, 0.8, 0.4))], y=0.07)
    _cta_cards(img, t, CTA_END[0], CTA_END[1], "WHAT'S YOUR ANSWER?",
               [("COMMENT", (0.45, 0.8, 1.0)), ("LIKE", (1.0, 0.6, 0.6)), ("SUBSCRIBE", (1.0, 0.8, 0.4))], y=0.05)
    # global fades: gentle open, long fade-out
    fade = 1.0
    if t < 0.4:
        fade *= 0.35 + 0.65 * smooth(t / 0.4)
    if t > DURATION - 2.5:
        fade *= smooth((DURATION - t) / 2.5)
    img = E.post(img * F32(fade), frame_no=n, **sh.post)
    if captions and _CAPS:
        txt, a = _captions(t, _CAPS)
        if txt:
            E.caption(img, txt, a)
    return E.to_u8(img)
