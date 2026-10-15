"""Audio builder: narration (edge-tts -> gTTS -> espeak-ng -> silent fallback), cinematic synth score,
sound design (boom / silence drop / whooshes / ambience) and sidechain ducking.

Outputs (in ./build):  audio.wav   captions.json   narration_timing.json
"""
import os, sys, json, math, subprocess, asyncio, shutil, re, wave
import numpy as np
from scipy import signal
from . import script as SC
from .timeline import SHOTS, DURATION

SR = 44100
BUILD = os.environ.get("BUILD_DIR", os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "build"))
TMP = os.path.join(BUILD, "tts")
RNG = np.random.default_rng(1234)


# --------------------------------------------------------------------------- helpers
def read_wav(path):
    with wave.open(path, "rb") as w:
        n, ch, sw, sr = w.getnframes(), w.getnchannels(), w.getsampwidth(), w.getframerate()
        raw = w.readframes(n)
    a = np.frombuffer(raw, dtype={1: np.uint8, 2: np.int16, 4: np.int32}[sw]).astype(np.float32)
    a = (a - 128) / 128 if sw == 1 else a / float(2 ** (8 * sw - 1))
    if ch > 1:
        a = a.reshape(-1, ch).mean(axis=1)
    if sr != SR:
        a = signal.resample_poly(a, SR, sr).astype(np.float32)
    return a


def write_wav(path, stereo):
    st = np.clip(stereo, -1, 1)
    pcm = (st * 32767).astype(np.int16)
    with wave.open(path, "wb") as w:
        w.setnchannels(2); w.setsampwidth(2); w.setframerate(SR)
        w.writeframes(pcm.tobytes())


def ffmpeg_to_wav(src, dst):
    subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-i", src, "-ar", str(SR), "-ac", "1", dst], check=True)


def lp(x, f, order=2):
    return signal.sosfilt(signal.butter(order, f, "low", fs=SR, output="sos"), x)


def hp(x, f, order=2):
    return signal.sosfilt(signal.butter(order, f, "high", fs=SR, output="sos"), x)


def bp(x, f0, f1, order=2):
    return signal.sosfilt(signal.butter(order, [f0, f1], "band", fs=SR, output="sos"), x)


def put(buf, t, clip, gain=1.0):
    i = int(t * SR)
    if i >= len(buf):
        return
    j = min(len(buf), i + len(clip))
    buf[i:j] += clip[:j - i] * gain


def env_ad(n, a, d, power=2.0):
    t = np.arange(n) / SR
    return np.minimum(t / max(a, 1e-4), 1.0) * np.exp(-t / d) ** power if False else np.minimum(t / max(a, 1e-4), 1.0) * np.exp(-t * power / max(d, 1e-4))


def reverb(x, seconds=2.6, wet=0.3, damp=3500, seed=0):
    """Stereo reverb from a synthetic decaying-noise IR. x: mono or (n,2). Returns (n,2)."""
    r = np.random.default_rng(seed)
    n = int(seconds * SR)
    t = np.arange(n) / SR
    outs = []
    xs = [x, x] if x.ndim == 1 else [x[:, 0], x[:, 1]]
    for ch in range(2):
        ir = r.standard_normal(n) * np.exp(-t * 6.9 / seconds)
        ir = lp(ir, damp, 1)
        ir[:int(0.012 * SR)] *= np.linspace(0, 1, int(0.012 * SR))
        ir /= np.sqrt((ir ** 2).sum()) + 1e-9
        outs.append(signal.oaconvolve(xs[ch], ir)[:len(xs[ch])])
    wet_sig = np.stack(outs, 1)
    dry = np.stack(xs, 1)
    return dry * (1 - wet * 0.5) + wet_sig * wet * 2.2


# --------------------------------------------------------------------------- TTS
def tts_edge(text, out_mp3, voice, rate):
    import edge_tts
    async def go():
        await edge_tts.Communicate(text, voice, rate=rate).save(out_mp3)
    asyncio.run(go())


def tts_gtts(text, out_mp3):
    from gtts import gTTS
    gTTS(text=text, lang="en", tld="co.in" if os.environ.get("GTTS_INDIA") else "com").save(out_mp3)


def tts_espeak(text, out_wav):
    exe = shutil.which("espeak-ng") or shutil.which("espeak")
    if not exe:
        raise RuntimeError("espeak not installed")
    subprocess.run([exe, "-v", "en-us+m3", "-s", "150", "-p", "40", "-w", out_wav, text], check=True)


def synth_line(i, text):
    os.makedirs(TMP, exist_ok=True)
    wav = os.path.join(TMP, "line%03d.wav" % i)
    if os.path.exists(wav):
        return read_wav(wav), "cache"
    voice = os.environ.get("VOICE", SC.VOICE)
    errs = []
    mode = os.environ.get("TTS", "auto")
    chain = ["edge", "gtts", "espeak"] if mode == "auto" else [mode]
    for engine in chain:
        try:
            if engine == "edge":
                mp3 = wav[:-4] + ".mp3"; tts_edge(text, mp3, voice, SC.VOICE_RATE); ffmpeg_to_wav(mp3, wav)
            elif engine == "gtts":
                mp3 = wav[:-4] + ".mp3"; tts_gtts(text, mp3); ffmpeg_to_wav(mp3, wav)
            elif engine == "espeak":
                raw = wav[:-4] + "_raw.wav"; tts_espeak(text, raw); ffmpeg_to_wav(raw, wav)
            return read_wav(wav), engine
        except Exception as e:  # noqa
            errs.append("%s: %s" % (engine, str(e)[:80]))
    print("  [tts] all engines failed for line %d (%s) -> silent placeholder" % (i, "; ".join(errs)), file=sys.stderr)
    est = max(1.5, len(text.split()) / 2.5)
    return np.zeros(int(est * SR), np.float32), "silent"


def tempo(x, factor):
    """Speed up speech with ffmpeg atempo (keeps pitch)."""
    if factor <= 1.005:
        return x
    tmp_in, tmp_out = os.path.join(TMP, "_t_in.wav"), os.path.join(TMP, "_t_out.wav")
    with wave.open(tmp_in, "wb") as w:
        w.setnchannels(1); w.setsampwidth(2); w.setframerate(SR)
        w.writeframes((np.clip(x, -1, 1) * 32767).astype(np.int16).tobytes())
    subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-i", tmp_in, "-filter:a", "atempo=%.4f" % factor, tmp_out], check=True)
    return read_wav(tmp_out)


def build_narration():
    total = int(DURATION * SR)
    track = np.zeros(total, np.float32)
    timing = []
    cursor = 0.0
    lines = SC.NARRATION
    clips = []
    for i, (start, text) in enumerate(lines):
        x, eng = synth_line(i, text)
        # trim leading/trailing silence
        thr = 0.005
        nz = np.where(np.abs(x) > thr)[0]
        if len(nz):
            x = x[max(nz[0] - int(0.03 * SR), 0): nz[-1] + int(0.08 * SR)]
        clips.append((start, text, x, eng))
    print("  [tts] engines used:", sorted(set(c[3] for c in clips)))
    for i, (start, text, x, eng) in enumerate(clips):
        t0 = max(start, cursor + 0.25)
        nxt = clips[i + 1][0] if i + 1 < len(clips) else DURATION - 0.2
        slot = max(nxt - t0 - 0.25, 1.0)
        dur = len(x) / SR
        if dur > slot:
            f = min(dur / slot, 1.14)
            x = tempo(x, f)
            dur = len(x) / SR
        t1 = t0 + dur
        if t1 > DURATION - 0.1:
            x = x[:int((DURATION - 0.1 - t0) * SR)]
            t1 = DURATION - 0.1
        fade = np.minimum(1.0, np.minimum(np.arange(len(x)) / (0.02 * SR), (len(x) - np.arange(len(x))) / (0.04 * SR)))
        put(track, t0, x * fade)
        timing.append(dict(start=round(t0, 3), end=round(t1, 3), text=text, engine=eng, shifted=round(t0 - start, 2)))
        cursor = t1
    return track, timing


def make_captions(timing, max_words=11):
    caps = []
    for it in timing:
        parts = re.split(r"(?<=[\.\?\!:;,])\s+", it["text"].replace("...", "…"))
        chunks, cur = [], ""
        for p in parts:
            if cur and len((cur + " " + p).split()) > max_words:
                chunks.append(cur); cur = p
            else:
                cur = (cur + " " + p).strip()
        if cur:
            chunks.append(cur)
        total_chars = sum(len(c) for c in chunks)
        t = it["start"]; dur = it["end"] - it["start"]
        for c in chunks:
            d = dur * len(c) / total_chars
            caps.append(dict(start=round(t, 3), end=round(t + d - 0.04, 3), text=c))
            t += d
    return caps


# --------------------------------------------------------------------------- synth music
NOTE = {n: 440 * 2 ** ((m - 69) / 12) for n, m in zip(
    ["C2", "D2", "E2", "F2", "G2", "A2", "B2", "C3", "D3", "E3", "F3", "G3", "A3", "B3", "C4", "D4", "E4", "G4", "A4", "C5", "E5"],
    [36, 38, 40, 41, 43, 45, 47, 48, 50, 52, 53, 55, 57, 59, 60, 62, 64, 67, 69, 72, 76])}
CHORDS = [["A2", "E3", "A3", "C4"], ["F2", "C3", "F3", "A3"], ["C3", "G3", "C4", "E4"], ["G2", "D3", "G3", "B3"]]


def pad_segment(freqs, dur, gain=1.0):
    n = int(dur * SR)
    t = np.arange(n) / SR
    out = np.zeros(n, np.float32)
    for f in freqs:
        for det in (-0.0035, 0.0, 0.004):
            ff = f * (1 + det)
            ph = RNG.random() * 6.28
            for k, a in ((1, 1.0), (2, 0.45), (3, 0.22), (4, 0.12), (5, 0.06)):
                out += (a / len(freqs) * 0.33 * np.sin(2 * np.pi * ff * k * t + ph * k)).astype(np.float32)
    # slow swell + tremolo
    out *= (0.75 + 0.25 * np.sin(2 * np.pi * t / dur * 1.0 + 1.0)).astype(np.float32) * gain
    return out


def build_music():
    n = int(DURATION * SR)
    pad = np.zeros(n, np.float32)
    seg = 12.0
    t0 = 0.0
    idx = 0
    while t0 < DURATION:
        ch = CHORDS[idx % 4]
        freqs = [NOTE[x] for x in ch]
        s = pad_segment(freqs, seg + 5.0)
        m = len(s)
        a = np.minimum(np.arange(m) / (4.0 * SR), 1.0)
        r = np.minimum((m - np.arange(m)) / (5.0 * SR), 1.0)
        put(pad, t0, s * a * r * 0.9)
        t0 += seg
        idx += 1
    pad = lp(pad, 1800, 2)
    # sub drone
    t = np.arange(n) / SR
    drone = (np.sin(2 * np.pi * 55 * t) * 0.5 + np.sin(2 * np.pi * 82.41 * t) * 0.22 + np.sin(2 * np.pi * 110.3 * t) * 0.12)
    drone *= (0.6 + 0.4 * np.sin(2 * np.pi * t / 23.0))
    drone = drone.astype(np.float32)
    # space wind
    wind = lp(RNG.standard_normal(n).astype(np.float32), 500, 2) * (0.5 + 0.5 * np.sin(2 * np.pi * t / 17.0 + 1)) * 0.5
    # sparse plucks (A minor pentatonic) - denser in the middle scenes
    plucks = np.zeros(n, np.float32)
    scale = [NOTE[k] for k in ("A3", "C4", "D4", "E4", "G4", "A4", "C5", "E5")]
    tt = 4.0
    while tt < DURATION - 3:
        dens = 0.55 if tt < 60 else (1.0 if tt < 360 else 0.7)
        f = RNG.choice(scale)
        dur = 3.0
        m = int(dur * SR); x = np.arange(m) / SR
        note = (np.sin(2 * np.pi * f * x) + 0.3 * np.sin(2 * np.pi * f * 2.01 * x)) * np.exp(-x * 1.7)
        put(plucks, tt, note.astype(np.float32), 0.10)
        tt += RNG.uniform(2.2, 4.6) / dens
    # heartbeat pulse in the darkest stretch (scene 3) and the finale lead-in
    beat = np.zeros(n, np.float32)
    for (a, b) in ((60, 100), (440, 470)):
        tb = float(a)
        while tb < b:
            m = int(0.5 * SR); x = np.arange(m) / SR
            thump = np.sin(2 * np.pi * (48 + 40 * np.exp(-x * 14)) * x) * np.exp(-x * 9)
            put(beat, tb, thump.astype(np.float32), 0.35)
            put(beat, tb + 0.22, thump.astype(np.float32), 0.22)
            tb += 1.35
    music = pad * 0.9 + drone * 0.16 + wind * 0.10 + plucks * 1.0 + beat * 0.6
    st = reverb(music, 3.0, 0.28, 4000, seed=3)
    # width: tiny Haas delay on one channel
    d = int(0.011 * SR)
    st[d:, 1] = st[:-d, 1] * 0.9 + st[d:, 1] * 0.1
    # global music arc
    tc = np.arange(n) / SR
    arc = np.interp(tc, [0, 12, 19.6, 20.0, 24.0, 40, 60, 120, 180, 270, 360, 420, 468, 480],
                    [0.5, 0.8, 1.0, 0.0, 0.0, 0.55, 0.45, 0.75, 0.85, 0.7, 0.9, 1.0, 0.8, 0.3])
    st *= arc[:, None].astype(np.float32)
    return st


# --------------------------------------------------------------------------- sound design
def noise(n):
    return RNG.standard_normal(n).astype(np.float32)


def whoosh(dur=1.3, up=True):
    n = int(dur * SR)
    x = noise(n)
    t = np.linspace(0, 1, n)
    fc = (300 + 4500 * (t if up else 1 - t) ** 2)
    out = np.zeros(n, np.float32)
    blocks = 40
    for b in range(blocks):
        s, e = b * n // blocks, (b + 1) * n // blocks
        f = float(fc[(s + e) // 2])
        out[s:e] = bp(x[max(0, s - 2000):e], f * 0.7, min(f * 1.4, 18000), 2)[-(e - s):]
    env = np.sin(np.pi * t) ** 2
    return (out * env * 0.5).astype(np.float32)


def boom():
    dur = 3.5
    n = int(dur * SR)
    t = np.arange(n) / SR
    f = 28 + 90 * np.exp(-t * 3.2)
    ph = 2 * np.pi * np.cumsum(f) / SR
    sub = np.sin(ph) * np.exp(-t * 1.5)
    burst = lp(noise(n), 700, 2) * np.exp(-t * 6)
    crack = hp(noise(n), 1500, 2) * np.exp(-t * 25) * 0.25
    x = (sub * 1.0 + burst * 0.8 + crack).astype(np.float32)
    return x


def riser(dur):
    n = int(dur * SR)
    t = np.linspace(0, 1, n)
    x = noise(n)
    out = np.zeros(n, np.float32)
    blocks = 50
    for b in range(blocks):
        s, e = b * n // blocks, (b + 1) * n // blocks
        f = 200 + 5000 * (b / blocks) ** 2.2
        out[s:e] = bp(x[max(0, s - 2000):e], f * 0.8, min(f * 1.3, 18000), 2)[-(e - s):]
    tone = np.sin(2 * np.pi * np.cumsum(80 + 500 * t ** 3) / SR)
    return ((out * 0.5 + tone * 0.25) * (t ** 2.2)).astype(np.float32)


def hit(freq=62, dur=1.6):
    n = int(dur * SR); x = np.arange(n) / SR
    s = np.sin(2 * np.pi * (freq + 50 * np.exp(-x * 25)) * x) * np.exp(-x * 3.2)
    return (s + lp(noise(n), 400, 2) * np.exp(-x * 14) * 0.5).astype(np.float32)


def tick():
    n = int(0.12 * SR); x = np.arange(n) / SR
    return (np.sin(2 * np.pi * 1320 * x) * np.exp(-x * 40) * 0.6 + np.sin(2 * np.pi * 1980 * x) * np.exp(-x * 55) * 0.3).astype(np.float32)


def ambience(kind, dur):
    n = int(dur * SR); t = np.arange(n) / SR
    if kind == "crickets":
        out = np.zeros(n, np.float32)
        for f0, rate in ((4300, 21.0), (4800, 24.0), (3900, 18.0)):
            grp = (np.sin(2 * np.pi * (0.6 + RNG.random() * 0.5) * t + RNG.random() * 6) > 0.1).astype(np.float32)
            pulse = (np.sin(2 * np.pi * rate * t) > 0.2).astype(np.float32)
            out += np.sin(2 * np.pi * f0 * t) * grp * pulse * 0.05
        out = lp(out, 7000, 1)
        return out + lp(noise(n), 900, 2) * 0.03 * (0.5 + 0.5 * np.sin(2 * np.pi * t / 9))
    if kind == "waves":
        env = 0.35 + 0.65 * (0.5 + 0.5 * np.sin(2 * np.pi * t / 7.0 - 1.0)) ** 2
        return (lp(noise(n), 900, 2) * env * 0.5).astype(np.float32)
    if kind == "wind":
        return (bp(noise(n), 200, 900, 2) * (0.4 + 0.6 * (0.5 + 0.5 * np.sin(2 * np.pi * t / 11 + 2))) * 0.4).astype(np.float32)
    if kind == "rumble":
        return (lp(np.cumsum(noise(n)) * 0.0008, 90, 2)).astype(np.float32) * 6
    if kind == "underwater":
        x = lp(noise(n), 380, 2) * 0.6
        for _ in range(int(dur * 1.4)):
            ts = RNG.uniform(0, dur - 0.3); m = int(0.12 * SR); xx = np.arange(m) / SR
            put(x, ts, (np.sin(2 * np.pi * RNG.uniform(500, 1400) * xx) * np.exp(-xx * 30) * 0.12).astype(np.float32))
        return x.astype(np.float32)
    return np.zeros(n, np.float32)


AMBIENCE = [  # (t0, t1, kind, gain)
    (0, 480, "rumble", 0.55),
    (27, 35, "crickets", 0.9), (27, 35, "wind", 0.6),
    (50, 58, "crickets", 1.0), (58, 66, "crickets", 1.0), (58, 66, "wind", 0.6),
    (66, 74, "waves", 1.0), (66, 74, "wind", 0.5),
    (83, 92, "wind", 0.7), (83, 92, "crickets", 0.7), (92, 100, "crickets", 0.7),
    (140, 152, "waves", 1.0),
    (322, 336, "underwater", 1.0),
]

# (time, sound, gain)
SFX = [
    (17.4, "riser", 1.0, 2.6),        # tension rise right before the drop
    (20.0, "boom", 1.0),
    (13.0, "whoosh_up", 0.5), (27.0, "whoosh_down", 0.4), (35.0, "whoosh_up", 0.4),
    (50.0, "hit", 0.6), (100.0, "hit", 0.7), (180.0, "hit", 0.7), (258.4, "riser", 0.7, 1.6), (270.0, "hit", 0.8),
    (359.0, "riser", 0.8, 1.0), (360.0, "hit", 0.9), (398.0, "whoosh_up", 0.5), (421.0, "whoosh_up", 0.6), (426.0, "whoosh_up", 0.7),
    (455.0, "riser", 0.6, 1.0), (468.0, "whoosh_down", 0.5),
]
for s in SHOTS:  # whoosh on most shot changes (soft) + tick/hit on titles
    if s.t0 > 0 and s.t0 not in (13, 20, 27, 35, 50, 100, 180, 270, 360, 398, 426, 468) and int(s.t0) % 2 == 0:
        SFX.append((s.t0 - 0.05, "whoosh_up" if int(s.t0) % 4 == 0 else "whoosh_down", 0.22))
    for ti in s.titles:
        SFX.append((ti["a"], "hit" if ti["hit"] else "tick", 0.35 if ti["hit"] else 0.5))
    for fa in s.facts:
        SFX.append((fa["a"], "tick", 0.35))


def build_sfx():
    n = int(DURATION * SR)
    st = np.zeros(n, np.float32)
    lib = {}
    for ev in SFX:
        t, name, gain = ev[0], ev[1], ev[2]
        if name == "riser":
            key = ("riser", ev[3]); lib.setdefault(key, riser(ev[3])); clip = lib[key]
        elif name.startswith("whoosh"):
            key = name; lib.setdefault(key, whoosh(1.3, name.endswith("up"))); clip = lib[key]
        elif name == "boom":
            lib.setdefault("boom", boom()); clip = lib["boom"]
        elif name == "hit":
            lib.setdefault("hit", hit()); clip = lib["hit"]
        else:
            lib.setdefault("tick", tick()); clip = lib["tick"]
        put(st, t, clip, gain)
    amb = np.zeros(n, np.float32)
    for (a, b, kind, g) in AMBIENCE:
        x = ambience(kind, b - a)
        fi = np.minimum(1, np.arange(len(x)) / (1.5 * SR)) * np.minimum(1, (len(x) - np.arange(len(x))) / (1.5 * SR))
        put(amb, a, x * fi, g)
    # the "silence" after the drop: ambience ducked out 20.3 -> 24.0
    tc = np.arange(n) / SR
    silence = np.interp(tc, [19.95, 20.3, 23.5, 25.5], [1, 0.0, 0.0, 1.0])
    amb *= silence
    st_sfx = reverb(st, 2.2, 0.22, 5000, seed=5)
    amb_st = np.stack([amb, np.roll(amb, int(0.013 * SR))], 1)
    return st_sfx, amb_st


# --------------------------------------------------------------------------- mix
def main():
    os.makedirs(BUILD, exist_ok=True)
    print("[audio] narration ...")
    narr, timing = build_narration()
    json.dump(timing, open(os.path.join(BUILD, "narration_timing.json"), "w"), indent=1)
    caps = make_captions(timing)
    json.dump(caps, open(os.path.join(BUILD, "captions.json"), "w"), indent=1)
    shifted = [t for t in timing if abs(t["shifted"]) > 0.6]
    print("  lines: %d, shifted >0.6s: %d" % (len(timing), len(shifted)))
    print("[audio] music ...")
    music = build_music()
    print("[audio] sound design ...")
    sfx, amb = build_sfx()
    # narration -> light eq + room
    nar = hp(narr, 70, 2)
    nar = nar / (np.percentile(np.abs(nar[np.abs(nar) > 0.01]), 95) + 1e-6) * 0.42 if np.any(np.abs(nar) > 0.01) else nar
    nar_st = reverb(nar, 1.1, 0.07, 6000, seed=9)
    # sidechain duck from narration envelope
    env = np.abs(narr)
    env = signal.sosfilt(signal.butter(1, 6, "low", fs=SR, output="sos"), env)
    duck = 1.0 - 0.6 * np.clip(env / 0.12, 0, 1)
    duck = signal.sosfilt(signal.butter(1, 1.2, "low", fs=SR, output="sos"), duck)
    duck = np.clip(duck, 0.35, 1.0).astype(np.float32)
    mix = music * (0.62 * duck)[:, None] + amb * (0.35 * (0.6 + 0.4 * duck))[:, None] + sfx * 0.85 + nar_st * 1.0
    # master: gentle soft clip, normalise to -1 dBFS
    mix = np.tanh(mix * 1.1) / np.tanh(1.1)
    mix *= 0.89 / (np.abs(mix).max() + 1e-9)
    # fade out
    n = len(mix)
    mix *= np.interp(np.arange(n) / SR, [0, 0.3, DURATION - 3.0, DURATION], [0.0, 1, 1, 0])[:, None].astype(np.float32)
    out = os.path.join(BUILD, "audio.wav")
    write_wav(out, mix.astype(np.float32))
    print("[audio] wrote", out, "%.1fs" % (n / SR))


if __name__ == "__main__":
    main()
