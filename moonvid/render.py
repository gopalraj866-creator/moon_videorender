"""Chunked renderer.

  python -m moonvid.render --chunk 3 --chunks 12 --out build/chunk_03.mp4
  python -m moonvid.render --still 20.5 --out frame.png          (preview one frame)
  python -m moonvid.render --assemble build --audio build/audio.wav --out moon_video.mp4
"""
import os
for _v in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ[_v] = "1"          # avoid fork + threadpool deadlocks
import argparse, sys, subprocess, time, glob, multiprocessing as mp
import numpy as np, cv2
cv2.setNumThreads(1)


def _init():
    cv2.setNumThreads(1)
    os.environ.setdefault("OMP_NUM_THREADS", "1")


def _work(n):
    from . import timeline as TL
    return TL.render_frame(n)


def render_chunk(a, b, out, workers, scale, captions):
    from . import engine as E, timeline as TL
    from .assets import get_assets
    E.set_scale(scale)
    get_assets()
    TL.load_captions(captions)
    W, H = E.RW, E.RH
    OW, OH = 1920, 1080
    vf = []
    if (W, H) != (OW, OH):
        vf = ["-vf", "scale=%d:%d:flags=lanczos" % (OW, OH)]
    cmd = ["ffmpeg", "-y", "-loglevel", "error", "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", "%dx%d" % (W, H), "-r", str(TL.FPS), "-i", "-"] + vf + \
          ["-c:v", "libx264", "-preset", os.environ.get("X264_PRESET", "medium"), "-crf", os.environ.get("CRF", "17"), "-pix_fmt", "yuv420p",
           "-g", "30", "-keyint_min", "30", "-sc_threshold", "0", "-movflags", "+faststart", out]
    proc = subprocess.Popen(cmd, stdin=subprocess.PIPE)
    t0 = time.time()
    total = b - a
    ctx = mp.get_context("fork")
    with ctx.Pool(workers, initializer=_init) as pool:
        for k, frame in enumerate(pool.imap(_work, range(a, b), chunksize=2)):
            proc.stdin.write(np.ascontiguousarray(frame).tobytes())
            if k % 30 == 0:
                el = time.time() - t0
                print("  frame %d/%d  %.1f fps  eta %.0fs" % (k, total, (k + 1) / max(el, 1e-6), (total - k) / max((k + 1) / max(el, 1e-6), 1e-6)), flush=True)
    proc.stdin.close()
    if proc.wait() != 0:
        sys.exit("ffmpeg failed")
    print("chunk done: %s  (%.0fs)" % (out, time.time() - t0))


def assemble(chunk_dir, audio, out):
    files = sorted(glob.glob(os.path.join(chunk_dir, "chunk_*.mp4")))
    if not files:
        sys.exit("no chunks found in " + chunk_dir)
    lst = os.path.join(chunk_dir, "concat.txt")
    open(lst, "w").write("".join("file '%s'\n" % os.path.abspath(f) for f in files))
    tmp = os.path.join(chunk_dir, "video_only.mp4")
    subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-f", "concat", "-safe", "0", "-i", lst, "-c", "copy", tmp], check=True)
    if audio and os.path.exists(audio):
        subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-i", tmp, "-i", audio, "-c:v", "copy", "-c:a", "aac", "-b:a", "256k",
                        "-shortest", "-movflags", "+faststart", out], check=True)
    else:
        os.replace(tmp, out)
    print("final video:", out)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--chunk", type=int)
    ap.add_argument("--chunks", type=int, default=12)
    ap.add_argument("--start-frame", type=int)
    ap.add_argument("--end-frame", type=int)
    ap.add_argument("--out", default="out.mp4")
    ap.add_argument("--workers", type=int, default=os.cpu_count() or 2)
    ap.add_argument("--scale", type=float, default=float(os.environ.get("RENDER_SCALE", "1.0")))
    ap.add_argument("--captions", default=os.environ.get("CAPTIONS_JSON", "build/captions.json"))
    ap.add_argument("--no-captions", action="store_true")
    ap.add_argument("--still", type=float)
    ap.add_argument("--assemble")
    ap.add_argument("--audio")
    a = ap.parse_args()
    if a.assemble:
        return assemble(a.assemble, a.audio, a.out)
    from . import engine as E, timeline as TL
    if a.still is not None:
        E.set_scale(a.scale)
        TL.load_captions(None if a.no_captions else a.captions)
        fr = TL.render_frame(int(a.still * TL.FPS))
        cv2.imwrite(a.out, cv2.cvtColor(fr, cv2.COLOR_RGB2BGR))
        return print("wrote", a.out)
    if a.chunk is not None:
        per = -(-TL.TOTAL_FRAMES // a.chunks)
        s, e = a.chunk * per, min((a.chunk + 1) * per, TL.TOTAL_FRAMES)
    else:
        s, e = a.start_frame or 0, a.end_frame or TL.TOTAL_FRAMES
    os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
    print("rendering frames %d-%d (%.1fs-%.1fs) scale=%.2f workers=%d" % (s, e, s / 30, e / 30, a.scale, a.workers))
    render_chunk(s, e, a.out, a.workers, a.scale, None if a.no_captions else a.captions)


if __name__ == "__main__":
    main()
