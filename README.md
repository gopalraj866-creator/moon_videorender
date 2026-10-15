# What If the Moon Disappeared? - 8 minute 3D simulation (GitHub Actions render)

1920x1080, 30 fps, 8:00 long, 7 scenes exactly as in your brief, narration + music + sound design, burned-in captions.
No GPU and no PC needed: everything is rendered by a Python/numpy/OpenCV software 3D engine on GitHub's free runners.

## Run it (3 steps)
1. Create a new GitHub repository and upload everything in this folder (keep the `.github/workflows/render.yml` path).
2. Open the **Actions** tab -> **Render "What If the Moon Disappeared?"** -> **Run workflow**.
   - `quality = preview` first (about 15 min total, 960x540 scaled to 1080p) to check the look.
   - `quality = full` for the final 1920x1080 render (12 parallel jobs, about 25-40 min each run).
3. When it finishes, download the **moon_video_final** artifact: `What_If_The_Moon_Disappeared.mp4`, `video_description.txt`, `captions.json`.

Pipeline: `prepare` (download textures, narration via edge-tts, synth score and sound design) -> 12 parallel `render` jobs
(40 seconds each) -> `assemble` (concatenate + mux audio). A failed chunk can be re-run on its own from the Actions UI.

## What is real, what is procedural
| Thing | Source |
|---|---|
| Earth day map, clouds, ocean mask, elevation relief | NASA-derived 4k maps (webgl-earth) |
| Earth night city lights | NASA Visible Earth "Earth at night" (three.js examples) |
| Moon | LRO-based map + procedural crater/fractal detail for close-ups (tries an 8k Solar System Scope map first) |
| Mars, Venus, Mercury, Jupiter, Saturn (+ rings), Uranus, Neptune | threex.planets NASA/JPL maps |
| Stars, Milky Way, Sun surface, city / forest / ocean / desert night scenes, coral spawn | procedural (seeded, identical in every job) |
| Narration | edge-tts neural voice (fallbacks: gTTS, then espeak-ng) |
| Music + sound design | synthesised in numpy: pads, drone, boom at 0:20 then ~3 s of silence, risers, whooshes, ambience, sidechain ducking under narration |

## Timeline
| Time | Scene | Notes |
|---|---|---|
| 0:00-0:20 | Mystery | Earth + Moon, push toward the Moon, title in the first second, a tiny flicker teases the vanish |
| 0:20-0:50 | Disappearance | Moon vanishes at exactly 0:20 (bass drop, silence), empty space, dark sky, "what would change first?" prompt on screen at 0:30 |
| 0:50-1:40 | A darker night | city, forest, ocean, desert go from moonlit to dark; night-side Earth; lux facts |
| 1:40-3:00 | Tides | tidal bulges, gravity arrows, spring/neap, Moon fades -> solar-only, two-beach comparison, day-length brake |
| 3:00-4:30 | Axial tilt | Sun + glowing axis, seasons, stable vs wandering tilt (labelled illustrative), extremes, Mars, timescale bar |
| 4:30-6:00 | What else | lunar + solar eclipse, phases, calendars (Diwali, Eid, CNY...), coral spawning, Moon drifting away, empty orbit |
| 6:00-8:00 | Final question | survives/orbit/atmosphere, giant-impact origin, pull-away to the solar system, Jupiter, Saturn, CTA, closing |

A new shot starts every 6-20 seconds (the whole video has 45 shots).

## Customise
* **Narration text and timing**: `moonvid/script.py` (each line has a start time; the audio builder auto-fits or nudges lines).
* **Shots, titles, on-screen facts**: `moonvid/timeline.py` (one line per shot).
* **Voice**: run the workflow with `voice` (e.g. `en-IN-PrabhatNeural`) or edit `VOICE` in `script.py`.
* **Local test**: `pip install -r requirements.txt && python -m moonvid.assets && python -m moonvid.audio`, then
  `python -m moonvid.render --still 20.5 --scale 0.5 --out frame.png` to preview any second, or
  `python -m moonvid.render --start-frame 0 --end-frame 300 --out test.mp4`.
* **More realism**: drop larger equirectangular maps into `assets/` (`earth_day.png`, `moon.png`, ...), they are used automatically.

## Science notes used in the script
Lunar tides vanish but solar tides (about 46 % as strong) stay; the Moon is about 1.2 % of Earth's mass; Earth would not leave orbit or lose its air;
the Moon stabilises the tilt (about 22.1-24.5 deg over ~41,000 years) and without it the tilt could drift far more over millions of years
(models disagree on how far - the video says so); lunar eclipses and total solar eclipses would end; the Moon recedes about 3.8 cm/year;
tidal friction adds roughly 2 ms per century to the day.

## Troubleshooting
* Narration silent or robotic: edge-tts can be blocked on some runners; the log shows which engine was used per line. Re-run `prepare`.
* Out of time: reduce quality to `preview`, or raise `CHUNKS` and the matrix list in `render.yml`.
