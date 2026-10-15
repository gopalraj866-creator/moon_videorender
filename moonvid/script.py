"""Narration script. (start_seconds, text). Timings line up with the shot list in timeline.py.
Edit freely: the audio builder auto-fits lines (small speed-up, or pushes the next line) and writes captions."""

NARRATION = [
    # --- Scene 1: the mystery
    (0.8, "Imagine looking up at the night sky... and suddenly, the Moon is gone."),
    (7.4, "What would happen to our oceans, our nights, and life on Earth?"),
    (13.6, "For four and a half billion years, it has been right there, quietly shaping our world."),
    # --- Scene 2: the disappearance (silence after the drop at 20.0)
    (24.2, "Gone. No flash. No explosion. Just empty space, where our Moon used to be."),
    (30.8, "Before we go on: what do you think would change first? The tides, the night sky, or Earth's climate? Tell us in the comments."),
    (43.0, "Here's the good news. Earth would not fall apart. The Moon has just over one percent of Earth's mass, so our planet would simply keep orbiting the Sun. But some changes would begin right away."),
    # --- Scene 3: a darker night
    (58.5, "The first thing you would notice is the dark. The Moon makes no light of its own. It reflects sunlight, and on a full-Moon night, that borrowed glow is enough to light up a whole landscape."),
    (74.5, "Take it away, and nights get far darker. A full Moon delivers roughly a tenth of a lux. Starlight alone? About a hundred times less."),
    (85.5, "Cities would still glow with their own lights. But forests, deserts and oceans would be truly dark. The stars would still be there, easier to see, weather permitting."),
    (97.5, "Many animals, though, use moonlight to hunt, hide, and navigate."),
    # --- Scene 4: tides
    (102.5, "Now, the oceans. Tides are caused by gravity, and mostly the Moon's."),
    (108.5, "The Moon pulls hardest on the side of Earth facing it, and least on the far side. The result is two bulges of water, and as Earth spins beneath them, we get two high tides every day."),
    (124.0, "But the Sun creates tides too, about forty-six percent as strong as the Moon's. When the Sun and Moon line up, we get big spring tides. When they pull at right angles, weaker neap tides."),
    (139.5, "Without the Moon, the lunar tides would vanish. Only the solar tides would remain."),
    (146.0, "So tides wouldn't stop. They would be much smaller, with local effects depending on the shape of each coastline."),
    (155.0, "Tide pools, salt marshes, mangroves and estuaries all depend on rising and falling water. Harbors and fishing would have to adapt."),
    (164.5, "Tides also act like a brake on Earth's spin, lengthening our day by about two milliseconds each century. Without the Moon, that braking would be much weaker."),
    # --- Scene 5: axial tilt
    (181.0, "Next, a quieter but far bigger effect: Earth's tilt."),
    (185.5, "Our planet spins on an axis tilted about twenty-three and a half degrees. That tilt gives us our seasons."),
    (194.0, "The Moon acts like a stabilizer. Its gravity helps keep that tilt steady, wobbling only between about twenty-two and twenty-four and a half degrees over forty thousand years."),
    (213.0, "Remove the Moon, and that anchor is gone. The Sun and the other planets could slowly tug Earth's axis around, and the tilt could drift far more over millions of years."),
    (226.0, "Simulations disagree on how far. Some show modest changes, others show wild swings."),
    (233.0, "A larger tilt would mean extreme seasons. Almost no tilt would mean almost no seasons at all. Either way, climate zones and ecosystems would slowly reshape."),
    (246.0, "Mars, which has only two tiny moons, is known to wobble dramatically."),
    (259.0, "But look at the timescale. Millions of years. This would not be an instant catastrophe, and humans would never see it happen."),
    # --- Scene 6: what else changes
    (271.0, "Then there's the sky itself. Lunar eclipses happen when the Moon slides through Earth's shadow and turns a deep red. Without the Moon, they would be gone."),
    (283.0, "Solar eclipses, too. By a cosmic coincidence, the Moon appears almost exactly the same size as the Sun, so it can cover it perfectly. No Moon, no total eclipses."),
    (295.5, "The phases of the Moon, new, crescent, quarter, gibbous, full, would be gone as well. We would lose the oldest clock in human history."),
    (309.0, "Many calendars are built on it. The Islamic and Jewish calendars follow the Moon, and festivals like Diwali, Eid and Chinese New Year are timed by it. People would need new ways to keep time."),
    (324.0, "Nature keeps lunar time too. Corals on the Great Barrier Reef spawn in sync after a full Moon. Some animals navigate by moonlight, and others hunt or hide by it."),
    (337.0, "The Moon is also slowly drifting away from us, about three point eight centimeters a year, roughly as fast as your fingernails grow."),
    (349.0, "Remove it, and the sky would lose its most familiar landmark. Humanity would lose its first calendar, its first clock, and its first destination in space."),
    # --- Scene 7: the final question
    (361.0, "So, would Earth still survive?"),
    (364.0, "Yes. Earth would not suddenly fly out of orbit. Our planet's path is controlled by the Sun's gravity, and the Moon is only about one percent of Earth's mass."),
    (376.5, "Earth would not lose its atmosphere either. Air is held by Earth's own gravity, not the Moon's."),
    (387.0, "Life would continue. But tides would shrink, nights would darken, and over very long timescales, Earth's climate could become less stable."),
    (399.0, "And it's worth remembering where the Moon came from. About four and a half billion years ago, a Mars-sized body called Theia is thought to have struck the young Earth. The debris formed the Moon."),
    (415.0, "Our Moon is a rare, oversized companion, and we're lucky to have it."),
    (426.5, "Now, zoom out. Earth is one small planet, circling one ordinary star."),
    (441.0, "Jupiter and Saturn have scores of moons. Mars has two tiny ones. But no other rocky planet has a companion quite like ours."),
    (457.0, "So if the Moon vanished tonight, Earth would survive. But it would never be the same."),
    (464.2, "What do you think would be the biggest change? Tell us in the comments, and subscribe for more journeys through space and science."),
    (474.0, "The Moon is more than a light in the sky."),
]

VOICE = "en-US-GuyNeural"          # override with env VOICE (any edge-tts voice, e.g. en-GB-RyanNeural, en-IN-PrabhatNeural)
VOICE_RATE = "-4%"
