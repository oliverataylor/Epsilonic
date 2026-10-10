import ast
import io
import json
import sys
import tempfile
import time
import tokenize
import unittest
import wave
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import epsilonic as app


class CompositionTests(unittest.TestCase):
    def test_all_styles_stay_active_and_bounded(self):
        for style in app.STYLES:
            with self.subTest(style=style):
                composer = app.Composer(style, seed=7, sample_rate=8000)
                windows = []
                largest_jump = 0
                largest_internal_jump = 0
                previous = np.zeros(2)
                for index in range(2400):
                    block = composer.render(1000)
                    self.assertTrue(np.isfinite(block).all())
                    self.assertLess(float(np.max(np.abs(block))), 1)
                    self.assertLess(len(composer.voices), 150)
                    self.assertLess(sum(v.audio.nbytes for v in composer.voices), 32_000_000)
                    largest_jump = max(largest_jump, float(np.max(np.abs(block[0] - previous))))
                    largest_internal_jump = max(largest_internal_jump, float(np.max(np.abs(np.diff(block, axis=0)))))
                    previous = block[-1]
                    if index > 8:
                        windows.append(float(np.sqrt(np.mean(block ** 2))))
                self.assertGreater(min(windows), .004)
                self.assertLessEqual(largest_jump, largest_internal_jump * 1.05 + 1e-6)
                self.assertGreater(composer.chord_index, 60)

    def test_render_is_independent_of_block_boundaries(self):
        for style in app.STYLES:
            a = app.Composer(style, seed=9, sample_rate=8000)
            b = app.Composer(style, seed=9, sample_rate=8000)
            expected = np.concatenate([a.render(1000) for _ in range(96)])
            sizes = [127, 2903, 500, 17, 6453] * 9 + [6000]
            actual = np.concatenate([b.render(n) for n in sizes])
            np.testing.assert_allclose(actual, expected, atol=2e-7)

    def test_style_changes_preserve_tails_and_timeline(self):
        composer = app.Composer(seed=22, sample_rate=8000)
        for style in app.STYLES * 4:
            composer.requested_style = style
            frames_before = composer.frame
            for _ in range(40):
                audio = composer.render(1000)
                self.assertGreater(float(np.sqrt(np.mean(audio ** 2))), .001)
            self.assertEqual(composer.style, style)
            self.assertEqual(composer.frame, frames_before + 40000)

    def test_seed_reproducibility_and_variation(self):
        a, b, c = [app.Composer(seed=seed, sample_rate=8000) for seed in (42, 42, 43)]
        first, same, different = [composer.render(32000) for composer in (a, b, c)]
        np.testing.assert_array_equal(first, same)
        self.assertFalse(np.array_equal(first, different))

    def test_original_bossa_piano_pattern(self):
        composer = app.Composer(app.STYLES[1], seed=3, sample_rate=8000)
        recorded = []
        composer.note = lambda *args, **kwargs: recorded.append((args, kwargs))
        composer.drum = lambda *args, **kwargs: None
        duration = composer.schedule_bossa(0)
        piano = [args for args, kwargs in recorded if args[1] == 0]
        self.assertEqual(duration, 2.4)
        self.assertEqual(len(piano), 36)
        expected = [round(step * .6 * 8000) for step in (0, .5, .75, 1.25, 1.5, 2, 2.5, 2.75, 3.5)]
        self.assertEqual(sorted(set(args[0] for args in piano)), expected)

    def test_original_ethereal_instruments_and_rhythm(self):
        composer = app.Composer(app.STYLES[0], seed=4, sample_rate=8000)
        recorded = []
        composer.note = lambda *args, **kwargs: recorded.append((args, kwargs))
        composer.drum = lambda *args, **kwargs: None
        self.assertEqual(composer.schedule_ethereal(0), 4.0)
        programs = [args[1] for args, kwargs in recorded]
        self.assertEqual(set(programs), {89, 52, 33, 10, 27})
        self.assertEqual(programs.count(33), 4)
        self.assertEqual(programs.count(10), 12)
        self.assertEqual(programs.count(89), programs.count(52))
        melody = [args for args, kwargs in recorded if args[1] == 10]
        guitar = [args for args, kwargs in recorded if args[1] == 27]
        self.assertEqual(len(guitar), 6)
        for pluck, bell in zip(guitar, melody[::2]):
            self.assertEqual(pluck[0], bell[0])
            self.assertEqual(pluck[2] + 12, bell[2])
        for first, echo in zip(melody[::2], melody[1::2]):
            self.assertEqual(first[2], echo[2])
            self.assertEqual(echo[0] - first[0], 4000)
            self.assertLess(echo[4], first[4])

    def test_haze_has_stable_harmonic_cycle(self):
        composer = app.Composer(app.STYLES[2], seed=4, sample_rate=8000)
        composer.note = lambda *args, **kwargs: None
        composer.drum = lambda *args, **kwargs: None
        composer.guitar_cloud = lambda *args, **kwargs: None
        names = []
        lengths = []
        for index in range(24):
            composer.chord_index = index
            lengths.append(composer.schedule_haze(0))
            names.append(composer.chord_name)
        self.assertEqual(names[:12], names[12:])
        self.assertEqual(names[:12], ["D", "F#m", "Bm", "G", "F#m", "Em", "D", "Em", "Bm", "A", "A6", "Em7"])
        self.assertEqual(len(set(lengths)), 1)

    def test_ethereal_hook_follows_chords_and_varies(self):
        composer = app.Composer(app.STYLES[0], seed=21, sample_rate=8000)
        phrases = []
        composer.drum = lambda *args, **kwargs: None
        for _ in range(16):
            recorded = []
            composer.note = lambda *args, **kwargs: recorded.append(args)
            composer.schedule_chord()
            pitches = [args[2] for args in recorded if args[1] == 10][::2]
            chord = {pitch % 12 for pitch in app.chord_to_notes(composer.chord_name)}
            self.assertTrue(all(pitch % 12 in chord for pitch in pitches))
            self.assertLessEqual(max(abs(a - b) for a, b in zip(pitches, pitches[1:])), 7)
            phrases.append(pitches)
        self.assertEqual(phrases[:4], phrases[4:8])
        self.assertNotEqual(phrases[:4], phrases[8:12])
        self.assertEqual(phrases[:4], phrases[12:16])

    def test_all_styles_have_drums_and_latin_samples_exist(self):
        for style in app.STYLES:
            composer = app.Composer(style, seed=4, sample_rate=8000)
            hits = []
            composer.note = lambda *args, **kwargs: None
            composer.guitar_cloud = lambda *args, **kwargs: None
            composer.drum = lambda *args, **kwargs: hits.append(args)
            composer.schedule_chord()
            pitches = {hit[1] for hit in hits}
            required = {36, 37, 60, 61, 62, 63, 64, 70, 75, 54} if style == app.STYLES[1] else {36, 37, 42} if style == app.STYLES[0] else {36, 38, 42}
            self.assertTrue(required <= pitches)
            for pitch in pitches:
                self.assertIn(f'128_{pitch}', composer.bank)
            if style == app.STYLES[0]:
                self.assertEqual(hits, [(0, 36, 35, .7), (16000, 37, 38, .7)] + [(step * 4000, 42, 15, .7) for step in range(8)])
            if style == app.STYLES[2]:
                beat = 60 / 116 if style == app.STYLES[2] else 1.0
                self.assertEqual([hit[0] for hit in hits if hit[1] == 38], [round(2 * beat * 8000)])
                self.assertEqual([hit[0] for hit in hits if hit[1] == 36], [0])
                self.assertEqual(sum(hit[1] == 42 for hit in hits), 16 if style == app.STYLES[2] else 12)

    def test_ethereal_contains_sustained_pads(self):
        composer = app.Composer(app.STYLES[0], seed=4, sample_rate=8000)
        programs = []
        original = composer.sample
        def capture(program, *args, **kwargs):
            programs.append(program)
            return original(program, *args, **kwargs)
        composer.sample = capture
        composer.schedule_ethereal(0)
        self.assertGreaterEqual(programs.count(89), 3)
        self.assertIn(52, programs)
        self.assertIn(10, programs)
        self.assertNotIn(0, programs)
        self.assertIn(27, programs)

    def test_large_reverb_retains_tail_and_bossa_stays_dry(self):
        composer = app.Composer(app.STYLES[2], seed=4, sample_rate=8000)
        impulse = np.zeros((8000, 2), dtype=np.float32)
        impulse[0] = 1
        for delay in composer.hall:
            delay.process(impulse)
        for _ in range(3):
            tail = sum(delay.process(np.zeros_like(impulse)) for delay in composer.hall)
        self.assertGreater(float(np.max(np.abs(tail))), 1e-5)
        composer.style = app.STYLES[1]
        composer.add(0, np.ones(8), 1)
        self.assertFalse(composer.voices[-1].spacious)

    def test_restored_interface_and_names(self):
        self.assertEqual(app.STYLES, ("Ethereal Feel", "Bossa Staccato", "Shoegaze Haze"))
        source = (ROOT / "epsilonic.py").read_text(encoding="utf-8")
        tree = ast.parse(source)
        gui = next(node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name == "run_gui")
        strings = [node.value for node in ast.walk(gui) if isinstance(node, ast.Constant) and isinstance(node.value, str)]
        self.assertIn("1000x520", strings)
        self.assertIn(" ~ EPSILONIC ~ ", strings)
        for unwanted in ("Warm strings", "Space to pause", "Always unfolding", "G E N E R A T I V E"):
            self.assertFalse(any(unwanted.lower() in value.lower() for value in strings))

    def test_export_duration_format_and_fades(self):
        with tempfile.TemporaryDirectory() as folder:
            path = app.render_file(Path(folder) / "test.wav", app.STYLES[2], .23, 1)
            with wave.open(str(path), "rb") as source:
                self.assertEqual(source.getframerate(), 44100)
                self.assertEqual(source.getnchannels(), 2)
                self.assertEqual(source.getsampwidth(), 2)
                self.assertEqual(source.getnframes(), round(.23 * 44100))
                audio = np.frombuffer(source.readframes(source.getnframes()), dtype="<i2")
            self.assertTrue(np.any(audio))
            np.testing.assert_array_equal(audio[:2], 0)
            np.testing.assert_array_equal(audio[-2:], 0)


class PlaybackTests(unittest.TestCase):
    def engine(self):
        engine = app.AudioEngine(seed=2)
        for _ in range(16):
            engine.blocks.put((np.ones((2048, 2), dtype=np.float32) * .2, app.STYLES[0], "Cmaj7"))
        return engine

    def output(self, engine, frames):
        output = np.empty((frames, 2), dtype=np.float32)
        engine.callback(output, frames, None, False)
        return output

    def test_variable_callback_sizes_consume_without_gaps(self):
        engine = self.engine()
        for frames in (128, 3000, 17, 4096, 9000):
            audio = self.output(engine, frames)
            self.assertGreater(float(audio.min()), 0)
        self.assertEqual(engine.underruns, 0)
        self.assertEqual(engine.played_frames, 16241)

    def test_pause_freezes_position_and_resume_ramps(self):
        engine = self.engine()
        self.output(engine, 2048)
        engine.paused = True
        fading = self.output(engine, 2048)
        self.assertGreater(float(fading[0, 0]), 0)
        self.assertEqual(float(fading[-1, 0]), 0)
        position = engine.played_frames
        self.assertFalse(np.any(self.output(engine, 4096)))
        self.assertEqual(engine.played_frames, position)
        engine.paused = False
        resumed = self.output(engine, 2048)
        self.assertLess(float(resumed[0, 0]), .001)
        self.assertGreater(float(resumed[-1, 0]), .1)
        self.assertEqual(engine.underruns, 0)

    def test_mute_keeps_music_advancing(self):
        engine = self.engine()
        self.output(engine, 2048)
        engine.volume = 0
        self.output(engine, 2048)
        position = engine.played_frames
        self.assertFalse(np.any(self.output(engine, 2048)))
        self.assertEqual(engine.played_frames, position + 2048)

    def test_starvation_recovers(self):
        engine = app.AudioEngine()
        self.assertFalse(np.any(self.output(engine, 2048)))
        self.assertEqual(engine.underruns, 1)
        engine.blocks.put((np.ones((2048, 2), dtype=np.float32) * .2, app.STYLES[1], "Dm7"))
        self.assertGreater(float(self.output(engine, 2048).max()), .1)
        self.assertEqual(engine.audible_style, app.STYLES[1])

    def test_worker_is_bounded_and_stops_while_paused(self):
        import threading

        engine = app.AudioEngine(seed=3)
        engine.paused = True
        engine.worker = threading.Thread(target=engine.produce)
        engine.worker.start()
        self.assertTrue(engine.ready.wait(10))
        time.sleep(.3)
        self.assertLessEqual(engine.blocks.qsize(), engine.blocks.maxsize)
        engine.close()
        self.assertFalse(engine.worker.is_alive())
        self.assertIsNone(engine.error)

    def test_no_comments_or_docstrings(self):
        for path in [ROOT / "epsilonic.py", *ROOT.joinpath("tests").glob("*.py")]:
            source = path.read_text(encoding="utf-8")
            comments = [token for token in tokenize.generate_tokens(io.StringIO(source).readline) if token.type == tokenize.COMMENT]
            self.assertEqual(comments, [], str(path))
            for node in ast.walk(ast.parse(source)):
                if isinstance(node, (ast.Module, ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef)):
                    self.assertIsNone(ast.get_docstring(node), str(path))


if __name__ == "__main__":
    unittest.main()
