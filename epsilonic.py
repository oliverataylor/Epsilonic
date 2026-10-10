import argparse
import math
import queue
import random
import threading
import wave
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

import numpy as np
from scipy.signal import butter, sosfilt


BASE_DIR = Path(__file__).resolve().parent
SAMPLE_RATE = 44100
BLOCK_SIZE = 2048
STYLES = ("Ethereal Feel", "Bossa Staccato", "Shoegaze Haze")
THEMES = {
    STYLES[0]: {"text": "#ffffff", "bg": "#112215", "pop_bg": "#112215", "sel_bg": "#1b3822", "active_btn": "#1b3822", "image": "epsilonic.png"},
    STYLES[1]: {"text": "#fcd797", "bg": "#1a1512", "pop_bg": "#613b25", "sel_bg": "#1a1512", "active_btn": "#613b25", "image": "epsilonic2.png"},
    STYLES[2]: {"text": "#38cfff", "bg": "#020b38", "pop_bg": "#05267a", "sel_bg": "#124bce", "active_btn": "#124bce", "image": "epsilonic3.png"},
}

PROGRESSIONS = [['Cmaj7', 'A7', 'Dm7', 'G7'], ['Gmaj7', 'E7', 'Am7', 'D7'], ['Fmaj7', 'G7', 'Cmaj7', 'C6'], ['Amaj7', 'B7', 'Emaj7', 'E6'], ['Dmaj7', 'Em7', 'F#m7', 'Em7'], ['Cmaj7', 'D7', 'Gmaj7', 'G6'], ['Bbmaj7', 'C7', 'Fmaj7', 'F6'], ['Emaj7', 'F#7', 'Bmaj7', 'B6']]

NOTE_MAP = {'C': 60, 'C#': 61, 'Db': 61, 'D': 62, 'D#': 63, 'Eb': 63, 'E': 64, 'F': 65, 'F#': 66, 'G': 67, 'G#': 68, 'Ab': 68, 'A': 69, 'A#': 70, 'Bb': 70, 'B': 71}

def parse_root(ch):
    return (ch[:2], ch[2:]) if len(ch) >= 2 and ch[:2] in NOTE_MAP else (ch[0], ch[1:])

def bossa_voicing(ch, step_index, chord_idx):
    root, quality = parse_root(ch)
    base = NOTE_MAP.get(root, 60)
    voicing = [base, base + (3 if 'm' in quality and 'maj' not in quality else 4), base + 7]
    top_note = base + (14 if step_index in [0, 1, 2] else 12 if step_index in [3, 4, 5] else 10 if '7' in quality and 'maj' not in quality else 11)
    if chord_idx == 1:
        top_note += 7
    return voicing + [top_note]

def chord_to_notes(ch):
    root, quality = parse_root(ch)
    base = NOTE_MAP.get(root, 60)
    notes = [base, base + (3 if 'm' in quality and 'maj' not in quality else 4), base + 7]
    if 'maj7' in quality:
        notes.append(base + 11)
    elif '7' in quality:
        notes.append(base + 10)
    return notes

SCALES = {'Cmaj7': [60, 62, 64, 67, 69, 72], 'G7': [55, 59, 62, 65, 67, 71], 'Dm7': [57, 60, 62, 65, 69, 72], 'Am7': [57, 60, 64, 67, 69, 72], 'D7': [54, 57, 60, 62, 66, 69], 'Fm7': [56, 60, 63, 65, 68, 72], 'Bb7': [55, 58, 62, 65, 68, 70], 'Ebmaj7': [55, 58, 62, 63, 67, 70], 'Bm7': [59, 62, 64, 66, 69, 71], 'E7': [56, 59, 62, 64, 68, 70], 'Amaj7': [57, 61, 64, 68, 69, 73], 'C#m7': [56, 60, 61, 64, 68, 71], 'F#7': [54, 58, 61, 64, 66, 70], 'Bmaj7': [58, 61, 63, 66, 70, 71], 'Gm7': [55, 58, 62, 65, 67, 70], 'C7': [55, 58, 60, 64, 67, 70], 'Fmaj7': [57, 60, 64, 65, 69, 72], 'Cm7': [55, 58, 60, 63, 67, 70], 'F7': [53, 57, 60, 63, 65, 69], 'Bbmaj7': [53, 57, 58, 62, 65, 69], 'Db7': [53, 56, 59, 61, 65, 68], 'F#m7': [54, 57, 61, 64, 66, 69], 'Emaj7': [56, 59, 63, 64, 68, 71], 'Gmaj7': [55, 59, 62, 66, 67, 71], 'Em7': [55, 59, 60, 64, 67, 71], 'C6': [60, 62, 64, 67, 69, 72], 'E6': [56, 59, 61, 64, 68, 71], 'G6': [55, 59, 62, 64, 67, 71], 'F6': [53, 57, 60, 62, 65, 69], 'B6': [56, 59, 63, 66, 68, 71]}

@dataclass
class Voice:
    start: int
    audio: np.ndarray
    spacious: float = 0.0


@lru_cache(maxsize=1)
def instrument_bank():
    path = BASE_DIR / "instruments.npz"
    if not path.is_file():
        raise RuntimeError("Keep instruments.npz beside epsilonic.py")
    with np.load(path, allow_pickle=False) as source:
        return {name: source[name].astype(np.float32) / 32767 for name in source.files if name != "sample_rate"}


class Delay:
    def __init__(self, seconds, feedback, sample_rate):
        self.buffer = np.zeros((round(seconds * sample_rate), 2), dtype=np.float32)
        self.position = 0
        self.feedback = feedback
        self.sos = butter(1, min(3600, sample_rate * .35), fs=sample_rate, output="sos")
        self.state = np.zeros((len(self.sos), 2, 2))

    def process(self, audio):
        result = np.empty_like(audio)
        offset = 0
        while offset < len(audio):
            count = min(len(audio) - offset, len(self.buffer) - self.position)
            target = slice(self.position, self.position + count)
            source = slice(offset, offset + count)
            delayed, self.state = sosfilt(self.sos, self.buffer[target], axis=0, zi=self.state)
            result[source] = delayed
            self.buffer[target] = audio[source] + delayed[:, ::-1] * self.feedback
            self.position = (self.position + count) % len(self.buffer)
            offset += count
        return result


class Diffuser:
    def __init__(self, seconds, sample_rate):
        self.buffer = np.zeros((round(seconds * sample_rate), 2), dtype=np.float32)
        self.position = 0

    def process(self, audio):
        result = np.empty_like(audio)
        offset = 0
        while offset < len(audio):
            count = min(len(audio) - offset, len(self.buffer) - self.position)
            target = slice(self.position, self.position + count)
            source = slice(offset, offset + count)
            delayed = self.buffer[target].copy()
            output = delayed - .65 * audio[source]
            self.buffer[target] = audio[source] + .65 * output
            result[source] = output
            self.position = (self.position + count) % len(self.buffer)
            offset += count
        return result


class Composer:
    def __init__(self, style=STYLES[0], seed=None, sample_rate=SAMPLE_RATE):
        if style not in STYLES:
            raise ValueError("Unknown style")
        if sample_rate < 8000:
            raise ValueError("Sample rate must be at least 8000 Hz")
        self.sample_rate = sample_rate
        self.rng = np.random.default_rng(seed)
        self.random = random.Random(seed)
        self.style = style
        self.requested_style = style
        self.frame = 0
        self.next_chord = 0
        self.chord_index = 0
        self.progression = PROGRESSIONS[0]
        self.last_lead = None
        self.voices = []
        self.chord_name = ""
        self.bank = instrument_bank()
        self.delays = [Delay(s, f, sample_rate) for s, f in ((.071, .72), (.089, .74), (.113, .76), (.137, .78), (.173, .79), (.199, .80))]
        self.echo = Delay(.387, .48, sample_rate)
        self.hall = [Delay(seconds, 10 ** (-3 * seconds / 6.8), sample_rate) for seconds in (.067, .083, .109, .139, .173, .211)]
        self.hall_filter = butter(1, 180, btype="highpass", fs=sample_rate, output="sos")
        self.hall_state = np.zeros((len(self.hall_filter), 2, 2))
        self.diffusers = [Diffuser(seconds, sample_rate) for seconds in (.0083, .0179, .0307)]
        self.master_filter = butter(1, 28, btype="highpass", fs=sample_rate, output="sos")
        self.master_state = np.zeros((len(self.master_filter), 2, 2))
        self.cabinet = butter(2, [85, min(4300, sample_rate * .40)], btype="bandpass", fs=sample_rate, output="sos")
        self.fuzz_cabinet = butter(2, [95, min(6200, sample_rate * .43)], btype="bandpass", fs=sample_rate, output="sos")
        self.piano_filter = butter(1, min(3200, sample_rate * .4), fs=sample_rate, output="sos")

    def sample(self, program, pitch, hold, release=.3, bend=0.0, detune=0.0, phase=0.0):
        keys = [int(name.split("_")[1]) for name in self.bank if name.startswith(f"{program}_")]
        key = min(keys, key=lambda value: abs(value - pitch))
        source = self.bank[f"{program}_{key}"]
        duration = hold + release
        t = np.arange(max(2, round(duration * self.sample_rate)), dtype=np.float64) / self.sample_rate
        motion = detune + bend * (np.sin(2 * np.pi * .17 * t + phase) + .22 * np.sin(2 * np.pi * .43 * t + phase))
        speed = 22050 / self.sample_rate * 2 ** ((pitch - key + motion) / 12)
        positions = np.cumsum(np.broadcast_to(speed, t.shape)) - np.asarray(speed)
        audio = np.interp(positions, np.arange(len(source)), source, left=0, right=0)
        fade = np.minimum(t / .003, 1) * np.clip((duration - t - 1 / self.sample_rate) / max(release, .001), 0, 1)
        return (audio * fade).astype(np.float32)

    def add(self, start, mono, gain, pan=.5):
        stereo = np.asarray([math.cos(pan * math.pi / 2), math.sin(pan * math.pi / 2)], dtype=np.float32)
        send = .45 if self.style == STYLES[2] else .38 if self.style == STYLES[0] else 0.0
        self.voices.append(Voice(int(start), mono[:, None] * (stereo * gain), send))

    def note(self, start, program, pitch, hold, velocity=80, pan=.5, release=.35, gain=1.0, bend=0.0, detune=0.0):
        if (self.style == STYLES[0] and program in (89, 52, 33, 10)) or (self.style == STYLES[1] and program in (0, 32)):
            gain = 1.0
            pan = .5
        audio = self.sample(program, pitch, hold, release, bend, detune, self.rng.uniform(0, 6.28))
        if program == 0 and self.style == STYLES[0]:
            audio = sosfilt(self.piano_filter, audio).astype(np.float32)
        if self.style == STYLES[0] and program == 27:
            audio = sosfilt(self.cabinet, audio).astype(np.float32)
        if self.style == STYLES[0] and program in (89, 52):
            attack = .35 if program == 89 else .55
            audio *= np.minimum(np.arange(len(audio)) / (self.sample_rate * attack), 1)
        self.add(start, audio, (velocity / 100) ** 1.5 * gain, pan)

    def drum(self, start, pitch, velocity, gain=1.0):
        audio = self.bank[f"128_{pitch}"]
        if self.sample_rate != 22050:
            count = round(len(audio) * self.sample_rate / 22050)
            audio = np.interp(np.arange(count) * 22050 / self.sample_rate, np.arange(len(audio)), audio).astype(np.float32)
        pan = .65 if pitch in (60, 62, 63, 67, 75) else .35 if pitch in (61, 64, 68, 70, 78, 79) else .57 if pitch in (42, 46, 54, 69) else .5
        if self.style == STYLES[0] or (self.style == STYLES[1] and pitch in (36, 37, 42, 67, 68, 69, 78, 79)):
            gain = 1.0
            pan = .5
        self.add(start, audio, (velocity / 100) ** 1.5 * gain, pan)
        self.voices[-1].spacious = .035 if self.style != STYLES[1] and pitch in (38, 45, 49) else 0.0

    def classic_drums(self, start, beat, gentle=False):
        at = lambda step: start + round(step * beat * self.sample_rate)
        level = .68 if gentle else .82
        self.drum(at(0), 36, 76, level)
        if self.chord_index % 2 == 1:
            self.drum(at(3.5), 36, 49, level)
        self.drum(at(2), 38, 66 if gentle else 75, level)
        for step in range(16):
            if gentle and step % 4 == 1:
                continue
            velocity = (53, 30, 43, 34)[step % 4]
            self.drum(at(step * .25), 42, velocity, level * .65)
        if self.chord_index % 16 == 0:
            self.drum(at(0), 49, 37, level * .35)

    def schedule_bossa(self, start):

        beat = .60

        index = self.chord_index % 4

        if index == 0:

            self.progression = self.random.choice(PROGRESSIONS)

        chord = self.progression[index]

        self.chord_name = chord

        scale = SCALES.get(chord, [60, 62, 64, 67, 69, 72])

        at = lambda step: start + round(step * beat * self.sample_rate)

        for i, step in enumerate((0, .5, .75, 1.25, 1.5, 2, 2.5, 2.75, 3.5)):

            for pitch in bossa_voicing(chord, i, index):

                self.note(at(step), 0, pitch, .12 * beat, self.random.randint(80, 92), release=.10, gain=2.3)

        bass = [pitch - 24 for pitch in scale]

        for i, step in enumerate((0, 1, 1.5, 2, 3, 3.5)):

            self.note(at(step), 32, bass[0] + 1 if i == 5 else bass[i % len(bass)], .22 * beat, 78, release=.08, gain=1.3)

        for step in (0, 2.5):

            self.drum(at(step), 36, 80, .7)

        for step in (.5, .75, 1.5, 2.25, 2.75, 3.5):

            self.drum(at(step), 37, 74, .75)

        for step, pitch in ((.25, 79), (.5, 78), (1.25, 79), (1.5, 78), (2.25, 79), (2.5, 78), (3.25, 79), (3.5, 78)):

            self.drum(at(step), pitch, 46, .6)

        for step, pitch in ((.5, 67), (1.5, 68), (2.5, 67), (3.5, 68)):

            self.drum(at(step), pitch, 54, .7)

        for step in range(16):

            self.drum(at(step * .25), 69 if step % 2 == 0 else 42, 54 if step % 2 == 0 else 28, .6)

        for step, pitch, velocity in ((.25, 62, 48), (.75, 63, 57), (1.5, 64, 54), (2.25, 62, 43), (2.75, 63, 61), (3.5, 64, 57)):
            self.drum(at(step), pitch, velocity, .25)
        for step, pitch in ((.5, 60), (1.75, 61), (2.5, 60), (3.75, 61)):
            self.drum(at(step), pitch, 43, .21)
        for step in (0, 1.5, 3):
            self.drum(at(step), 75, 45, .17)
        for step in range(8):
            self.drum(at(step * .5), 70, 34 if step % 2 else 42, .19)
        for step in (1, 3):
            self.drum(at(step), 54, 38, .15)
        return 4 * beat



    def guitar_cloud(self, start, notes, duration, take):
        sr = self.sample_rate
        beat = duration / 4
        total = round((duration + 1.2) * sr)
        acoustic = np.zeros(total, dtype=np.float32)
        fuzz = np.zeros(total, dtype=np.float32)
        for step, strength in ((0, 1.0), (.5, .49), (1, .77), (1.5, .57), (2, .92), (2.5, .52), (3, .76), (3.5, .60)):
            order = notes if step % 1 == 0 else tuple(reversed(notes))
            for i, pitch in enumerate(order):
                offset = round((step * beat + i * .011 + take * .012) * sr)
                length = min(1.6, (total - offset) / sr)
                guitar = self.sample(24, pitch, max(.05, length - .45), .45, .028, -.007 if take == 0 else .007, (start + offset) / sr * 2 * np.pi * .17)
                guitar = sosfilt(self.cabinet, guitar)
                count = min(len(guitar), total - offset)
                acoustic[offset:offset + count] += guitar[:count] * strength / len(notes)
                electric = self.sample(27, pitch, max(.05, length - .35), .35, .04, -.006 if take == 0 else .006, (start + offset) / sr * 2 * np.pi * .17)
                electric = sosfilt(self.cabinet, electric)
                overdrive = np.tanh(electric * (20 if take == 0 else 16)) * .28
                distorted = np.tanh(overdrive * 5.2) * .21
                count = min(len(electric), total - offset)
                fuzz[offset:offset + count] += (distorted[:count] * .95 + electric[:count] * .12) * strength / len(notes)
        driven = np.clip(fuzz * 6.8, -.26, .26)
        driven = sosfilt(self.fuzz_cabinet, driven).astype(np.float32)
        t = np.arange(total) / sr
        envelope = np.minimum(t / .006, 1) * np.clip((total / sr - t - 1 / sr) / .3, 0, 1)
        self.add(start, driven * envelope, .38, .18 if take == 0 else .82)
        self.add(start, acoustic * envelope, 1.65, .5)
        self.voices[-1].spacious = .08

    def schedule_haze(self, start):
        beat = 60 / 116
        sequence = (
            (38, (50, 57, 62, 66), "D"),
            (42, (54, 57, 61, 66), "F#m"),
            (35, (54, 59, 62, 66), "Bm"),
            (43, (55, 59, 62, 67), "G"),
            (42, (54, 57, 61, 66), "F#m"),
            (40, (52, 55, 59, 64), "Em"),
            (38, (50, 57, 62, 66), "D"),
            (40, (52, 55, 59, 64), "Em"),
            (35, (54, 59, 62, 66), "Bm"),
            (33, (52, 57, 61, 64), "A"),
            (33, (54, 57, 61, 64), "A6"),
            (40, (55, 59, 62, 64), "Em7"),
        )
        index = self.chord_index % len(sequence)
        root, notes, self.chord_name = sequence[index]
        duration = 4 * beat
        at = lambda step: start + round(step * beat * self.sample_rate)
        for take in range(2):
            self.guitar_cloud(start, notes, duration, take)
        for step, accent in ((0, 72), (1, 60), (2, 68), (3, 60)):
            self.note(at(step), 33, root, beat * .90, accent, release=.12, gain=1.10)
        self.classic_drums(start, beat)
        return duration

    def schedule_ethereal(self, start):
        index = self.chord_index % 4
        if index == 0:
            self.progression = ["Cmaj7", "Am7", "Fmaj7", "G6"]
        chord = self.progression[index]
        self.chord_name = chord
        notes = chord_to_notes(chord)
        scale = SCALES.get(chord, [60, 62, 64, 67, 69, 72])
        at = lambda seconds: start + round(seconds * self.sample_rate)
        for i, pitch in enumerate(notes):
            self.note(at(i * .15), 89, pitch, 4 - i * .15, 32, .25 + i * .15, release=1.2, gain=2.0)
        choir_variant = 0
        for i, pitch in enumerate(notes):
            offset = .1 + i * .1
            self.note(at(offset), 52, pitch + 12 + choir_variant, 3.9 - offset, 18, .20 + i * .18, release=1.4, gain=2.8)
        root = NOTE_MAP[parse_root(chord)[0]] - 24
        for step, pitch in enumerate((root, root + 7, root, root + 7)):
            self.note(at(step), 33, pitch, .95, 42, release=.25, gain=1.0)
        phrase = (
            (76, 79, 76, 72, 76, 79),
            (76, 81, 84, 81, 79, 76),
            (77, 81, 84, 81, 77, 76),
            (79, 83, 86, 83, 79, 74),
        )[index]
        if (self.chord_index // 4) % 4 == 2:
            phrase = (phrase[0], phrase[1], phrase[3], phrase[2], *phrase[4:])
        rhythm = ((0, .65), (.75, .20), (1, .45), (1.5, .85), (2.5, .45), (3, .85))
        for i, (step, duration) in enumerate(rhythm):
            pitch = phrase[i]
            velocity = (46, 40, 44, 48, 40, 43)[i]
            self.last_lead = pitch
            self.note(at(step), 10, pitch, duration, velocity, .5, release=.55, gain=1.0)
            self.note(at(step), 27, pitch - 12, duration, 48, .57, release=.65, gain=.65, bend=.015)
            self.note(at(step + .5), 10, pitch, min(duration, 3.8 - step - .5), int(velocity * .35), .5, release=.35, gain=1.0)
        self.drum(at(0), 36, 35, .7)
        self.drum(at(2), 37, 38, .7)
        for step in range(8):
            self.drum(at(step * .5), 42, 15, .7)
        return 4.0

    def schedule_chord(self):
        if self.style != self.requested_style:
            self.style = self.requested_style
            self.chord_index = 0
            self.last_lead = None
        start = self.next_chord
        if self.style == STYLES[1]:
            duration = self.schedule_bossa(start)
        elif self.style == STYLES[2]:
            duration = self.schedule_haze(start)
        else:
            duration = self.schedule_ethereal(start)
        self.next_chord += round(duration * self.sample_rate)
        self.chord_index += 1

    def render(self, frames=BLOCK_SIZE):
        if frames <= 0:
            raise ValueError("Frame count must be positive")
        end = self.frame + frames
        while self.next_chord < end:
            self.schedule_chord()
        audio = np.zeros((frames, 2), dtype=np.float32)
        spacious = np.zeros_like(audio)
        remaining = []
        for voice in self.voices:
            voice_end = voice.start + len(voice.audio)
            left, right = max(self.frame, voice.start), min(end, voice_end)
            if right > left:
                section = voice.audio[left - voice.start:right - voice.start]
                audio[left - self.frame:right - self.frame] += section
                if voice.spacious:
                    spacious[left - self.frame:right - self.frame] += section * voice.spacious
            if voice_end > end:
                remaining.append(voice)
        self.voices = remaining
        wet = sum(delay.process(audio) for delay in self.delays) / len(self.delays)
        echoes = self.echo.process(audio)
        spacious, self.hall_state = sosfilt(self.hall_filter, spacious, axis=0, zi=self.hall_state)
        hall = sum(delay.process(spacious) for delay in self.hall) / math.sqrt(len(self.hall))
        for diffuser in self.diffusers:
            hall = diffuser.process(hall)
        mixed = audio + wet * .28 + echoes * .04 + hall * 1.8
        mixed, self.master_state = sosfilt(self.master_filter, mixed, axis=0, zi=self.master_state)
        self.frame = end
        return np.tanh(mixed * 1.4).astype(np.float32)


class AudioEngine:
    def __init__(self, style=STYLES[0], seed=None, device=None):
        self.composer = Composer(style, seed)
        self.device = device
        self.blocks = queue.Queue(maxsize=48)
        self.stop_event = threading.Event()
        self.ready = threading.Event()
        self.paused = False
        self.volume = .65
        self.gain = 0.0
        self.stream = None
        self.error = None
        self.underruns = 0
        self.played_frames = 0
        self.current = np.empty((0, 2), dtype=np.float32)
        self.offset = 0
        self.audible_style = style
        self.audible_chord = ""
        self.worker = None

    def produce(self):
        try:
            while not self.stop_event.is_set():
                if self.blocks.full():
                    self.stop_event.wait(.01)
                    continue
                block = self.composer.render()
                self.blocks.put_nowait((block, self.composer.style, self.composer.chord_name))
                if self.blocks.qsize() >= 32:
                    self.ready.set()
        except Exception as exc:
            self.error = str(exc)
            self.ready.set()

    def callback(self, outdata, frames, time_info, status):
        outdata.fill(0)
        if status:
            self.underruns += 1
        written = 0
        while written < frames:
            target = 0.0 if self.paused else self.volume
            if self.paused and self.gain <= .00001:
                break
            if self.offset >= len(self.current):
                try:
                    self.current, self.audible_style, self.audible_chord = self.blocks.get_nowait()
                    self.offset = 0
                except queue.Empty:
                    self.underruns += 1
                    self.gain = 0.0
                    break
            count = min(frames - written, len(self.current) - self.offset)
            ramp_frames = max(1, round(SAMPLE_RATE * .025))
            delta = np.clip(target - self.gain, -.65, .65)
            step = math.copysign(1 / ramp_frames, delta) if delta else 0
            if self.paused:
                count = min(count, max(1, math.ceil(self.gain * ramp_frames)))
            gains = self.gain + step * np.arange(1, count + 1)
            gains = np.clip(gains, min(self.gain, target), max(self.gain, target))
            outdata[written:written + count] = self.current[self.offset:self.offset + count] * gains[:, None]
            self.gain = float(gains[-1])
            self.offset += count
            written += count
            self.played_frames += count

    def start(self):
        import sounddevice as sd

        self.worker = threading.Thread(target=self.produce, name="epsilonic-composer", daemon=True)
        self.worker.start()
        if not self.ready.wait(15):
            self.close()
            raise RuntimeError("Audio preparation timed out")
        if self.error:
            self.close()
            raise RuntimeError(self.error)
        if self.stop_event.is_set():
            return
        try:
            self.stream = sd.OutputStream(samplerate=SAMPLE_RATE, blocksize=BLOCK_SIZE, channels=2, dtype="float32", latency="high", device=self.device, callback=self.callback)
            self.stream.start()
        except Exception:
            self.close()
            raise

    def close(self):
        self.stop_event.set()
        if self.stream is not None:
            try:
                self.stream.stop()
            finally:
                self.stream.close()
                self.stream = None
        if self.worker is not None:
            self.worker.join(timeout=5)


def render_file(path, style, seconds, seed=None):
    if not math.isfinite(seconds) or seconds <= 0:
        raise ValueError("Duration must be a finite positive number")
    composer = Composer(style, seed)
    total = round(seconds * SAMPLE_RATE)
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    with destination.open("wb") as raw, wave.open(raw, "wb") as output:
        output.setnchannels(2)
        output.setsampwidth(2)
        output.setframerate(SAMPLE_RATE)
        written = 0
        while written < total:
            count = min(BLOCK_SIZE, total - written)
            audio = composer.render(count)
            positions = np.arange(written, written + count)
            fade = np.minimum(np.clip(positions / (SAMPLE_RATE * .025), 0, 1), np.clip((total - 1 - positions) / (SAMPLE_RATE * .05), 0, 1))
            output.writeframesraw((audio * fade[:, None] * .65 * 32767).astype("<i2").tobytes())
            written += count
    return destination


def run_gui(style, seed=None, device=None):
    import tkinter as tk
    from tkinter import ttk
    from PIL import Image, ImageOps, ImageTk

    root = tk.Tk()
    root.title("Epsilonic Player")
    root.geometry("1000x520")
    root.resizable(False, False)
    engine = AudioEngine(style, seed, device)
    images = {}
    for name, theme in THEMES.items():
        path = BASE_DIR / theme["image"]
        if name == STYLES[1] and not path.exists():
            path = BASE_DIR / "epsilonic2.jpg"
        try:
            with Image.open(path) as picture:
                centering = (.5, .91) if name == STYLES[2] else (.5, .5)
                images[name] = ImageTk.PhotoImage(ImageOps.fit(picture.convert("RGB"), (1000, 520), method=Image.Resampling.LANCZOS, centering=centering))
        except OSError:
            pass
    main_frame = tk.Frame(root)
    main_frame.pack(fill="both", expand=True)
    bg_label = tk.Label(main_frame)
    bg_label.place(x=0, y=0, relwidth=1, relheight=1)
    ttk_style = ttk.Style()
    ttk_style.theme_use("clam")
    title = tk.Label(main_frame, text=" ~ EPSILONIC ~ ", font=("Courier", 26, "bold"))
    title.pack(pady=35)
    label = tk.Label(main_frame, text=" Playback Style: ", font=("Courier", 14, "bold"))
    label.pack(pady=5)
    selector = ttk.Combobox(main_frame, values=STYLES, state="readonly", width=32, font=("Courier", 13, "bold"), justify="center")
    selector.set(style)
    selector.pack(pady=5)
    pause = tk.Button(main_frame, text="PAUSE", font=("Courier", 14, "bold"), relief="flat", width=12, bd=0)
    pause.pack(pady=25)
    status = tk.Label(main_frame, text="Preparing audio…", font=("Courier", 12), wraplength=900)
    status.pack(pady=25)
    started = threading.Event()
    startup_error = []
    closing = False

    def popup_theme():
        theme = THEMES[selector.get()]
        for suffix, color in (("background", theme["pop_bg"]), ("foreground", theme["text"]), ("selectBackground", theme["sel_bg"]), ("selectForeground", theme["text"])):
            root.option_add("*TCombobox*Listbox." + suffix, color)
        root.option_add("*TCombobox*Listbox.font", ("Courier", 13, "bold"))
        try:
            popup = root.tk.call("ttk::combobox::PopdownWindow", selector)
            root.tk.call(f"{popup}.f.l", "configure", "-background", theme["pop_bg"], "-foreground", theme["text"], "-selectbackground", theme["sel_bg"], "-selectforeground", theme["text"], "-font", ("Courier", 13, "bold"), "-borderwidth", 0, "-highlightthickness", 0)
        except tk.TclError:
            pass

    def apply_theme():
        theme = THEMES[selector.get()]
        picture = images.get(selector.get())
        bg_label.configure(image=picture if picture else "", bg=theme["bg"])
        root.configure(bg=theme["bg"])
        main_frame.configure(bg=theme["bg"])
        for widget in (title, label, status):
            widget.configure(bg=theme["bg"], fg=theme["text"])
        pause.configure(bg=theme["bg"], fg=theme["text"], activebackground=theme["active_btn"], activeforeground=theme["text"])
        ttk_style.configure("TCombobox", fieldbackground=theme["bg"], background=theme["bg"], foreground=theme["text"], arrowcolor=theme["text"], bordercolor="white", darkcolor="white", lightcolor="white", selectbackground=theme["bg"], selectforeground=theme["text"])
        ttk_style.map("TCombobox", fieldbackground=[("readonly", theme["bg"])], foreground=[("readonly", theme["text"])], background=[("readonly", theme["bg"])], arrowcolor=[("readonly", theme["text"])])
        popup_theme()

    def change_style(event=None):
        engine.composer.requested_style = selector.get()
        selector.selection_clear()
        root.focus_set()
        apply_theme()

    def toggle():
        engine.paused = not engine.paused
        pause.configure(text="RESUME" if engine.paused else "PAUSE")

    def boot():
        try:
            engine.start()
        except Exception as exc:
            startup_error.append(str(exc))
        finally:
            started.set()

    def refresh():
        if closing:
            return
        if startup_error or engine.error:
            status.configure(text="Audio error: " + (startup_error[0] if startup_error else engine.error))
        elif started.is_set():
            elapsed = engine.played_frames // SAMPLE_RATE
            minutes, seconds = divmod(elapsed, 60)
            chord = engine.audible_chord or "—"
            pending = selector.get() != engine.audible_style
            status.configure(text="Changing to: " + selector.get() if pending else f"{chord} - {minutes}:{seconds:02d}")
        root.after(200, refresh)

    def close():
        nonlocal closing
        closing = True
        engine.stop_event.set()
        root.destroy()

    pause.configure(command=toggle)
    selector.bind("<<ComboboxSelected>>", change_style)
    selector.bind("<ButtonPress-1>", lambda event: root.after(5, popup_theme))
    root.protocol("WM_DELETE_WINDOW", close)
    apply_theme()
    starter = threading.Thread(target=boot, name="epsilonic-startup", daemon=True)
    starter.start()
    root.after(100, refresh)
    try:
        root.mainloop()
    finally:
        engine.stop_event.set()
        starter.join(timeout=16)
        engine.close()


def main():
    parser = argparse.ArgumentParser(description="Epsilonic · continuously evolving music")
    parser.add_argument("--style", choices=STYLES, default=STYLES[0])
    parser.add_argument("--seed", type=int)
    parser.add_argument("--render", metavar="WAV")
    parser.add_argument("--render-wall", metavar="WAV")
    parser.add_argument("--seconds", type=float, default=60)
    parser.add_argument("--device", type=int)
    parser.add_argument("--list-devices", action="store_true")
    args = parser.parse_args()
    if not math.isfinite(args.seconds) or args.seconds <= 0:
        parser.error("--seconds must be a finite positive number")
    if args.list_devices:
        import sounddevice as sd

        print(sd.query_devices())
    elif args.render or args.render_wall:
        print(render_file(args.render or args.render_wall, STYLES[2] if args.render_wall else args.style, args.seconds, args.seed))
    else:
        try:
            run_gui(args.style, args.seed, args.device)
        except ImportError as exc:
            parser.exit(1, f"Missing dependency: {exc}. Run python -m pip install -r requirements.txt\n")


if __name__ == "__main__":
    main()
