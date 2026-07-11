import random
import time
import pretty_midi
import pygame
import os
import threading
import tkinter as tk
from tkinter import ttk
from PIL import Image, ImageTk

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
TRACK_A = os.path.join(BASE_DIR, "dream_A.mid")
IMAGE_A = "epsilonic.png"
IMAGE_B = "epsilonic2.jpg" if os.path.exists(os.path.join(BASE_DIR, "epsilonic2.jpg")) else "epsilonic2.png"
SOUNDFONT_FILE = os.path.join(BASE_DIR, "default.sf2")
NUMBER_OF_PROGRESSIONS = 100
current_style = "Ethereal Drift"  
is_paused = False  

PROGRESSIONS = [
    ["Cmaj7", "A7", "Dm7", "G7"], ["Gmaj7", "E7", "Am7", "D7"],         
    ["Fmaj7", "G7", "Cmaj7", "C6"], ["Amaj7", "B7", "Emaj7", "E6"],       
    ["Dmaj7", "Em7", "F#m7", "Em7"], ["Cmaj7", "D7", "Gmaj7", "G6"],       
    ["Bbmaj7", "C7", "Fmaj7", "F6"], ["Emaj7", "F#7", "Bmaj7", "B6"]       
]

NOTE_MAP = {
    "C": 60, "C#": 61, "Db": 61, "D": 62, "D#": 63, "Eb": 63,
    "E": 64, "F": 65, "F#": 66, "G": 67, "G#": 68, "Ab": 68,
    "A": 69, "A#": 70, "Bb": 70, "B": 71
}

THEMES = {
    "Bossa Staccato": {
        "text": "#fcd797", "bg": "#1a1512", "pop_bg": "#613b25", 
        "sel_bg": "#1a1512", "active_btn": "#613b25", "img_key": "bossa"
    },
    "Ethereal Drift": {
        "text": "#ffffff", "bg": "#112215", "pop_bg": "#112215", 
        "sel_bg": "#1b3822", "active_btn": "#1b3822", "img_key": "drift"
    }
}

def parse_root(ch):
    return (ch[:2], ch[2:]) if len(ch) >= 2 and ch[:2] in NOTE_MAP else (ch[0], ch[1:])

def chord_to_b_string_voicing(ch, step_index, chord_idx):
    root, quality = parse_root(ch)
    base = NOTE_MAP.get(root, 60)
    voicing = [base, base + (3 if "m" in quality and "maj" not in quality else 4), base + 7]
    top_note = base + (14 if step_index in [0, 1, 2] else 12 if step_index in [3, 4, 5] else 10 if "7" in quality and "maj" not in quality else 11)
    if chord_idx == 1: top_note += 7  
    return voicing + [top_note]

def chord_to_notes(ch):
    root, quality = parse_root(ch)
    base = NOTE_MAP.get(root, 60)
    notes = [base, base + (3 if "m" in quality and "maj" not in quality else 4), base + 7]
    if "maj7" in quality: notes.append(base + 11)
    elif "7" in quality: notes.append(base + 10)
    return notes

SCALES = {
    "Cmaj7": [60, 62, 64, 67, 69, 72], "G7":    [55, 59, 62, 65, 67, 71],
    "Dm7":   [57, 60, 62, 65, 69, 72], "Am7":   [57, 60, 64, 67, 69, 72],
    "D7":    [54, 57, 60, 62, 66, 69], "Fm7":   [56, 60, 63, 65, 68, 72],
    "Bb7":   [55, 58, 62, 65, 68, 70], "Ebmaj7":[55, 58, 62, 63, 67, 70],
    "Bm7":   [59, 62, 64, 66, 69, 71], "E7":    [56, 59, 62, 64, 68, 70],
    "Amaj7": [57, 61, 64, 68, 69, 73], "C#m7":  [56, 60, 61, 64, 68, 71],
    "F#7":   [54, 58, 61, 64, 66, 70], "Bmaj7":  [58, 61, 63, 66, 70, 71],
    "Gm7":   [55, 58, 62, 65, 67, 70], "C7":    [55, 58, 60, 64, 67, 70],
    "Fmaj7": [57, 60, 64, 65, 69, 72], "Cm7":   [55, 58, 60, 63, 67, 70],
    "F7":    [53, 57, 60, 63, 65, 69], "Bbmaj7":[53, 57, 58, 62, 65, 69],
    "Db7":   [53, 56, 59, 61, 65, 68], "F#m7":  [54, 57, 61, 64, 66, 69],
    "Emaj7": [56, 59, 63, 64, 68, 71], "Gmaj7":  [55, 59, 62, 66, 67, 71],
    "Em7":   [55, 59, 60, 64, 67, 71], "C6":     [60, 62, 64, 67, 69, 72],
    "E6":    [56, 59, 61, 64, 68, 71], "G6":     [55, 59, 62, 64, 67, 71],
    "F6":    [53, 57, 60, 62, 65, 69], "B6":     [56, 59, 63, 66, 68, 71]
}

def make_song():
    pm = pretty_midi.PrettyMIDI()
    is_bossa = current_style == "Bossa Staccato"
    
    piano_inst = pretty_midi.Instrument(program=0 if is_bossa else 89)
    choir_inst = pretty_midi.Instrument(program=0 if is_bossa else 52)
    bass = pretty_midi.Instrument(program=32 if is_bossa else 33)
    time_scale = 0.60 if is_bossa else 1.0

    drums = pretty_midi.Instrument(program=0, is_drum=True)
    celesta_lead = pretty_midi.Instrument(program=8)
    music_box_lead = pretty_midi.Instrument(program=10)

    # Dynamic target inclusion for control changes
    targets = [piano_inst, choir_inst] + ([] if is_bossa else [celesta_lead, music_box_lead])
    for inst in targets:
        inst.control_changes.append(pretty_midi.ControlChange(91, 127, 0))
        inst.control_changes.append(pretty_midi.ControlChange(93, 127, 0))

    beat = 0.0
    for _ in range(NUMBER_OF_PROGRESSIONS):
        for chord_idx, ch in enumerate(random.choice(PROGRESSIONS)):
            scale = SCALES.get(ch, [60, 62, 64, 67, 69, 72])
            
            if is_bossa:
                rhythm_steps = [0.0, 0.5, 0.75, 1.25, 1.5, 2.0, 2.5, 2.75, 3.5]
                for idx, step in enumerate(rhythm_steps):
                    for n in chord_to_b_string_voicing(ch, step_index=idx, chord_idx=chord_idx):
                        piano_inst.notes.append(pretty_midi.Note(
                            velocity=random.randint(80, 92), pitch=n,
                            start=(beat + step) * time_scale, end=(beat + step + 0.12) * time_scale
                        ))

                bass_scale = [n - 24 for n in scale]
                for i, step in enumerate([0.0, 1.0, 1.5, 2.0, 3.0, 3.5]):
                    bp = (bass_scale[0] + 1) if i == 5 else bass_scale[i % len(bass_scale)]
                    bass.notes.append(pretty_midi.Note(velocity=78, pitch=bp, start=(beat + step) * time_scale, end=(beat + step + 0.22) * time_scale))

                drums.notes.append(pretty_midi.Note(velocity=80, pitch=36, start=beat * time_scale, end=(beat + 0.1) * time_scale))
                drums.notes.append(pretty_midi.Note(velocity=80, pitch=36, start=(beat + 2.5) * time_scale, end=(beat + 2.6) * time_scale))
                
                for r_step in [0.5, 0.75, 1.5, 2.25, 2.75, 3.5]:
                    drums.notes.append(pretty_midi.Note(velocity=74, pitch=37, start=(beat + r_step) * time_scale, end=(beat + r_step + 0.1) * time_scale))
                
                for c_step, cuica_pitch in [(0.25, 79), (0.5, 78), (1.25, 79), (1.5, 78), (2.25, 79), (2.5, 78), (3.25, 79), (3.5, 78)]:
                    drums.notes.append(pretty_midi.Note(velocity=46, pitch=cuica_pitch, start=(beat + c_step) * time_scale, end=(beat + c_step + 0.12) * time_scale))
                
                for a_step, agogo_pitch in [(0.5, 67), (1.5, 68), (2.5, 67), (3.5, 68)]:
                    drums.notes.append(pretty_midi.Note(velocity=54, pitch=agogo_pitch, start=(beat + a_step) * time_scale, end=(beat + a_step + 0.10) * time_scale))

                for step in range(16):
                    drums.notes.append(pretty_midi.Note(
                        velocity=54 if step % 2 == 0 else 28, pitch=69 if step % 2 == 0 else 42, 
                        start=(beat + (step * 0.25)) * time_scale, end=(beat + (step * 0.25) + 0.05) * time_scale
                    ))
            else:
                chord_notes = chord_to_notes(ch)
                for i, n in enumerate(chord_notes):
                    piano_inst.notes.append(pretty_midi.Note(velocity=32, pitch=n, start=(beat + (i * 0.15)) * time_scale, end=(beat + 4.0) * time_scale))

                choir_variant = random.choice([0, 12, -12])
                for i, n in enumerate(chord_notes):
                    choir_inst.notes.append(pretty_midi.Note(velocity=18, pitch=n + 12 + choir_variant, start=(beat + 0.1 + (i * 0.1)) * time_scale, end=(beat + 3.9) * time_scale))

                bass_scale = [n - 24 for n in scale]
                bass_idx = 0
                for step in range(4):
                    bass.notes.append(pretty_midi.Note(velocity=42, pitch=bass_scale[bass_idx % len(bass_scale)], start=(beat + step) * time_scale, end=(beat + step + 0.95) * time_scale))
                    if step != 3:
                        choices = [c for c, cond in [(-1, bass_idx > 0), (1, bass_idx < len(bass_scale) - 1)] if cond]
                        bass_idx += random.choice(choices if choices else [0])

                offsets = ([0, 1, 2, 4, 3, 2] if chord_idx == 0 else 
                           [2, 3, 4, 5, 4, 2] if chord_idx == 1 else 
                           [4, 3, 2, 3, 1, 2] if chord_idx == 2 else [3, 2, 1, 0, 1, 0])

                for idx, (l_step, duration) in enumerate([(0.0, 0.45), (0.5, 0.45), (1.0, 0.45), (1.5, 0.85), (2.5, 0.45), (3.0, 0.85)]):
                    lead_pitch = scale[offsets[idx] % len(scale)] + 12
                    v_vel = random.randint(40, 48)
                    sp, ep = (beat + l_step) * time_scale, (beat + l_step + duration) * time_scale
                    
                    music_box_lead.notes.append(pretty_midi.Note(velocity=v_vel, pitch=lead_pitch, start=sp, end=ep))
                    music_box_lead.notes.append(pretty_midi.Note(velocity=int(v_vel * 0.35), pitch=lead_pitch, start=sp + 0.5, end=ep + 0.5))

                drums.notes.append(pretty_midi.Note(velocity=35, pitch=36, start=beat * time_scale, end=(beat + 0.1) * time_scale))
                drums.notes.append(pretty_midi.Note(velocity=38, pitch=37, start=(beat + 2.0) * time_scale, end=(beat + 2.1) * time_scale))
                for step in range(8):
                    drums.notes.append(pretty_midi.Note(velocity=15, pitch=42, start=(beat + (step * 0.5)) * time_scale, end=(beat + (step * 0.5) + 0.1) * time_scale))

            beat += 4.0 

    pm.instruments.extend([piano_inst, choir_inst, bass, drums, celesta_lead, music_box_lead])
    return pm

def init_mixer_with_soundfont():
    pygame.mixer.quit()
    if os.path.exists(SOUNDFONT_FILE):
        try:
            pygame.mixer.init(buffer=1024, soundfont=SOUNDFONT_FILE)
            return
        except: pass
    pygame.mixer.init(buffer=1024)

def audio_worker():
    init_mixer_with_soundfont()
    make_song().write(TRACK_A)
    pygame.mixer.music.load(TRACK_A)
    pygame.mixer.music.play(-1)  
    while True: time.sleep(1.0)

def toggle_pause(btn_pause):
    global is_paused
    if pygame.mixer.get_init():
        is_paused = not is_paused
        pygame.mixer.music.pause() if is_paused else pygame.mixer.music.unpause()
        btn_pause.config(text="RESUME" if is_paused else "PAUSE")

def trigger_track_regeneration():
    if pygame.mixer.get_init():
        pygame.mixer.music.stop()
        pygame.mixer.music.unload()  
        make_song().write(TRACK_A)
        pygame.mixer.music.load(TRACK_A)
        if not is_paused: pygame.mixer.music.play(-1)

def crop_to_aspect(image_path, tw, th):
    if not os.path.exists(image_path): return None
    try:
        img = Image.open(image_path)
        img_w, img_h = img.size
        ta, ia = tw / th, img_w / img_h
        if ia > ta:
            img = img.resize((int(th * ia), th), Image.Resampling.LANCZOS)
            img = img.crop(((img.size[0] - tw) // 2, 0, (img.size[0] - tw) // 2 + tw, th))
        else:
            img = img.resize((tw, int(tw / ia)), Image.Resampling.LANCZOS)
            img = img.crop((0, (img.size[1] - th) // 2, tw, (img.size[1] - th) // 2 + th))
        return ImageTk.PhotoImage(img)
    except Exception as e:
        print(f"Error processing image {image_path}: {e}")
        return None

def setup_gui():
    root = tk.Tk()
    root.title("Epsilonic Player")
    root.geometry("1000x520")  
    root.resizable(False, False)

    images = {
        "drift": crop_to_aspect(os.path.join(BASE_DIR, IMAGE_A), 1000, 520),
        "bossa": crop_to_aspect(os.path.join(BASE_DIR, IMAGE_B), 1000, 520)
    }

    main_frame = tk.Frame(root)
    main_frame.pack(fill="both", expand=True)

    bg_label = tk.Label(main_frame)
    bg_label.place(x=0, y=0, relwidth=1, relheight=1)

    style = ttk.Style()
    style.theme_use('clam')  

    lbl_title = tk.Label(main_frame, text=" ~ EPSILONIC ~ ", font=("Courier", 26, "bold"))
    lbl_title.pack(pady=35)

    lbl_style = tk.Label(main_frame, text=" Playback Style: ", font=("Courier", 14, "bold"))
    lbl_style.pack(pady=5)

    cb_style = ttk.Combobox(main_frame, values=["Ethereal Drift", "Bossa Staccato"], state="readonly", width=32, font=("Courier", 13, "bold"), justify="center")
    cb_style.set(current_style)
    cb_style.pack(pady=5)

    btn_pause = tk.Button(main_frame, text="PAUSE", font=("Courier", 14, "bold"), relief="flat", width=12, bd=0, command=lambda: toggle_pause(btn_pause))
    btn_pause.pack(pady=25)

    lbl_info = tk.Label(main_frame, text="Made by Oliver Taylor", font=("Courier", 12, "italic", "bold"))
    lbl_info.pack(pady=25)

    def safely_update_dropdown_popup():
        t = THEMES[cb_style.get()]
        for target in ['*TCombobox*Listbox.background', '*TCombobox*Listbox.foreground', '*TCombobox*Listbox.selectBackground', '*TCombobox*Listbox.selectForeground']:
            root.option_add(target, t["pop_bg"] if "background" in target.lower() and "select" not in target.lower() else t["text"] if "foreground" in target.lower() and "select" not in target.lower() else t["sel_bg"] if "background" in target.lower() else t["text"])
        root.option_add('*TCombobox*Listbox.font', ("Courier", 13, "bold"))
        
        try:
            popdown = root.tk.eval(f'ttk::combobox::PopdownWindow {cb_style}')
            if popdown:
                root.tk.call(f'{popdown}.f.l', 'configure', '-background', t["pop_bg"], '-foreground', t["text"], '-selectbackground', t["sel_bg"], '-selectforeground', t["text"], '-font', ("Courier", 13, "bold"), '-borderwidth', 0, '-highlightthickness', 0)
                root.tk.call(f'{popdown}', 'configure', '-background', '#ffffff', '-borderwidth', 1, '-relief', 'flat')
        except tk.TclError: pass

    def apply_theme(style_name):
        t = THEMES[style_name]
        img = images.get(t["img_key"])
        
        bg_label.config(image=img if img else "", bg=t["bg"])
        if img: bg_label.image = img

        root.config(bg=t["bg"])
        main_frame.config(bg=t["bg"])
        for lbl in [lbl_title, lbl_style, lbl_info]: lbl.config(bg=t["bg"], fg=t["text"])
        btn_pause.config(bg=t["bg"], fg=t["text"], activebackground=t["active_btn"], activeforeground=t["text"])
        
        style.configure("TCombobox", fieldbackground=t["bg"], background=t["bg"], foreground=t["text"], arrowcolor=t["text"], bordercolor="#ffffff", darkcolor="#ffffff", lightcolor="#ffffff", focusfill=t["bg"], selectbackground=t["bg"], selectforeground=t["text"])
        style.map("TCombobox", fieldbackground=[('readonly', t["bg"])], foreground=[('readonly', t["text"])], background=[('readonly', t["bg"])], arrowcolor=[('readonly', t["text"])])
        safely_update_dropdown_popup()

    def on_style_changed(event):
        global current_style
        current_style = cb_style.get()
        apply_theme(current_style)
        cb_style.selection_clear()
        root.focus()
        threading.Thread(target=trigger_track_regeneration, daemon=True).start()

    cb_style.bind("<<ComboboxSelected>>", on_style_changed)
    cb_style.bind("<ButtonPress-1>", lambda e: root.after(5, safely_update_dropdown_popup))
    
    apply_theme(current_style) 

    def on_close():
        pygame.mixer.quit()
        if os.path.exists(TRACK_A):
            try: os.remove(TRACK_A)
            except: pass
        root.destroy()
        os._exit(0)

    root.protocol("WM_DELETE_WINDOW", on_close)
    root.mainloop()

if __name__ == "__main__":
    threading.Thread(target=audio_worker, daemon=True).start()
    setup_gui()
