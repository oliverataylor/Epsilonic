# Epsilonic

Epsilonic is a desktop generative-music player by Oliver Taylor. It creates looping MIDI music in two styles:

- **Ethereal Drift** — an ambient, layered arrangement.
- **Bossa Staccato** — a rhythmic bossa-inspired arrangement.

## Run it

Install Python 3.10 or later, then open PowerShell in this folder and run:

```powershell
pip install -r requirements.txt
python epsilonic.py
```

Use the dropdown to change the style, **PAUSE** to stop or resume playback, and close the window when you are done. Epsilonic creates a temporary `dream_A.mid` file while it runs and removes it on close.

## Files

- `epsilonic.py` — application source code
- `epsilonic.png` — Ethereal Drift background
- `epsilonic2.png` — Bossa Staccato background
- `requirements.txt` — Python packages required to run the app
