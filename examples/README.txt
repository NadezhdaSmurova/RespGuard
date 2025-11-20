# RespiraGuard Audio Prototype

Prototype: audio-based analysis of cough, wheeze and breathing rate from a single video file.

> ⚠️ **Disclaimer:**  
> This is **NOT** a medical device and does **NOT** provide a diagnosis.  
> Research / hackathon prototype for demonstration only.

---

## What it does

- Loads a video with a child (or any person) and extracts the audio.
- Estimates:
  - **Cough level** – sharp audio spikes (onset envelope).
  - **Wheeze level** – dominance of mid frequencies (≈400–2000 Hz) over low frequencies.
  - **Breathing rate** – rough breaths-per-minute estimate from the RMS envelope.
- Counts **episodes per last 60 seconds**:
  - cough episodes,
  - wheeze episodes.
- Computes a simple **risk alert**: `LOW` / `MEDIUM` / `HIGH` based on:
  - number of cough episodes,  
  - number of wheeze episodes,  
  - breathing rate (fast breathing).
- Draws a **dashboard overlay** on the video and saves it as `mp4`.

---

## Repository structure

```text
src/
  audio_dashboard.py    # main script with audio pipeline + overlay
notebooks/
  demo_colab.ipynb      # optional Google Colab demo
examples/
  README.md             # description of expected input test videos
requirements.txt        # Python dependencies
LICENSE
README.md
.gitignore


Usage

Put your test video into examples/, for example: examples/input_video2.mp4.

Run:

python -m src.audio_dashboard examples/input_video2.mp4 --output output_med_dashboard.mp4


The script will:

extract audio,

compute features,

add overlay with:

cough level,

wheeze level,

breathing rate,

episodes in the last 60 seconds,

risk alert,

save the result to output_med_dashboard.mp4.

How the logic works (high-level)

Cough level – normalized librosa.onset.onset_strength.

Wheeze level – ratio of energy in 400–2000 Hz band to low-frequency band
with local normalization and non-linear compression.

Breath rate – peak detection over smoothed RMS envelope in a ±10 s window.

Episodes per last minute are counted using a small EpisodeCounter helper with:

a probability/level threshold,

sliding time window (60 s),

minimal separation between episodes.

Risk alert:

HIGH if:

cough episodes ≥ 6, or

wheeze episodes ≥ 2, or

breathing rate > 35 / min

MEDIUM if:

cough episodes ≥ 3, or

wheeze episodes ≥ 1, or

breathing rate in [30, 35]

otherwise LOW.

All thresholds are experimental and tuned manually for demo purposes.

Limitations

Works only with clear audio and limited background noise.

Thresholds and logic are not validated clinically.

Does not distinguish between different types of cough/wheeze.

For research / hackathon use only.

Roadmap / ideas

Replace hand-crafted features with trained classifiers (cough / wheeze models).

Add video-based breathing estimation (chest motion).

Calibrate thresholds on real annotated data.

Build a small web or mobile UI around this prototype.


