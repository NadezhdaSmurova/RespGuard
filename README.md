RespGuard — Early Breathing Distress Prototype

⚠️ Not a medical device.
Prototype for research and hackathon experiments only.

RespGuard analyzes cough, wheeze, and breathing patterns directly from ordinary video/audio — no sensors or wearables needed.
Goal: explore how early warning signals could look if a camera understood breathing.


1️⃣ Cough / wheeze spikes
https://github.com/NadezhdaSmurova/RespGuard/blob/main/assets/demo.mp4

🔒 Privacy & Child Safety

This demo uses publicly available videos of children with strict privacy protection measures applied.
To prevent identification, the child’s face is fully masked with a solid light-blue circle, and the original voice is pitch-shifted by +4 semitones.
These transformations hide all biometric and personally identifiable features while keeping the respiratory sounds (coughs, wheezes, breathing patterns) acoustically intact for analysis.

🌟 What RespGuard Does

For each input .mp4 video:

extracts the audio,
computes:
Cough level (onset spikes)
Wheeze level (mid-band spectral energy: 400–2000 Hz)
Breathing rate (RMS peak cycles)
counts episodes per last 60 seconds (cough & wheeze),
estimates a simple RISK ALERT: LOW / MEDIUM / HIGH,
adds a visual dashboard overlay including:
three risk bars (BREATH / WHEEZE / COUGH),
total & recent episodes,
small disclaimer: Experimental demo. Not for medical use.

🛠 Quick Start
pip install -r requirements.txt
python RespGuard.py


Input folder: examples/input
Output folder: examples/output

Example:

input:  examples/input/video.mp4
output: examples/output/output_video.mp4

📂 Repository Structure
RespGuard.py          # main script
examples/input/       # put test videos here
examples/output/      # results appear here
requirements.txt
README.md

🧠 How the risk logic works.

HIGH risk if:

cough ≥ 6/min, OR
wheeze ≥ 2/min, OR
breathing rate > 35/min

MEDIUM if:

cough ≥ 3/min, OR
wheeze ≥ 1/min, OR
breathing 30–35/min

LOW otherwise.

All thresholds are experimental and not clinically validated.

⚠ Important Note
RespGuard is a concept prototype.
It must not be used for diagnosis or safety decisions.

📁 Project Status
✔ Prototype of audio-based breathing and cough analysis
✔ Works with stored video files (.mp4)
✔ Experimental detection logic based on onset and spectral energy
✔ Visual dashboard overlay with per-parameter risk bars
🔜 Planned: integration with live IP cameras (e.g. Tapo)
🔜 Planned: replacement of handcrafted logic with trained ML models