# RespGuard — Prototype of Early Breathing Distress Detection

> ⚠️ **Disclaimer:** This is **not a medical device** and does **not** provide any medical diagnosis.  
> Created for research and hackathon demonstration purposes only.

RespGuard is an experimental prototype that analyzes **cough, wheezing and breathing patterns** directly from ordinary video/audio — no wearable sensors, no additional hardware. The goal is to empower parents and caregivers by giving them **early warning signals** when a child's breathing might show signs of potential distress.

### 🌟 Why this matters

More than 250 million children every year suffer from respiratory conditions. Many parents already use cameras to monitor their children, but **no camera today understands how a child breathes**.

RespGuard attempts to change that.

It uses:
- 🎙 Audio signal analysis (cough, wheeze intensity, breathing rate)  
- ⏱ Episode detection logic within moving time windows  
- 🟢 Visual risk indicator overlay on the video  
- 🇵 Simple and portable — just run Python on a short video clip  

🚀 In the future, the system may be extended to real-time monitoring using IP cameras, local edge processing, or integrated with more advanced ML audio/video recognition models.

---

### 📁 Current Project Status

- ✔ Prototype of audio-based breathing and cough detection
- ✔ Works with stored video files (MP4)
- 🔬 Experimental detection logic based on onset & spectral energy
- 🔜 Future: integration with live IP camera (Tapo), trained models

---

### 🛠 Quick Start

```bash
pip install -r requirements.txt
python audio_dashboard.py examples/sample_video.mp4 --no-notebook
