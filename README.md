# 🎙️ Voice Isolation Project

### 🔊 Overview
**Voice Isolation** is a deep learning–based system designed to **separate individual speaker voices** from a mixed or overlapping audio signal.  
It uses **pre-trained speech separation and diarization models** to identify, isolate, and export clean voice tracks for each speaker.

This project can be applied to:
- Meeting transcription and speaker analysis  
- Podcast editing and background noise removal  
- AI training datasets for speech models  
- Music vocal/instrument separation (with modifications)

---

## 🧠 Features
- 🎧 **Multi-speaker separation** – Isolates multiple voices from a single recording  
- 🧍‍♂️ **Speaker diarization** – Detects “who spoke when”  
- 🔊 **Audio enhancement** – Removes noise and improves clarity  
- 💾 **Export options** – Saves separated tracks as individual `.wav` files  
- 🧩 **Model flexibility** – Supports models like `SpeechBrain`, `Pyannote`, or `Sudo rm -rf Demucs`  
- ⚙️ **Customizable pipeline** – Easily extend for 2, 3, or N-speaker separation

---

## 🏗️ Project Structure
voice-isolation/
├── data/
│ ├── input_audio/ # Raw mixed audio files
│ ├── separated_audio/ # Output directory for isolated voices
├── models/
│ ├── your_model.ckpt # Pre-trained model weights
│ └── config.yaml # Model configuration file
├── pitchnet/
│ ├── scripts/
│ │ ├── separate_audio.py # Main script to run separation
│ ├── utils/
│ │ ├── audio_utils.py # Helper functions
│ └── init.py
├── requirements.txt
├── README.md
└── main.py
