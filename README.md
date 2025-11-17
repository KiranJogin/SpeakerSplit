SpeakerSplit – AI Voice Separation System
🔊 AI-Powered Multi-Speaker Voice Separation & Playback

SpeakerSplit is an AI-based audio processing application designed to separate mixed audio recordings containing multiple speakers into clean, isolated audio tracks.
Using advanced signal processing and machine learning clustering techniques, the system identifies different speakers, segments the audio, and reconstructs individual speaker tracks that users can play or download.

Built with Streamlit for the interface and leveraging powerful audio processing libraries like librosa, scipy, and scikit-learn, SpeakerSplit enables seamless audio upload, processing configuration, playback, and analysis.

🚀 Features

Separate multiple overlapping voices from a single audio source

Automatic speaker identification & clustering (2–6 speakers supported)

Noise reduction, audio normalization & spectral filtering

Individual speaker track export & playback dashboard

Visual audio analysis (waveform & spectrogram views)

User-configurable settings (noise level, speaker count, chunk size)

🏗 System Architecture
Frontend (UI Layer) – Streamlit Application
Component	Description
Framework	Streamlit
Layout	Multi-tab interface
Tabs	Upload & Process, Playback Dashboard, Analysis
State Management	Session state for audio, separated tracks, sample rate
Configurable Controls	Sidebar for noise reduction, normalization, speaker count
Backend Architecture
Folder / Module	Responsibility
audio_processor.py	Preprocessing (noise removal, normalization, filtering)
speaker_separator.py	Feature extraction, clustering, separation logic
app.py	Main orchestrator & UI integration
Processing Pipeline

Upload audio & convert format (soundfile/pydub)

Apply preprocessing (noise profile estimation, normalization)

Extract features (MFCC, spectral contrast, chroma, zero-crossing rate)

Segment audio into time-based chunks (~1 second)

Cluster speaker segments (KMeans / hierarchical clustering)

Rebuild separated tracks for each speaker

Output playback & download options

🔬 Core Signal Processing Methods
Technique	Purpose
STFT (Short-Time Fourier Transform)	Time–frequency analysis
Median-based noise profiling	Background noise reduction
Band-pass filtering (80 Hz – 8 kHz)	Isolate human voice
Spectral masking	Remove noise outside threshold
1-second chunk segmentation	Granular clustering per speaker
🧠 Machine Learning Components
Component	Description
Feature Extraction	MFCC (13), spectral contrast (7), chroma (12), ZCR
Clustering Algorithms	KMeans & Hierarchical clustering
Distance Metrics	Cosine / Euclidean
Speaker Count	Configurable (2–6 speakers)
📂 Data Storage Strategy

Temporary storage via tempfile

In-memory processing via NumPy arrays

Streamlit session state persistence

No database or permanent storage

📦 Dependencies
Audio & Signal Processing

librosa

soundfile

pydub

scipy.signal

numpy

Machine Learning & Clustering

scikit-learn

scipy.cluster.hierarchy

scipy.spatial.distance

UI & Visualization

streamlit

matplotlib

🖥 User Interface

Streamlit-based interactive web dashboard

Upload audio, view spectrograms & waveforms, play separated tracks

Sidebar for intelligent configuration

⚙️ Audio Parameters
Parameter	Default
Sample Rate	Handled automatically by librosa
FFT Size	2048
Hop Length	512
Noise Threshold	1.5× baseline
Normalization	95% peak limit
📌 Use Cases

Podcast voice separation

Meeting / Lecture audio clarity

Interview audio enhancement

Audio forensics & evidence analysis

Content creation & editing

🏁 Future Enhancements

Name-based speaker labeling

Real-time microphone streaming support

Speech-to-text & per-speaker summary

Automated transcription export
