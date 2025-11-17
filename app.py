import streamlit as st
import numpy as np
import librosa
import librosa.display
import soundfile as sf
from pydub import AudioSegment
import matplotlib.pyplot as plt
from scipy import signal
import io
import os
import tempfile
from pathlib import Path

from audio_processor import AudioProcessor
from speaker_separator import SpeakerSeparator

st.set_page_config(
    page_title="SpeakerSplit - AI Voice Separation",
    page_icon="🎙️",
    layout="wide"
)

st.title("🎙️ SpeakerSplit")
st.markdown("**AI-Powered Individual Voice Separation and Playback System**")
st.markdown("---")

if 'processed_audio' not in st.session_state:
    st.session_state.processed_audio = None
if 'separated_speakers' not in st.session_state:
    st.session_state.separated_speakers = None
if 'original_audio' not in st.session_state:
    st.session_state.original_audio = None
if 'sample_rate' not in st.session_state:
    st.session_state.sample_rate = None

st.sidebar.header("⚙️ Settings")
noise_reduction = st.sidebar.checkbox("Enable Noise Reduction", value=True)
normalize_audio = st.sidebar.checkbox("Normalize Audio", value=True)
num_speakers = st.sidebar.slider("Expected Number of Speakers", 2, 6, 2)

st.sidebar.markdown("---")
st.sidebar.markdown("### About")
st.sidebar.info(
    "SpeakerSplit separates mixed audio recordings into individual speaker tracks. "
    "Upload your audio file containing multiple speakers, and the system will automatically "
    "identify and isolate each voice."
)

tab1, tab2, tab3 = st.tabs(["📤 Upload & Process", "🎧 Playback Dashboard", "📊 Analysis"])

with tab1:
    st.header("Upload Audio File")
    st.markdown("Supported formats: WAV, MP3, M4A, FLAC, OGG")
    
    uploaded_file = st.file_uploader(
        "Choose an audio file",
        type=["wav", "mp3", "m4a", "flac", "ogg"],
        help="Upload a multi-speaker audio recording"
    )
    
    if uploaded_file is not None:
        st.success(f"✅ File uploaded: {uploaded_file.name}")
        
        col1, col2 = st.columns([2, 1])
        
        with col1:
            st.audio(uploaded_file, format=f'audio/{uploaded_file.type.split("/")[-1]}')
        
        with col2:
            st.metric("File Size", f"{len(uploaded_file.getvalue()) / (1024*1024):.2f} MB")
        
        if st.button("🚀 Process Audio", type="primary", use_container_width=True):
            with st.spinner("Processing audio... This may take a few minutes."):
                try:
                    progress_bar = st.progress(0)
                    status_text = st.empty()
                    
                    status_text.text("Loading audio file...")
                    progress_bar.progress(10)
                    
                    with tempfile.NamedTemporaryFile(delete=False, suffix=Path(uploaded_file.name).suffix) as tmp_file:
                        tmp_file.write(uploaded_file.getvalue())
                        tmp_path = tmp_file.name
                    
                    audio, sr = librosa.load(tmp_path, sr=None, mono=False)
                    
                    if audio.ndim > 1:
                        audio = np.mean(audio, axis=0)
                    
                    st.session_state.original_audio = audio
                    st.session_state.sample_rate = sr
                    
                    status_text.text("Preprocessing audio...")
                    progress_bar.progress(30)
                    
                    processor = AudioProcessor(sr)
                    processed = processor.preprocess(
                        audio,
                        reduce_noise=noise_reduction,
                        normalize=normalize_audio
                    )
                    st.session_state.processed_audio = processed
                    
                    status_text.text("Detecting and separating speakers...")
                    progress_bar.progress(50)
                    
                    separator = SpeakerSeparator(sr, num_speakers=num_speakers)
                    separated_speakers = separator.separate_speakers(processed)
                    
                    st.session_state.separated_speakers = separated_speakers
                    
                    status_text.text("Finalizing...")
                    progress_bar.progress(100)
                    
                    os.unlink(tmp_path)
                    
                    st.success(f"✅ Successfully separated {len(separated_speakers)} speakers!")
                    st.balloons()
                    
                except Exception as e:
                    st.error(f"❌ Error processing audio: {str(e)}")
                    st.exception(e)

with tab2:
    st.header("Separated Speaker Tracks")
    
    if st.session_state.separated_speakers is not None and st.session_state.sample_rate is not None:
        speakers = st.session_state.separated_speakers
        sr = st.session_state.sample_rate
        
        st.markdown(f"**{len(speakers)} speakers detected**")
        st.markdown("---")
        
        for idx, (speaker_label, speaker_audio) in enumerate(speakers.items(), 1):
            with st.expander(f"🎤 Speaker {idx}", expanded=True):
                col1, col2 = st.columns([3, 1])
                
                with col1:
                    audio_bytes = io.BytesIO()
                    sf.write(audio_bytes, speaker_audio, sr, format='WAV')
                    audio_bytes.seek(0)
                    st.audio(audio_bytes, format='audio/wav')
                
                with col2:
                    duration = len(speaker_audio) / sr
                    st.metric("Duration", f"{duration:.1f}s")
                    st.metric("Samples", f"{len(speaker_audio):,}")
                    
                    wav_bytes = io.BytesIO()
                    sf.write(wav_bytes, speaker_audio, sr, format='WAV')
                    wav_bytes.seek(0)
                    
                    st.download_button(
                        label="⬇️ Download WAV",
                        data=wav_bytes,
                        file_name=f"speaker_{idx}.wav",
                        mime="audio/wav",
                        use_container_width=True
                    )
                
                fig, ax = plt.subplots(figsize=(10, 2))
                time = np.linspace(0, duration, len(speaker_audio))
                ax.plot(time, speaker_audio, linewidth=0.5, color='#1f77b4')
                ax.set_xlabel('Time (s)')
                ax.set_ylabel('Amplitude')
                ax.set_title(f'Speaker {idx} Waveform')
                ax.grid(True, alpha=0.3)
                st.pyplot(fig)
                plt.close()
        
        st.markdown("---")
        col1, col2, col3 = st.columns(3)
        with col2:
            if st.button("🗑️ Clear Results", use_container_width=True):
                st.session_state.separated_speakers = None
                st.session_state.processed_audio = None
                st.session_state.original_audio = None
                st.rerun()
    else:
        st.info("👆 Upload and process an audio file to see separated speaker tracks here.")

with tab3:
    st.header("Audio Analysis & Visualization")
    
    if st.session_state.original_audio is not None and st.session_state.sample_rate is not None:
        audio = st.session_state.original_audio
        sr = st.session_state.sample_rate
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("Original Audio Waveform")
            fig, ax = plt.subplots(figsize=(10, 4))
            time = np.linspace(0, len(audio) / sr, len(audio))
            ax.plot(time, audio, linewidth=0.5)
            ax.set_xlabel('Time (s)')
            ax.set_ylabel('Amplitude')
            ax.set_title('Full Audio Waveform')
            ax.grid(True, alpha=0.3)
            st.pyplot(fig)
            plt.close()
        
        with col2:
            st.subheader("Spectrogram")
            fig, ax = plt.subplots(figsize=(10, 4))
            D = librosa.amplitude_to_db(np.abs(librosa.stft(audio)), ref=np.max)
            img = librosa.display.specshow(D, sr=sr, x_axis='time', y_axis='hz', ax=ax, cmap='viridis')
            ax.set_title('Spectrogram')
            fig.colorbar(img, ax=ax, format='%+2.0f dB')
            st.pyplot(fig)
            plt.close()
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric("Sample Rate", f"{sr} Hz")
        with col2:
            st.metric("Duration", f"{len(audio) / sr:.2f}s")
        with col3:
            st.metric("Total Samples", f"{len(audio):,}")
        
        if st.session_state.separated_speakers is not None:
            st.markdown("---")
            st.subheader("Speaker Timeline")
            
            fig, ax = plt.subplots(figsize=(12, 6))
            
            for idx, (speaker_label, speaker_audio) in enumerate(st.session_state.separated_speakers.items(), 1):
                energy = np.abs(speaker_audio)
                smoothed = signal.medfilt(energy, kernel_size=2001)
                time = np.linspace(0, len(speaker_audio) / sr, len(speaker_audio))
                ax.plot(time, smoothed + (idx - 1) * 0.3, label=f'Speaker {idx}', linewidth=1)
            
            ax.set_xlabel('Time (s)')
            ax.set_ylabel('Speaker Activity')
            ax.set_title('Speaker Activity Timeline')
            ax.legend()
            ax.grid(True, alpha=0.3)
            st.pyplot(fig)
            plt.close()
    else:
        st.info("👆 Upload and process an audio file to see detailed analysis here.")

st.markdown("---")
st.markdown(
    "<div style='text-align: center; color: #666;'>"
    "SpeakerSplit v1.0 | AI-Powered Voice Separation System"
    "</div>",
    unsafe_allow_html=True
)
