import numpy as np
import librosa
from scipy import signal
from scipy.ndimage import median_filter

class AudioProcessor:
    def __init__(self, sample_rate):
        self.sample_rate = sample_rate
    
    def preprocess(self, audio, reduce_noise=True, normalize=True):
        processed = audio.copy()
        
        if reduce_noise:
            processed = self.reduce_noise(processed)
        
        if normalize:
            processed = self.normalize_audio(processed)
        
        return processed
    
    def reduce_noise(self, audio):
        stft = librosa.stft(audio)
        magnitude = np.abs(stft)
        phase = np.angle(stft)
        
        noise_profile = np.median(magnitude[:, :10], axis=1, keepdims=True)
        
        threshold = 1.5
        mask = magnitude > (noise_profile * threshold)
        
        cleaned_magnitude = magnitude * mask
        
        cleaned_stft = cleaned_magnitude * np.exp(1j * phase)
        cleaned_audio = librosa.istft(cleaned_stft)
        
        return cleaned_audio
    
    def normalize_audio(self, audio):
        max_val = np.max(np.abs(audio))
        if max_val > 0:
            normalized = audio / max_val * 0.95
        else:
            normalized = audio
        return normalized
    
    def apply_bandpass_filter(self, audio, lowcut=80, highcut=8000):
        nyquist = self.sample_rate / 2
        low = lowcut / nyquist
        high = highcut / nyquist
        
        b, a = signal.butter(4, [low, high], btype='band')
        filtered = signal.filtfilt(b, a, audio)
        
        return filtered
    
    def detect_voice_activity(self, audio, frame_length=2048, hop_length=512):
        energy = librosa.feature.rms(y=audio, frame_length=frame_length, hop_length=hop_length)[0]
        
        threshold = np.mean(energy) * 1.5
        
        voice_frames = energy > threshold
        
        return voice_frames, energy
    
    def enhance_audio(self, audio):
        filtered = self.apply_bandpass_filter(audio)
        
        enhanced = self.reduce_noise(filtered)
        
        enhanced = self.normalize_audio(enhanced)
        
        return enhanced
