import numpy as np
import librosa
from scipy.cluster.hierarchy import linkage, fcluster
from scipy.spatial.distance import pdist, squareform
from sklearn.cluster import KMeans
from scipy import signal

class SpeakerSeparator:
    def __init__(self, sample_rate, num_speakers=2):
        self.sample_rate = sample_rate
        self.num_speakers = num_speakers
        self.hop_length = 512
        self.n_fft = 2048
    
    def extract_features(self, audio):
        mfcc = librosa.feature.mfcc(
            y=audio,
            sr=self.sample_rate,
            n_mfcc=13,
            hop_length=self.hop_length
        )
        
        spectral_contrast = librosa.feature.spectral_contrast(
            y=audio,
            sr=self.sample_rate,
            hop_length=self.hop_length
        )
        
        chroma = librosa.feature.chroma_stft(
            y=audio,
            sr=self.sample_rate,
            hop_length=self.hop_length
        )
        
        zcr = librosa.feature.zero_crossing_rate(
            y=audio,
            hop_length=self.hop_length
        )
        
        features = np.vstack([mfcc, spectral_contrast, chroma, zcr])
        
        return features.T
    
    def segment_audio(self, audio, segment_duration=1.0):
        segment_samples = int(segment_duration * self.sample_rate)
        
        num_segments = len(audio) // segment_samples
        
        segments = []
        segment_indices = []
        
        for i in range(num_segments):
            start = i * segment_samples
            end = start + segment_samples
            
            segment = audio[start:end]
            
            energy = np.sum(segment ** 2)
            
            if energy > 0.001:
                segments.append(segment)
                segment_indices.append((start, end))
        
        return segments, segment_indices
    
    def cluster_segments(self, features, num_speakers):
        if len(features) < num_speakers:
            num_speakers = max(1, len(features))
        
        kmeans = KMeans(n_clusters=num_speakers, random_state=42, n_init='auto')
        labels = kmeans.fit_predict(features)
        
        return labels
    
    def separate_speakers(self, audio):
        segments, segment_indices = self.segment_audio(audio, segment_duration=0.5)
        
        if len(segments) == 0:
            return {"speaker_1": audio}
        
        segment_features = []
        for segment in segments:
            features = self.extract_features(segment)
            avg_features = np.mean(features, axis=0)
            segment_features.append(avg_features)
        
        segment_features = np.array(segment_features)
        
        labels = self.cluster_segments(segment_features, self.num_speakers)
        
        speaker_audio = {}
        for speaker_id in range(self.num_speakers):
            speaker_audio[f"speaker_{speaker_id + 1}"] = np.zeros_like(audio)
        
        for idx, label in enumerate(labels):
            start, end = segment_indices[idx]
            speaker_key = f"speaker_{label + 1}"
            speaker_audio[speaker_key][start:end] = segments[idx]
        
        speaker_audio = {k: v for k, v in speaker_audio.items() if np.sum(np.abs(v)) > 0}
        
        if len(speaker_audio) == 0:
            speaker_audio = {"speaker_1": audio}
        
        for speaker_key in speaker_audio:
            speaker_audio[speaker_key] = self.apply_spectral_masking(
                audio,
                speaker_audio[speaker_key]
            )
        
        return speaker_audio
    
    def apply_spectral_masking(self, original, speaker_signal):
        stft_original = librosa.stft(original, n_fft=self.n_fft, hop_length=self.hop_length)
        stft_speaker = librosa.stft(speaker_signal, n_fft=self.n_fft, hop_length=self.hop_length)
        
        magnitude_original = np.abs(stft_original)
        magnitude_speaker = np.abs(stft_speaker)
        
        mask = magnitude_speaker / (magnitude_original + 1e-10)
        mask = np.clip(mask, 0, 1)
        
        masked_stft = stft_original * mask
        
        enhanced_audio = librosa.istft(masked_stft, hop_length=self.hop_length)
        
        if len(enhanced_audio) < len(original):
            enhanced_audio = np.pad(enhanced_audio, (0, len(original) - len(enhanced_audio)))
        elif len(enhanced_audio) > len(original):
            enhanced_audio = enhanced_audio[:len(original)]
        
        return enhanced_audio
    
    def smooth_labels(self, labels, window_size=5):
        smoothed = signal.medfilt(labels.astype(float), kernel_size=window_size)
        return smoothed.astype(int)
