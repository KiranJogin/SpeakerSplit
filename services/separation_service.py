# app/services/separation_service.py
import os
import torch
import torchaudio
import tempfile
from typing import List
from speechbrain.pretrained import SepformerSeparation as Separator

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# Load once
SEP_MODEL = Separator.from_hparams(
    source="speechbrain/sepformer-whamr",
    savedir="pretrained_sepformer",
    run_opts={"device": DEVICE}
)

def _save_tensor_to_wav(tensor: torch.Tensor, sample_rate: int) -> str:
    """
    Save a 1-D waveform tensor to a temporary .wav file and return its path.
    tensor: [time] or [1, time]
    """
    if tensor.ndim == 1:
        tensor = tensor.unsqueeze(0)  # [1, time]
    tensor = tensor.detach().cpu()
    fd, path = tempfile.mkstemp(suffix=".wav")
    os.close(fd)  # we'll write to the path with torchaudio
    torchaudio.save(path, tensor, sample_rate=sample_rate)
    return path

def separate_speakers(input_path: str) -> List[str]:
    """
    Separate mixture into sources. Returns list of temp .wav file paths.
    Raises on hard failure so caller can fallback.
    """
    # est shape is [batch, num_spk, time] or [num_spk, time]
    est = SEP_MODEL.separate_file(path=input_path)

    if est.ndim == 3:
        est = est.squeeze(0)  # [N, T]

    num_sources = int(est.size(0))
    if num_sources == 0:
        return []

    # SepFormer-WHAMR works at 8k internally
    sr = getattr(getattr(SEP_MODEL, "hparams", None), "sample_rate", 8000) or 8000
    sr = int(sr)

    out_paths = []
    for i in range(num_sources):
        wav_path = _save_tensor_to_wav(est[i], sr)
        out_paths.append(wav_path)

    return out_paths
