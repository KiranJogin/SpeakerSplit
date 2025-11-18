
import os
import torch
from pyannote.audio import Pipeline
from dotenv import load_dotenv

load_dotenv()

HUGGINGFACE_TOKEN = os.getenv("HUGGINGFACE_TOKEN")
if not HUGGINGFACE_TOKEN:
    raise RuntimeError("Missing HUGGINGFACE_TOKEN in .env")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

try:
    pipeline = Pipeline.from_pretrained(
        "pyannote/speaker-diarization-3.1",
        use_auth_token=HUGGINGFACE_TOKEN
    )
    pipeline.to(device)
except Exception as e:
    raise RuntimeError(
        f"Failed to load diarization pipeline. Check Hugging Face token and model access.\nError: {e}"
    )

def diarize_audio(file_path: str):
    """Perform speaker diarization and return segments."""
    diarization = pipeline(file_path)
    segments = []
    for turn, _, speaker in diarization.itertracks(yield_label=True):
        segments.append({
            "start": turn.start,
            "end": turn.end,
            "speaker": speaker
        })
    return segments
