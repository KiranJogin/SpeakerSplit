from services.diarization_service import diarize_audio
from services.audio_utils import convert_to_wav_mono
from services.align_service import assign_speaker_to_words, group_words_to_turns

import whisper
import torch
import tempfile
import os
import datetime
import subprocess
from pathlib import Path

device = "cuda" if torch.cuda.is_available() else "cpu"
model = whisper.load_model("base", device=device)

SAVE_ROOT = Path("saved_transcription")
SAVE_ROOT.mkdir(exist_ok=True)

def extract_audio_segment(input_path, start, end, output_path):
    """Extract a specific audio segment (in seconds)."""
    command = [
        "ffmpeg",
        "-y",
        "-i",
        input_path,
        "-ss",
        str(start),
        "-to",
        str(end),
        "-c",
        "copy",
        output_path,
    ]
    subprocess.run(command, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

def _transcribe_from_path(tmp_path: str):
    """Internal: main pipeline given a temp wav path."""
    wav_path = convert_to_wav_mono(tmp_path)
    result = model.transcribe(wav_path, word_timestamps=True)

    # collect word-level timestamps
    words = []
    for seg in result["segments"]:
        for w in seg.get("words", []):
            words.append({"word": w["word"], "start": w["start"], "end": w["end"]})

    # diarization
    speaker_segments = diarize_audio(wav_path)

    # alignment: assign speaker labels to each word
    aligned = assign_speaker_to_words(words, speaker_segments)
    turns = group_words_to_turns(aligned)

    # session folder under saved_transcription/
    session_name = datetime.datetime.now().strftime("session_%Y-%m-%d_%H-%M-%S")
    session_dir = SAVE_ROOT / session_name
    session_dir.mkdir(parents=True, exist_ok=True)

    # save full transcript
    full_text_path = session_dir / "full_transcript.txt"
    with open(full_text_path, "w", encoding="utf-8") as f:
        for t in turns:
            f.write(f"{t['speaker']}: {t['text']}\n\n")

    # save each turn's text + audio
    for idx, t in enumerate(turns, start=1):
        sp_dir = session_dir / t["speaker"]
        sp_dir.mkdir(exist_ok=True)

        txt_path = sp_dir / f"line_{idx:02d}.txt"
        with open(txt_path, "w", encoding="utf-8") as f:
            f.write(t["text"])

        audio_path = sp_dir / f"line_{idx:02d}.wav"
        extract_audio_segment(wav_path, t["start"], t["end"], str(audio_path))

        # for Streamlit we store the real filesystem path
        t["audio_path"] = str(audio_path)

    # create drama-style script
    script_lines = [f"{t['speaker']}: {t['text']}" for t in turns]
    drama_script = "\n\n".join(script_lines)

    # cleanup temp wavs
    if os.path.exists(tmp_path):
        os.remove(tmp_path)
    if os.path.exists(wav_path):
        os.remove(wav_path)

    return {
        "status": "success",
        "session": session_name,
        "transcription": turns,
        "formatted_script": drama_script,
        "session_dir": str(session_dir),
    }

def transcribe_streamlit(file_bytes: bytes):
    """
    Entry point used by Streamlit.
    Takes raw file bytes and returns transcription + diarization result.
    """
    with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp:
        tmp.write(file_bytes)
        tmp_path = tmp.name

    return _transcribe_from_path(tmp_path)
