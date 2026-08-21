"""
MuScriptor WebUI - Gradio Server Backend
=========================================

Fixed version with UnicodeDecodeError handling.
Uses gradio.Server for queuing, concurrency control, and API access.
"""

import os
import io
import json
import wave
import tempfile
import shutil
from pathlib import Path
from typing import Optional

from gradio import Server
from gradio.data_classes import FileData
from fastapi.responses import HTMLResponse, FileResponse
from fastapi import UploadFile, File, Form, HTTPException

# MuScriptor imports
from muscriptor.transcription_model import TranscriptionModel
from muscriptor.tokenizer.mt3 import MT3_FULL_PLUS_GROUP_NAMES, resolve_instrument_names
from muscriptor.events import NoteStartEvent, NoteEndEvent, ProgressEvent
from muscriptor.utils.audio import _read_wav_file, _read_non_wav_file
from muscriptor.utils.beats import BeatDetectionError, TempoDetection
from muscriptor.utils.download import ModelDownloadError

import dataclasses
import base64


def safe_read_text(path, encodings=None):
    if encodings is None:
        encodings = ["utf-8", "latin-1", "cp1252", "iso-8859-1"]
    for encoding in encodings:
        try:
            return path.read_text(encoding=encoding)
        except (UnicodeDecodeError, UnicodeError):
            continue
    return path.read_bytes().decode("utf-8", errors="replace")


def safe_json_load(path):
    return json.loads(safe_read_text(path))


import muscriptor.transcription_model as tm_module

def _fixed_config_from_json(path):
    data = safe_json_load(path)
    return tm_module._ModelConfig(**{field: data[field] for field in tm_module._CONFIG_FIELDS})

if hasattr(tm_module, "_config_from_json"):
    tm_module._config_from_json = _fixed_config_from_json

model = None
WEB_DIR = Path(__file__).resolve().parent


def clear_corrupted_cache():
    cache_dir = Path.home() / ".cache" / "huggingface"
    cleared = []
    if cache_dir.exists():
        for pattern in ["hub/models--MuScriptor--*", "hub/models--muscriptor-*"]:
            for item in cache_dir.glob(pattern):
                try:
                    shutil.rmtree(item)
                    cleared.append(str(item.name))
                except Exception:
                    pass
    return cleared


def load_model(model_size="medium", device="auto", dtype=None, retry=True):
    global model
    if model is not None:
        return model
    max_retries = 2 if retry else 1
    for attempt in range(max_retries):
        try:
            if attempt > 0:
                print("Clearing corrupted cache and retrying...")
                clear_corrupted_cache()
            model = TranscriptionModel.load_model(weights_path=model_size, device=device, dtype=dtype)
            return model
        except UnicodeDecodeError as e:
            if attempt < max_retries - 1:
                continue
            raise
        except ModelDownloadError as e:
            raise ValueError(f"Error loading model: {e}")


def event_to_dict(ev):
    if isinstance(ev, NoteStartEvent):
        return {"type": "start", **dataclasses.asdict(ev)}
    return {"type": "end", "end_time": ev.end_time, "start_event_index": ev.start_event_index}


app = Server(title="muscriptor")


@app.api(name="transcribe")
def transcribe(audio_file: FileData, instruments: str = "", detect_tempo: str = "best-effort") -> FileData:
    m = load_model()
    instrument_names = None
    if instruments and instruments.strip():
        tokens = [n.strip() for n in instruments.split(",") if n.strip()]
        instrument_names = resolve_instrument_names(tokens)
    audio_path = Path(audio_file["path"])
    with open(audio_path, "rb") as f:
        data = f.read()
    try:
        wav, sr = _read_wav_file(io.BytesIO(data))
    except (wave.Error, EOFError):
        try:
            wav, sr = _read_non_wav_file(io.BytesIO(data))
        except Exception as e:
            raise ValueError(f"Could not decode audio: {e}")
    tempo_mode = {"true": True, "false": False, "best-effort": "best-effort"}.get(detect_tempo, "best-effort")
    try:
        midi_bytes = m.transcribe_to_midi((wav, sr), instruments=instrument_names, detect_tempo=tempo_mode)
    except BeatDetectionError as e:
        raise ValueError(f"Beat detection error: {e}")
    output_path = audio_path.with_suffix(".mid")
    output_path.write_bytes(midi_bytes)
    return FileData(path=str(output_path))


@app.get("/health")
async def health():
    return {"status": "ok", "model_loaded": model is not None}


@app.get("/instruments")
async def list_instruments():
    return {"instruments": list(MT3_FULL_PLUS_GROUP_NAMES.keys())}


@app.get("/", response_class=HTMLResponse)
async def homepage():
    html_path = WEB_DIR / "index.html"
    with open(html_path, "r", encoding="utf-8") as f:
        return f.read()


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="MuScriptor WebUI Server")
    parser.add_argument("--model", default="medium")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--dtype", default=None)
    parser.add_argument("--port", type=int, default=7860)
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--share", action="store_true")
    parser.add_argument("--clear-cache", action="store_true", dest="clear_cache")
    args = parser.parse_args()
    if getattr(args, "clear_cache", False):
        print("Clearing model cache...")
        clear_corrupted_cache()
    try:
        print(f"Loading MuScriptor model ({args.model})...")
        load_model(args.model, args.device, args.dtype)
        print("Model loaded successfully!")
    except Exception as e:
        print(f"Warning: Could not preload model: {e}")
    print(f"
🎵 MuScriptor WebUI starting on http://{args.host}:{args.port}")
    app.launch(server_name=args.host, server_port=args.port, share=getattr(args, "share", False), show_error=True)
