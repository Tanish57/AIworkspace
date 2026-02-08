import os
import io
import json
import base64
import requests
import numpy as np
import scipy.io.wavfile as wav
from pathlib import Path
from kokoro_onnx import Kokoro

class TTSEngine:
    def __init__(self, data_dir: str = "./data"):
        self.data_dir = Path(data_dir) / "kokoro"
        self.data_dir.mkdir(parents=True, exist_ok=True)
        
        self.model_path = self.data_dir / "kokoro-v0_19.onnx"
        self.voices_path = self.data_dir / "voices.json"
        
        self._ensure_models()
        
        print(f"[TTS] Loading Kokoro model from {self.model_path}...")
        self.kokoro = Kokoro(str(self.model_path), str(self.voices_path))
        print("[TTS] Model loaded successfully.")

    def _ensure_models(self):
        """Download model files if they don't exist."""
        # URLs for Kokoro v0.19 ONNX
        MODEL_URL = "https://github.com/thewh1teagle/kokoro-onnx/releases/download/model-files/kokoro-v0_19.onnx"
        VOICES_URL = "https://github.com/thewh1teagle/kokoro-onnx/releases/download/model-files/voices.bin"
        
        if not self.model_path.exists():
            print(f"[TTS] Downloading model to {self.model_path}...")
            self._download_file(MODEL_URL, self.model_path)
            
        # Update path for voices (use .bin usually required by kokoro-onnx < 0.5? or 0.5?)
        # kokoro-onnx 0.5.0 uses np.load, so it likely expects the bin/npy file.
        # The release has voices.bin.
        self.voices_path = self.data_dir / "voices.bin"
        
        if not self.voices_path.exists():
            print(f"[TTS] Downloading voices to {self.voices_path}...")
            self._download_file(VOICES_URL, self.voices_path)

    def _download_file(self, url: str, dest: Path):
        print(f"[TTS] Downloading {url} ...")
        response = requests.get(url, stream=True)
        response.raise_for_status()
        with open(dest, "wb") as f:
            for chunk in response.iter_content(chunk_size=8192):
                f.write(chunk)
        print(f"[TTS] Downloaded {dest.name}")

    def generate_audio(self, text: str, voice: str = "af_sarah") -> str:
        """
        Generate audio from text and return as base64 encoded WAV string.
        """
        try:
            # Generate audio (returns numpy array and sample rate)
            # Support splitting long text if needed, but Kokoro handles some max length.
            # Ideally verify text length.
            
            samples, sample_rate = self.kokoro.create(
                text, 
                voice=voice, 
                speed=1.0, 
                lang="en-us"
            )
            
            # Convert to WAV in-memory
            byte_io = io.BytesIO()
            wav.write(byte_io, sample_rate, samples)
            wav_bytes = byte_io.getvalue()
            
            # Encode to base64
            base64_audio = base64.b64encode(wav_bytes).decode("utf-8")
            return base64_audio
            
        except Exception as e:
            print(f"[TTS] Error generating audio: {e}")
            return None

    def get_available_voices(self):
        # voices.json structure: { voice_name: [embedding] }
        if self.voices_path.exists():
            with open(self.voices_path, "r") as f:
                data = json.load(f)
                return list(data.keys())
        return []
