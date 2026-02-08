import os
import base64
from tts_engine import TTSEngine

def test_tts():
    print("Initializing TTS Engine...")
    try:
        engine = TTSEngine()
    except Exception as e:
        print(f"FAILED to initialize engine: {e}")
        return

    text = "Hello, this is a test of Kokoro TTS."
    print(f"Generating audio for: '{text}'")
    
    try:
        audio_b64 = engine.generate_audio(text)
        
        if not audio_b64:
            print("FAILED: No audio generated (None returned).")
            return
            
        # Decode and save to check validity
        audio_bytes = base64.b64decode(audio_b64)
        print(f"Generated {len(audio_bytes)} bytes of audio.")
        
        with open("test_output.wav", "wb") as f:
            f.write(audio_bytes)
            
        print("SUCCESS: Audio generated and saved to test_output.wav")
        
    except Exception as e:
        print(f"FAILED during generation: {e}")

if __name__ == "__main__":
    test_tts()
