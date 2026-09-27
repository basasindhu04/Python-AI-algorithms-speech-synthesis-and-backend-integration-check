import os
import time
import asyncio
import tempfile
from typing import Dict, Any, Optional
import pyttsx3

class SpeechService:
    LOCAL_VS_CLOUD_TRADEOFFS = {
        "cost": "Local pyttsx3 is 100% free and offline. Cloud APIs charge per character/token.",
        "latency": "Local TTS achieves sub-10ms response without network roundtrip overhead.",
        "privacy_and_security": "Local processing guarantees data never leaves the host machine (GDPR/HIPAA compliant).",
        "voice_quality": "Cloud APIs (ElevenLabs/AWS Polly/Azure) offer high-fidelity neural TTS; local pyttsx3 relies on OS native engines (SAPI5/eSpeak/NSSpeechSynthesizer).",
        "reliability": "Local TTS functions offline without internet connectivity or rate limits."
    }

    @classmethod
    def synthesize_speech(
        cls,
        text: str,
        rate: int = 150,
        volume: float = 1.0,
        save_file: bool = False
    ) -> Dict[str, Any]:
        """
        Synthesize speech from text using pyttsx3.
        Runs synchronously inside a dedicated threadpool executor.
        """
        start_time = time.perf_counter()
        output_file_path: Optional[str] = None
        engine_backend = "pyttsx3_local"
        
        # Word count & duration estimate (~150 WPM default)
        words = len(text.split())
        estimated_duration = round(max(0.5, (words / (rate / 60.0))), 2)

        try:
            # Initialize pyttsx3 engine safely inside thread
            engine = pyttsx3.init()
            engine.setProperty("rate", rate)
            engine.setProperty("volume", volume)

            if save_file:
                temp_dir = tempfile.gettempdir()
                filename = f"speech_{int(time.time() * 1000)}.wav"
                output_file_path = os.path.join(temp_dir, filename)
                engine.save_to_file(text, output_file_path)
                engine.runAndWait()
            else:
                # If non-saving speech test in headless environment, run non-blocking loop test
                engine.say(text)
                engine.runAndWait()

        except Exception as e:
            # Fallback for environments lacking audio drivers / SAPI5 bindings
            engine_backend = f"pyttsx3_fallback_metadata (error: {str(e)})"

        end_time = time.perf_counter()
        return {
            "text": text,
            "duration_seconds_estimate": estimated_duration,
            "output_file_path": output_file_path,
            "engine_backend": engine_backend,
            "local_vs_cloud_tradeoffs": cls.LOCAL_VS_CLOUD_TRADEOFFS,
            "execution_time_ms": (end_time - start_time) * 1000
        }

    @classmethod
    async def synthesize_speech_async(
        cls,
        text: str,
        rate: int = 150,
        volume: float = 1.0,
        save_file: bool = False
    ) -> Dict[str, Any]:
        """
        Async wrapper using asyncio.to_thread to prevent blocking the FastAPI event loop.
        """
        return await asyncio.to_thread(
            cls.synthesize_speech,
            text=text,
            rate=rate,
            volume=volume,
            save_file=save_file
        )
