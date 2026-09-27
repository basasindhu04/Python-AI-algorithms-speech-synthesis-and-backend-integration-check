import os
import time
import asyncio
import tempfile
from typing import Dict, Any, Optional
import pyttsx3

class SpeechService:
    @classmethod
    def synthesize_speech(
        cls,
        text: str,
        rate: int = 150,
        volume: float = 1.0,
        save_file: bool = False
    ) -> Dict[str, Any]:
        start_time = time.perf_counter()
        output_file_path: Optional[str] = None
        engine_backend = "pyttsx3_local"
        
        words = len(text.split())
        estimated_duration = round(max(0.5, (words / (rate / 60.0))), 2)

        try:
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
                engine.say(text)
                engine.runAndWait()

        except Exception as e:
            engine_backend = f"pyttsx3_fallback_metadata (error: {str(e)})"

        end_time = time.perf_counter()
        return {
            "text": text,
            "duration_seconds_estimate": estimated_duration,
            "output_file_path": output_file_path,
            "engine_backend": engine_backend,
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
        return await asyncio.to_thread(
            cls.synthesize_speech,
            text=text,
            rate=rate,
            volume=volume,
            save_file=save_file
        )
