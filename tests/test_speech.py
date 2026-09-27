import asyncio
import pytest
from app.services.speech_service import SpeechService

def test_speech_synthesis_metadata():
    text = "Testing local pyttsx3 speech synthesis engine."
    result = SpeechService.synthesize_speech(text, rate=150, save_file=False)
    
    assert result["text"] == text
    assert result["duration_seconds_estimate"] > 0.0
    assert "cost" in result["local_vs_cloud_tradeoffs"]
    assert "latency" in result["local_vs_cloud_tradeoffs"]

def test_speech_synthesis_async():
    text = "Async speech synthesis invocation test."
    result = asyncio.run(SpeechService.synthesize_speech_async(text, rate=160, save_file=False))
    
    assert result["text"] == text
    assert result["execution_time_ms"] >= 0.0

