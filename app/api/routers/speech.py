from fastapi import APIRouter, HTTPException, status
from app.schemas import SpeechSynthesisRequest, SpeechSynthesisResponse
from app.services.speech_service import SpeechService

router = APIRouter(prefix="/speech", tags=["Speech Synthesis"])

@router.post("/synthesize", response_model=SpeechSynthesisResponse, status_code=status.HTTP_200_OK)
async def synthesize_speech_endpoint(request: SpeechSynthesisRequest):
    """
    Synthesize text to speech locally via pyttsx3 in an async non-blocking thread execution model.
    """
    try:
        result = await SpeechService.synthesize_speech_async(
            text=request.text,
            rate=request.rate,
            volume=request.volume,
            save_file=request.save_file
        )
        return SpeechSynthesisResponse(**result)
    except ValueError as ve:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(ve))
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Speech synthesis error: {str(e)}"
        )
