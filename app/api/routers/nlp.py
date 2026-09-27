from fastapi import APIRouter, HTTPException, status
from app.schemas import TextSimilarityRequest, TextSimilarityResponse
from app.services.nlp_service import NLPService

router = APIRouter(prefix="/nlp", tags=["NLP & Vector Embeddings"])

@router.post("/similarity", response_model=TextSimilarityResponse, status_code=status.HTTP_200_OK)
def analyze_text_similarity(request: TextSimilarityRequest):
    """
    Compute document vector similarity scores against a search query using Gensim TF-IDF or Vector Average.
    """
    try:
        documents_data = [doc.model_dump() for doc in request.documents]
        result = NLPService.analyze_similarity(
            query=request.query,
            documents=documents_data,
            method=request.method,
            top_k=request.top_k
        )
        return TextSimilarityResponse(**result)
    except ValueError as ve:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(ve))
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"NLP service processing failure: {str(e)}"
        )
