from fastapi import APIRouter, HTTPException, status
from app.schemas import RecommendationRequest, RecommendationResponse
from app.services.recommendation_service import RecommendationService

router = APIRouter(prefix="/recommendations", tags=["Recommendation Systems"])

@router.post("", response_model=RecommendationResponse, status_code=status.HTTP_200_OK)
def get_recommendations(request: RecommendationRequest):
    """
    Generate personalized item recommendations using Collaborative, Content-Based, or Hybrid filtering.
    """
    try:
        ratings_data = [r.model_dump() for r in request.ratings]
        result = RecommendationService.get_recommendations(
            ratings_data=ratings_data,
            target_user_id=request.target_user_id,
            strategy=request.strategy,
            item_features=request.item_features,
            top_k=request.top_k
        )
        return RecommendationResponse(**result)
    except ValueError as ve:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(ve))
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Recommendation engine error: {str(e)}"
        )
