import pytest
from app.services.recommendation_service import RecommendationService

@pytest.fixture
def sample_ratings():
    return [
        {"user_id": "u1", "item_id": "i1", "rating": 5.0},
        {"user_id": "u1", "item_id": "i2", "rating": 4.0},
        {"user_id": "u2", "item_id": "i1", "rating": 5.0},
        {"user_id": "u2", "item_id": "i3", "rating": 5.0},
        {"user_id": "u3", "item_id": "i2", "rating": 2.0},
    ]

@pytest.fixture
def sample_features():
    return {
        "i1": [1.0, 0.0, 0.5],
        "i2": [0.8, 0.2, 0.1],
        "i3": [0.0, 1.0, 0.9],
    }

def test_collaborative_recommendations(sample_ratings):
    res = RecommendationService.get_recommendations(
        ratings_data=sample_ratings,
        target_user_id="u1",
        strategy="collaborative",
        top_k=2
    )
    assert res["target_user_id"] == "u1"
    assert res["strategy_used"] == "collaborative"
    assert len(res["recommendations"]) <= 2
    # u1 hasn't rated i3, and u2 (who is similar to u1) rated i3 high
    item_ids = [rec["item_id"] for rec in res["recommendations"]]
    assert "i3" in item_ids

def test_content_based_recommendations(sample_ratings, sample_features):
    res = RecommendationService.get_recommendations(
        ratings_data=sample_ratings,
        target_user_id="u1",
        strategy="content_based",
        item_features=sample_features,
        top_k=1
    )
    assert res["strategy_used"] == "content_based"
    assert len(res["recommendations"]) == 1

def test_cold_start_user(sample_ratings):
    res = RecommendationService.get_recommendations(
        ratings_data=sample_ratings,
        target_user_id="u_new",
        strategy="collaborative",
        top_k=2
    )
    assert len(res["recommendations"]) > 0
    assert "Cold-start" in res["recommendations"][0]["explanation"]
