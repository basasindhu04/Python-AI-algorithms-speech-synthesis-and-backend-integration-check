from fastapi.testclient import TestClient
from app.main import app

client = TestClient(app)

def test_health_check():
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json() == {"status": "healthy"}

def test_root():
    response = client.get("/")
    assert response.status_code == 200
    assert response.json()["status"] == "online"

def test_api_grid_a_star():
    payload = {
        "start": [0, 0],
        "goal": [1, 2],
        "grid": [
            [0, 0, 0],
            [1, 0, 0]
        ],
        "heuristic": "manhattan",
        "allow_diagonal": False
    }
    response = client.post("/api/v1/search/grid-a-star", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert data["path_found"] is True
    assert len(data["path"]) > 0

def test_api_recommendations():
    payload = {
        "ratings": [
            {"user_id": "u1", "item_id": "i1", "rating": 5.0},
            {"user_id": "u2", "item_id": "i1", "rating": 4.0},
            {"user_id": "u2", "item_id": "i2", "rating": 5.0}
        ],
        "target_user_id": "u1",
        "strategy": "collaborative",
        "top_k": 5
    }
    response = client.post("/api/v1/recommendations", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert data["target_user_id"] == "u1"
    assert "recommendations" in data

def test_api_nlp_similarity():
    payload = {
        "query": "A* search algorithm",
        "documents": [
            {"doc_id": "d1", "text": "A* search algorithm for graphs and grids"},
            {"doc_id": "d2", "text": "Random text about cooking recipes"}
        ],
        "method": "tfidf",
        "top_k": 2
    }
    response = client.post("/api/v1/nlp/similarity", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert data["results"][0]["doc_id"] == "d1"

def test_api_speech_synthesis():
    payload = {
        "text": "Hello world, testing speech synthesis API endpoint.",
        "rate": 150,
        "volume": 1.0,
        "save_file": False
    }
    response = client.post("/api/v1/speech/synthesize", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert data["text"] == payload["text"]
    assert "engine_backend" in data
