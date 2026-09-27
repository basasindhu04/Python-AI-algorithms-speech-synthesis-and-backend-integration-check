import pytest
from app.services.nlp_service import NLPService

@pytest.fixture
def sample_docs():
    return [
        {"doc_id": "doc1", "text": "Artificial intelligence and machine learning engineering"},
        {"doc_id": "doc2", "text": "Heuristic search algorithms A* pathfinding and graphs"},
        {"doc_id": "doc3", "text": "Speech synthesis local text to speech audio generation"},
    ]

def test_gensim_tfidf_similarity(sample_docs):
    query = "machine learning AI"
    res = NLPService.analyze_similarity(query, sample_docs, method="tfidf", top_k=2)
    
    assert res["query"] == query
    assert res["method_used"] == "tfidf"
    assert len(res["results"]) == 2
    # doc1 should rank top for AI/ML query
    assert res["results"][0]["doc_id"] == "doc1"
    assert res["results"][0]["similarity_score"] > 0.0

def test_vector_average_similarity(sample_docs):
    query = "speech synthesis audio"
    res = NLPService.analyze_similarity(query, sample_docs, method="vector_average", top_k=1)
    
    assert res["method_used"] == "vector_average"
    assert len(res["results"]) == 1
    assert res["results"][0]["doc_id"] == "doc3"
