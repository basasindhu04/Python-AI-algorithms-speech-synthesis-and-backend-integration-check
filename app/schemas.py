from typing import List, Dict, Any, Optional, Tuple, Literal
from pydantic import BaseModel, Field, ConfigDict

# --- Heuristic Search & Graph Schemas ---

class GridSearchRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")
    
    start: Tuple[int, int] = Field(..., description="(row, col) start coordinate")
    goal: Tuple[int, int] = Field(..., description="(row, col) goal coordinate")
    grid: List[List[int]] = Field(
        ..., 
        description="2D binary grid where 0 is walkable and 1 is an obstacle"
    )
    heuristic: Literal["euclidean", "manhattan", "chebyshev"] = Field(
        default="manhattan", 
        description="Heuristic function type for A* search"
    )
    allow_diagonal: bool = Field(
        default=False, 
        description="Whether diagonal movement is allowed in grid search"
    )

class GraphEdge(BaseModel):
    source: str
    target: str
    weight: float = Field(default=1.0, ge=0.0)

class GraphSearchRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")
    
    start: str
    goal: str
    edges: List[GraphEdge]
    heuristics: Optional[Dict[str, float]] = Field(
        default=None, 
        description="Precomputed heuristic h(n) values to goal for each node"
    )

class SearchResponse(BaseModel):
    path: List[Any]
    cost: float
    nodes_expanded: int
    execution_time_ms: float
    heuristic_used: str
    path_found: bool

# --- Recommendation System Schemas ---

class UserItemRating(BaseModel):
    user_id: str
    item_id: str
    rating: float = Field(..., ge=0.0, le=5.0)

class RecommendationRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")
    
    ratings: List[UserItemRating]
    target_user_id: str
    strategy: Literal["collaborative", "content_based", "hybrid"] = Field(
        default="collaborative",
        description="Recommendation strategy algorithm"
    )
    item_features: Optional[Dict[str, List[float]]] = Field(
        default=None,
        description="Feature vectors for items (used in content-based / hybrid strategies)"
    )
    top_k: int = Field(default=5, ge=1, le=50)

class RecommendedItem(BaseModel):
    item_id: str
    predicted_score: float
    explanation: str

class RecommendationResponse(BaseModel):
    target_user_id: str
    strategy_used: str
    recommendations: List[RecommendedItem]
    execution_time_ms: float

# --- NLP & Embedding Schemas ---

class TextDocument(BaseModel):
    doc_id: str
    text: str

class TextSimilarityRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")
    
    query: str
    documents: List[TextDocument]
    method: Literal["tfidf", "vector_average"] = Field(
        default="tfidf",
        description="NLP similarity algorithm"
    )
    top_k: int = Field(default=5, ge=1, le=50)

class DocumentSimilarityScore(BaseModel):
    doc_id: str
    text: str
    similarity_score: float

class TextSimilarityResponse(BaseModel):
    query: str
    method_used: str
    results: List[DocumentSimilarityScore]
    execution_time_ms: float

# --- Speech Synthesis Schemas ---

class SpeechSynthesisRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")
    
    text: str = Field(..., min_length=1, max_length=2000, description="Text to synthesize")
    rate: int = Field(default=150, ge=50, le=300, description="Speech rate (words per minute)")
    volume: float = Field(default=1.0, ge=0.0, le=1.0, description="Volume level between 0.0 and 1.0")
    save_file: bool = Field(default=False, description="Save audio output to WAV file")

class SpeechSynthesisResponse(BaseModel):
    text: str
    duration_seconds_estimate: float
    output_file_path: Optional[str] = None
    engine_backend: str
    local_vs_cloud_tradeoffs: Dict[str, str]
    execution_time_ms: float
