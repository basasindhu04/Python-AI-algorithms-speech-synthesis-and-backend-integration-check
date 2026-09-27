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
