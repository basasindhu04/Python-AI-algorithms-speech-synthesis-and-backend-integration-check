from fastapi import APIRouter, HTTPException, status
from app.schemas import GridSearchRequest, GraphSearchRequest, SearchResponse
from app.services.search_service import SearchService

router = APIRouter(prefix="/search", tags=["Heuristic Search & Graphs"])

@router.post("/grid-a-star", response_model=SearchResponse, status_code=status.HTTP_200_OK)
def run_grid_a_star(request: GridSearchRequest):
    """
    Execute A* pathfinding algorithm on a 2D grid matrix.
    """
    try:
        result = SearchService.grid_a_star(
            start=request.start,
            goal=request.goal,
            grid=request.grid,
            heuristic_type=request.heuristic,
            allow_diagonal=request.allow_diagonal
        )
        return SearchResponse(**result)
    except ValueError as ve:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(ve))
    except Exception as e:
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=f"A* Search failed: {str(e)}")

@router.post("/graph-a-star", response_model=SearchResponse, status_code=status.HTTP_200_OK)
def run_graph_a_star(request: GraphSearchRequest):
    """
    Execute A* search on a weighted graph with custom node heuristic map.
    """
    try:
        edges_data = [edge.model_dump() for edge in request.edges]
        result = SearchService.graph_a_star(
            start=request.start,
            goal=request.goal,
            edges=edges_data,
            heuristics=request.heuristics
        )
        return SearchResponse(**result)
    except ValueError as ve:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(ve))
    except Exception as e:
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=f"Graph Search failed: {str(e)}")
