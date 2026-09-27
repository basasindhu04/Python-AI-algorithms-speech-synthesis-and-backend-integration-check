import pytest
from app.services.search_service import SearchService

def test_grid_a_star_simple_path():
    grid = [
        [0, 0, 0],
        [0, 1, 0],
        [0, 0, 0]
    ]
    start = (0, 0)
    goal = (2, 2)
    
    result = SearchService.grid_a_star(start, goal, grid, heuristic_type="manhattan")
    assert result["path_found"] is True
    assert result["path"][0] == start
    assert result["path"][-1] == goal
    assert result["cost"] == 4.0

def test_grid_a_star_blocked_goal():
    grid = [
        [0, 0, 1],
        [0, 1, 1],
        [0, 0, 1]
    ]
    start = (0, 0)
    goal = (0, 2)
    
    result = SearchService.grid_a_star(start, goal, grid)
    assert result["path_found"] is False
    assert result["path"] == []

def test_graph_a_star():
    edges = [
        {"source": "A", "target": "B", "weight": 2.0},
        {"source": "B", "target": "C", "weight": 3.0},
        {"source": "A", "target": "C", "weight": 10.0},
    ]
    heuristics = {"A": 4.0, "B": 2.0, "C": 0.0}
    
    result = SearchService.graph_a_star(start="A", goal="C", edges=edges, heuristics=heuristics)
    assert result["path_found"] is True
    assert result["path"] == ["A", "B", "C"]
    assert result["cost"] == 5.0
