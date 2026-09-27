import heapq
import math
import time
from typing import List, Tuple, Dict, Any, Optional
import networkx as nx

class SearchService:
    @staticmethod
    def calculate_heuristic(
        node: Tuple[int, int], 
        goal: Tuple[int, int], 
        heuristic_type: str = "manhattan"
    ) -> float:
        dr = abs(node[0] - goal[0])
        dc = abs(node[1] - goal[1])
        
        if heuristic_type == "manhattan":
            return float(dr + dc)
        elif heuristic_type == "euclidean":
            return math.sqrt(dr * dr + dc * dc)
        elif heuristic_type == "chebyshev":
            return float(max(dr, dc))
        else:
            raise ValueError(f"Unsupported heuristic type: {heuristic_type}")
