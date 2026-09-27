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

    @classmethod
    def grid_a_star(
        cls,
        start: Tuple[int, int],
        goal: Tuple[int, int],
        grid: List[List[int]],
        heuristic_type: str = "manhattan",
        allow_diagonal: bool = False
    ) -> Dict[str, Any]:
        start_time = time.perf_counter()
        rows = len(grid)
        if rows == 0:
            raise ValueError("Grid cannot be empty")
        cols = len(grid[0])
        
        if not (0 <= start[0] < rows and 0 <= start[1] < cols):
            raise ValueError(f"Start node {start} out of grid bounds ({rows}x{cols})")
        if not (0 <= goal[0] < rows and 0 <= goal[1] < cols):
            raise ValueError(f"Goal node {goal} out of grid bounds ({rows}x{cols})")
        if grid[start[0]][start[1]] == 1 or grid[goal[0]][goal[1]] == 1:
            return {
                "path": [],
                "cost": 0.0,
                "nodes_expanded": 0,
                "execution_time_ms": (time.perf_counter() - start_time) * 1000,
                "heuristic_used": heuristic_type,
                "path_found": False
            }

        directions = [(0, 1, 1.0), (1, 0, 1.0), (0, -1, 1.0), (-1, 0, 1.0)]
        if allow_diagonal:
            diag_cost = math.sqrt(2.0)
            directions.extend([
                (1, 1, diag_cost), (1, -1, diag_cost),
                (-1, 1, diag_cost), (-1, -1, diag_cost)
            ])

        open_set: List[Tuple[float, float, Tuple[int, int]]] = []
        h_start = cls.calculate_heuristic(start, goal, heuristic_type)
        heapq.heappush(open_set, (h_start, 0.0, start))

        g_score: Dict[Tuple[int, int], float] = {start: 0.0}
        came_from: Dict[Tuple[int, int], Tuple[int, int]] = {}
        visited = set()
        nodes_expanded = 0
        path_found = False

        while open_set:
            f, current_g, current = heapq.heappop(open_set)

            if current in visited:
                continue

            visited.add(current)
            nodes_expanded += 1

            if current == goal:
                path_found = True
                break

            for dr, dc, step_cost in directions:
                neighbor = (current[0] + dr, current[1] + dc)

                if 0 <= neighbor[0] < rows and 0 <= neighbor[1] < cols:
                    if grid[neighbor[0]][neighbor[1]] == 1:
                        continue

                    tentative_g = current_g + step_cost
                    if neighbor not in g_score or tentative_g < g_score[neighbor]:
                        g_score[neighbor] = tentative_g
                        came_from[neighbor] = current
                        h = cls.calculate_heuristic(neighbor, goal, heuristic_type)
                        heapq.heappush(open_set, (tentative_g + h, tentative_g, neighbor))

        path = []
        final_cost = 0.0
        if path_found:
            curr = goal
            while curr in came_from:
                path.append(curr)
                curr = came_from[curr]
            path.append(start)
            path.reverse()
            final_cost = g_score[goal]

        end_time = time.perf_counter()
        return {
            "path": path,
            "cost": final_cost,
            "nodes_expanded": nodes_expanded,
            "execution_time_ms": (end_time - start_time) * 1000,
            "heuristic_used": heuristic_type,
            "path_found": path_found
        }
