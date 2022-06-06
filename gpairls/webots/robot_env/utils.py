"""
Utility functions
"""

import numpy as np


def compute_shortest_path_bfs(occupancy_grid, start, goal):
    """
    Compute the shortest path from start to goal cell. The occupancy
    grid is a 2D array where 0 means the cell is free and 1 means it is occupied.
    """

    # check if start and goal are the same
    if np.array_equal(start, goal):
        return [start]

    # create a queue for BFS
    queue = []
    queue.append(start)

    # keep track of visited nodes
    visited = []
    visited.append(start)

    # keep track of parent of each node
    parents = {}
    parents[start] = None

    while queue:
        # pop the first node from the queue
        node = queue.pop(0)

        # if the node is the goal, then return the path
        if np.array_equal(node, goal):
            path = []
            node = goal
            while node is not None:
                path.append(node)
                node = parents[node]
            return path[::-1]

        # get the neighbors of the node (8-connected)
        neighbors = [
            (node[0] + dx, node[1] + dy)
            for dx, dy in [
                (-1, -1),
                (0, -1),
                (-1, 0),
                (1, 0),
                (0, 1),
                (-1, 1),
                (1, 1),
                (1, -1),
            ]
        ]
        neighbors = filter(
            lambda n: 0 <= n[0] < occupancy_grid.shape[0]
            and 0 <= n[1] < occupancy_grid.shape[1]
            and occupancy_grid[n[0], n[1]] == 0,
            neighbors,
        )

        # add neighbors to the queue and mark as visited
        for neighbor in neighbors:
            if neighbor not in visited:
                visited.append(neighbor)
                parents[neighbor] = node
                queue.append(neighbor)

    # no path found
    return []


def compute_shortest_path_astar(occupancy_grid, start, goal):
    """
    Compute the shortest path from start to goal cell using A*. The occupancy
    grid is a 2D array where 0 means the cell is free and 1 means it is occupied.
    """

    def heuristic(a, b):
        """
        Compute the heuristic value H(n) using the Euclidean distance.
        """
        return np.linalg.norm(np.array(a) - np.array(b))

    # check if start and goal are the same
    if np.array_equal(start, goal):
        return [start]

    # create a queue for BFS
    queue = []
    queue.append(start)

    # keep track of visited nodes
    visited = []
    visited.append(start)

    # keep track of parent of each node
    parents = {}
    parents[start] = None

    # keep track of the cost of getting to each node
    costs = {}
    costs[start] = 0

    # keep track of the heuristic value of each node
    heuristics = {}
    heuristics[start] = heuristic(start, goal)

    while queue:
        # pop the first node from the queue
        node = queue.pop(0)

        # if the node is the goal, then return the path
        if np.array_equal(node, goal):
            path = []
            node = goal
            while node is not None:
                path.append(node)
                node = parents[node]
            return path[::-1]

        # get the neighbors of the node (8-connected)
        neighbors = [
            (node[0] + dx, node[1] + dy)
            for dx, dy in [
                (-1, -1),
                (0, -1),
                (-1, 0),
                (1, 0),
                (0, 1),
                (-1, 1),
                (1, 1),
                (1, -1),
            ]
        ]
        neighbors = filter(
            lambda n: 0 <= n[0] < occupancy_grid.shape[0]
            and 0 <= n[1] < occupancy_grid.shape[1]
            and occupancy_grid[n[0], n[1]] == 0,
            neighbors,
        )

        # add neighbors to the queue and mark as visited
        for neighbor in neighbors:
            if neighbor not in visited:
                visited.append(neighbor)
                parents[neighbor] = node
                costs[neighbor] = costs[node] + 1
                heuristics[neighbor] = heuristic(neighbor, goal)
                queue.append(neighbor)

    # no path found
    return []
