"""
Utility functions
"""

from typing import List, Tuple

import numpy as np
from scipy.signal import convolve2d


# fmt: off
# diffs for 8-connectivity
CONN_8 = [
    (-1, -1), (0, -1), (1, -1),
    (-1,  0),          (1,  0),
    (-1,  1), (0,  1), (1,  1),
]
# fmt: on

# diffs for 4-connectivity
CONN_4 = list(filter(lambda n: 0 in n, CONN_8))


def get_neighbors(
    node: Tuple[int, int],
    max_sizes: Tuple[int, int],
    connectivity: List[Tuple[int, int]] = CONN_8,
) -> List[Tuple[int, int]]:
    neighbors = [(node[0] + dx, node[1] + dy) for dx, dy in connectivity]
    neighbors = filter(
        lambda n: 0 <= n[0] < max_sizes[0] and 0 <= n[1] < max_sizes[1],
        neighbors,
    )
    return list(neighbors)


def compute_advice_grid(occ_grid, goal_cell) -> np.ndarray:
    """
    Use BFS to compute advice. Direction is represented as the angle
    (0 degrees is "down", in radians)
    """
    pf = compute_potential_field(occ_grid, goal_cell)
    advice_grid = np.empty_like(pf)

    # number of steps to look ahead
    N_STEPS = 30

    for x in range(pf.shape[0]):
        for y in range(pf.shape[1]):

            if occ_grid[x, y] == 1:
                advice_grid[x, y] = 0
                continue

            cur_pos = np.array([x, y], dtype=np.float32)
            diffs = []
            pos = (x, y)
            for i in range(N_STEPS):
                neighbors = get_neighbors(pos, pf.shape)
                pos = neighbors[np.argmin([pf[n] for n in neighbors])]
                diff = cur_pos - np.array(pos)
                diff /= np.linalg.norm(diff) + 1e-6
                diffs.append(diff)

            # weighted average
            weight = np.expand_dims(np.linspace(1.0, 0.5, N_STEPS), -1)
            avg_diff = (weight * np.array(diffs)).sum(0) / weight.sum()

            # convert to angle (rad)
            angle = np.arctan2(avg_diff[1], avg_diff[0])
            advice_grid[x, y] = angle

    return advice_grid


def compute_potential_field(occ_grid, goal_cell) -> np.ndarray:
    """Use BFS to compute potential field (higher = worse)"""

    pf = np.full_like(occ_grid, np.inf, dtype=np.float32)

    cur_potential = 0
    queue = set([goal_cell])
    next_queue = set()
    visited = set()

    while queue:
        for node in queue:
            pf[node] = min(pf[node], cur_potential)
            # get the neighbors of the node (8-connected)
            neighbors = get_neighbors(node, pf.shape, CONN_8)
            neighbors = filter(
                lambda n: n not in queue and n not in visited and occ_grid[n] != 1,
                neighbors,
            )
            next_queue.update(neighbors)
            visited.add(node)

        queue, next_queue = next_queue, queue
        next_queue.clear()
        cur_potential += 1

    # smooth (also repels around obstacles due to inf)
    SMOOTH_SIZE = 13
    max_pf_val = pf[~np.isinf(pf)].max()
    pf_smoothed = convolve2d(
        np.nan_to_num(pf, posinf=max_pf_val),
        np.full((SMOOTH_SIZE, SMOOTH_SIZE), 1.0 / SMOOTH_SIZE**2),
        mode="same",
        boundary="fill",
        fillvalue=max_pf_val * 3,
    )
    pf[~np.isinf(pf)] = pf_smoothed[~np.isinf(pf)]

    # # add potential around obstacles
    # OBSTACLE_MAX_DIST = 10
    # obs_queue = set([tuple(n) for n in np.argwhere(occ_grid == 1)])
    # next_obs_queue = set()
    # cur_dist = 0

    # while obs_queue and cur_dist <= OBSTACLE_MAX_DIST:
    #     for node in obs_queue:
    #         pf[node] += 1 * (OBSTACLE_MAX_DIST - cur_dist)
    #         # get the neighbors of the node (8-connected)
    #         neighbors = get_neighbors(node, pf.shape, CONN_8)
    #         neighbors = filter(
    #             lambda n: n not in obs_queue and occ_grid[n] != 1,
    #             neighbors,
    #         )
    #         next_obs_queue.update(neighbors)

    #     obs_queue, next_obs_queue = next_obs_queue, obs_queue
    #     next_obs_queue.clear()
    #     cur_dist += 1

    return pf
