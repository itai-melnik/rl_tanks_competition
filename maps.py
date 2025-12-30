import numpy as np
from constants import GRID_SIZE

# Map definitions
# 0: Empty
# 1: Wall

# Base grid
MAP_EMPTY = np.zeros((GRID_SIZE, GRID_SIZE), dtype=int)

# Add border walls
MAP_EMPTY[0, :] = 1
MAP_EMPTY[-1, :] = 1
MAP_EMPTY[:, 0] = 1
MAP_EMPTY[:, -1] = 1

# Helper indices relative to GRID_SIZE so maps stay well-formed when we
# change the arena size (e.g. 11x11, 15x15, etc.).
MID = GRID_SIZE // 2
INNER_START = 2
INNER_END = GRID_SIZE - 2  # exclusive upper bound for inner structures

MAP_CROSS = MAP_EMPTY.copy()
# Horizontal and vertical bars crossing near the center.
MAP_CROSS[MID, INNER_START:INNER_END] = 1
MAP_CROSS[INNER_START:INNER_END, MID] = 1
# Clear center for movement
MAP_CROSS[MID, MID] = 0

MAP_FOUR_PILLARS = MAP_EMPTY.copy()
# Place four pillars roughly around the center.
PILLAR_OFFSET = 2
MAP_FOUR_PILLARS[MID - PILLAR_OFFSET, MID - PILLAR_OFFSET] = 1
MAP_FOUR_PILLARS[MID - PILLAR_OFFSET, MID + PILLAR_OFFSET] = 1
MAP_FOUR_PILLARS[MID + PILLAR_OFFSET, MID - PILLAR_OFFSET] = 1
MAP_FOUR_PILLARS[MID + PILLAR_OFFSET, MID + PILLAR_OFFSET] = 1

MAP_DIAGONAL = MAP_EMPTY.copy()
for i in range(INNER_START, INNER_END):
    MAP_DIAGONAL[i, i] = 1
    MAP_DIAGONAL[i, GRID_SIZE - 1 - i] = 1
# Clear corners (of the inner diagonal region) and center
MAP_DIAGONAL[MID, MID] = 0
MAP_DIAGONAL[INNER_START, INNER_START] = 0
MAP_DIAGONAL[INNER_START, GRID_SIZE - 1 - INNER_START] = 0
MAP_DIAGONAL[GRID_SIZE - 1 - INNER_START, INNER_START] = 0
MAP_DIAGONAL[GRID_SIZE - 1 - INNER_START, GRID_SIZE - 1 - INNER_START] = 0


# ============================================================================
# UNSEEN MAPS (for evaluation only - not in training pool)
# ============================================================================

# Maze-like corridors
MAP_CORRIDORS = MAP_EMPTY.copy()
# Horizontal walls with gaps
MAP_CORRIDORS[4, 2:6] = 1
MAP_CORRIDORS[4, 9:13] = 1
MAP_CORRIDORS[10, 2:6] = 1
MAP_CORRIDORS[10, 9:13] = 1
# Vertical walls with gaps
MAP_CORRIDORS[2:6, 7] = 1
MAP_CORRIDORS[9:13, 7] = 1

# L-shaped walls
MAP_L_WALLS = MAP_EMPTY.copy()
# Top-left L
MAP_L_WALLS[3, 3:7] = 1
MAP_L_WALLS[3:7, 3] = 1
# Bottom-right L
MAP_L_WALLS[11, 8:12] = 1
MAP_L_WALLS[8:12, 11] = 1

# Scattered obstacles (random-looking but fixed)
MAP_SCATTERED = MAP_EMPTY.copy()
scatter_positions = [
    (3, 5), (3, 9), (5, 3), (5, 11),
    (7, 6), (7, 8),
    (9, 3), (9, 11), (11, 5), (11, 9)
]
for y, x in scatter_positions:
    if 1 <= y < GRID_SIZE-1 and 1 <= x < GRID_SIZE-1:
        MAP_SCATTERED[y, x] = 1

# Ring/donut shape
MAP_RING = MAP_EMPTY.copy()
# Outer ring walls (but with gaps)
for i in range(4, 11):
    MAP_RING[4, i] = 1
    MAP_RING[10, i] = 1
    MAP_RING[i, 4] = 1
    MAP_RING[i, 10] = 1
# Clear corners for movement
MAP_RING[4, 4] = 0
MAP_RING[4, 10] = 0
MAP_RING[10, 4] = 0
MAP_RING[10, 10] = 0
# Clear center
MAP_RING[7, 7] = 0

# ============================================================================
# MAP POOLS
# ============================================================================

# Training pool (what agent sees during training)
MAP_POOL = {
    "empty": MAP_EMPTY,
    "cross": MAP_CROSS,
    "pillars": MAP_FOUR_PILLARS,
    "diagonal": MAP_DIAGONAL,
}

# Unseen maps (for evaluation only)
MAP_POOL_UNSEEN = {
    "corridors": MAP_CORRIDORS,
    "l_walls": MAP_L_WALLS,
    "scattered": MAP_SCATTERED,
    "ring": MAP_RING,
}

def get_random_map(rng):
    """Selects a random map from the training pool."""
    name = rng.choice(list(MAP_POOL.keys()))
    return MAP_POOL[name], name

def get_random_unseen_map(rng):
    """Selects a random map from the unseen pool (for evaluation)."""
    name = rng.choice(list(MAP_POOL_UNSEEN.keys()))
    return MAP_POOL_UNSEEN[name], name


