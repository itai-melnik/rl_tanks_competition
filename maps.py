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


# Phase 4: ALL MAPS ENABLED ✓
MAP_POOL = {
    "empty": MAP_EMPTY,
    "cross": MAP_CROSS,
    "pillars": MAP_FOUR_PILLARS,
    "diagonal": MAP_DIAGONAL,
}

def get_random_map(rng):
    """Selects a random map from the pool."""
    name = rng.choice(list(MAP_POOL.keys()))
    return MAP_POOL[name], name


