# Action Space
ACTION_MOVE_FORWARD = 0
ACTION_MOVE_BACKWARD = 1
ACTION_TURN_LEFT = 2
ACTION_TURN_RIGHT = 3
ACTION_SHOOT = 4
ACTION_DO_NOTHING = 5
ACTION_SHIELD = 6

ACTION_NAMES = [
    "MOVE_FORWARD",
    "MOVE_BACKWARD",
    "TURN_LEFT",
    "TURN_RIGHT",
    "SHOOT",
    "DO_NOTHING",
    "SHIELD",
]

# Grid
GRID_SIZE = 15

# Tank stats / game tuning
MAX_HP = 100

# Damage
DAMAGE_PER_HIT = 10

# Gun cooldown (in game steps)
GUN_COOLDOWN_STEPS = 3

# Minimum spawn distance between tanks (Manhattan)
MIN_SPAWN_DISTANCE = 4

# Ammo & reload
MAX_AMMO = 6          # shots before needing a reload
RELOAD_STEPS = 8      # steps to fully reload from 0 ammo

# Shield ability
SHIELD_DURATION_STEPS = 5   # how many steps shield stays active
SHIELD_COOLDOWN_STEPS = 20  # cooldown before shield can be used again

# Anti-camping: how many consecutive steps on same tile before we start penalizing
STAY_STEPS_THRESHOLD = 6

# Orientation
DIR_UP = 0
DIR_RIGHT = 1
DIR_DOWN = 2
DIR_LEFT = 3

"""
Direction vectors for movement.

We represent tank positions as (x, y) where:
  - x is the column index (0 .. GRID_SIZE-1), increasing to the right
  - y is the row index    (0 .. GRID_SIZE-1), increasing downward

The environment code uses:
  - x movement from DX
  - y movement from DY
"""
DY = [-1, 0, 1, 0]  # change in y for [UP, RIGHT, DOWN, LEFT]
DX = [0, 1, 0, -1]  # change in x for [UP, RIGHT, DOWN, LEFT]


