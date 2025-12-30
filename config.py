import torch
from bots import BaseBot, RandomBot, AggressiveBot, CamperBot, EvadeBot

# Random Seed (for reproducibility)
SEED = 42



# Training Hyperparameters - Phase 5b (Balanced Self-Play)
NUM_EPISODES = 1500          # Moderate - don't overtrain
MAX_STEPS_PER_EPISODE = 100  # Truncate episode after this many steps

LEARNING_RATE = 2e-4         # Conservative - preserve existing skills
GAMMA = 0.99                 # Discount factor
BATCH_SIZE = 128
BUFFER_SIZE = 100000         # Large buffer
MIN_BUFFER_SIZE = 2000       # Warmup before training

# Epsilon Greedy Schedule
EPS_START = 0.15             # Lower exploration - agent is already good
EPS_END = 0.05               # Low final epsilon
EPS_DECAY = 0.999            # Decay rate per episode (multiplicative)
# Or linear decay steps:
EPS_DECAY_STEPS = 1200       # Moderate decay

# DQN Specifics
TARGET_UPDATE_FREQ = 500     # More stable target updates
TRAIN_FREQ = 1               # Train every N steps (or episodes)

# Visualization (pygame)
# When True, train.py and evaluate.py will render the environment using pygame.
# Default is False for fast, headless training/evaluation; enable manually
# when you want to inspect behavior visually.
USE_PYGAME_RENDER = False

# Checkpointing
LOAD_CHECKPOINT = True  # Set to True to resume training from agent.pt
CHECKPOINT_DIR = "checkpoints"
MODEL_SAVE_PATH = "checkpoints/agent.pt"
MODEL_CONFIG_PATH = "checkpoints/model_config.json"

# Device
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
if torch.backends.mps.is_available():
    DEVICE = "mps"


