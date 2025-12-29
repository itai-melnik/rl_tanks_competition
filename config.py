import torch

# Random Seed (for reproducibility)
SEED = 42

# Training Hyperparameters
NUM_EPISODES = 1000    # Total episodes to train
MAX_STEPS_PER_EPISODE = 100  # Truncate episode after this many steps

LEARNING_RATE = 1e-3
GAMMA = 0.99                 # Discount factor
BATCH_SIZE = 64
BUFFER_SIZE = 10000          # Replay buffer size
MIN_BUFFER_SIZE = 1000       # Minimum buffer size before training starts

# Epsilon Greedy Schedule
EPS_START = 1.0
EPS_END = 0.05
EPS_DECAY = 0.9999           # Decay rate per episode (multiplicative)
# Or linear decay steps:
EPS_DECAY_STEPS = 1000       # Decay linearly over this many episodes (alternative)

# DQN Specifics
TARGET_UPDATE_FREQ = 100     # Update target network every N episodes (or steps)
TRAIN_FREQ = 1               # Train every N steps (or episodes)

# Visualization (pygame)
# When True, train.py and evaluate.py will render the environment using pygame.
# Default is False for fast, headless training/evaluation; enable manually
# when you want to inspect behavior visually.
USE_PYGAME_RENDER = True

# Checkpointing
LOAD_CHECKPOINT = True  # Set to True to resume training from agent.pt
CHECKPOINT_DIR = "checkpoints"
MODEL_SAVE_PATH = "checkpoints/agent.pt"
MODEL_CONFIG_PATH = "checkpoints/model_config.json"

# Device
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
if torch.backends.mps.is_available():
    DEVICE = "mps"


