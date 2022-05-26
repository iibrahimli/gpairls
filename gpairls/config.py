"""
RL training config
"""

from pathlib import Path


# directory with training logs
LOG_DIR = Path(__file__).parent.parent.resolve() / "logs"

# directory with model weights
MODEL_DIR = LOG_DIR / "model"

# random seed for reproducibility
SEED = 42

# number of steps to train each agent
TRAINING_STEPS = 20_000

# number of steps to collect data before starting training
INIT_STEPS = 1000

# encoder type
ENCODER_TYPE = "cnn"

# dimensionality of feature vector from encoder
ENCODER_FEATURE_DIM = 4

# actor and critic hidden dimensions
HIDDEN_DIM = 16

REPLAY_BUFFER_CAPACITY = 10_000

BATCH_SIZE = 32

CRITIC_LR = 1e-3

ACTOR_LR = 1e-3

ENCODER_LR = 1e-3

# frequency of logging (steps)
LOG_FREQ = 100

# frequency of evaluation (episodes)
EVAL_FREQ = 1