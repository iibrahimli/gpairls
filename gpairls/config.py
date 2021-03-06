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
TRAINING_STEPS = 2000

# number of steps to collect data before starting training
INIT_STEPS = 500

# encoder type
ENCODER_TYPE = "cnn"

# dimensionality of feature vector from encoder
ENCODER_FEATURE_DIM = 32

# actor and critic hidden dimensions
HIDDEN_DIM = 16

REPLAY_BUFFER_CAPACITY = 64

BATCH_SIZE = 16

CRITIC_LR = 1e-3

ACTOR_LR = 1e-3

ENCODER_LR = 1e-3

# frequency of logging (steps)
LOG_FREQ = 250

# frequency of evaluation (steps)
EVAL_FREQ = 250