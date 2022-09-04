"""
RL training config
"""

from pathlib import Path


# directory with training logs
LOG_DIR = Path(__file__).parent.parent.resolve() / "logs"

# directory with model weights
MODEL_DIR = LOG_DIR / "model"

# directory for eval trajectories
TRAJECTORY_DIR = LOG_DIR / "trajectories"
TRAJECTORY_DIR.mkdir(parents=True, exist_ok=True)

# random seed for reproducibility
SEED = 42

# number of steps to train each agent
TRAINING_STEPS = 20_000_000

# number of steps to collect data before starting training
INIT_STEPS = 1000

# encoder type
ENCODER_TYPE = "cnn"

# dimensionality of feature vector from encoder
ENCODER_FEATURE_DIM = 32

# number of conv layers in the CNN encoder
ENCODER_NUM_LAYERS = 3

# number of filters in each encoder conv layer
ENCODER_NUM_FILTERS = 64

# actor and critic hidden dimensions
HIDDEN_DIM = 16

# neurons in reward decoder hidden layers
DECODER_DIM = 64

# neurons in transition model hidden layer
TRANSITION_MODEL_DIM = 32

REPLAY_BUFFER_CAPACITY = 10_000

BATCH_SIZE = 256

CRITIC_LR = 1e-3

ACTOR_LR = 1e-3

# is also the decoder LR
ENCODER_LR = 1e-3

# frequency of logging (steps)
LOG_FREQ = 10000

# frequency of evaluation (steps)
EVAL_FREQ = 10000

# frequency of W&B logging
WANDB_LOG_FREQ = 10