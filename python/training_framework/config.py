import torch

# Global aiconfig
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# Model configuration
EMBEDDING_MODEL_NAME = "all-MiniLM-L6-v2"
GENERATION_MODEL_NAME = "gpt2"

# Data configuration
DATA_PATH = "data/" # Directory for training data
DB_CONNECTION_STRING = ""
DB_SCHEMA = "public"

# Training configuration
TRAIN_BATCH_SIZE = 8
EVAL_BATCH_SIZE = 8
NUM_EPOCHS = 3
LEARNING_RATE = 5e-5
WEIGHT_DECAY = 0.01

# Output configuration
MODEL_OUTPUT_DIR = "models/"
LOG_DIR = "logs/"