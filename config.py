# config.py

import os

# Paths
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_PATH = os.path.join(BASE_DIR, "data", "translit_dataset.csv")
CHECKPOINT_PATH = os.path.join(BASE_DIR, "checkpoints", "seq2seq_model.pt")

# Model + Training Hyperparams
EMBED_SIZE = 256
HIDDEN_SIZE = 512
NUM_LAYERS = 1
DROPOUT = 0.2

BATCH_SIZE = 32
NUM_EPOCHS = 50
LEARNING_RATE = 0.001
TEACHER_FORCING_RATIO = 0.5
MAX_LENGTH = 30
