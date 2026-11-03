"""Constants shared by the data pipeline, the model and the training loop."""

MODALITIES = ("text", "audio", "vision")
LETTERS = {"l": "text", "a": "audio", "v": "vision"}
SUBSETS = ("l", "v", "a", "lv", "la", "va", "lva")

# Epochs of raw-score training before the memory bank is built and editing starts.
BANK_WARMUP_EPOCHS = 3
# Weight of the direction loss on the edit.
LAMBDA_DIR = 0.045
LABEL_MIN, LABEL_MAX = -3.0, 3.0
