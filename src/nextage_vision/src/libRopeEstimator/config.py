RANDOM_SEED = 42
# ---------------------
# -- Data parameters --
# ---------------------
DATA_PATH = "../data/tangled_rope_dataset_augmented_direction_depth.pkl"

# Data Augmentation
SKIP_ORIGINAL_VARIANT = True # if True, only augmented variants are used

MIRROR_DATASET = True # Mirror horizontally and vertically

ROTATE_DATASET = True
ROTATE_ANGLE_RANGE = 20
ROTATE_ANGLE_STEP = 20

AUGMENT_SCALE_AND_SHIFT = True
AUGMENT_SCALE_RANGE = (1, 1)
AUGMENT_COUNT_SCALE_SHIFT = 3

# ----------------------
# -- Model parameters --
# ----------------------
IMAGE_CHANNELS = 3
DIRECTION_CHANNELS = 2
IMAGE_HEIGHT = 256
IMAGE_WIDTH = 256
BOX_RADIUS = 12 # ROI box for intersection detection
ROI_POOL_SIZE = 14 # ROI Align output size

# -------------------------
# -- Training parameters --
# -------------------------
DEVICE = "cuda"  # "cuda" or "cpu"
BATCH_SIZE = 32
LEARNING_RATE = 0.00005
SMOOTHING_FACTOR = 0.1 # label smoothing for type classification

EPOCHS = 300 # 400
TYPE_WARMUP_EPOCHS = 0 # after 100 epochs, start training the type head

# Dataset Split Ratios
TRAIN_SPLIT = 0.8
TEST_SPLIT = 0.02
VAL_SPLIT = 0.18

# Loss Ratios
COORDINATE_LOSS_WEIGHT = 1
TYPE_LOSS_WEIGHT = 1
LENGTH_MAP_LOSS_WEIGHT = 20
DIRECTION_LOSS_WEIGHT = 5