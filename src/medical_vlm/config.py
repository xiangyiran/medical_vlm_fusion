"""Original experiment constants and seed-specific Google Drive paths."""

import os

import torch


SEED = int(os.environ.get("MEDICAL_VLM_SEED", "95"))
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

VISION_MODEL_NAME = "microsoft/rad-dino"
TEXT_MODEL_NAME = "microsoft/BiomedVLP-CXR-BERT-general"

DRIVE_SOURCE_DIR = "/content/drive/MyDrive/Dissertation/medical_vlm_fusion"
ARTIFACTS_DIR = os.path.join(DRIVE_SOURCE_DIR, "artifacts")
DATA_DIR = os.path.join(ARTIFACTS_DIR, "data", "iu_xray_chexpert14")
PROCESSED_CSV = os.path.join(DATA_DIR, "iu_xray_chexpert14_binary_u_zero.csv")

MAX_TEXT_LEN = 512
BATCH_SIZE = 8
NUM_WORKERS = 2

EPOCHS = 8
LR = 1e-4
WEIGHT_DECAY = 1e-4
GRAD_CLIP = 1.0
POS_WEIGHT_MAX = 20.0
FREEZE_IMAGE_ENCODER = True
FREEZE_TEXT_ENCODER = True
USE_AMP = torch.cuda.is_available()

HIDDEN_DIM = 512
NUM_HEADS = 8
FUSION_LAYERS = 1
DROPOUT = 0.1
ALIGNMENT_LOSS_WEIGHT = 0.1

TRAIN_RATIO = 0.70
VAL_RATIO = 0.15
TEST_RATIO = 0.15

ONLY_FRONTAL = True
DROP_ALL_ZERO_LABEL_ROWS = False
DEBUG_FRACTION = 1.0
KEEP_MODELS_IN_MEMORY = False

PROFILE_GFLOPS = True
FORCE_REPROFILE_GFLOPS = False
PROFILE_WARMUP_STEPS = 3
PROFILE_LATENCY_REPEAT = 10

LABEL_COLS = [
    "No Finding",
    "Enlarged Cardiomediastinum",
    "Cardiomegaly",
    "Lung Lesion",
    "Lung Opacity",
    "Edema",
    "Consolidation",
    "Pneumonia",
    "Atelectasis",
    "Pneumothorax",
    "Pleural Effusion",
    "Pleural Other",
    "Fracture",
    "Support Devices",
]

DRIVE_PROJECT_DIR = ARTIFACTS_DIR
EXPERIMENT_NAME = "late_fusion"
EXPERIMENT_ID = f"exp_seed_{SEED}"
CKPT_DIR = os.path.join(ARTIFACTS_DIR, "checkpoints", EXPERIMENT_ID)
RESULTS_ROOT_DIR = os.path.join(ARTIFACTS_DIR, "results")
RESULTS_DIR = os.path.join(RESULTS_ROOT_DIR, EXPERIMENT_ID)
LOCAL_CKPT_DIR = CKPT_DIR
LOCAL_RESULTS_DIR = RESULTS_DIR

os.makedirs(CKPT_DIR, exist_ok=True)
os.makedirs(RESULTS_DIR, exist_ok=True)
