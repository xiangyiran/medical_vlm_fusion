"""IU X-ray loading, patient-level split, tokenization, and DataLoaders."""

import os

import kagglehub
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from IPython.display import display
from PIL import Image
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, Dataset
from tqdm.auto import tqdm
from transformers import AutoImageProcessor, AutoTokenizer

from .config import *
from .utils import recursive_find_images, seed_everything

# Mount Google Drive; downloadable caches use Colab's default local storage.
try:
    from google.colab import drive
    if not os.path.ismount("/content/drive"):
        drive.mount("/content/drive")
except Exception as e:
    print("Google Drive is not available in this environment.")
    print("Reason:", repr(e))

dataset_root = kagglehub.dataset_download("raddar/chest-xrays-indiana-university")
print("Path to dataset files:", dataset_root)

seed_everything(SEED)

# ===== Load processed labels and images =====
# This cell expects a processed IU X-ray CSV with CheXpert-14 binary labels.
if not os.path.exists(PROCESSED_CSV):
    raise FileNotFoundError(
        f"Processed CSV not found: {PROCESSED_CSV}\n"
        "Run your IU X-ray CheXpert-14 preprocessing notebook first."
    )

all_images = recursive_find_images(dataset_root)
image_path_map = {p.name: str(p) for p in tqdm(all_images, desc="Indexing image files")}

processed_df = pd.read_csv(PROCESSED_CSV)
processed_df.columns = [c.strip() for c in processed_df.columns]

required_cols = ["uid", "filename", "projection", "text"] + LABEL_COLS
missing_cols = [c for c in required_cols if c not in processed_df.columns]
if missing_cols:
    raise ValueError(f"Missing required columns: {missing_cols}")

for c in LABEL_COLS:
    processed_df[c] = processed_df[c].fillna(0).astype(np.float32)

df = processed_df.copy()
df["filename"] = df["filename"].astype(str)
df["projection"] = df["projection"].fillna("").astype(str)

if ONLY_FRONTAL:
    df = df[df["projection"].str.upper().isin(["PA", "AP", "FRONTAL"])].copy()

df["image_path"] = df["filename"].map(image_path_map)
df = df[df["image_path"].notna()].copy()
df["text"] = df["text"].fillna("").astype(str)
df = df[df["text"].str.len() > 0].copy()

if DROP_ALL_ZERO_LABEL_ROWS:
    df = df[df[LABEL_COLS].sum(axis=1) > 0].copy()

if DEBUG_FRACTION < 1.0:
    df = df.sample(frac=DEBUG_FRACTION, random_state=SEED).copy()

print("Final df shape:", df.shape)
display(df[LABEL_COLS].sum().sort_values(ascending=False).to_frame("positive_count"))
display(df[["uid", "filename", "projection", "text"] + LABEL_COLS].head())


# ===== Patient-level split by uid =====
# This prevents images from the same report/patient id appearing in different splits.
uid_list = df["uid"].unique()
train_uids, temp_uids = train_test_split(
    uid_list,
    test_size=(1.0 - TRAIN_RATIO),
    random_state=SEED,
    shuffle=True,
)

relative_test_ratio = TEST_RATIO / (VAL_RATIO + TEST_RATIO)
val_uids, test_uids = train_test_split(
    temp_uids,
    test_size=relative_test_ratio,
    random_state=SEED,
    shuffle=True,
)

split_map = {u: "train" for u in train_uids}
split_map.update({u: "validate" for u in val_uids})
split_map.update({u: "test" for u in test_uids})

df["split"] = df["uid"].map(split_map)
print(df["split"].value_counts())


# ===== Tokenizer / processor =====
# The processor handles X-ray images, and the tokenizer handles report text.
image_processor = AutoImageProcessor.from_pretrained(VISION_MODEL_NAME)
tokenizer = AutoTokenizer.from_pretrained(TEXT_MODEL_NAME, trust_remote_code=True)

print("Tokenizer pad token:", tokenizer.pad_token)
print("Number of labels:", len(LABEL_COLS))


# Dataset class and DataLoaders for image-text-label batches.
class IndianaMultimodalDataset(Dataset):
    """Return one IU X-ray image, one report text, and one CheXpert-14 multi-label target."""

    def __init__(self, df, tokenizer, image_processor, max_text_len):
        """Store the dataframe and preprocessing tools."""
        self.df = df.reset_index(drop=True)
        self.tokenizer = tokenizer
        self.image_processor = image_processor
        self.max_text_len = max_text_len

    def __len__(self):
        """Return the number of samples."""
        return len(self.df)

    def __getitem__(self, idx):
        """Load and preprocess one image-text-label sample."""
        row = self.df.iloc[idx]

        image = Image.open(row["image_path"]).convert("RGB")
        image_inputs = self.image_processor(images=image, return_tensors="pt")

        text_inputs = self.tokenizer(
            row["text"],
            padding="max_length",
            truncation=True,
            max_length=self.max_text_len,
            return_tensors="pt",
        )

        labels = torch.tensor(row[LABEL_COLS].values.astype(np.float32), dtype=torch.float32)

        return {
            "pixel_values": image_inputs["pixel_values"].squeeze(0),
            "input_ids": text_inputs["input_ids"].squeeze(0),
            "attention_mask": text_inputs["attention_mask"].squeeze(0),
            "labels": labels,
        }


train_df = df[df["split"] == "train"].copy()
val_df = df[df["split"] == "validate"].copy()
test_df = df[df["split"] == "test"].copy()

train_ds = IndianaMultimodalDataset(train_df, tokenizer, image_processor, MAX_TEXT_LEN)
val_ds = IndianaMultimodalDataset(val_df, tokenizer, image_processor, MAX_TEXT_LEN)
test_ds = IndianaMultimodalDataset(test_df, tokenizer, image_processor, MAX_TEXT_LEN)

pin_memory = torch.cuda.is_available()
train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, num_workers=NUM_WORKERS, pin_memory=pin_memory)
val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS, pin_memory=pin_memory)
test_loader = DataLoader(test_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS, pin_memory=pin_memory)

print("train/val/test:", len(train_ds), len(val_ds), len(test_ds))


# ===== Loss weight for multi-label imbalance =====
# Positive labels are rare, so pos_weight gives positive examples more weight in BCE loss.
train_labels_np = train_df[LABEL_COLS].values.astype(np.float32)
pos_counts = train_labels_np.sum(axis=0)
neg_counts = len(train_labels_np) - pos_counts
raw_pos_weight = neg_counts / np.clip(pos_counts, 1, None)
pos_weight = torch.tensor(np.clip(raw_pos_weight, 1.0, POS_WEIGHT_MAX).astype(np.float32), device=DEVICE)
criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

print("pos_weight:", {c: float(w) for c, w in zip(LABEL_COLS, pos_weight.detach().cpu().numpy())})


# Encoder wrappers, common fusion utilities, and single-modality baselines.
