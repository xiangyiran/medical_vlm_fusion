# Install dependencies from requirements.txt before running this Python source file.

import os
import re
import json
from pathlib import Path

import numpy as np
import pandas as pd
from tqdm.auto import tqdm
from IPython.display import display

import kagglehub

try:
    from google.colab import drive
    drive.mount("/content/drive")
except Exception as e:
    print("Google Drive mount failed. If you are not in Colab, this is expected.")
    print("Reason:", repr(e))


# =========================================================
# Config
# =========================================================
DATASET_NAME = "raddar/chest-xrays-indiana-university"

# Save all processed files inside the project artifacts directory.
PROJECT_DIR = "/content/drive/MyDrive/Dissertation/medical_vlm_fusion"
ARTIFACTS_DIR = os.path.join(PROJECT_DIR, "artifacts")
SAVE_DIR = os.path.join(ARTIFACTS_DIR, "data", "iu_xray_chexpert14")
os.makedirs(SAVE_DIR, exist_ok=True)

TEXT_MODE = "findings_impression"   # options: findings, impression, findings_impression
ONLY_FRONTAL = True

# Uncertainty handling for the binary CSV used by the formal training notebook.
# CheXpert raw labels use:
#   1  = positive
#   0  = negative
#  -1  = uncertain
# NaN = unmentioned
#
# For a simple multi-label BCE classifier, U-Zero is usually the cleanest first version:
# uncertain and unmentioned are converted to 0.
UNCERTAIN_POLICY = "u_zero"  # options: u_zero, u_one, ignore_not_supported_for_binary_csv

RAW_OUTPUT_CSV = os.path.join(SAVE_DIR, "iu_xray_chexpert14_raw.csv")
BINARY_OUTPUT_CSV = os.path.join(SAVE_DIR, "iu_xray_chexpert14_binary_u_zero.csv")
SUMMARY_OUTPUT_CSV = os.path.join(SAVE_DIR, "iu_xray_chexpert14_label_summary.csv")

CHEXPERT_LABELS = [
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

print("SAVE_DIR:", SAVE_DIR)
print("RAW_OUTPUT_CSV:", RAW_OUTPUT_CSV)
print("BINARY_OUTPUT_CSV:", BINARY_OUTPUT_CSV)


# =========================================================
# Download IU X-ray dataset
# =========================================================
dataset_root = kagglehub.dataset_download(DATASET_NAME)
print("Path to dataset files:", dataset_root)


# =========================================================
# Utility functions
# =========================================================
def normalize_text(x):
    """Normalize whitespace in one report field."""
    if pd.isna(x):
        return ""
    x = str(x)
    x = x.replace("\n", " ")
    x = re.sub(r"\s+", " ", x)
    return x.strip()

def recursive_find_one(root, filename):
    """Find one named file recursively below the dataset root."""
    matches = list(Path(root).rglob(filename))
    if not matches:
        raise FileNotFoundError(f"Could not find {filename} under {root}")
    return str(matches[0])

def recursive_find_images(root):
    """Find all supported image files below the dataset root."""
    exts = {".png", ".jpg", ".jpeg"}
    return [p for p in Path(root).rglob("*") if p.suffix.lower() in exts]

def choose_text(row, mode="findings_impression"):
    """Choose and combine report text using the configured original mode."""
    findings = normalize_text(row.get("findings", ""))
    impression = normalize_text(row.get("impression", ""))
    indication = normalize_text(row.get("indication", ""))

    if mode == "findings":
        return findings if findings else impression
    elif mode == "impression":
        return impression if impression else findings
    else:
        parts = []
        if findings:
            parts.append("Findings: " + findings)
        if impression:
            parts.append("Impression: " + impression)
        if not parts and indication:
            parts.append("Indication: " + indication)
        return " ".join(parts)

def split_sentences(text):
    """Split normalized report text with the original simple rule."""
    text = normalize_text(text).lower()
    # simple sentence splitter; enough for IU report preprocessing
    parts = re.split(r"[.;]\s+|\n+", text)
    return [p.strip() for p in parts if p.strip()]


# =========================================================
# Locate and read IU files
# =========================================================
reports_csv = recursive_find_one(dataset_root, "indiana_reports.csv")
projections_csv = recursive_find_one(dataset_root, "indiana_projections.csv")
all_images = recursive_find_images(dataset_root)

print("reports_csv:", reports_csv)
print("projections_csv:", projections_csv)
print("num image files found:", len(all_images))

reports_df = pd.read_csv(reports_csv)
proj_df = pd.read_csv(projections_csv)

reports_df.columns = [c.strip().lower() for c in reports_df.columns]
proj_df.columns = [c.strip().lower() for c in proj_df.columns]

print("reports_df shape:", reports_df.shape)
print("proj_df shape:", proj_df.shape)
print("reports columns:", reports_df.columns.tolist())
print("projections columns:", proj_df.columns.tolist())


# =========================================================
# Build one sample per UID
# =========================================================
image_path_map = {p.name: str(p) for p in tqdm(all_images, desc="Indexing image files")}

for col in ["mesh", "problems", "indication", "comparison", "findings", "impression"]:
    if col not in reports_df.columns:
        reports_df[col] = ""
    reports_df[col] = reports_df[col].fillna("").astype(str)

proj_df["filename"] = proj_df["filename"].astype(str)
proj_df["projection"] = proj_df["projection"].fillna("").astype(str)

if ONLY_FRONTAL:
    proj_df = proj_df[proj_df["projection"].str.upper().isin(["PA", "AP", "FRONTAL"])].copy()

# Keep one image per study to match the current single-image multimodal model.
proj_df = proj_df.sort_values(["uid", "filename"]).groupby("uid", as_index=False).first()

df = reports_df.merge(proj_df[["uid", "filename", "projection"]], on="uid", how="inner").copy()
df["image_path_at_preprocessing"] = df["filename"].map(image_path_map)
df = df[df["image_path_at_preprocessing"].notna()].copy()

df["text"] = df.apply(lambda r: choose_text(r, TEXT_MODE), axis=1)
df["text_len"] = df["text"].str.len()
df = df[df["text_len"] > 0].copy()

print("Processed samples before labeling:", df.shape)
display(df[["uid", "filename", "projection", "text"]].head())


# =========================================================
# Lightweight CheXpert-style rule labeler
# =========================================================
# Output label meanings:
#   1    positive mention
#   0    negative mention
#  -1    uncertain mention
#   NaN  unmentioned
#
# This is intentionally simple and transparent for a dissertation.
# It follows the CheXpert 14-label format, but it is not a full clone of the official Stanford parser.

LABEL_PATTERNS = {
    "Enlarged Cardiomediastinum": [
        r"enlarged cardiomediastinum", r"cardiomediastinal enlargement",
        r"enlarged cardiac silhouette", r"enlarged mediastinum",
        r"widened mediastinum", r"mediastinal widening"
    ],
    "Cardiomegaly": [
        r"cardiomegaly", r"cardiac enlargement", r"enlarged heart",
        r"heart is enlarged", r"mildly enlarged heart", r"cardiac silhouette is enlarged"
    ],
    "Lung Lesion": [
        r"lung lesion", r"pulmonary lesion", r"pulmonary nodule", r"lung nodule",
        r"nodule", r"mass", r"granuloma", r"pulmonary mass"
    ],
    "Lung Opacity": [
        r"lung opacity", r"pulmonary opacity", r"airspace opacity", r"opacity",
        r"opacification", r"infiltrate", r"airspace disease", r"haziness",
        r"linear density", r"patchy density"
    ],
    "Edema": [
        r"edema", r"pulmonary edema", r"vascular congestion",
        r"interstitial edema", r"congestive heart failure", r"chf",
        r"fluid overload"
    ],
    "Consolidation": [
        r"consolidation", r"consolidative", r"airspace consolidation"
    ],
    "Pneumonia": [
        r"pneumonia", r"pneumonitis", r"infection", r"infectious process"
    ],
    "Atelectasis": [
        r"atelectasis", r"atelectatic", r"volume loss", r"subsegmental atelectasis",
        r"bibasilar atelectasis", r"linear atelectasis"
    ],
    "Pneumothorax": [
        r"pneumothorax", r"ptx"
    ],
    "Pleural Effusion": [
        r"pleural effusion", r"effusion", r"pleural fluid", r"costophrenic blunting",
        r"blunting of the costophrenic angle"
    ],
    "Pleural Other": [
        r"pleural thickening", r"pleural plaque", r"pleural plaques",
        r"pleural scarring", r"pleural calcification"
    ],
    "Fracture": [
        r"fracture", r"rib fracture", r"clavicle fracture", r"compression fracture",
        r"deformity of .* rib", r"old fracture"
    ],
    "Support Devices": [
        r"tube", r"line", r"catheter", r"picc", r"port-a-cath", r"pacemaker",
        r"defibrillator", r"endotracheal", r"tracheostomy", r"ng tube",
        r"enteric tube", r"central venous", r"chest tube", r"device"
    ],
}

NO_FINDING_POSITIVE_PATTERNS = [
    r"no acute cardiopulmonary abnormality",
    r"no acute cardiopulmonary disease",
    r"no acute disease",
    r"no active disease",
    r"no acute findings",
    r"no significant abnormality",
    r"normal chest",
    r"unremarkable chest",
    r"clear lungs",
]

NEGATION_PATTERNS = [
    r"\bno\b", r"\bnot\b", r"\bwithout\b", r"\babsent\b", r"\bnegative for\b",
    r"\bfree of\b", r"\bno evidence of\b", r"\bno radiographic evidence of\b"
]

UNCERTAIN_PATTERNS = [
    r"\bpossible\b", r"\bpossibly\b", r"\bprobable\b", r"\bprobably\b",
    r"\bsuggest\b", r"\bsuggests\b", r"\bsuspicious\b", r"\bquestionable\b",
    r"\bmay represent\b", r"\bcould represent\b", r"\bcannot exclude\b",
    r"\bcan't exclude\b", r"\bconcern for\b", r"\blikely\b"
]

def regex_search(pattern, text):
    """Return whether a case-insensitive regular expression matches."""
    return re.search(pattern, text, flags=re.IGNORECASE) is not None

def has_any(patterns, text):
    """Return whether any configured pattern matches the text."""
    return any(regex_search(p, text) for p in patterns)

def classify_sentence_for_label(sentence, label_patterns):
    """
    Classify one sentence for one observation.
    Priority:
    - If label not mentioned: None
    - If uncertain wording exists near/in the sentence: -1
    - If negation wording exists: 0
    - Otherwise: 1
    """
    if not has_any(label_patterns, sentence):
        return None

    if has_any(UNCERTAIN_PATTERNS, sentence):
        return -1

    if has_any(NEGATION_PATTERNS, sentence):
        return 0

    return 1

def aggregate_mentions(mentions):
    """
    CheXpert-style aggregation:
    positive > uncertain > negative > unmentioned
    """
    if len(mentions) == 0:
        return np.nan
    if 1 in mentions:
        return 1
    if -1 in mentions:
        return -1
    if 0 in mentions:
        return 0
    return np.nan

def label_report_chexpert_style(text):
    """Extract the original transparent CheXpert-style report labels."""
    sentences = split_sentences(text)
    out = {}

    # First label all abnormal observations except No Finding.
    abnormal_positive_or_uncertain = False

    for label, patterns in LABEL_PATTERNS.items():
        mentions = []
        for sent in sentences:
            pred = classify_sentence_for_label(sent, patterns)
            if pred is not None:
                mentions.append(pred)

        value = aggregate_mentions(mentions)
        out[label] = value

        if value in [1, -1]:
            abnormal_positive_or_uncertain = True

    # No Finding: positive only for explicit normal/no-acute-disease reports.
    text_low = normalize_text(text).lower()
    explicit_no_finding = has_any(NO_FINDING_POSITIVE_PATTERNS, text_low)

    if explicit_no_finding and not abnormal_positive_or_uncertain:
        out["No Finding"] = 1
    elif abnormal_positive_or_uncertain:
        out["No Finding"] = 0
    else:
        out["No Finding"] = np.nan

    # Return in official label order.
    return {label: out.get(label, np.nan) for label in CHEXPERT_LABELS}


# =========================================================
# Apply labeler
# =========================================================
label_rows = []
for text in tqdm(df["text"].tolist(), desc="Extracting CheXpert-style labels"):
    label_rows.append(label_report_chexpert_style(text))

label_df = pd.DataFrame(label_rows)
df_raw = pd.concat([df.reset_index(drop=True), label_df.reset_index(drop=True)], axis=1)

# Reorder columns for readability.
base_cols = [
    "uid", "filename", "projection",
    "image_path_at_preprocessing",
    "text", "text_len",
    "findings", "impression", "indication",
    "mesh", "problems"
]
base_cols = [c for c in base_cols if c in df_raw.columns]
df_raw = df_raw[base_cols + CHEXPERT_LABELS]

print("Raw labeled dataframe:", df_raw.shape)
display(df_raw.head())


# =========================================================
# Create binary training CSV
# =========================================================
def convert_raw_to_binary(x, uncertain_policy="u_zero"):
    """Convert one raw label using the configured uncertainty policy."""
    if pd.isna(x):
        return 0
    x = int(x)
    if x == 1:
        return 1
    if x == 0:
        return 0
    if x == -1:
        if uncertain_policy == "u_one":
            return 1
        # default: U-Zero
        return 0
    return 0

df_binary = df_raw.copy()

for label in CHEXPERT_LABELS:
    df_binary[label] = df_binary[label].apply(lambda x: convert_raw_to_binary(x, UNCERTAIN_POLICY)).astype(int)

df_binary["num_positive_labels"] = df_binary[CHEXPERT_LABELS].sum(axis=1)
df_binary["has_any_positive"] = (df_binary["num_positive_labels"] > 0).astype(int)

print("Binary dataframe:", df_binary.shape)
print("Rows with at least one positive label:", int(df_binary["has_any_positive"].sum()))
display(df_binary[["uid", "filename", "projection", "num_positive_labels"] + CHEXPERT_LABELS].head())


# =========================================================
# Label summary
# =========================================================
summary_rows = []

for label in CHEXPERT_LABELS:
    raw_counts = df_raw[label].value_counts(dropna=False).to_dict()
    binary_pos = int(df_binary[label].sum())
    summary_rows.append({
        "label": label,
        "raw_positive_1": int(raw_counts.get(1.0, 0) + raw_counts.get(1, 0)),
        "raw_negative_0": int(raw_counts.get(0.0, 0) + raw_counts.get(0, 0)),
        "raw_uncertain_-1": int(raw_counts.get(-1.0, 0) + raw_counts.get(-1, 0)),
        "raw_unmentioned_nan": int(df_raw[label].isna().sum()),
        "binary_positive_after_policy": binary_pos,
        "binary_prevalence_after_policy": binary_pos / max(len(df_binary), 1),
    })

summary_df = pd.DataFrame(summary_rows)
display(summary_df)


# =========================================================
# Save to Google Drive
# =========================================================
df_raw.to_csv(RAW_OUTPUT_CSV, index=False)
df_binary.to_csv(BINARY_OUTPUT_CSV, index=False)
summary_df.to_csv(SUMMARY_OUTPUT_CSV, index=False)

metadata = {
    "dataset_name": DATASET_NAME,
    "text_mode": TEXT_MODE,
    "only_frontal": ONLY_FRONTAL,
    "uncertain_policy": UNCERTAIN_POLICY,
    "num_samples": int(len(df_binary)),
    "labels": CHEXPERT_LABELS,
    "raw_output_csv": RAW_OUTPUT_CSV,
    "binary_output_csv": BINARY_OUTPUT_CSV,
    "summary_output_csv": SUMMARY_OUTPUT_CSV,
}

metadata_path = os.path.join(SAVE_DIR, "iu_xray_chexpert14_metadata.json")
with open(metadata_path, "w") as f:
    json.dump(metadata, f, indent=2)

print("Saved files:")
print("RAW_OUTPUT_CSV:", RAW_OUTPUT_CSV)
print("BINARY_OUTPUT_CSV:", BINARY_OUTPUT_CSV)
print("SUMMARY_OUTPUT_CSV:", SUMMARY_OUTPUT_CSV)
print("metadata_path:", metadata_path)


# =========================================================
# Quick quality check
# =========================================================
print("Label prevalence after binary conversion:")
display(summary_df[["label", "binary_positive_after_policy", "binary_prevalence_after_policy"]])

print("\nExample rows with positive labels:")
cols = ["uid", "filename", "projection", "num_positive_labels"] + CHEXPERT_LABELS + ["text"]
display(df_binary[df_binary["num_positive_labels"] > 0][cols].head(10))
