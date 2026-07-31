# Medical VLM Fusion

This repository contains a Colab-oriented implementation of multimodal medical image classification on the Indiana University chest X-ray dataset. It includes data preprocessing, seven main-study model configurations, bidirectional cross-attention ablations, three-seed evaluation, and image-to-text (I2T) attention visualisation.

The original notebook logic has been organised into reusable modules under `src/medical_vlm/`. Model selection, random seed, and execution action are passed explicitly through the command-line entry points used by the notebooks.

## Repository layout

```text
medical_vlm_fusion/
├── notebooks/
│   ├── 01_main_training_and_testing.ipynb
│   ├── 02_ablation_training_and_testing.ipynb
│   └── 03_results_display.ipynb
├── src/
│   ├── main_experiment.py
│   ├── ablation_experiment.py
│   ├── preprocessing/
│   │   └── preprocess_iu_xray_chexpert14.py
│   └── medical_vlm/
│       ├── config.py
│       ├── data.py
│       ├── utils.py
│       ├── experiments/
│       ├── models/
│       ├── training/
│       └── visualization/
├── artifacts/
├── requirements.txt
└── README.md
```

The main experiment and ablation notebooks call `src/main_experiment.py` and `src/ablation_experiment.py` directly. The results notebook imports the I2T visualisation functions from `src/medical_vlm/visualization/i2t.py`.

## Main-study models

The main study evaluates the following seven configurations:

- Image-only
- Text-only
- Concatenation fusion
- Average fusion
- Gated fusion
- Self-attention fusion
- Bidirectional cross-attention

Experiments use seeds `42`, `95`, and `1024`, producing 21 main model/seed jobs in total.

## Ablation models

The ablation study compares the full bidirectional cross-attention model with:

- Image-to-text attention only
- Text-to-image attention only
- No cross-attention
- No residual update
- No layer normalisation
- Bidirectional cross-attention with alignment loss

The full bidirectional reference model is trained by the main-study notebook and reused by the ablation workflow.

## Colab and Google Drive setup

The project currently uses this fixed Google Drive location:

```text
/content/drive/MyDrive/Dissertation/medical_vlm_fusion
```

Copy or clone the complete repository into that location before running the notebooks. If you use a different location, update the project paths in the notebooks and `src/medical_vlm/config.py`.

In Colab:

1. Mount Google Drive.
2. Select a GPU runtime. The experiments were prepared for an A100 runtime.
3. Install the dependencies:

```bash
pip install --no-cache-dir -r requirements.txt
```

The preprocessing dataset and pretrained model weights are downloaded into Colab's temporary runtime storage. They may need to be downloaded again after the runtime is reset.

## Artifact layout

Persistent project outputs use the following layout:

```text
/content/drive/MyDrive/Dissertation/medical_vlm_fusion/artifacts/
├── data/
│   └── iu_xray_chexpert14/
├── checkpoints/
│   ├── exp_seed_42/
│   ├── exp_seed_95/
│   └── exp_seed_1024/
└── results/
    ├── exp_seed_42/
    ├── exp_seed_95/
    ├── exp_seed_1024/
    └── three_seed_average/
```

The current repository snapshot includes processed data, saved result files, histories, tables, plots, and previously generated visualisation outputs under `artifacts/`. These files are currently tracked by Git.

Trained `.pt` checkpoint files are **not included** in the current repository snapshot. The `artifacts/checkpoints/` directory contains only a placeholder until training is run. Training creates the seed-specific checkpoint directories and model files shown above.

## Notebook workflow

### 1. Main study

Open `notebooks/01_main_training_and_testing.ipynb`.

This notebook:

- mounts Google Drive;
- installs project dependencies;
- runs IU X-ray preprocessing; and
- provides 21 independent training cells for seven models across three seeds.

Each training job also evaluates the selected model and saves its checkpoint, history, profiling information, and result JSON.

### 2. Ablation study

Open `notebooks/02_ablation_training_and_testing.ipynb` after completing the main study.

Important details about the current notebook:

- the three full bidirectional reference commands are commented out because those models are produced by the main-study notebook;
- the six seed-42 ablation commands use `--action train`; and
- the seed-95 and seed-1024 ablation commands currently use `--action test` to reuse existing saved outputs.

For a complete ablation retraining from scratch, change the seed-95 and seed-1024 ablation commands from:

```bash
--action test
```

to:

```bash
--action train
```

The current `test` workflow first reuses an existing saved result when one is available. It is not a replacement for training when neither a result nor a checkpoint exists.

### 3. Results and visualisation

Open `notebooks/03_results_display.ipynb` after the required result files have been generated or restored.

This notebook:

- displays preprocessing metadata and label statistics;
- validates the main-study and ablation result files;
- computes arithmetic means across seeds `42`, `95`, and `1024`;
- generates paper tables and plots; and
- renders I2T attention visualisations for a selected seed and sample.

The scalar result tables and existing plots can be read from the included result files. Generating a new I2T visualisation requires the trained `bidirectional_cross_attention` checkpoint for the selected `I2T_SEED`. Because checkpoints are not included in this repository snapshot, run the corresponding main-study training job first or provide the correct checkpoint in:

```text
artifacts/checkpoints/exp_seed_<SEED>/
```

Without that checkpoint, the I2T section raises a `FileNotFoundError`.

## Command-line examples

Main-study training:

```bash
python src/main_experiment.py \
  --seed 42 \
  --model image_only \
  --action train
```

Ablation training:

```bash
python src/ablation_experiment.py \
  --seed 42 \
  --model bicross_i2t_only \
  --action train
```

Valid seed values are `42`, `95`, and `1024`. The notebooks explicitly control model and seed execution order; `TRAIN_MODEL_FLAGS` is not used.

## Reproducing all experiments from scratch

To produce a new complete set of outputs:

1. Place the project at the required Google Drive path.
2. Run preprocessing in the first notebook.
3. Run all 21 main-study training cells.
4. In the ablation notebook, keep the full reference cells commented because the reference checkpoints already come from the main study.
5. Change all active seed-95 and seed-1024 ablation commands to `--action train`.
6. Run all 18 ablation training cells.
7. Run the results notebook to calculate the three-seed summaries and visualisations.

## Preprocessing

`src/preprocessing/preprocess_iu_xray_chexpert14.py` downloads the Indiana University chest X-ray dataset through `kagglehub` and produces CheXpert-14 labels. It retains the configured report selection, rule-based labelling, uncertainty policy, frontal-image selection, and output filenames.

Processed CSV and metadata files are written to:

```text
artifacts/data/iu_xray_chexpert14/
```

The downloaded source dataset remains in Colab's temporary cache rather than inside the Git repository.
