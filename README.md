# FrameSense — Image Quality & Encoder Reliability

FrameSense is a polished dark-navy research demo for comparing frozen ViT, CLIP and DINOv2 image encoders with saved binary logistic-regression classifiers trained on a sample of the VizWiz-QualityIssues dataset. The interface uses a deep blue palette, soft sage accents, restrained borders and responsive result panels.

## Interface design

The app uses a dark navy background with subtle blue/teal gradients, sage-green action buttons and high-contrast typography. Theme settings live in `.streamlit/config.toml`; interface CSS is defined near the top of `app.py`.

## What the app does

- Accepts one image upload (JPG, PNG, WEBP or BMP).
- Extracts an embedding with each selected pretrained encoder.
- Loads the corresponding saved baseline classifier for four tasks: framing (FRM), blur (BLR), rotation (ROT) and unrecognizable images.
- Displays the predicted class and positive-class probability at the shared baseline threshold of 0.50.
- Allows results to be downloaded as a CSV.

The code matches the training notebook's feature path:
- ViT: CLS token from `ViTForImageClassification.vit(...).last_hidden_state`.
- CLIP: `CLIPModel.visual_projection(vision_model(...).pooler_output)`.
- DINOv2: CLS token from `AutoModel(...).last_hidden_state`.

## Required project files

Put the 12 baseline `.joblib` files in `models/`. The app looks for:

```text
models/
  FRM_ViT_baseline.joblib
  FRM_CLIP_baseline.joblib
  FRM_DINOv2_baseline.joblib
  BLR_ViT_baseline.joblib
  BLR_CLIP_baseline.joblib
  BLR_DINOv2_baseline.joblib
  ROT_ViT_baseline.joblib
  ROT_CLIP_baseline.joblib
  ROT_DINOv2_baseline.joblib
  UNRECOGNIZABLE_ViT_baseline.joblib
  UNRECOGNIZABLE_CLIP_baseline.joblib
  UNRECOGNIZABLE_DINOv2_baseline.joblib
```

By default, the app downloads the public pretrained backbone checkpoints from Hugging Face on first use. If matching saved backbone folders are available, it uses `backbones/vit`, `backbones/clip` and `backbones/dinov2` instead. Keep those directories out of Git if they are large; model weights can make a repository exceed host file limits.

## Local run

Use Python 3.11, then run:

```bash
python -m venv .venv
# Windows PowerShell:
.venv\Scripts\Activate.ps1
# macOS/Linux:
source .venv/bin/activate

pip install -r requirements.txt
streamlit run app.py
```

The first inference downloads the pretrained weights if they are not already cached; it can take a few minutes. Inference runs on CPU for compatibility. The app loads one encoder at a time to reduce peak memory usage.

## Prepare the deployment bundle from Google Colab

The trained classifiers currently live in the project's Google Drive folder, so they must be copied into this package before running it. See `PACKAGE_FROM_COLAB.md` for the one-cell packaging workflow. That workflow also pins the installed scikit-learn version used to create the saved pipelines.

## Deploy on Streamlit Community Cloud

1. Complete the Colab packaging step and download the resulting ZIP.
2. Extract it and push the files (including `models/`) to a GitHub repository. Do **not** commit large backbone checkpoint folders unless you intentionally use Git LFS.
3. In Streamlit Community Cloud, create an app from the repository and select `app.py` as the entry point.
4. Wait for the first startup and backbone downloads to finish. The first image inference will be slower than later runs.

## Research limitations

This is a demonstration, not a production accessibility tool. The displayed probabilities are classifier scores and are not guaranteed to be calibrated confidence estimates. Validation metrics were measured on the project's held-out sample; an individual uploaded image may behave differently. Keep the common 0.50 threshold for the baseline comparison rather than presenting tuned thresholds as the main result.
