# FrameSense — Image Quality Assessment for Accessibility

FrameSense is a Streamlit research demo that assesses four image-quality issues using saved classifiers built on pretrained image encoders. It is based on the VizWiz-QualityIssues dataset, which contains real-world photos captured by people who are blind or have low vision.

The project explores whether an uploaded image may have **poor framing**, **blur**, **rotation**, or **unrecognisable content**. FrameSense is a research prototype, not a validated accessibility or safety tool.

## What FrameSense does

- Accepts a single image upload in a supported image format.
- Extracts visual features using the encoder selected for each task.
- Applies the corresponding saved full-data logistic-regression classifier.
- Uses a separate decision threshold for each task.
- Displays the predictions and associated positive-class scores.
- Provides plain-language guidance and a results export where available in the interface.

## Models and decision thresholds

The current app uses these selected model artefacts:

| Image-quality task | Encoder | Decision threshold |
|---|---|---:|
| Poor framing (`FRM`) | CLIP | 0.42 |
| Blur (`BLR`) | CLIP | 0.53 |
| Rotation (`ROT`) | DINOv2 | 0.76 |
| Unrecognisable content (`UNRECOGNIZABLE`) | CLIP | 0.86 |

A positive prediction is made when the classifier's positive-class score meets the task's threshold. The score should not be interpreted as a guaranteed or calibrated probability that the image has the issue.

## Model files

Place the four selected classifier files in the `models/` directory:

```text
models/
  FRM_CLIP_full.joblib
  BLR_CLIP_full.joblib
  ROT_DINOv2_full.joblib
  UNRECOGNIZABLE_CLIP_full.joblib
```

The app's model-loading code must use these filenames. If you change model artefacts or filenames, update the corresponding app configuration as well.

The `.joblib` files contain saved classifier pipelines and their task metadata, including the selected threshold. Only use trusted model files because loading joblib files can execute code.

## Feature extraction

The classifiers were trained on frozen embeddings extracted from pretrained image encoders. The app must use the same encoder, feature extraction path, image conversion, preprocessing, and feature dimensions as the training workflow.

- **CLIP:** 512-dimensional visual embeddings.
- **DINOv2:** 768-dimensional visual embeddings.

The selected models use CLIP for framing, blur, and unrecognisability, and DINOv2 for rotation. The image encoder checkpoints may be downloaded from Hugging Face on first use if they are not already cached. The first inference can therefore take longer, especially on CPU.

## Run locally

Use a compatible Python environment (Python 3.11 is recommended if it matches the pinned dependencies).

```bash
python -m venv .venv

# Windows PowerShell
.venv\\Scripts\\Activate.ps1

# macOS/Linux
source .venv/bin/activate

pip install -r requirements.txt
streamlit run app.py
```

The app will be available at the local URL printed by Streamlit. The first run may need to download pretrained encoder weights and can take several minutes.

## Deploy on Streamlit Community Cloud

1. Push `app.py`, `requirements.txt`, the Streamlit configuration, and the four required model files to the GitHub repository.
2. In Streamlit Community Cloud, create or open the app linked to that repository.
3. Set `app.py` as the app entry point.
4. Wait for dependency installation and pretrained checkpoint downloads to finish.
5. Upload a sample image and check that all four task results render.

Avoid committing large backbone checkpoint folders unless you deliberately manage them with Git LFS. Keep private data, credentials, and unrelated local files out of the repository.

## Testing checklist

Before a presentation or release, verify:

- The deployed app starts without errors.
- All four model files load.
- An uploaded image produces results for all four tasks.
- Each task displays the intended threshold and label.
- The result table, chart, guidance, and export work as expected.
- At least one additional image can be processed without a crash.

A generated image can be used for a basic smoke test, but it is not a ground-truth dataset example and cannot establish model accuracy. Record any demo screenshots only after confirming the app output.

## Research context and limitations

FrameSense is a demonstration of image-quality classification for an accessibility-related research project. Its predictions may be wrong, and its output should not be relied on as the sole basis for accessibility, safety, or other important decisions.

The thresholds are task-specific and were selected during the project's training/validation workflow. Reported validation metrics describe performance on that validation setup; they do not guarantee performance on a new uploaded image. Do not describe the prototype as production-ready or as a substitute for human judgement.

## Project structure

```text
.
├── app.py
├── requirements.txt
├── .streamlit/
│   └── config.toml
└── models/
    ├── FRM_CLIP_full.joblib
    ├── BLR_CLIP_full.joblib
    ├── ROT_DINOv2_full.joblib
    └── UNRECOGNIZABLE_CLIP_full.joblib
```

Additional project files, notebooks, reports, and packaging instructions may be present depending on the development environment.
