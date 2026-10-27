from __future__ import annotations

import gc
import os
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
import streamlit as st
import torch
from PIL import Image, ImageOps, UnidentifiedImageError

# -----------------------------------------------------------------------------
# FrameSense: image-quality and visual-encoder reliability demo.
# Uses the same frozen-encoder feature definitions as the training notebook:
#   ViT:   ViTForImageClassification.vit(...).last_hidden_state[:, 0, :]
#   CLIP:  CLIPModel.visual_projection(vision_model(...).pooler_output)
#   DINOv2: AutoModel(...).last_hidden_state[:, 0, :]
# Selected full-data classifiers use task-specific thresholds; the UNRECOGNIZABLE threshold
# is intentionally set to 0.80 for a recall-oriented demo operating point.
# -----------------------------------------------------------------------------

APP_DIR = Path(__file__).resolve().parent
MODELS_DIR = APP_DIR / "models"
DEVICE = torch.device("cpu")  # reliable for local laptops and hosted demos

ENCODER_INFO = {
    "ViT": {
        "hub_id": "google/vit-base-patch16-224",
        "local_candidates": [APP_DIR / "backbones" / "vit", APP_DIR / "backbones" / "ViT"],
        "description": "ViT-Base / Patch16, using the CLS-token embedding.",
    },
    "CLIP": {
        "hub_id": "openai/clip-vit-base-patch32",
        "local_candidates": [APP_DIR / "backbones" / "clip", APP_DIR / "backbones" / "CLIP"],
        "description": "CLIP ViT-B/32, using the projected visual pooled embedding.",
    },
    "DINOv2": {
        "hub_id": "facebook/dinov2-base",
        "local_candidates": [APP_DIR / "backbones" / "dinov2", APP_DIR / "backbones" / "DINOv2"],
        "description": "DINOv2-Base, using the CLS-token embedding.",
    },
}

TASK_INFO = {
    "FRM": {"label": "Poor framing", "explanation": "The image framing or composition may make the subject difficult to inspect."},
    "BLR": {"label": "Blurriness", "explanation": "The image may be blurred or lack useful visual detail."},
    "ROT": {"label": "Rotation", "explanation": "The image may be rotated relative to an upright orientation."},
    "UNRECOGNIZABLE": {"label": "Unrecognizable image", "explanation": "The image may be difficult to interpret reliably."},
}
MODEL_ORDER = ["ViT", "CLIP", "DINOv2"]
TASK_ORDER = ["FRM", "BLR", "ROT", "UNRECOGNIZABLE"]

# Best selected full-data model for each task.
TASK_MODELS = {
    "FRM": "CLIP",
    "BLR": "CLIP",
    "ROT": "DINOv2",
    "UNRECOGNIZABLE": "CLIP",
}
TASK_THRESHOLDS = {
    "FRM": 0.42,
    "BLR": 0.53,
    "ROT": 0.76,
    "UNRECOGNIZABLE": 0.80,
}
TASK_MODEL_FILES = {
    task: f"{task}_{encoder}_full.joblib"
    for task, encoder in TASK_MODELS.items()
}

st.set_page_config(
    page_title="FrameSense | Image Quality Reliability",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown(
    """
    <style>
      :root {
        --navy-950: #07101e;
        --navy-900: #0a1424;
        --navy-850: #0d192b;
        --navy-800: #111f34;
        --navy-700: #1b2c45;
        --text-main: #f2f6fc;
        --text-muted: #a7b5c9;
        --sage: #a8c8a1;
        --sage-deep: #7eaa88;
        --line: rgba(169, 193, 220, .16);
      }

      .stApp {
        background:
          radial-gradient(ellipse at 8% 0%, rgba(50, 88, 133, .24), transparent 36%),
          radial-gradient(ellipse at 95% 9%, rgba(91, 132, 111, .12), transparent 29%),
          linear-gradient(145deg, #07101e 0%, #0a1424 52%, #0b1628 100%);
        color: var(--text-main);
      }
      [data-testid="stHeader"] { background: rgba(7, 16, 30, .72); }
      [data-testid="stToolbar"] { right: 1rem; }
      [data-testid="stSidebar"] {
        background: linear-gradient(180deg, #0d192b 0%, #091322 100%);
        border-right: 1px solid var(--line);
      }
      [data-testid="stSidebar"] > div:first-child { padding-top: 1.2rem; }
      .block-container {
        padding-top: 2rem;
        padding-bottom: 3rem;
        max-width: 1360px;
      }
      h1, h2, h3, h4 { color: var(--text-main) !important; letter-spacing: -.025em; }
      h2 { margin-top: .55rem; }
      p, li, label, .stMarkdown { color: #d3dceb; }
      [data-testid="stCaptionContainer"] p,
      .small-note { color: var(--text-muted) !important; }

      .hero {
        position: relative;
        overflow: hidden;
        padding: 2.1rem 2.15rem 1.85rem;
        border: 1px solid rgba(168, 200, 161, .25);
        border-radius: 24px;
        background:
          radial-gradient(circle at 84% 15%, rgba(168, 200, 161, .13), transparent 26%),
          linear-gradient(118deg, rgba(20, 43, 70, .97) 0%, rgba(14, 34, 55, .96) 56%, rgba(20, 51, 57, .94) 100%);
        box-shadow: 0 22px 60px rgba(0, 0, 0, .22);
        margin-bottom: 1.15rem;
      }
      .hero:after {
        content: "";
        position: absolute;
        right: -52px; top: -94px;
        width: 255px; height: 255px;
        border: 1px solid rgba(168, 200, 161, .16);
        border-radius: 50%;
        box-shadow: 0 0 0 28px rgba(168, 200, 161, .025), 0 0 0 58px rgba(168, 200, 161, .02);
        pointer-events: none;
      }
      .hero-kicker {
        display: inline-flex; align-items: center; gap: .55rem;
        color: #c4d9c0; font-size: .72rem; font-weight: 750;
        letter-spacing: .16em; text-transform: uppercase; margin-bottom: .9rem;
      }
      .hero-dot {
        width: 7px; height: 7px; border-radius: 50%; background: var(--sage);
        box-shadow: 0 0 12px rgba(168, 200, 161, .85);
      }
      .hero h1 { margin: 0 0 .55rem 0; font-size: clamp(2rem, 4vw, 3rem); line-height: 1.08; font-weight: 760; }
      .hero h1 span { color: var(--sage); }
      .hero p { margin: 0; max-width: 740px; color: #c0cfe0; font-size: 1.03rem; line-height: 1.65; }
      .hero-tags { display: flex; flex-wrap: wrap; gap: .5rem; margin-top: 1.35rem; }
      .hero-tag {
        padding: .34rem .68rem; border-radius: 100px;
        border: 1px solid rgba(196, 217, 192, .23);
        background: rgba(6, 16, 29, .28); color: #e2eee0;
        font-size: .76rem; letter-spacing: .015em;
      }
      .section-eyebrow {
        color: var(--sage); font-size: .72rem; font-weight: 750;
        letter-spacing: .14em; text-transform: uppercase;
      }
      .result-card {
        border: 1px solid var(--line); border-radius: 16px; padding: 1rem;
        background: linear-gradient(145deg, rgba(20, 36, 57, .9), rgba(13, 26, 43, .92));
      }
      [data-testid="stMetric"] {
        background: rgba(17, 31, 52, .68);
        border: 1px solid var(--line); border-radius: 14px;
        padding: .9rem 1rem;
      }
      [data-testid="stMetricLabel"] { color: var(--text-muted) !important; }
      [data-testid="stMetricValue"] { color: var(--text-main) !important; }
      [data-testid="stFileUploader"] section {
        background: rgba(15, 30, 49, .7) !important;
        border: 1px dashed rgba(168, 200, 161, .55) !important;
        border-radius: 18px !important;
      }
      [data-testid="stFileUploader"] section:hover { border-color: var(--sage) !important; }
      [data-testid="stFileUploaderDropzoneInstructions"] div { color: #dce6f3 !important; }
      .stButton > button, .stFormSubmitButton > button,
      [data-testid="stDownloadButton"] button {
        border: 1px solid rgba(168, 200, 161, .72) !important;
        border-radius: 11px !important;
        background: linear-gradient(135deg, #b7d4af, #91b99b) !important;
        color: #0a1a22 !important; font-weight: 750 !important;
        min-height: 2.8rem; box-shadow: 0 7px 20px rgba(95, 145, 107, .15);
        transition: transform .16s ease, box-shadow .16s ease;
      }
      .stButton > button:hover, .stFormSubmitButton > button:hover,
      [data-testid="stDownloadButton"] button:hover {
        transform: translateY(-1px); box-shadow: 0 10px 24px rgba(95, 145, 107, .24);
        border-color: #c4e0bc !important;
      }
      [data-testid="stBaseButton-secondary"] {
        background: rgba(25, 44, 68, .8) !important; color: #e6edf7 !important;
        border: 1px solid var(--line) !important;
      }
      [data-testid="stDataFrame"], [data-testid="stTable"] {
        border: 1px solid var(--line); border-radius: 12px; overflow: hidden;
      }
      [data-testid="stAlert"] { border-radius: 14px; }
      hr { border-color: var(--line) !important; }
      div[data-testid="stVerticalBlockBorderWrapper"] {
        background: rgba(15, 29, 47, .72); border-color: var(--line) !important;
        border-radius: 16px;
      }
      @media (max-width: 700px) {
        .block-container { padding: 1rem 1rem 2rem; }
        .hero { padding: 1.35rem; border-radius: 18px; }
        .hero h1 { font-size: 2rem; }
      }
    </style>
    """,
    unsafe_allow_html=True,
)

st.markdown(
    """
    <div class="hero">
      <div class="hero-kicker"><span class="hero-dot"></span> FrameSense · Visual Reliability Lab</div>
      <h1>Frame<span>Sense</span></h1>
      <p>Image quality and encoder reliability. Compare ViT, CLIP and DINOv2 on the same photo to explore four quality signals that matter for assistive computer vision.</p>
      <div class="hero-tags">
        <span class="hero-tag">CLIP · DINOv2</span>
        <span class="hero-tag">4 quality tasks</span>
        <span class="hero-tag">Task-specific thresholds</span>
      </div>
    </div>
    """,
    unsafe_allow_html=True,
)


st.markdown(
    '<div class="section-eyebrow">Image-quality assessment</div>',
    unsafe_allow_html=True,
)
st.write(
    "Upload a photo to assess four image-quality issues. FrameSense uses the selected "
    "full-data classifier for each task and applies its task-specific decision threshold."
)

with st.sidebar:
    st.markdown(
        '<div class="hero-kicker"><span class="hero-dot"></span> Project notes</div>',
        unsafe_allow_html=True,
    )
    st.header("About this demo")
    st.write("**Study pipeline**")
    st.write("Frozen visual encoder → saved classifier pipeline")
    st.markdown("**Selected models**")
    st.write("CLIP: framing, blur and unrecognizability")
    st.write("DINOv2: rotation")
    st.markdown("**Decision thresholds**")
    for task in TASK_ORDER:
        st.write(f"· {TASK_INFO[task]['label']}: {TASK_THRESHOLDS[task]:.2f}")
    st.markdown("**Tasks in this study**")
    for task in TASK_ORDER:
        st.write(f"· {TASK_INFO[task]['label']}")
    st.divider()
    st.caption("First inference can take longer while pretrained backbone weights download.")
    st.caption("Research demonstration only — not a safety-critical image-quality checker.")


def classifier_path(task: str, encoder: str | None = None) -> Path:
    """Return the selected full-data model file for a task."""
    return MODELS_DIR / TASK_MODEL_FILES[task]


def load_classifier(task: str, encoder: str | None = None) -> tuple[Any, float]:
    """Load the saved pipeline and use the app-configured task threshold."""
    path = classifier_path(task)
    if not path.is_file():
        raise FileNotFoundError(
            f"Missing selected model: {path.name}. Put the four *_full.joblib files in the app's models/ folder."
        )

    saved_data = joblib.load(path)
    if not isinstance(saved_data, dict) or "pipeline" not in saved_data:
        raise ValueError(f"Unexpected model-file format in {path.name}; expected a dictionary containing 'pipeline'.")

    pipeline = saved_data["pipeline"]
    # Use app-configured thresholds so deliberate operating-point changes are
    # reflected consistently in predictions and the interface. The saved model
    # metadata remains unchanged.
    threshold = float(TASK_THRESHOLDS[task])
    return pipeline, threshold


def encoder_source(encoder: str) -> str:
    for candidate in ENCODER_INFO[encoder]["local_candidates"]:
        if candidate.is_dir() and any(candidate.iterdir()):
            return str(candidate)
    return ENCODER_INFO[encoder]["hub_id"]


def embedding_for_image(image: Image.Image, encoder: str) -> np.ndarray:
    """Compute one image embedding with the same feature path as training."""
    source = encoder_source(encoder)

    if encoder == "ViT":
        from transformers import AutoImageProcessor, ViTForImageClassification

        processor = AutoImageProcessor.from_pretrained(source)
        backbone = ViTForImageClassification.from_pretrained(source)
        backbone = backbone.to(DEVICE).eval()
        inputs = processor(images=image, return_tensors="pt")
        inputs = {key: value.to(DEVICE) for key, value in inputs.items()}
        with torch.inference_mode():
            output = backbone.vit(**inputs)
            features = output.last_hidden_state[:, 0, :]

    elif encoder == "CLIP":
        from transformers import CLIPModel, CLIPProcessor

        processor = CLIPProcessor.from_pretrained(source)
        backbone = CLIPModel.from_pretrained(source)
        backbone = backbone.to(DEVICE).eval()
        inputs = processor(images=image, return_tensors="pt")
        pixel_values = inputs["pixel_values"].to(DEVICE)
        with torch.inference_mode():
            output = backbone.vision_model(pixel_values=pixel_values)
            features = backbone.visual_projection(output.pooler_output)

    elif encoder == "DINOv2":
        from transformers import AutoImageProcessor, AutoModel

        processor = AutoImageProcessor.from_pretrained(source)
        backbone = AutoModel.from_pretrained(source)
        backbone = backbone.to(DEVICE).eval()
        inputs = processor(images=image, return_tensors="pt")
        inputs = {key: value.to(DEVICE) for key, value in inputs.items()}
        with torch.inference_mode():
            output = backbone(**inputs)
            features = output.last_hidden_state[:, 0, :]

    else:
        raise ValueError(f"Unsupported encoder: {encoder}")

    array = features.detach().cpu().numpy().astype(np.float32)
    expected_dim = int(array.shape[1])

    # Drop the large backbone from memory before loading the next encoder.
    del backbone, processor, inputs, output, features
    gc.collect()

    return array.reshape(1, expected_dim)


def score_image(image: Image.Image) -> tuple[pd.DataFrame, dict[str, np.ndarray]]:
    """Score each task with its selected encoder and saved classifier pipeline."""
    rows: list[dict[str, Any]] = []
    embedding_cache: dict[str, np.ndarray] = {}

    progress = st.progress(0, text="Preparing image assessment…")
    total = len(TASK_ORDER)

    for index, task in enumerate(TASK_ORDER):
        encoder = TASK_MODELS[task]
        progress.progress(index / max(total, 1), text=f"Checking {TASK_INFO[task]['label'].lower()} with {encoder}…")

        if encoder not in embedding_cache:
            embedding_cache[encoder] = embedding_for_image(image, encoder)
        vector = embedding_cache[encoder]

        classifier, configured_threshold = load_classifier(task, encoder)
        model_dim = getattr(classifier, "n_features_in_", None)
        if model_dim is not None and vector.shape[1] != int(model_dim):
            raise ValueError(
                f"Feature dimension mismatch for {encoder}/{task}: encoder returned {vector.shape[1]} features, "
                f"but the classifier expects {model_dim}. Check the checkpoint and feature extraction used during training."
            )

        probabilities = classifier.predict_proba(vector)[0]
        classes = list(classifier.classes_)
        if 1 not in classes:
            raise ValueError(f"Classifier for {task}/{encoder} does not have positive class 1.")
        positive_probability = float(probabilities[classes.index(1)])
        # Apply the task-specific threshold configured for this app release.
        threshold = configured_threshold
        predicted = int(positive_probability >= threshold)

        rows.append({
            "Encoder": encoder,
            "Task": task,
            "Image-quality issue": TASK_INFO[task]["label"],
            "Predicted result": "Issue detected" if predicted else "No issue detected",
            "P(issue)": positive_probability,
            "Threshold": threshold,
        })
        progress.progress((index + 1) / total, text=f"Finished {TASK_INFO[task]['label'].lower()}")

    progress.empty()
    result_df = pd.DataFrame(rows)
    result_df["Task"] = pd.Categorical(result_df["Task"], TASK_ORDER, ordered=True)
    result_df["Encoder"] = pd.Categorical(result_df["Encoder"], MODEL_ORDER, ordered=True)
    result_df = result_df.sort_values(["Task", "Encoder"]).reset_index(drop=True)

    # Keep the existing summary-card/chart structure while showing only the tasks
    # each selected encoder actually handles. NaN means that encoder was not used for that task.
    probability_grid: dict[str, np.ndarray] = {}
    for encoder in sorted(set(TASK_MODELS.values()), key=MODEL_ORDER.index):
        encoder_rows = result_df[result_df["Encoder"] == encoder].set_index("Task")
        probability_grid[encoder] = np.array([
            float(encoder_rows.loc[task, "P(issue)"]) if task in encoder_rows.index else np.nan
            for task in TASK_ORDER
        ], dtype=float)

    embedding_cache.clear()
    gc.collect()
    return result_df, probability_grid


expected_files = [classifier_path(task) for task in TASK_ORDER]
missing_files = [p.name for p in expected_files if not p.is_file()]
if missing_files:
    st.warning(
        "Some selected full-data model files are missing from `models/`. "
        "Add the four *_full.joblib files before running inference. Missing files: "
        + ", ".join(missing_files)
    )


st.info(
    "Selected models: CLIP for poor framing, blur and unrecognizability; DINOv2 for rotation. "
    "Each task uses its own saved decision threshold."
)

uploaded = st.file_uploader(
    "Upload an image",
    type=["png", "jpg", "jpeg", "webp", "bmp"],
    help="Use a real photo. Images are processed in memory for this session.",
)

if uploaded is not None:
    try:
        original = Image.open(uploaded)
        image = ImageOps.exif_transpose(original).convert("RGB")
        left, right = st.columns([1, 1.3], gap="large")
        with left:
            st.image(image, caption="Uploaded image", use_container_width=True)
        with right:
            st.subheader("Image details")
            st.write(f"Dimensions: **{image.width} × {image.height}** pixels")
            st.write(f"Format: **{original.format or 'Unknown'}**")
            st.write("The same upload is evaluated independently by each selected encoder.")

        with st.form("inference_form"):
            run_button = st.form_submit_button(
                "Assess image quality",
                type="primary",
                use_container_width=True,
                disabled=bool(missing_files),
            )

        if run_button:
            try:
                with st.spinner("Running image-quality models. The first run can take a few minutes while weights download."):
                    results, score_grid = score_image(image)
                st.session_state["vizwiz_results"] = results
                st.session_state["vizwiz_scores"] = score_grid
                st.session_state["vizwiz_image_name"] = uploaded.name
            except Exception as exc:
                st.error(f"Inference failed: {exc}")
                st.exception(exc)

    except (UnidentifiedImageError, OSError, ValueError) as exc:
        st.error(f"Could not read the uploaded image: {exc}")

if "vizwiz_results" in st.session_state:
    results = st.session_state["vizwiz_results"]
    scores = st.session_state["vizwiz_scores"]

    st.divider()
    st.header("Prediction results")
    st.caption(
        f"Image: {st.session_state.get('vizwiz_image_name', 'uploaded image')} · "
        "task-specific decision thresholds are shown in the results table"
    )

    cols = st.columns(len(TASK_ORDER))
    for col, task in zip(cols, TASK_ORDER):
        frame = results[results["Task"].astype(str) == task]
        issue_flagged = not frame.empty and frame.iloc[0]["Predicted result"] == "Issue detected"
        with col:
            with st.container(border=True):
                st.markdown(f"<div class='section-eyebrow'>{TASK_INFO[task]['label']}</div>", unsafe_allow_html=True)
                st.metric("Assessment", "Possible issue" if issue_flagged else "No issue flagged")
                st.caption(f"{TASK_MODELS[task]} · threshold {TASK_THRESHOLDS[task]:.2f}")

    st.subheader("Side-by-side task results")
    display = results.copy()
    display["P(issue)"] = display["P(issue)"].map(lambda x: f"{x:.1%}")
    display["Threshold"] = display["Threshold"].map(lambda x: f"{x:.2f}")
    st.dataframe(
        display[["Task", "Image-quality issue", "Encoder", "Predicted result", "P(issue)", "Threshold"]],
        use_container_width=True,
        hide_index=True,
    )

    st.subheader("Positive-class model scores")
    chart_data = results.set_index("Image-quality issue")[["P(issue)"]]
    chart_data = chart_data.reindex([TASK_INFO[t]["label"] for t in TASK_ORDER])
    st.bar_chart(chart_data, y_label="Model score")

    
    st.subheader("What do the results mean?")
    st.caption(
        "Plain-language explanations based on the selected encoders. "
        "These are estimates, not confirmed image-quality judgements."
    )

    guidance = {
        "FRM": (
            "Poor framing",
            "Check whether the main subject is cut off or difficult to locate. "
            "Try centring the subject and including the whole object."
        ),
        "BLR": (
            "Possible blur",
            "Check whether the subject and important details look sharp. "
            "Try holding the phone steady, improving the lighting, "
            "or moving closer without excessive digital zoom."
        ),
        "ROT": (
            "Possible rotation",
            "Check whether the subject appears tilted or sideways. "
            "If it does, straighten the phone and take another photo."
        ),
        "UNRECOGNIZABLE": (
            "Possible recognition difficulty",
            "Check whether the subject is visible and its important details "
            "can be seen. Try better lighting, reducing obstructions, "
            "or framing the subject more clearly."
        ),
    }

    for task in TASK_ORDER:
        task_rows = results[
            results["Task"].astype(str) == task
        ].copy()

        if task_rows.empty:
            continue

        task_rows["P(issue)"] = task_rows["P(issue)"].astype(float)
        n_models = len(task_rows)
        n_flagged = int(
            (task_rows["Predicted result"] == "Issue detected").sum()
        )

        probabilities = task_rows["P(issue)"].to_numpy()
        task_threshold = float(task_rows["Threshold"].iloc[0])
        near_threshold = bool(np.any(np.abs(probabilities - task_threshold) <= 0.05))

        task_label, advice = guidance[task]

        if n_models == 1:
            if n_flagged == 1:
                status = "Possible issue flagged"
                summary = (
                    "The selected encoder flagged this issue. "
                    "Check the image before deciding whether to retake it."
                )
            else:
                status = "No issue flagged by this model"
                summary = (
                    "The selected encoder did not cross the decision threshold. "
                    "This does not guarantee that the image is clear."
                )
        elif n_flagged == n_models:
            status = "All selected models flagged this issue"
            summary = (
                "The selected encoders agree that this issue may be present, "
                "but agreement does not guarantee correctness."
            )
        elif n_flagged > n_models / 2:
            status = "Most selected models flagged this issue"
            summary = (
                f"{n_flagged} of {n_models} selected encoders flagged this issue. "
                "Consider checking the image for the problem described below."
            )
        elif n_flagged == 0:
            status = "No selected model flagged this issue"
            summary = (
                "None of the selected encoders crossed the threshold. "
                "The issue may still be present."
            )
        else:
            status = "The models disagree"
            summary = (
                f"{n_flagged} of {n_models} encoders flagged this issue. "
                "Treat the result cautiously because the encoders do not agree."
            )

        with st.container(border=True):
            st.markdown(f"### {task_label}")
            st.markdown(f"**{status}**")
            st.write(summary)

            if near_threshold:
                st.info(
                    f"The model score is close to this task's {task_threshold:.2f} decision threshold. "
                    "Small score differences can change that model's label, "
                    "so interpret this result cautiously."
                )

            st.markdown("**What to check**")
            st.write(advice)

            score_columns = st.columns(n_models)
            for col, (_, row) in zip(score_columns, task_rows.iterrows()):
                with col:
                    st.metric(
                        label=f'{row["Encoder"]} score',
                        value=f'{row["P(issue)"]:.1%}',
                        help="Estimated issue score; not guaranteed to be calibrated.",
                    )

            st.caption(
                "Scores are model outputs, not guaranteed calibrated "
                "probabilities. Suggested actions are general guidance."
            )
    csv = results.to_csv(index=False).encode("utf-8")
    st.download_button(
        "Download these predictions as CSV",
        data=csv,
        file_name="framesense_image_quality_predictions.csv",
        mime="text/csv",
    )

st.divider()
st.caption(
    "Study: Comparing the Reliability of ViT, CLIP and DINOv2 for Image-Quality Assessment in Assistive Computer Vision."
)
