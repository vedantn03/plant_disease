"""Streamlit UI for diagnosis, model trust, and operational evidence."""

import base64
import io
import json
import os
import sys
from pathlib import Path

# Ensure project root is in sys.path so `from src.model import ...` resolves
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import requests
import streamlit as st
from PIL import Image
from app.ui_helpers import (
    SUPPORTED_EXPLANATION_LANGUAGES,
    explanation_text,
    import_optional_explainer,
    load_saved_mlops_evidence,
    optional_explainer_available,
    parse_prediction_trust,
    prediction_explanation_values,
    saved_explainer_figures,
)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
API_URL          = os.environ.get("API_URL", "http://127.0.0.1:8000")
DEMO_SAMPLES_DIR = PROJECT_ROOT / "demo_samples"
LOGS_FILE        = PROJECT_ROOT / "logs" / "requests.jsonl"
REPORTS_DIR      = PROJECT_ROOT / "reports"
FIGURES_DIR      = REPORTS_DIR / "figures"


def _render_saved_explainer_figure(kind: str, streamlit_module, language: str) -> None:
    """Show repository-generated offline figures when an optional package is absent."""
    streamlit_module.info(explanation_text(
        "explainer_not_installed", language, name=kind.upper()
    )[0])
    figures = saved_explainer_figures(FIGURES_DIR, kind)
    if not figures:
        streamlit_module.caption("Not generated yet.")
        return
    for figure in figures[:4]:
        streamlit_module.image(
            figure,
            caption=explanation_text("offline_figure", language, name=figure.name)[0],
            use_container_width=True,
        )


def _render_trust_panel(
    payload: dict, original_image: Image.Image | None, language: str
) -> None:
    """Render API-returned trust fields and background-neutralized output only."""
    trust = parse_prediction_trust(payload)
    details = trust["bg_check_details"]
    st.markdown(f"**{explanation_text('trust_heading', language)[0]}**")
    label_col, confidence_col, ood_col, bg_col = st.columns(4)
    label_col.metric(explanation_text("returned_label", language)[0], trust["label"])
    confidence = trust["confidence"]
    confidence_col.metric(
        explanation_text("confidence_label", language)[0],
        f"{confidence:.1%}" if confidence is not None else "not available",
    )
    ood_col.metric(explanation_text("ood_status", language)[0], str(trust["ood_status"]))
    bg_col.metric(
        explanation_text("background_sensitivity", language)[0], trust["bg_check"]
    )

    quality = trust["quality"]
    if quality is None:
        st.caption(explanation_text("quality_not_available", language)[0])
    else:
        brightness = quality.get("brightness")
        blur = quality.get("blur_score")
        st.caption(explanation_text(
            "quality_values",
            language,
            brightness=brightness if brightness is not None else "not available",
            blur=blur if blur is not None else "not available",
        )[0])

    original_label = details.get("original_label") or trust["label"]
    original_confidence = details.get("original_confidence", confidence)
    neutralized_label = details.get("neutralized_label")
    neutralized_confidence = details.get("neutralized_confidence")
    neutralized_b64 = details.get("neutralized_image_base64")
    image_col, neutralized_col = st.columns(2)
    with image_col:
        original_confidence_display = (
            f"{original_confidence:.1%}"
            if isinstance(original_confidence, (int, float)) else "not available"
        )
        st.caption(explanation_text(
            "original_prediction",
            language,
            label=original_label,
            confidence=original_confidence_display,
        )[0])
        if original_image is not None:
            st.image(original_image, use_container_width=True)
        else:
            st.info("Original image: not available.")
    with neutralized_col:
        if neutralized_b64 and neutralized_label:
            neutralized_confidence_display = (
                f"{neutralized_confidence:.1%}"
                if isinstance(neutralized_confidence, (int, float)) else "not available"
            )
            st.caption(explanation_text(
                "neutralized_prediction",
                language,
                label=neutralized_label,
                confidence=neutralized_confidence_display,
            )[0])
            try:
                st.image(
                    Image.open(io.BytesIO(base64.b64decode(neutralized_b64))),
                    use_container_width=True,
                )
            except Exception:
                st.info("Neutralized image: not available from the API response.")
        else:
            st.caption(explanation_text("neutralized_not_available", language)[0])
            st.info(
                "The API did not return a neutralized image for this background sensitivity result."
            )

    change = details.get("confidence_change")
    if isinstance(change, (int, float)):
        st.caption(explanation_text(
            "confidence_change", language, change=f"{change:+.4f}"
        )[0])
    else:
        st.caption("Confidence change after mean-fill neutralization: not available.")

# ---------------------------------------------------------------------------
# Page config
# ---------------------------------------------------------------------------
st.set_page_config(
    page_title="Crop Disease Diagnostic System",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# ---------------------------------------------------------------------------
# CSS — clean, professional, no emojis
# ---------------------------------------------------------------------------
st.markdown("""
<style>
/* Layout */
.block-container { padding-top:1.1rem; padding-bottom:2rem; max-width:1240px; }

/* Status badges */
.badge-ok     { background:#15803d; color:#fff; padding:3px 11px; border-radius:3px;
                font-size:0.8rem; font-weight:700; display:inline-block; letter-spacing:.05em; }
.badge-warn   { background:#b45309; color:#fff; padding:3px 11px; border-radius:3px;
                font-size:0.8rem; font-weight:700; display:inline-block; letter-spacing:.05em; }
.badge-danger { background:#9f1239; color:#fff; padding:3px 11px; border-radius:3px;
                font-size:0.8rem; font-weight:700; display:inline-block; letter-spacing:.05em; }

/* Advisory cards */
.chem-card {
    border-left:4px solid #16a34a; padding:9px 14px; margin-bottom:10px;
    border-radius:0 4px 4px 0; background:rgba(22,163,74,.07);
    line-height:1.6; word-wrap:break-word; overflow-wrap:break-word;
}
.cultural-card {
    border-left:4px solid #2563eb; padding:7px 13px; margin-bottom:8px;
    border-radius:0 4px 4px 0; background:rgba(37,99,235,.07);
    line-height:1.55; word-wrap:break-word; overflow-wrap:break-word;
}
.disclaimer-box {
    border-left:4px solid #94a3b8; padding:10px 14px; font-size:0.8rem;
    margin-top:14px; border-radius:0 4px 4px 0; background:rgba(148,163,184,.07);
    line-height:1.5; word-wrap:break-word;
}

/* XAI captions */
.xai-title   { font-size:0.88rem; font-weight:600; margin-bottom:3px; }
.xai-caption { font-size:0.77rem; color:#64748b; line-height:1.45; margin-top:5px;
                word-wrap:break-word; }

/* Pipeline section cards */
.pipeline-card {
    border:1px solid rgba(100,116,139,.25); border-radius:6px; padding:14px 18px;
    margin-bottom:12px; background:rgba(100,116,139,.04);
}
.pipeline-card h4 { margin:0 0 6px 0; font-size:1rem; }
.pipeline-card p  { margin:0; font-size:0.86rem; line-height:1.6; color:#475569; }

/* Status dot */
.dot-green { display:inline-block; width:8px; height:8px; border-radius:50%;
             background:#16a34a; margin-right:5px; vertical-align:middle; }
.dot-amber { display:inline-block; width:8px; height:8px; border-radius:50%;
             background:#d97706; margin-right:5px; vertical-align:middle; }

/* Component table */
table.comp { width:100%; border-collapse:collapse; font-size:0.86rem; }
table.comp th { background:#1e3a5f; color:#fff; padding:7px 12px; text-align:left; font-weight:600; }
table.comp td { padding:7px 12px; border-bottom:1px solid rgba(100,116,139,.18); vertical-align:top;
                word-wrap:break-word; }
table.comp tr:nth-child(even) td { background:rgba(100,116,139,.04); }
</style>
""", unsafe_allow_html=True)

# ---------------------------------------------------------------------------
# Header
# ---------------------------------------------------------------------------
st.title("Crop Leaf Disease Diagnostic & Treatment Advisory System")
st.caption(
    "ResNet-18 deep learning classifier · 26 crop-disease classes · "
    "Grad-CAM / SHAP / LIME explanation methods · Source-aware advisory knowledge base"
)

# Backend health — small line, no jargon
_backend_ok = False
_health_payload = None
try:
    _hr = requests.get(f"{API_URL}/health", timeout=2)
    if _hr.status_code == 200:
        _health_payload = _hr.json()
    if _hr.status_code == 200 and _health_payload.get("status") == "ok":
        _device = _health_payload.get("device", "unknown").upper()
        _backend_ok = True
        st.caption(
            f"API: connected ({API_URL}) · Inference device: {_device} · "
            f"Serving model version: {_health_payload.get('model_version', 'not available')}"
        )
    else:
        st.warning(f"API returned HTTP {_hr.status_code}.")
except Exception:
    st.error(
        f"Cannot reach API at `{API_URL}`. "
        "Start the backend with: "
        "`cd plant_disease && ..\\.venv\\Scripts\\uvicorn.exe src.api:app --host 127.0.0.1 --port 8000`"
    )

# ---------------------------------------------------------------------------
# Tabs
# ---------------------------------------------------------------------------
tab_dx, tab_tel, tab_mlops = st.tabs([
    "Diagnostic Screening",
    "Production Telemetry",
    "MLOps System",
])

explanation_language = st.selectbox(
    "Explanation language / स्पष्टीकरण भाषा",
    list(SUPPORTED_EXPLANATION_LANGUAGES),
    key="explanation_language",
)
language_code = SUPPORTED_EXPLANATION_LANGUAGES[explanation_language]
_, is_reviewed_language = explanation_text("explanation_scope", language_code)

CROPS_LIST = [
    "Auto (infer from image)", "Apple", "Blueberry", "Cherry", "Corn",
    "Grape", "Orange", "Peach", "Pepper", "Potato", "Raspberry",
    "Soybean", "Squash", "Strawberry", "Tomato",
]

# ===========================================================================
# TAB 1 — DIAGNOSTIC SCREENING
# ===========================================================================
with tab_dx:
    # Wide right column to prevent text clipping
    col_in, col_out = st.columns([1, 1.5], gap="large")

    # ── LEFT: Input ──────────────────────────────────────────────────────────
    with col_in:
        st.subheader("Input")

        manifest_path = DEMO_SAMPLES_DIR / "manifest.json"
        demo_samples  = []
        if manifest_path.exists():
            try:
                demo_samples = json.loads(manifest_path.read_text())
            except Exception:
                pass

        sample_opts   = ["Upload an image"] + [f"Demo: {s['class_name']}" for s in demo_samples]
        chosen_sample = st.selectbox("Image source", sample_opts)

        selected_crop  = st.selectbox("Crop type (constrains inference)", CROPS_LIST)
        api_crop_param = None if "Auto" in selected_crop else selected_crop.lower()

        current_image = None
        if chosen_sample == "Upload an image":
            uf = st.file_uploader("Leaf photograph (JPEG / PNG)", type=["jpg", "jpeg", "png"])
            if uf:
                current_image = Image.open(uf).convert("RGB")
        else:
            idx = sample_opts.index(chosen_sample) - 1
            sp  = DEMO_SAMPLES_DIR / demo_samples[idx]["filename"]
            if sp.exists():
                current_image = Image.open(sp).convert("RGB")
            else:
                st.warning(f"Demo file not found: {demo_samples[idx]['filename']}")

        if current_image:
            st.image(current_image, caption="Image to diagnose", use_container_width=True)

        with st.expander("Field conditions (optional)", expanded=False):
            temp       = st.number_input("Temperature (°C)", value=24.0, step=0.5)
            hum        = st.number_input("Relative Humidity (%)", value=85.0, step=1.0)
            use_weather = st.checkbox("Include weather risk assessment", value=True)

        run_btn = st.button(
            "Run Diagnosis", type="primary", use_container_width=True,
            disabled=(current_image is None),
        )

    # ── RIGHT: Output ────────────────────────────────────────────────────────
    with col_out:
        st.subheader("Result")

        if run_btn and current_image:
            with st.spinner("Running inference..."):
                buf = io.BytesIO()
                current_image.save(buf, format="JPEG")
                buf.seek(0)
                payload = {}
                if use_weather:
                    payload["temperature_c"] = str(temp)
                    payload["humidity_pct"]   = str(hum)
                if api_crop_param:
                    payload["crop"] = api_crop_param
                try:
                    res = requests.post(
                        f"{API_URL}/predict",
                        data=payload,
                        files={"file": ("leaf.jpg", buf, "image/jpeg")},
                        timeout=15,
                    )
                    if res.status_code == 200:
                        st.session_state["last_diagnosis"] = res.json()
                        st.session_state["last_image"]     = current_image
                        st.session_state.pop("shap_map",  None)
                        st.session_state.pop("lime_fig",  None)
                        st.session_state.pop("shap_mass", None)
                    else:
                        st.error(f"Inference error: HTTP {res.status_code} — {res.text}")
                except Exception as e:
                    st.error(f"API error: {e}")

        if "last_diagnosis" not in st.session_state:
            st.info("Select or upload a leaf image on the left, then click **Run Diagnosis**.")
        else:
            data   = st.session_state["last_diagnosis"]
            status = data.get("status", "ok")
            if not is_reviewed_language:
                st.info(explanation_text("draft_translation_notice", language_code)[0])
            _render_trust_panel(data, st.session_state.get("last_image"), language_code)
            st.write("")

            # Rejected / OOD
            if status == "rejected_not_a_leaf":
                st.markdown('<span class="badge-danger">REJECTED — NON-LEAF INPUT</span>', unsafe_allow_html=True)
                st.error(
                    f"{data.get('message', 'Non-leaf image detected.')}  \n\n"
                    "Chemical spray recommendations are suppressed for non-leaf inputs."
                )

            # Uncertain
            elif status == "uncertain":
                st.markdown('<span class="badge-warn">LOW CONFIDENCE</span>', unsafe_allow_html=True)
                st.warning(data.get("message", "Model confidence is below the selective threshold."))
                st.metric("Confidence", f"{data.get('confidence', 0):.1%}")

            # Valid diagnosis
            else:
                crop_disp = data.get("crop", "").capitalize()
                cond_disp = data.get("condition", "").replace("_", " ").title()
                conf_val  = data.get("confidence", 0.0)

                st.markdown('<span class="badge-ok">DIAGNOSIS COMPLETE</span>', unsafe_allow_html=True)
                st.write("")
                m1, m2, m3 = st.columns(3)
                m1.metric("Crop",       crop_disp)
                m2.metric("Condition",  cond_disp)
                m3.metric("Confidence", f"{conf_val:.1%}")

                if data.get("coverage_caveat"):
                    st.info(data["coverage_caveat"])

                # ── EXPLAINABILITY ──────────────────────────────────────
                with st.container(border=True):
                    st.markdown(f"**{explanation_text('explainability_title', language_code)[0]}**")
                    st.caption(explanation_text("explanation_scope", language_code)[0])
                    explanation_values = prediction_explanation_values(data)
                    confidence_value = explanation_values["confidence"]
                    confidence_display = (
                        f"{confidence_value:.1%}"
                        if isinstance(confidence_value, (int, float))
                        else "not available"
                    )
                    st.caption(explanation_text(
                        "prediction_line",
                        language_code,
                        label=explanation_values["label"],
                        confidence=confidence_display,
                    )[0])
                    with st.expander("Original English explanation and evidence"):
                        st.write("Where the model looked — Explainability")
                        st.write(
                            "Attribution maps are visual aids. They show image regions that "
                            "influenced this model output; they do not prove a disease, explain "
                            "its cause, or establish that the model relied only on leaf tissue."
                        )
                        st.json(prediction_explanation_values(data))
                    st.write("")
                    col_cam, col_shap, col_lime = st.columns(3)

                    # Grad-CAM
                    with col_cam:
                        st.markdown('<div class="xai-title">Grad-CAM</div>', unsafe_allow_html=True)
                        if data.get("gradcam_png_base64"):
                            cam_bytes = base64.b64decode(data["gradcam_png_base64"])
                            st.image(Image.open(io.BytesIO(cam_bytes)), use_container_width=True)
                            st.caption(explanation_text("gradcam_caption", language_code)[0])
                        else:
                            st.caption(explanation_text("no_gradcam", language_code)[0])

                    # SHAP
                    with col_shap:
                        st.markdown('<div class="xai-title">SHAP</div>', unsafe_allow_html=True)
                        if "shap_map" in st.session_state:
                            st.image(st.session_state["shap_map"], use_container_width=True)
                            mass = st.session_state.get("shap_mass")
                            mass_txt = f" Leaf attribution mass: {mass:.1%}." if mass else ""
                            st.caption(
                                "This attribution visualization reflects feature contributions "
                                "for this model output; it does not confirm the diagnosis or its "
                                f"cause.{mass_txt}"
                            )
                        elif not optional_explainer_available("shap"):
                            _render_saved_explainer_figure("shap", st, language_code)
                        else:
                            st.button("Compute SHAP", key="shap_btn", use_container_width=True)
                            if st.session_state.get("shap_btn"):
                                with st.spinner("Computing Shapley values (~5 s)..."):
                                    try:
                                        import torch
                                        shap_lib = import_optional_explainer("shap")
                                        if shap_lib is None:
                                            raise ImportError("shap is not installed")
                                        from src.model import build_model, preprocess_image
                                        ckpt = torch.load(
                                            PROJECT_ROOT / "models" / "best_model.pth",
                                            map_location="cpu", weights_only=False,
                                        )
                                        sh_model = build_model(
                                            num_classes=len(ckpt["class_to_idx"]), pretrained=False
                                        )
                                        sh_model.load_state_dict(ckpt["model_state_dict"])
                                        sh_model.eval()
                                        bg   = torch.zeros(10, 3, 224, 224)
                                        t    = preprocess_image(st.session_state["last_image"])
                                        exp  = shap_lib.GradientExplainer(sh_model, bg)
                                        sv, _ = exp.shap_values(t, ranked_outputs=1)
                                        sv   = np.squeeze(sv[0] if isinstance(sv, list) else sv)
                                        smap = np.abs(sv).mean(axis=0)
                                        smap = smap / (smap.max() + 1e-8)
                                        st.session_state["shap_mass"] = float((smap > smap.mean()).mean())
                                        fig, ax = plt.subplots(figsize=(4, 4))
                                        ax.imshow(np.array(st.session_state["last_image"].resize((224, 224))))
                                        ax.imshow(smap, cmap="hot", alpha=0.55)
                                        ax.axis("off")
                                        plt.tight_layout(pad=0)
                                        b2 = io.BytesIO()
                                        fig.savefig(b2, format="PNG", bbox_inches="tight")
                                        plt.close(fig)
                                        b2.seek(0)
                                        st.session_state["shap_map"] = Image.open(b2).copy()
                                        st.rerun()
                                    except ImportError:
                                        _render_saved_explainer_figure("shap", st, language_code)
                                    except Exception as e:
                                        st.error(f"SHAP explanation unavailable: {e}")

                    # LIME
                    with col_lime:
                        st.markdown('<div class="xai-title">LIME</div>', unsafe_allow_html=True)
                        if "lime_fig" in st.session_state:
                            st.image(st.session_state["lime_fig"], use_container_width=True)
                            st.caption(
                                "This LIME visualization reflects the method's local feature "
                                "contributions; it does not confirm the diagnosis or explain "
                                "disease cause."
                            )
                        elif not optional_explainer_available("lime.lime_image"):
                            _render_saved_explainer_figure("lime", st, language_code)
                        else:
                            st.button("Compute LIME", key="lime_btn", use_container_width=True)
                            if st.session_state.get("lime_btn"):
                                with st.spinner("Computing LIME (~10 s)..."):
                                    try:
                                        import torch
                                        from torchvision import transforms
                                        lime_image = import_optional_explainer("lime.lime_image")
                                        if lime_image is None:
                                            raise ImportError("lime is not installed")
                                        from skimage.segmentation import mark_boundaries
                                        from src.model import build_model
                                        ckpt = torch.load(
                                            PROJECT_ROOT / "models" / "best_model.pth",
                                            map_location="cpu", weights_only=False,
                                        )
                                        l_model = build_model(
                                            num_classes=len(ckpt["class_to_idx"]), pretrained=False
                                        )
                                        l_model.load_state_dict(ckpt["model_state_dict"])
                                        l_model.eval()
                                        tf = transforms.Compose([
                                            transforms.ToTensor(),
                                            transforms.Normalize(
                                                [0.485, 0.456, 0.406], [0.229, 0.224, 0.225]
                                            ),
                                        ])
                                        def batch_pred(imgs):
                                            b = torch.stack([tf(i) for i in imgs], 0)
                                            with torch.no_grad():
                                                return torch.softmax(l_model(b), 1).numpy()
                                        img_np = np.array(
                                            st.session_state["last_image"].resize((224, 224))
                                        )
                                        exp = lime_image.LimeImageExplainer()
                                        result = exp.explain_instance(
                                            img_np, batch_pred,
                                            top_labels=1, hide_color=None, num_samples=100,
                                        )
                                        tmp, msk = result.get_image_and_mask(
                                            result.top_labels[0],
                                            positive_only=True, num_features=5, hide_rest=True,
                                        )
                                        fig, ax = plt.subplots(figsize=(4, 4))
                                        ax.imshow(mark_boundaries(tmp, msk))
                                        ax.axis("off")
                                        plt.tight_layout(pad=0)
                                        b3 = io.BytesIO()
                                        fig.savefig(b3, format="PNG", bbox_inches="tight")
                                        plt.close(fig)
                                        b3.seek(0)
                                        st.session_state["lime_fig"] = Image.open(b3).copy()
                                        st.rerun()
                                    except ImportError:
                                        _render_saved_explainer_figure("lime", st, language_code)
                                    except Exception as e:
                                        st.error(f"LIME explanation unavailable: {e}")

                # ── ADVICE PASSPORT ──────────────────────────────────────────
                advisory = data.get("advisory", {})
                trace = data.get("advice_trace", {})
                with st.container(border=True):
                    st.markdown(f"**{explanation_text('advice_passport', language_code)[0]}**")
                    st.caption(
                        f"Tier: {trace.get('tier', 'unknown')} · "
                        f"Evidence: {trace.get('evidence_grade', 'unknown')} · "
                        f"Knowledge base: {trace.get('kb_version', data.get('kb_version', 'unknown'))}"
                    )
                    if trace.get("source_url"):
                        st.markdown(f"Source: [{trace.get('kb_entry_id', 'cited entry')}]({trace['source_url']})")
                    if trace.get("retrieved_on"):
                        st.caption(f"Source retrieved: {trace['retrieved_on']}")
                    if trace.get("evidence_grade") == "C":
                        st.warning(explanation_text("no_reviewed_advice", language_code)[0])
                    elif advisory.get("message"):
                        st.info(advisory["message"])
                    for key, heading in (("scouting", "Scouting"), ("cultural", "Cultural guidance")):
                        items = advisory.get(key) or []
                        if items:
                            st.markdown(f"**{heading}**")
                            for item in items:
                                text = item.get("text", "") if isinstance(item, dict) else str(item)
                                st.markdown(text)
                                if isinstance(item, dict) and item.get("source_location"):
                                    st.caption(item["source_location"])
                    chemicals = advisory.get("chemical_options") or []
                    if chemicals and trace.get("tier") == "full":
                        st.markdown("**Chemical options (quoted source text)**")
                        for item in chemicals:
                            st.markdown(f"> {item.get('quote', '')}")
                            if item.get("source_location"):
                                st.caption(item["source_location"])
                    label_reminder = advisory.get("label_reminder", "Consult a local agronomist.")
                    if trace.get("evidence_grade") == "C" and not isinstance(label_reminder, dict):
                        label_reminder = explanation_text("consult_expert", language_code)[0]
                    if isinstance(label_reminder, dict):
                        st.caption(label_reminder.get("quote", "Consult a local agronomist."))
                        if label_reminder.get("source_location"):
                            st.caption(label_reminder["source_location"])
                    else:
                        st.caption(label_reminder)
                    weather_note = advisory.get("weather_note", "risk unknown")
                    if weather_note == "risk unknown":
                        weather_note = explanation_text("risk_unknown", language_code)[0]
                    st.caption(weather_note)
                    for hint in data.get("confusable_hints", []):
                        st.info(hint.get("hint", "May be confused with an unlisted class."))
                        if hint.get("how_to_tell_apart") and hint.get("source_url"):
                            st.caption(hint["how_to_tell_apart"])
                    st.caption(f"Rules fired: {', '.join(trace.get('rules_fired') or []) or 'none'}")

                disc = data.get("disclaimer", "")
                if language_code != "en" and disc:
                    disc = explanation_text("advice_disclaimer", language_code)[0]
                if disc:
                    st.markdown(f'<div class="disclaimer-box">{disc}</div>', unsafe_allow_html=True)


# ===========================================================================
# TAB 2 — PRODUCTION TELEMETRY
# ===========================================================================
with tab_tel:
    st.subheader("Production Telemetry")
    st.caption(
        "Live monitoring of inference requests — latency, image quality, "
        "OOD rejection rate, and statistical drift signals."
    )

    if not LOGS_FILE.exists():
        st.info("No requests logged yet. Run a diagnosis in the Diagnostic Screening tab.")
    else:
        records = []
        with open(LOGS_FILE, encoding="utf-8") as fh:
            for line in fh:
                line = line.strip()
                if line:
                    try:
                        records.append(json.loads(line))
                    except Exception:
                        pass

        if not records:
            st.info("Log file is empty. Run a diagnosis to generate telemetry.")
        else:
            df = pd.DataFrame(records)

            # Summary metrics
            m1, m2, m3, m4 = st.columns(4)
            m1.metric("Total Requests", len(df))
            m2.metric(
                "Median Latency",
                f"{df['latency_ms'].median():.1f} ms" if "latency_ms" in df else "N/A",
            )
            m3.metric(
                "OOD Rejection Rate",
                f"{(df['status'] != 'ok').mean():.1%}" if "status" in df else "N/A",
            )
            m4.metric(
                "Mean Blur Score",
                f"{df['blur_score'].mean():.1f}" if "blur_score" in df else "N/A",
            )
            st.write("")

            # Charts
            ch1, ch2 = st.columns(2)
            with ch1:
                if "latency_ms" in df.columns and len(df) >= 2:
                    fig, ax = plt.subplots(figsize=(5, 3))
                    ax.plot(df["latency_ms"].values, lw=1.3, color="#2563eb")
                    ax.axhline(df["latency_ms"].median(), color="#d97706",
                               ls="--", lw=1, label="Median")
                    ax.set_xlabel("Request index", fontsize=9)
                    ax.set_ylabel("Latency (ms)", fontsize=9)
                    ax.set_title("Inference Latency", fontsize=10)
                    ax.legend(fontsize=8)
                    ax.grid(alpha=0.3)
                    fig.tight_layout()
                    st.pyplot(fig)
                    plt.close(fig)
            with ch2:
                if "confidence" in df.columns and len(df) >= 2:
                    fig, ax = plt.subplots(figsize=(5, 3))
                    ax.hist(df["confidence"].dropna(), bins=20,
                            color="#16a34a", edgecolor="white")
                    ax.set_xlabel("Model Confidence", fontsize=9)
                    ax.set_ylabel("Count", fontsize=9)
                    ax.set_title("Confidence Distribution", fontsize=10)
                    ax.grid(alpha=0.3)
                    fig.tight_layout()
                    st.pyplot(fig)
                    plt.close(fig)

            # Log table
            st.markdown("**Recent requests**")
            show = [c for c in ["timestamp", "predicted_class", "confidence",
                                 "latency_ms", "status"] if c in df.columns]
            st.dataframe(df.tail(20)[show], use_container_width=True)

            # Drift section
            st.write("")
            with st.container(border=True):
                st.markdown("**SIMULATION ONLY — Legacy Drift Trigger**")
                st.caption(
                    "This panel reads a legacy simulation artifact. It is not the real-log drift "
                    "monitor and does not establish a live retraining trigger. See Prometheus and "
                    "Grafana for the provisional request-log drift alert."
                )
                drift_f = REPORTS_DIR / "retraining_trigger.json"
                if drift_f.exists():
                    dt = json.loads(drift_f.read_text())
                    st.markdown(
                        f"Status: **{dt.get('status','OK')}**  |  "
                        f"Severity: **{dt.get('severity','LOW')}**  |  "
                        f"Action: {dt.get('action','—')}"
                    )
                    with st.expander("Full drift report"):
                        st.json(dt)
                else:
                    st.info("No legacy simulation artifact is present. The real-log monitor is the WP4 exporter.")


# ===========================================================================
# TAB 3 — MLOPS SYSTEM
# ===========================================================================
with tab_mlops:
    st.subheader("MLOps System")
    st.caption(
        "End-to-end automated machine learning infrastructure — "
        "data versioning, experiment tracking, CI pipeline, "
        "model registry, serving, monitoring, explainability, and cloud architecture."
    )

    st.subheader("Saved model and trust evidence")
    evidence = load_saved_mlops_evidence(REPORTS_DIR, _health_payload)
    st.write("Serving model version from /health:")
    if evidence["serving_model_version"] is None:
        st.info(evidence["serving_message"])
    else:
        st.write(str(evidence["serving_model_version"]))

    st.markdown("**Champion and candidate runs**")
    if evidence["models"]:
        st.dataframe(pd.DataFrame(evidence["models"]), use_container_width=True)
    else:
        st.info(evidence["models_message"])

    st.markdown("**Promotion decisions**")
    if evidence["promotions"]:
        st.dataframe(pd.DataFrame(evidence["promotions"]), use_container_width=True)
    else:
        st.info(evidence["promotions_message"])

    st.markdown("**Latest drift status**")
    if evidence["drift"] is not None:
        st.json(evidence["drift"])
    else:
        st.info(evidence["drift_message"])

    st.write("")

    # ── Data Pipeline ────────────────────────────────────────────────────────
    with st.expander("Data Pipeline", expanded=True):
        c1, c2 = st.columns([1, 1], gap="large")
        with c1:
            st.markdown("""
**Source**
PlantVillage dataset — 87,000 images across 38 crop-disease combinations.
Filtered to 26 agriculturally relevant classes for the Indian subcontinent.

**Deduplication**
SHA-256 hash comparison eliminates visually identical frames from multi-shot captures,
preventing label leakage between train and test sets.

**Group-aware stratified split**
Images from the same plant capture session (identified by `<crop>::<session_id>`) are
kept in the same split. This prevents the model from "memorising" a plant and
inflating validation accuracy — a common flaw in leaf disease benchmarks.

**Outcome**: 70 / 15 / 15 split with zero group leakage, confirmed by the data
validation suite.
""")
        with c2:
            st.markdown("""
**Data versioning (DVC)**
All pipeline stages — raw download, preprocessing, splits, model checkpoint — are
tracked in `dvc.yaml` with SHA-256 checksums. Any change to input data or code is
detected and only the affected stages re-run.

**Data validation — 7 automated checks**

| Check | Result |
|---|---|
| Schema (columns, dtypes) | PASSED |
| Zero null values | PASSED |
| Class distribution within ±5% | PASSED |
| Image file integrity | PASSED |
| Group leakage across splits | PASSED |
| Split ratio (70/15/15) | PASSED |
| Label consistency | PASSED |
""")
            val_rep = REPORTS_DIR / "data_validation_report.json"
            if st.button("View validation report", key="dv_btn"):
                if val_rep.exists():
                    dv = json.loads(val_rep.read_text())
                    st.success(f"{dv.get('status')} — {dv.get('checks_passed')}/{dv.get('checks_run')} checks passed")
                    st.json(dv)
                else:
                    st.info("Run `make validate` to generate the report.")

    # ── Experiment Tracking & Model Registry ─────────────────────────────────
    with st.expander("Experiment Tracking & Model Registry"):
        c1, c2 = st.columns([1, 1], gap="large")
        with c1:
            st.markdown("""
**MLflow experiment tracking**
Every training run logs:
- Hyperparameters (learning rate, batch size, epochs, weight decay)
- Per-epoch training loss and validation Macro-F1
- Final checkpoint artefact

Runs are stored in a local SQLite backend (`mlruns/`). The best run is promoted
to the **MLflow Model Registry** with stage `Production`, making rollback to any
prior version a single command.

**Training details**
- Architecture: ResNet-18, ImageNet pre-trained, last block fine-tuned
- Automatic Mixed Precision (AMP) — 1.8× training speed on GPU
- OneCycleLR scheduler — fast convergence in 15 epochs
- Reproducibility: fixed seeds (PyTorch + NumPy), deterministic CUDA ops
""")
        with c2:
            st.markdown("""
Candidate evidence and promotion decisions are shown above from saved reports.
The serving version is read from the API health response. TEST metrics are for
reporting only; selection is based on VAL evidence.
""")
            repro_f = REPORTS_DIR / "reproduce_check.json"
            if st.button("View reproducibility report", key="repro_btn"):
                if repro_f.exists():
                    rp = json.loads(repro_f.read_text())
                    st.success(
                        f"100% prediction agreement across two independent runs. "
                        f"Max logit delta: {rp.get('max_abs_logit_delta', 0):.2e}"
                    )
                    st.json(rp)
                else:
                    st.info("Run `make reproduce` to verify determinism.")

    # ── CI Pipeline ───────────────────────────────────────────────────────────
    with st.expander("Continuous Integration Pipeline"):
        st.markdown("""
**GitHub Actions** (`.github/workflows/ci.yml`) runs automatically on every push
and pull request. The pipeline:

1. Installs dependencies in a clean Python 3.11 environment
2. Runs the test suite configured for the repository
3. Fails the build if a test fails

Run the repository test suite locally with `pytest -q`.
""")

    # ── Serving & Deployment ─────────────────────────────────────────────────
    with st.expander("Serving & Deployment"):
        c1, c2 = st.columns([1, 1], gap="large")
        with c1:
            st.markdown("""
**FastAPI REST API** (`src/api.py`)
Asynchronous inference endpoint at `POST /predict`.

Request pipeline:
1. Receive multipart image + optional field parameters
2. Blur quality check (Laplacian variance) — rejects degraded images
3. OOD filter (Helmholtz Free Energy + Layer 1 heuristics)
4. ResNet-18 forward pass — returns top prediction + confidence
5. Grad-CAM generation — heatmap encoded as base64 PNG
6. Advisory knowledge-base lookup — returns chemical options, cultural controls
7. Append to request log (JSONL) for telemetry

""")
        with c2:
            st.markdown("""
**Docker containerisation**

```
# Build
docker build -t crop-diagnostic:latest .

# Run
docker-compose up
```

Multi-stage Dockerfile:
- Stage 1: install build dependencies
- Stage 2: copy only runtime requirements → minimal image size
- Health check: `GET /health` every 30 s

`docker-compose.yml` orchestrates the API container with volume mounts
for the model checkpoint and knowledge base, allowing hot-swapping
without rebuilding the image.
""")
        bench_f = REPORTS_DIR / "INFERENCE_BENCHMARK.md"
        if bench_f.exists():
            st.markdown("**Inference benchmark report**")
            st.markdown(bench_f.read_text()[:1400])

    # ── Monitoring & Retraining ───────────────────────────────────────────────
    with st.expander("Monitoring & Automated Retraining"):
        c1, c2 = st.columns([1, 1], gap="large")
        with c1:
            st.markdown("""
**Request telemetry** (Tab 2)
Every inference is appended to `logs/requests.jsonl`:
- Timestamp, predicted class, confidence score
- Inference latency (ms)
- Image blur score (Laplacian variance)
- OOD filter decision and reason code

**Drift detection**
The KS-test (Kolmogorov-Smirnov) and PSI (Population Stability Index) compare
the current window of blur scores against a reference baseline.

- PSI < 0.1 → distribution stable, no action
- PSI 0.1–0.2 → moderate drift, alert raised
- PSI > 0.2 → significant drift, retraining triggered
""")
        with c2:
            st.markdown("""
**Automated retraining — DAG definition**

The retraining workflow is defined as a DAG (Directed Acyclic Graph) in
`dags/plant_disease_retraining_dag.py` using Apache Airflow's task model.

The DAG defines 6 sequential tasks:

| Task | Operator | Purpose |
|---|---|---|
| `check_drift_trigger` | PythonOperator | Read drift report, decide if retraining needed |
| `data_audit_and_split` | BashOperator | Verify data integrity, re-run group split |
| `data_validation_suite` | PythonOperator | Run 7-check validation before training |
| `evaluate_ood_detection` | BashOperator | Re-evaluate OOD filter on new data |
| `benchmark_xai_explainability` | BashOperator | Re-run Grad-CAM / SHAP / LIME benchmark |
| `evaluate_promotion_gate` | BashOperator | Gate check before promoting new model |

**Note on Airflow**: The DAG is defined as code (`.py` file) and can be
deployed to any Airflow instance (local, AWS MWAA, Google Composer).
The simulation below runs each task locally without an Airflow server.
""")
        if st.button("Run DAG simulation (local executor)", key="dag_btn"):
            af_f = REPORTS_DIR / "airflow_dag_run.json"
            if af_f.exists():
                af = json.loads(af_f.read_text())
                st.success(
                    f"DAG `{af.get('dag_id')}` — {af.get('status')} "
                    f"(run: {af.get('execution_date','')})"
                )
                tasks_df = pd.DataFrame(af.get("tasks", []))
                if not tasks_df.empty:
                    st.dataframe(
                        tasks_df[["task_id", "operator", "status",
                                  "duration_seconds", "description"]],
                        use_container_width=True,
                    )
            else:
                st.info("Run `python scripts/run_airflow_simulation.py` to generate DAG results.")

    # ── Explainability & Responsible AI ──────────────────────────────────────
    with st.expander("Explainability & Responsible AI"):
        c1, c2 = st.columns([1, 1], gap="large")
        with c1:
            st.markdown("""
Attribution methods provide visual explanations for individual predictions; they do not
establish that a prediction relies only on leaf tissue. Background sensitivity is reported
separately in the Diagnostic Screening trust panel using the API's returned comparison.

SHAP and LIME visualizations are available on demand in the Diagnostic Screening tab.
""")
        with c2:
            xai_f = FIGURES_DIR / "xai_trio_comparison.png"
            if xai_f.exists():
                st.image(
                    Image.open(xai_f),
                    caption="Grad-CAM / SHAP / LIME side-by-side on same leaf",
                    use_container_width=True,
                )
            else:
                st.info("Run `python scripts/benchmark_xai.py` to generate this figure.")
            st.markdown("""
**Governance documents**
- ISO/IEC 23894 AI Risk Register — intended use, failure modes, risk mitigations
- Gebru Datasheet — dataset provenance, demographic scope, known limitations
- Regulatory advisory caveats — every chemical recommendation carries a label
  caveat and PHI (pre-harvest interval) sourced from UC IPM, Cornell, ICAR
""")
        if st.button("View XAI benchmark report", key="xai_btn"):
            xai_rep = REPORTS_DIR / "xai_benchmark.json"
            if xai_rep.exists():
                st.json(json.loads(xai_rep.read_text()))
            else:
                st.info("Run `python scripts/benchmark_xai.py` to generate the report.")

    # ── Cloud Architecture ────────────────────────────────────────────────────
    with st.expander("Cloud Architecture (AWS SageMaker)"):
        c1, c2 = st.columns([1, 1], gap="large")
        with c1:
            st.markdown("""
**SageMaker pipeline** (`pipelines/sagemaker_pipeline.py`)

The same workflow that runs locally is compiled into an AWS SageMaker Pipeline
definition — a JSON graph of managed steps:

| Step | SageMaker Component | Local Equivalent |
|---|---|---|
| Data prep | `ProcessingStep` (SKLearn container) | `scripts/preprocess.py` |
| Training | `TrainingStep` (PyTorch container) | `make train` |
| Evaluation | `ProcessingStep` | `make evaluate` |
| Model registration | `ModelStep` → Model Registry | MLflow Registry |

**How to deploy to AWS**
1. Upload data to S3 (`s3://your-bucket/plantvillage/`)
2. Run `python pipelines/sagemaker_pipeline.py` — creates pipeline in SageMaker Studio
3. Trigger from Studio UI or `pipeline.start()`
4. Model endpoint auto-created after promotion gate passes

This is the standard pattern used for production ML at AWS customers in
agriculture, healthcare, and finance.
""")
        with c2:
            st.markdown("""
**Local → Cloud mapping**

| Local component | AWS | GCP | Azure |
|---|---|---|---|
| DVC | S3 remote | GCS remote | Blob Storage |
| MLflow | SageMaker Experiments | Vertex AI Experiments | Azure ML |
| FastAPI | SageMaker Endpoint | Vertex AI Endpoint | Azure ML Endpoint |
| Docker | ECR + ECS | Artifact Registry | ACR + ACI |
| Airflow DAG | MWAA | Cloud Composer | Azure Data Factory |
| Prometheus | CloudWatch | Cloud Monitoring | Azure Monitor |
""")
        if st.button("View compiled SageMaker pipeline JSON", key="sm_btn"):
            sm_f = REPORTS_DIR / "sagemaker_pipeline_definition.json"
            if sm_f.exists():
                st.json(json.loads(sm_f.read_text()))
            else:
                st.info("Run `python scripts/compile_sagemaker_pipeline.py`.")
