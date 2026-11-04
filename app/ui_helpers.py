"""Pure UI helpers for trust data, language fallback, and optional explainability."""

from __future__ import annotations

import json
import importlib
import importlib.util
from pathlib import Path
from typing import Any


def import_optional_explainer(module_name: str) -> Any | None:
    """Import an optional XAI module on demand; return None when it is not installed."""
    try:
        return importlib.import_module(module_name)
    except ImportError:
        return None


def optional_explainer_available(module_name: str) -> bool:
    """Check optional package availability without importing it."""
    try:
        return importlib.util.find_spec(module_name) is not None
    except (ImportError, ModuleNotFoundError, ValueError):
        return False


def saved_explainer_figures(figures_dir: Path, explainer: str) -> list[Path]:
    """Find committed offline explanation figures without requiring XAI packages."""
    if explainer == "shap":
        return sorted((figures_dir / "shap").glob("compare_*.png"))
    if explainer == "lime":
        trio = figures_dir / "xai_trio_comparison.png"
        return [trio] if trio.is_file() else []
    return []


SUPPORTED_EXPLANATION_LANGUAGES = {
    "English": "en",
    "मराठी (Marathi)": "mr",
    "हिन्दी (Hindi)": "hi",
}

# These draft translations cover generic interface explanations only, not agronomic guidance.
# Source-backed advice stays in its cited language until a fluent reviewer approves a translation.
EXPLANATION_TEXT: dict[str, dict[str, str]] = {
    "en": {
        "explainability_title": "Where the model looked — Explainability",
        "explanation_scope": (
            "Attribution maps show image regions that influenced this model output. They do not "
            "prove a disease, explain its cause, or establish that the model relied only on leaf "
            "tissue. Background sensitivity is a separate check."
        ),
        "prediction_line": "Model output: {label} · Unscaled confidence: {confidence}",
        "gradcam_caption": (
            "Highlighted regions show where this model's activation was strong for this output. "
            "They do not prove the diagnosis is correct or explain why the disease occurred."
        ),
        "no_gradcam": "Grad-CAM is not available in the API response.",
        "advice_passport": "Advice passport",
        "draft_translation_notice": (
            "Marathi/Hindi interface text is a draft and needs fluent review. Cited advice remains "
            "in its source language."
        ),
        "no_reviewed_advice": (
            "No reviewed, source-backed advice is available for this prediction. Treatment advice "
            "is withheld; consult a local agricultural expert."
        ),
        "consult_expert": "Consult a local agricultural expert.",
        "risk_unknown": "Weather-related risk: unknown.",
        "confidence_label": "Unscaled confidence",
        "background_sensitivity": "Background sensitivity",
        "offline_figure": "{name} — saved offline comparison; not generated for this upload",
        "explainer_not_installed": (
            "{name} is not installed in this image; saved offline comparisons are shown below."
        ),
        "trust_heading": "Prediction trust and background sensitivity",
        "returned_label": "Returned label",
        "ood_status": "OOD status",
        "quality_not_available": "Image quality: not available from the API.",
        "quality_values": "Image quality values returned by API — brightness: {brightness}; blur score: {blur}.",
        "original_prediction": "Original prediction: {label}; confidence: {confidence}",
        "neutralized_prediction": "Mean-fill neutralized prediction: {label}; confidence: {confidence}",
        "neutralized_not_available": "Mean-fill neutralized prediction: not available.",
        "confidence_change": "Confidence change after mean-fill neutralization: {change}.",
        "advice_disclaimer": (
            "Decision support only; this prediction is not a substitute for advice from a local "
            "agricultural expert or an official product label."
        ),
    },
    "mr": {
        "explainability_title": "मॉडेलने प्रतिमेत कुठे लक्ष दिले — स्पष्टीकरण",
        "explanation_scope": (
            "हे वैशिष्ट्य-नकाशे मॉडेलच्या या निकालावर प्रभाव टाकलेल्या प्रतिमेतील भाग दाखवतात. "
            "ते रोग सिद्ध करत नाहीत, रोगाचे कारण सांगत नाहीत किंवा मॉडेलने फक्त पानाचाच वापर "
            "केला हे सिद्ध करत नाहीत. पार्श्वभूमी-संवेदनशीलता ही वेगळी तपासणी आहे."
        ),
        "prediction_line": "मॉडेलचा निकाल: {label} · न बदललेला विश्वास-स्तर: {confidence}",
        "gradcam_caption": (
            "हायलाइट केलेले भाग या निकालासाठी मॉडेलची सक्रियता कुठे जास्त होती ते दाखवतात. "
            "यामुळे निदान बरोबर आहे किंवा रोगाचे कारण समजते असे सिद्ध होत नाही."
        ),
        "no_gradcam": "API प्रतिसादात Grad-CAM उपलब्ध नाही.",
        "advice_passport": "सल्ल्याचा स्रोत आणि मर्यादा",
        "draft_translation_notice": (
            "मराठी/हिंदीतील मजकूर हा मसुदा आहे; वापरण्यापूर्वी भाषातज्ज्ञाकडून तपासून घ्या. "
            "स्रोतासह दिलेला सल्ला मूळ भाषेतच राहतो."
        ),
        "no_reviewed_advice": (
            "या अंदाजासाठी तपासलेला, स्रोताधारित सल्ला उपलब्ध नाही. उपचाराचा सल्ला दिलेला नाही; "
            "स्थानिक कृषी तज्ज्ञांचा सल्ला घ्या."
        ),
        "consult_expert": "स्थानिक कृषी तज्ज्ञांचा सल्ला घ्या.",
        "risk_unknown": "हवामानाशी संबंधित धोका: अज्ञात.",
        "confidence_label": "न बदललेला विश्वास-स्तर",
        "background_sensitivity": "पार्श्वभूमी-संवेदनशीलता",
        "offline_figure": "{name} — आधी जतन केलेली तुलना; या अपलोडसाठी तयार केलेली नाही",
        "explainer_not_installed": (
            "या image मध्ये {name} स्थापित नाही; खाली आधी जतन केलेल्या तुलना दाखवल्या आहेत."
        ),
        "trust_heading": "अंदाजावरील विश्वास आणि पार्श्वभूमी-संवेदनशीलता",
        "returned_label": "परत आलेले लेबल",
        "ood_status": "OOD स्थिती",
        "quality_not_available": "API कडून प्रतिमेची गुणवत्ता उपलब्ध नाही.",
        "quality_values": "API ने दिलेली प्रतिमा-गुणवत्ता — brightness: {brightness}; blur score: {blur}.",
        "original_prediction": "मूळ अंदाज: {label}; विश्वास-स्तर: {confidence}",
        "neutralized_prediction": "सरासरी-रंग पार्श्वभूमीनंतरचा अंदाज: {label}; विश्वास-स्तर: {confidence}",
        "neutralized_not_available": "सरासरी-रंग पार्श्वभूमीनंतरचा अंदाज उपलब्ध नाही.",
        "confidence_change": "सरासरी-रंग बदलानंतर विश्वास-स्तरातील बदल: {change}.",
        "advice_disclaimer": (
            "हा फक्त निर्णय-सहाय्यक अंदाज आहे; तो स्थानिक कृषी तज्ज्ञांच्या सल्ल्याचा किंवा "
            "अधिकृत उत्पादनाच्या लेबलाचा पर्याय नाही."
        ),
    },
    "hi": {
        "explainability_title": "मॉडल ने तस्वीर में कहाँ ध्यान दिया — व्याख्या",
        "explanation_scope": (
            "ये एट्रिब्यूशन नक्शे तस्वीर के उन हिस्सों को दिखाते हैं जिन्होंने इस मॉडल आउटपुट को "
            "प्रभावित किया। ये बीमारी साबित नहीं करते, उसका कारण नहीं बताते, और यह भी साबित नहीं "
            "करते कि मॉडल ने केवल पत्ती का उपयोग किया। बैकग्राउंड संवेदनशीलता एक अलग जाँच है।"
        ),
        "prediction_line": "मॉडल आउटपुट: {label} · बिना बदला confidence: {confidence}",
        "gradcam_caption": (
            "हाइलाइट किए गए हिस्से दिखाते हैं कि इस आउटपुट के लिए मॉडल की सक्रियता कहाँ अधिक थी। "
            "इससे निदान सही साबित नहीं होता और न ही बीमारी का कारण पता चलता है।"
        ),
        "no_gradcam": "API response में Grad-CAM उपलब्ध नहीं है।",
        "advice_passport": "सलाह का स्रोत और सीमाएँ",
        "draft_translation_notice": (
            "मराठी/हिंदी का इंटरफ़ेस पाठ मसौदा है; उपयोग से पहले भाषा विशेषज्ञ से जाँच कराएँ। "
            "स्रोत सहित सलाह मूल भाषा में ही रहती है।"
        ),
        "no_reviewed_advice": (
            "इस अनुमान के लिए समीक्षित, स्रोत-आधारित सलाह उपलब्ध नहीं है। उपचार संबंधी सलाह रोकी "
            "गई है; स्थानीय कृषि विशेषज्ञ से सलाह लें।"
        ),
        "consult_expert": "स्थानीय कृषि विशेषज्ञ से सलाह लें।",
        "risk_unknown": "मौसम से जुड़ा जोखिम: अज्ञात।",
        "confidence_label": "बिना बदला confidence",
        "background_sensitivity": "बैकग्राउंड संवेदनशीलता",
        "offline_figure": "{name} — पहले से सुरक्षित तुलना; इस अपलोड के लिए नहीं बनाई गई",
        "explainer_not_installed": (
            "इस image में {name} इंस्टॉल नहीं है; नीचे पहले से सुरक्षित तुलनाएँ दिखाई गई हैं।"
        ),
        "trust_heading": "अनुमान पर भरोसा और बैकग्राउंड संवेदनशीलता",
        "returned_label": "वापस मिला लेबल",
        "ood_status": "OOD स्थिति",
        "quality_not_available": "API से तस्वीर की गुणवत्ता उपलब्ध नहीं है।",
        "quality_values": "API द्वारा दिए गए तस्वीर-गुणवत्ता मान — brightness: {brightness}; blur score: {blur}।",
        "original_prediction": "मूल अनुमान: {label}; confidence: {confidence}",
        "neutralized_prediction": "औसत-रंग बैकग्राउंड के बाद अनुमान: {label}; confidence: {confidence}",
        "neutralized_not_available": "औसत-रंग बैकग्राउंड के बाद का अनुमान उपलब्ध नहीं है।",
        "confidence_change": "औसत-रंग बदलाव के बाद confidence में परिवर्तन: {change}।",
        "advice_disclaimer": (
            "यह केवल निर्णय-सहायता के लिए अनुमान है; यह स्थानीय कृषि विशेषज्ञ की सलाह या "
            "आधिकारिक उत्पाद लेबल का विकल्प नहीं है।"
        ),
    },
}


def explanation_text(key: str, language: str, **values: Any) -> tuple[str, bool]:
    """Return selected UI copy and whether a non-English translation is reviewed."""
    selected = EXPLANATION_TEXT.get(language, EXPLANATION_TEXT["en"])
    text = selected.get(key, EXPLANATION_TEXT["en"].get(key, key))
    try:
        text = text.format(**values)
    except (KeyError, ValueError):
        pass
    return text, language == "en"


def prediction_explanation_values(payload: dict[str, Any]) -> dict[str, Any]:
    """Expose prediction/evidence unchanged for explanation views."""
    label = payload.get("predicted_class")
    if not label and payload.get("crop") and payload.get("condition"):
        label = f"{payload['crop']}___{payload['condition']}"
    if not label:
        label = "not available"
    return {
        "label": label,
        "confidence": payload.get("confidence"),
        "gradcam_available": bool(payload.get("gradcam_png_base64")),
        "advice_trace": payload.get("advice_trace"),
    }


def parse_prediction_trust(payload: dict[str, Any]) -> dict[str, Any]:
    """Extract only fields the API returned; absent values remain unavailable."""
    crop = payload.get("crop")
    condition = payload.get("condition")
    label = payload.get("predicted_class")
    if not label and crop and condition:
        label = f"{crop}___{condition}"
    if not label:
        label = condition or "not available"
    confidence = payload.get("confidence")
    if not isinstance(confidence, (int, float)):
        confidence = None
    quality = payload.get("quality")
    if not isinstance(quality, dict):
        quality = None
    details = payload.get("bg_check_details")
    if not isinstance(details, dict):
        details = {}
    bg_check = payload.get("bg_check")
    if bg_check not in {"consistent", "inconsistent", "unknown"}:
        bg_check = "unknown"
    return {
        "label": label,
        "confidence": float(confidence) if confidence is not None else None,
        "quality": quality,
        "ood_status": payload.get("ood_status") or "unknown",
        "bg_check": bg_check,
        "bg_check_details": details,
    }


def _read_json(path: Path) -> tuple[Any | None, str | None]:
    if not path.exists():
        return None, "not generated yet"
    try:
        return json.loads(path.read_text(encoding="utf-8")), None
    except (OSError, json.JSONDecodeError) as exc:
        return None, f"could not read saved report: {exc}"


def _model_rows(reports_dir: Path) -> tuple[list[dict[str, Any]], str | None]:
    rows: list[dict[str, Any]] = []
    seen_run_ids: set[str] = set()
    seed_data, seed_error = _read_json(reports_dir / "seed_variance.json")
    if isinstance(seed_data, dict):
        for role, model in seed_data.get("models", {}).items():
            if not isinstance(model, dict):
                continue
            run_id = str(model.get("run_id") or "not available")
            if run_id != "not available":
                seen_run_ids.add(run_id)
            rows.append({"role": role, **model})

    candidate_files = sorted(reports_dir.glob("candidate_gate_*.json"))
    for report_path in candidate_files:
        report, _ = _read_json(report_path)
        if not isinstance(report, dict):
            continue
        for candidate in report.get("results", []):
            if not isinstance(candidate, dict):
                continue
            run_id = str(candidate.get("run_id") or "not available")
            if run_id in seen_run_ids:
                continue
            seen_run_ids.add(run_id)
            rows.append({
                "role": candidate.get("name") or "candidate",
                "run_id": run_id,
                "seed": candidate.get("seed", "not available"),
                "clean_macro_f1": candidate.get("clean_macro_f1", "not available"),
                "within_crop_macro_f1": candidate.get("within_crop_macro_f1", "not available"),
                "min_recall": candidate.get("min_recall", {}).get("value", "not available"),
                "ece": candidate.get("ece", "not available"),
                "report": report_path.name,
            })
    if rows:
        return rows, None
    if seed_error:
        return [], seed_error
    return [], "not generated yet"


def load_saved_mlops_evidence(reports_dir: Path, health: dict[str, Any] | None) -> dict[str, Any]:
    """Load only committed/generated report files and the supplied /health response."""
    model_rows, model_message = _model_rows(reports_dir)

    promotion_path = reports_dir / "promotion_log.jsonl"
    promotions = []
    promotion_message = None
    if not promotion_path.exists():
        promotion_message = "not generated yet"
    else:
        try:
            for line in promotion_path.read_text(encoding="utf-8").splitlines():
                if line.strip():
                    promotions.append(json.loads(line))
        except (OSError, json.JSONDecodeError) as exc:
            promotions = []
            promotion_message = f"could not read saved report: {exc}"
        if not promotions and promotion_message is None:
            promotion_message = "not generated yet"

    drift, drift_message = _read_json(reports_dir / "drift_latest.json")
    serving_version = None
    if isinstance(health, dict):
        serving_version = health.get("model_version")

    return {
        "models": model_rows,
        "models_message": model_message,
        "promotions": promotions,
        "promotions_message": promotion_message,
        "drift": drift,
        "drift_message": drift_message,
        "serving_model_version": serving_version,
        "serving_message": None if serving_version is not None else "not available from /health",
    }
