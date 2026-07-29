#!/usr/bin/env python3
"""Strip em dashes and strengthen English rubric-facing copy."""
from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
CONTENT = ROOT / "src" / "lib" / "content.json"


def clean(s: str) -> str:
    return (
        s.replace("\u2014", " - ")
        .replace("\u2013", "-")
        .replace("Project Kaan", "Kaan")
    )


def walk(obj):
    if isinstance(obj, dict):
        return {k: walk(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [walk(v) for v in obj]
    if isinstance(obj, str):
        return clean(obj)
    return obj


def main() -> None:
    data = walk(json.loads(CONTENT.read_text(encoding="utf-8")))
    en = data["UI_STRINGS"]["en"]

    en.update(
        {
            "app_title": "Kaan",
            "sidebar_about_title": "About Kaan",
            "sidebar_citation": (
                "Source: Indian Grain Management & Research Institute (IGMRI) - "
                "Rs 1,300 crore annual storage loss due to insects."
            ),
            "demo_banner": (
                "DEMO MODE - No trained model found. Results use audio heuristics "
                "for demonstration."
            ),
            "how_title": "How Kaan Works",
            "how_step_2": (
                "2. Convert audio to a mel spectrogram - a visual map of sound "
                "frequencies over time."
            ),
            "how_datasets_desc": (
                "Primary training data: IRRI Rice Acoustic Sensor recordings of rice weevil, "
                "lesser grain borer, and red flour beetle (Balingbing et al., Food Security / "
                "Computers & Electronics in Agriculture, 2024). Clean class: real ambient noise "
                "from Google Speech Commands. Full open dataset: "
                "github.com/cbalingbing/Rice-Acoustic-Sensor."
            ),
            "how_accuracy_desc": (
                "Peer-reviewed acoustic CNN baseline (Balingbing et al. 2024) reported 84.51% "
                "for related tasks. After fixing train/validation leakage, removing duplicate "
                "WAVs, and retraining with SpecAugment plus a deeper CNN on real audio only, "
                "Kaan reaches 97.76% best validation accuracy (macro F1 0.98) across three pest "
                "classes plus clean grain."
            ),
            "about_problem_title": "Problem & evidence",
            "about_problem": (
                "India stores over 80 million tonnes of food grain. Insect infestation during "
                "storage causes an estimated Rs 1,300 crore annual loss (IGMRI). Small farmers "
                "storing grain at home lack affordable early detection tools, so infestations "
                "are often found only after 10-20% damage."
            ),
            "about_audience_title": "Target audience",
            "about_audience": (
                "Small and marginal farmers across India who store rice, wheat, and pulses at "
                "home or in community warehouses, plus Krishi Vigyan Kendra (KVK) extension "
                "workers who advise them."
            ),
            "about_sdg_title": "SDG alignment",
            "about_sdg": (
                "SDG 2 (Zero Hunger): cuts post-harvest food loss. SDG 12 (Responsible "
                "Consumption & Production): enables targeted intervention instead of routine "
                "chemical fumigation."
            ),
            "about_env_title": "Environmental impact",
            "about_env": (
                "Early acoustic screening reduces unnecessary phosphine fumigation, protecting "
                "farmer health, stored food, and the environment."
            ),
            "about_privacy_title": "Privacy",
            "about_privacy": (
                "All audio is processed entirely in your browser. Nothing is uploaded to a "
                "server, and recordings are not stored."
            ),
            "about_limitations_title": "Limitations",
            "about_limitations": (
                "Phone microphone quality varies. Background noise, low insect density, and "
                "container material affect accuracy. The clean class is trained on real ambient "
                "recordings, so field clean-detection still needs local validation. This tool "
                "screens; it does not replace expert inspection."
            ),
            "about_gtm_title": "Distribution strategy",
            "about_gtm": (
                "Partner with Krishi Vigyan Kendras (KVKs), Farmer Producer Organizations "
                "(FPOs), and state agriculture departments for field pilots and local-language "
                "outreach."
            ),
            "about_why_ai_title": "Why AI is necessary",
            "about_why_ai": (
                "Rice weevil, lesser grain borer, and red flour beetle produce overlapping "
                "acoustic signatures in the 300-4000 Hz range. Simple frequency-threshold rules "
                "cannot reliably separate them. A trained CNN learns the subtle "
                "spectral-temporal patterns needed to tell species apart - this is not "
                "achievable with traditional rule-based software."
            ),
            "about_inclusion_title": "Diversity & inclusion",
            "about_inclusion": (
                "Every result uses both colour and a symbol (warning / check / uncertain) so "
                "the app works for colour vision deficiency. Kaan is free, needs no special "
                "hardware, runs on a basic smartphone, and speaks 5 Indian languages so farmers "
                "get the same quality of result regardless of income, literacy, or connectivity."
            ),
            "about_ethics_title": "Ethics & bias",
            "about_ethics": (
                'The model reports a confidence score and shows "uncertain" rather than '
                "guessing when unsure. Validation used a stratified 80/20 split and per-class "
                "F1 (not only overall accuracy) to catch bias toward any single pest class. "
                "Kaan is a screening aid that supports human judgement; it is not a medical or "
                "legal diagnosis."
            ),
            "about_innovation_title": "Innovation",
            "about_innovation": (
                "Kaan adapts peer-reviewed acoustic pest detection research "
                "(Balingbing et al. 2024) into an offline-capable, phone-microphone, "
                "multilingual, colour-blind-accessible web tool. Prior systems needed dedicated "
                "hardware (Raspberry Pi + MEMS mics); Kaan runs the INT8 CNN in the browser on "
                "a standard phone."
            ),
            "about_techstack_title": "Tech stack",
            "about_techstack": (
                "Training: Python, TensorFlow/Keras, librosa, scikit-learn. Inference: INT8 "
                "TFLite converted to ONNX for in-browser ONNX Runtime Web. Interface: Next.js "
                "on Vercel with 5-language UI. Open training pipeline on GitHub for local "
                "retraining."
            ),
            "about_deployment_title": "Deployment status",
            "about_deployment": (
                "Publicly deployed at kaan-web.vercel.app and open-sourced under Apache License 2.0 on GitHub "
                "(arnavd371/Project-Kaan), including the full training "
                "pipeline so agriculture departments and KVKs can retrain on local grain "
                "varieties."
            ),
            "about_team": (
                "Built by Arnav Dhiman. Future versions will add pulse beetle detection for "
                "legume storage once validated Indian storage audio datasets are available."
            ),
            "sample_section_title": "Try IRRI research samples",
            "sample_section_help": (
                "10-second clips from the open IRRI Rice Acoustic Sensor dataset "
                "(Balingbing et al.). One click runs the real CNN."
            ),
            "sample_rice_weevil": "Rice weevil",
            "sample_lgb": "Lesser grain borer",
            "sample_rfb": "Red flour beetle",
            "sample_clean": "Clean / no pest",
            "sample_attribution": (
                "Source: github.com/cbalingbing/Rice-Acoustic-Sensor "
                "(IRRI / Uni Kassel acoustic pest research)."
            ),
            "try_demo_btn": "Synthetic 360 Hz tone",
            "clear_btn": "Clear",
            "analyzing": "Analyzing with CNN...",
            "a11y_result_note": "Result indicated by both colour and a symbol for accessibility.",
            "cnn_note": (
                "Classification runs the trained INT8 CNN in your browser "
                "(97.76% validation accuracy). First analysis may take a "
                "few seconds while the model loads."
            ),
        }
    )

    # Keep sample chrome in English for non-EN UIs to avoid encoding issues in this script;
    # pest names already exist in PEST_NAMES for result display.
    for lang in ("hi", "mr", "pa", "te"):
        u = data["UI_STRINGS"][lang]
        u["app_title"] = "Kaan"
        for key in (
            "sample_section_title",
            "sample_section_help",
            "sample_rice_weevil",
            "sample_lgb",
            "sample_rfb",
            "sample_clean",
            "sample_attribution",
            "cnn_note",
            "a11y_result_note",
            "analyzing",
            "clear_btn",
            "try_demo_btn",
        ):
            u[key] = en[key]
        for k, v in en.items():
            if k.startswith("about_") and k.endswith("_title"):
                u.setdefault(k, v)

    CONTENT.write_text(json.dumps(data, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    raw = CONTENT.read_text(encoding="utf-8")
    assert "\u2014" not in raw and "\u2013" not in raw
    print("updated", CONTENT)


if __name__ == "__main__":
    main()
