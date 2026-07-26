"""Kaan (कान) -- Acoustic grain pest detector for Indian farmers."""

from __future__ import annotations

import hashlib
import io
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import librosa
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st

sys.path.insert(0, str(Path(__file__).resolve().parent))

from utils.inference import ProjectKaanPredictor, estimate_severity
from utils.language import LANGUAGES, get_advisory, get_pest_name, get_ui
from utils.visualizer import plot_spectrogram

_DEFAULTS = {
    "page": "intro",
    "lang": "en",
    "audio_bytes": None,
    "audio_id": None,
    "upload_file_id": None,
    "result": None,
    "severity": None,
    "pending_analyze": False,
}
for _key, _val in _DEFAULTS.items():
    if _key not in st.session_state:
        st.session_state[_key] = _val

GITHUB_REPO_URL = "https://github.com/arnavd371/Project-Kaan"

# Physical Intelligence design tokens
BG = "#F7F6F2"
SURFACE = "#ffffff"
TEXT = "#000000"
MUTED = "#666666"
LINE = "#d1d1d1"
DANGER = "#c0392b"
WARN = "#8a6d3b"
SUCCESS = "#1e7a3a"
INK = "#000000"
SHADOW = "5px 5px 0 #000000"
CHART = "#000000"

LANG_BUTTONS = [
    ("en", "English"),
    ("hi", "हिंदी"),
    ("mr", "मराठी"),
    ("pa", "ਪੰਜਾਬੀ"),
    ("te", "తెలుగు"),
]

MANIFESTO = {
    "en": (
        "Somewhere in rural India, a farmer is checking his grain by hand right now. "
        "He presses his palm into the sack, feels for heat, smells for something off. "
        "This is how it has been done for generations. It works, until it does not. "
        "By the time the smell is wrong, weeks of damage have already happened. "
        "We built Kaan because the farmer deserves a warning, not a discovery. "
        "Because early is the difference between saving the harvest and losing it. "
        "Because a phone is the only tool that reaches everywhere. "
        "Kaan listens to your grain so you do not have to wonder."
    ),
    "hi": (
        "भारत के किसी गांव में अभी एक किसान अपना अनाज हाथ से जांच रहा है। "
        "वह थैले में हाथ डालता है, गर्मी महसूस करता है, कुछ अजीब गंध की तलाश करता है। "
        "यही तरीका पीढ़ियों से चला आ रहा है। यह काम करता है, जब तक नहीं करता। "
        "जब तक गंध गलत लगती है, हफ्तों का नुकसान हो चुका होता है। "
        "हमने कान बनाया क्योंकि किसान एक चेतावनी का हकदार है, खोज का नहीं। "
        "क्योंकि जल्दी पता चलना फसल बचाने और गंवाने के बीच का फर्क है। "
        "क्योंकि फोन ही एकमात्र औजार है जो हर जगह पहुंचता है। "
        "कान आपके अनाज को सुनता है ताकि आपको अंदाजा न लगाना पड़े।"
    ),
    "mr": (
        "भारताच्या कुठल्यातरी खेड्यात आत्ता एक शेतकरी आपले धान्य हाताने तपासत आहे. "
        "तो पोत्यात हात घालतो, उष्णता जाणवतो, काहीतरी वेगळा वास शोधतो. "
        "हे पिढ्यानपिढ्या असेच होत आले आहे. हे काम करते, जोपर्यंत करत नाही. "
        "जेव्हा वास चुकीचा वाटतो, तोपर्यंत आठवड्यांचे नुकसान झालेले असते. "
        "आम्ही कान बनवले कारण शेतकऱ्याला इशारा मिळायला हवा, शोध नको. "
        "कारण लवकर कळणे म्हणजे पीक वाचवणे आणि गमावणे यातील फरक आहे. "
        "कारण फोन हेच एकमेव साधन आहे जे सर्वत्र पोहोचते. "
        "कान तुमचे धान्य ऐकतो जेणेकरून तुम्हाला अंदाज करावा लागणार नाही."
    ),
    "pa": (
        "ਭਾਰਤ ਦੇ ਕਿਸੇ ਪਿੰਡ ਵਿੱਚ ਹੁਣੇ ਇੱਕ ਕਿਸਾਨ ਆਪਣਾ ਅਨਾਜ ਹੱਥ ਨਾਲ ਜਾਂਚ ਰਿਹਾ ਹੈ। "
        "ਉਹ ਬੋਰੀ ਵਿੱਚ ਹੱਥ ਪਾਉਂਦਾ ਹੈ, ਗਰਮੀ ਮਹਿਸੂਸ ਕਰਦਾ ਹੈ, ਕੁਝ ਅਜੀਬ ਗੰਧ ਲੱਭਦਾ ਹੈ। "
        "ਇਹੀ ਤਰੀਕਾ ਪੀੜ੍ਹੀਆਂ ਤੋਂ ਚੱਲਦਾ ਆ ਰਿਹਾ ਹੈ। ਇਹ ਕੰਮ ਕਰਦਾ ਹੈ, ਜਦੋਂ ਤੱਕ ਨਹੀਂ ਕਰਦਾ। "
        "ਜਦੋਂ ਤੱਕ ਗੰਧ ਗਲਤ ਲੱਗਦੀ ਹੈ, ਹਫ਼ਤਿਆਂ ਦਾ ਨੁਕਸਾਨ ਹੋ ਚੁੱਕਾ ਹੁੰਦਾ ਹੈ। "
        "ਅਸੀਂ ਕਾਨ ਬਣਾਇਆ ਕਿਉਂਕਿ ਕਿਸਾਨ ਚੇਤਾਵਨੀ ਦਾ ਹੱਕਦਾਰ ਹੈ, ਖੋਜ ਦਾ ਨਹੀਂ। "
        "ਕਿਉਂਕਿ ਜਲਦੀ ਪਤਾ ਲੱਗਣਾ ਫਸਲ ਬਚਾਉਣ ਅਤੇ ਗੁਆਉਣ ਵਿਚਕਾਰ ਫਰਕ ਹੈ। "
        "ਕਿਉਂਕਿ ਫੋਨ ਹੀ ਇੱਕੋ ਔਜ਼ਾਰ ਹੈ ਜੋ ਹਰ ਜਗ੍ਹਾ ਪਹੁੰਚਦਾ ਹੈ। "
        "ਕਾਨ ਤੁਹਾਡਾ ਅਨਾਜ ਸੁਣਦਾ ਹੈ ਤਾਂ ਜੋ ਤੁਹਾਨੂੰ ਅੰਦਾਜ਼ਾ ਨਾ ਲਗਾਉਣਾ ਪਵੇ।"
    ),
    "te": (
        "భారతదేశంలోని ఏదో ఒక గ్రామంలో ఇప్పుడు ఒక రైతు తన ధాన్యాన్ని చేతితో తనిఖీ చేస్తున్నాడు. "
        "అతను సంచిలో చేయి వేస్తాడు, వేడిని అనుభవిస్తాడు, ఏదైనా వింత వాసన వస్తుందా అని చూస్తాడు. "
        "ఇది తరతరాలుగా జరుగుతూ వస్తోంది. ఇది పనిచేస్తుంది, అది పనిచేయనంత వరకు. "
        "వాసన తప్పుగా అనిపించే సమయానికి, వారాల నష్టం జరిగిపోయి ఉంటుంది. "
        "మేము కాన్ నిర్మించాము ఎందుకంటే రైతుకు హెచ్చరిక రావాలి, కనుగొనడం కాదు. "
        "ఎందుకంటే త్వరగా తెలుసుకోవడం పంటను కాపాడుకోవడానికి మరియు పోగొట్టుకోవడానికి మధ్య తేడా. "
        "ఎందుకంటే ఫోన్ మాత్రమే అన్నిచోట్లా చేరే ఏకైక సాధనం. "
        "కాన్ మీ ధాన్యాన్ని వింటుంది, మీరు అంచనా వేయకుండా ఉండేందుకు."
    ),
}

CTA_LISTEN = {
    "en": "Listen to your grain",
    "hi": "अपना अनाज सुनें",
    "mr": "तुमचे धान्य ऐका",
    "pa": "ਆਪਣਾ ਅਨਾਜ ਸੁਣੋ",
    "te": "మీ ధాన్యాన్ని వినండి",
}

BACK_LABEL = {
    "en": "Back to intro",
    "hi": "परिचय पर वापस जाएं",
    "mr": "परिचयाकडे परत जा",
    "pa": "ਜਾਣ-ਪਛਾਣ ਤੇ ਵਾਪਸ ਜਾਓ",
    "te": "పరిచయానికి తిరిగి వెళ్ళండి",
}

TAGLINE = "It hears what you cannot."

SEVERITY_LEVEL_NAMES = {
    "Early": {
        "en": "Early",
        "hi": "प्रारंभिक",
        "mr": "सुरुवातीचा",
        "pa": "ਸ਼ੁਰੂਆਤੀ",
        "te": "ప్రారంభ",
    },
    "Moderate": {
        "en": "Moderate",
        "hi": "मध्यम",
        "mr": "मध्यम",
        "pa": "ਦਰਮਿਆਨਾ",
        "te": "మధ్యస్థ",
    },
    "Severe": {
        "en": "Severe",
        "hi": "गंभीर",
        "mr": "गंभीर",
        "pa": "ਗੰਭੀਰ",
        "te": "తీవ్రమైన",
    },
}
SEVERITY_ORDER = ["Early", "Moderate", "Severe"]

st.set_page_config(
    page_title="Kaan",
    page_icon="data:image/svg+xml,<svg xmlns='http://www.w3.org/2000/svg'/>",
    layout="wide",
    initial_sidebar_state="collapsed",
)

CUSTOM_CSS = f"""
<style>
@import url('https://fonts.googleapis.com/css2?family=IBM+Plex+Mono:wght@400;500;600;700&family=Libre+Baskerville:wght@400;700&display=swap');

:root {{
  --pi-bg: {BG};
  --pi-surface: {SURFACE};
  --pi-text: {TEXT};
  --pi-muted: {MUTED};
  --pi-line: {LINE};
  --pi-shadow: {SHADOW};
  --pi-serif: "Libre Baskerville", "Times New Roman", Times, Georgia, serif;
  --pi-mono: "IBM Plex Mono", "Courier New", Courier, monospace;
}}

html, body, [data-testid="stAppViewContainer"], .stApp {{
  background: var(--pi-bg) !important;
  color: var(--pi-text) !important;
  font-family: var(--pi-mono) !important;
}}

[data-testid="stHeader"], [data-testid="stToolbar"] {{
  background: transparent !important;
}}

.block-container {{
  padding-top: 1.75rem !important;
  padding-bottom: 3.5rem !important;
  max-width: 56rem !important;
}}

[data-testid="stSidebar"] {{
  background: {BG} !important;
  border-right: 1px solid var(--pi-line) !important;
}}

[data-testid="stSidebar"] * {{
  font-family: var(--pi-mono) !important;
  color: var(--pi-text) !important;
}}

h1 {{
  font-family: var(--pi-serif) !important;
  color: #000 !important;
  font-weight: 700 !important;
  letter-spacing: -0.01em !important;
}}

h2, h3, h4 {{
  font-family: var(--pi-mono) !important;
  color: #000 !important;
  font-weight: 700 !important;
}}

p, label, li, .stMarkdown, .stCaption,
span:not([data-testid="stIconMaterial"]) {{
  font-family: var(--pi-mono) !important;
}}

[data-testid="stIconMaterial"][data-testid="stIconMaterial"][data-testid="stIconMaterial"] {{
  font-family: "Material Symbols Rounded" !important;
}}

.kaan-intro {{
  max-width: 44rem;
  margin: 0 auto;
  padding: 2rem 0.5rem 3rem;
  text-align: left;
}}

.kaan-name {{
  margin: 0;
  font-family: var(--pi-serif);
  font-size: clamp(3.5rem, 10vw, 5.5rem);
  font-weight: 700;
  line-height: 1;
  color: #000;
  text-align: center;
}}

.kaan-devanagari {{
  margin: 0.4rem 0 0;
  font-family: var(--pi-mono);
  font-size: 1.15rem;
  color: var(--pi-muted);
  text-align: center;
  letter-spacing: 0.12em;
}}

.kaan-tagline {{
  margin: 1.1rem 0 0;
  font-family: var(--pi-mono);
  font-size: 1rem;
  color: #000;
  text-align: center;
  font-weight: 500;
}}

.kaan-manifesto {{
  margin: 1.75rem 0 0;
  font-family: var(--pi-mono);
  font-size: 0.95rem;
  line-height: 1.75;
  color: var(--pi-muted);
  background: var(--pi-surface);
  border: 1px solid #000;
  box-shadow: var(--pi-shadow);
  border-radius: 0;
  padding: 1.25rem 1.35rem;
}}

.lang-row {{
  margin: 1.25rem 0 0.25rem;
}}

.kaan-tool-header {{
  margin-bottom: 1.35rem;
  padding-bottom: 1rem;
  border-bottom: 1px solid var(--pi-line);
}}

.kaan-tool-header h1 {{
  font-family: var(--pi-serif) !important;
  font-size: 2.6rem !important;
  margin: 0 !important;
}}

.kaan-tool-header p {{
  margin: 0.45rem 0 0;
  color: var(--pi-muted);
  font-family: var(--pi-mono);
  font-size: 0.9rem;
}}

.panel, .advisory, .demo-banner, .record-box {{
  background: var(--pi-surface);
  border: 1px solid #000;
  border-radius: 0;
  box-shadow: var(--pi-shadow);
  padding: 1rem 1.1rem;
  margin: 0.75rem 0 1rem;
}}

.panel-label {{
  display: block;
  font-family: var(--pi-mono);
  font-size: 0.72rem;
  letter-spacing: 0.08em;
  text-transform: uppercase;
  color: var(--pi-muted);
  font-weight: 700;
  margin-bottom: 0.35rem;
}}

.panel p, .advisory {{
  color: var(--pi-muted);
  font-family: var(--pi-mono);
  font-size: 0.88rem;
  line-height: 1.65;
}}

.record-box {{
  border-style: dashed;
  white-space: pre-line;
  color: #000;
  font-size: 0.88rem;
  line-height: 1.7;
}}

.demo-banner {{
  border-color: {WARN};
  box-shadow: 5px 5px 0 {WARN};
  color: #000;
  font-size: 0.88rem;
}}

.result-pest, .result-clean, .result-low {{
  background: var(--pi-surface);
  border-radius: 0;
  padding: 1.1rem 1.2rem;
  margin: 1rem 0;
}}

.result-pest {{
  border: 2px solid {DANGER};
  box-shadow: 5px 5px 0 {DANGER};
}}

.result-clean {{
  border: 2px solid {SUCCESS};
  box-shadow: 5px 5px 0 {SUCCESS};
}}

.result-low {{
  border: 2px solid {WARN};
  box-shadow: 5px 5px 0 {WARN};
}}

.result-pest h2, .result-clean h2, .result-low h2 {{
  margin: 0 0 0.3rem !important;
  font-family: var(--pi-mono) !important;
  font-size: 1rem !important;
  text-transform: uppercase;
  letter-spacing: 0.04em;
}}

.result-pest h2 {{ color: {DANGER} !important; }}
.result-clean h2 {{ color: {SUCCESS} !important; }}
.result-low h2 {{ color: {WARN} !important; }}

.result-pest h3, .result-clean h3, .result-low h3 {{
  margin: 0 !important;
  font-family: var(--pi-mono) !important;
  font-size: 0.95rem !important;
  color: #000 !important;
}}

.result-low p {{
  margin: 0.4rem 0 0;
  color: var(--pi-muted);
  font-family: var(--pi-mono);
  font-size: 0.85rem;
}}

.severity-box {{
  background: var(--pi-surface);
  border: 1px solid #000;
  box-shadow: var(--pi-shadow);
  border-radius: 0;
  padding: 1rem 1.1rem;
  margin: 0.75rem 0 1rem;
}}

.severity-meter {{
  display: flex;
  gap: 0.5rem;
  margin: 0.6rem 0 1rem;
}}

.severity-segment {{
  flex: 1;
  text-align: center;
  padding: 0.5rem 0.4rem;
  border: 1px solid #000;
  font-family: var(--pi-mono);
  font-size: 0.72rem;
  font-weight: 700;
  text-transform: uppercase;
  letter-spacing: 0.04em;
  color: var(--pi-muted);
  background: var(--pi-surface);
}}

.severity-segment.active {{
  color: #fff;
}}

.severity-level {{
  font-size: 1.6rem;
  font-weight: 700;
  margin: 0.2rem 0 0.6rem;
  font-family: var(--pi-mono);
  color: #000;
}}

.severity-box p {{
  margin: 0.35rem 0 0;
  color: var(--pi-muted);
  font-family: var(--pi-mono);
  font-size: 0.88rem;
  line-height: 1.6;
}}

.pi-timeline {{
  position: relative;
  margin: 1rem 0 1.5rem;
  padding-left: 1.35rem;
  border-left: 1px solid var(--pi-line);
}}

.pi-timeline-item {{
  position: relative;
  margin: 0 0 1.1rem;
}}

.pi-timeline-item::before {{
  content: "";
  position: absolute;
  left: -1.48rem;
  top: 0.35rem;
  width: 0.45rem;
  height: 0.45rem;
  background: #000;
}}

.pi-timeline-row {{
  display: flex;
  justify-content: space-between;
  gap: 1rem;
  align-items: baseline;
}}

.pi-timeline-title {{
  font-family: var(--pi-mono);
  font-weight: 700;
  font-size: 0.92rem;
  color: #000;
}}

.pi-timeline-meta {{
  font-family: var(--pi-mono);
  font-size: 0.78rem;
  color: var(--pi-muted);
}}

.pi-timeline-body {{
  margin: 0.35rem 0 0;
  color: var(--pi-muted);
  font-family: var(--pi-mono);
  font-size: 0.85rem;
  line-height: 1.65;
}}

.sidebar-card {{
  background: var(--pi-surface);
  border: 1px solid #000;
  box-shadow: var(--pi-shadow);
  border-radius: 0;
  padding: 0.9rem 1rem;
  margin: 0.75rem 0;
  font-family: var(--pi-mono);
  font-size: 0.82rem;
  line-height: 1.55;
  color: var(--pi-muted);
}}

.sidebar-logo {{
  font-family: var(--pi-serif) !important;
  font-size: 1.7rem !important;
  font-weight: 700 !important;
  margin: 0 !important;
  color: #000 !important;
}}

.stButton > button {{
  border-radius: 0 !important;
  border: 1px solid #000 !important;
  background: var(--pi-surface) !important;
  color: #000 !important;
  font-family: var(--pi-mono) !important;
  font-weight: 600 !important;
  box-shadow: 3px 3px 0 #000 !important;
  transition: transform 80ms ease, box-shadow 80ms ease !important;
}}

.stButton > button:hover {{
  transform: translate(1px, 1px);
  box-shadow: 2px 2px 0 #000 !important;
  background: #000 !important;
  color: {BG} !important;
}}

.stButton > button[kind="primary"],
.stButton > button[data-testid="baseButton-primary"] {{
  background: #000 !important;
  color: {BG} !important;
  border: 1px solid #000 !important;
  box-shadow: 3px 3px 0 #666 !important;
}}

.stButton > button[kind="primary"]:hover {{
  background: var(--pi-surface) !important;
  color: #000 !important;
  box-shadow: 2px 2px 0 #000 !important;
}}

[data-testid="stFileUploaderDropzone"] {{
  border-radius: 0 !important;
  border: 2px dashed #000 !important;
  background: var(--pi-surface) !important;
}}

[data-testid="stFileUploaderDropzone"] * {{
  font-family: var(--pi-mono) !important;
  color: var(--pi-muted) !important;
}}

div[data-baseweb="select"] > div {{
  background: var(--pi-surface) !important;
  border: 1px solid #000 !important;
  border-radius: 0 !important;
  color: #000 !important;
  font-family: var(--pi-mono) !important;
}}

[data-testid="stTabs"] [data-baseweb="tab-list"] {{
  gap: 1.1rem;
  border-bottom: 1px solid var(--pi-line);
}}

[data-testid="stTabs"] [data-baseweb="tab"] {{
  border-radius: 0;
  color: var(--pi-muted) !important;
  font-family: var(--pi-mono) !important;
  text-decoration: underline;
  text-underline-offset: 4px;
  background: transparent !important;
  padding-left: 0 !important;
  padding-right: 0 !important;
}}

[data-testid="stTabs"] [aria-selected="true"] {{
  color: #000 !important;
  font-weight: 700 !important;
  border-bottom: 2px solid #000 !important;
  text-decoration: none;
}}

.stProgress > div > div > div > div {{
  background: #000 !important;
  border-radius: 0 !important;
}}

.stProgress > div > div {{
  background: var(--pi-line) !important;
  border-radius: 0 !important;
}}

a {{
  color: #000 !important;
  text-decoration: underline !important;
  text-underline-offset: 3px;
  font-family: var(--pi-mono) !important;
}}

hr {{ border: none !important; border-top: 1px solid var(--pi-line) !important; }}

[data-testid="stImage"], [data-testid="stVegaLiteChart"] {{
  border: 1px solid #000;
  border-radius: 0;
  background: var(--pi-surface);
  padding: 0.35rem;
  box-shadow: var(--pi-shadow);
}}

.stCaption {{ color: var(--pi-muted) !important; font-family: var(--pi-mono) !important; }}

#MainMenu, footer {{ visibility: hidden; }}
</style>
"""

st.markdown(CUSTOM_CSS, unsafe_allow_html=True)

def page_chrome_css(page: str) -> str:
    """Kill Streamlit morph animations and hide sidebar on the intro screen."""
    hide_sidebar = ""
    if page == "intro":
        hide_sidebar = """
        [data-testid="stSidebar"],
        [data-testid="stSidebarCollapsedControl"],
        [data-testid="collapsedControl"] {
          display: none !important;
          width: 0 !important;
          min-width: 0 !important;
          visibility: hidden !important;
        }
        [data-testid="stAppViewContainer"] .main {
          margin-left: 0 !important;
        }
        """
    return f"""
<style>
[data-testid="stAppViewContainer"],
[data-testid="stAppViewContainer"] > .main,
section.main,
.stApp,
[data-testid="stVerticalBlock"],
[data-testid="stHorizontalBlock"],
[data-testid="element-container"],
[data-testid="stDecoration"],
[data-testid="stHeader"] {{
  animation: none !important;
  transition: none !important;
}}
{hide_sidebar}
</style>
"""


def _no_emdash(text: str) -> str:
    return text.replace("\u2014", " -- ").replace("\u2013", "-").replace(" -- ", " -- ").replace("–", "-")


@st.cache_resource
def load_predictor() -> ProjectKaanPredictor:
    return ProjectKaanPredictor()


@st.cache_data(show_spinner=False)
def generate_demo_audio() -> bytes:
    sr = 16000
    duration = 3  # short clip -- enough for demo, much faster to plot/analyze
    t = np.linspace(0, duration, sr * duration, endpoint=False)
    tone = 0.3 * np.sin(2 * np.pi * 360 * t)
    rng = np.random.default_rng(0)
    noise = 0.01 * rng.standard_normal(len(tone))
    audio = (tone + noise).astype(np.float32)
    import soundfile as sf

    buffer = io.BytesIO()
    sf.write(buffer, audio, sr, format="WAV")
    buffer.seek(0)
    return buffer.read()


def plot_waveform(audio_bytes: bytes) -> plt.Figure:
    y, sr = librosa.load(io.BytesIO(audio_bytes), sr=16000, mono=True)
    # Downsample plot points for speed on long recordings
    max_points = 4000
    if len(y) > max_points:
        step = max(1, len(y) // max_points)
        y_plot = y[::step]
        times = (np.arange(len(y_plot)) * step) / sr
    else:
        y_plot = y
        times = np.arange(len(y)) / sr
    fig, ax = plt.subplots(figsize=(10, 2.4), facecolor=BG)
    ax.set_facecolor(SURFACE)
    ax.plot(times, y_plot, color=INK, linewidth=0.6)
    ax.set_xlabel("Time (seconds)", color=INK, fontfamily="monospace")
    ax.set_ylabel("Amplitude", color=INK, fontfamily="monospace")
    ax.set_title("Audio Waveform", color=INK, fontsize=11, fontweight="bold", fontfamily="monospace")
    ax.tick_params(colors=INK)
    for spine in ax.spines.values():
        spine.set_color(INK)
        spine.set_linewidth(1.2)
    plt.tight_layout()
    return fig


def fig_to_image(fig: plt.Figure) -> bytes:
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=100, facecolor=fig.get_facecolor(), bbox_inches="tight")
    plt.close(fig)
    buf.seek(0)
    return buf.read()


@st.cache_data(show_spinner=False)
def cached_waveform_png(audio_bytes: bytes) -> bytes:
    return fig_to_image(plot_waveform(audio_bytes))


@st.cache_data(show_spinner=False)
def cached_spectrogram_png(audio_bytes: bytes) -> bytes:
    return fig_to_image(plot_spectrogram(audio_bytes))


def set_audio(audio_bytes: bytes, *, analyze: bool = False) -> None:
    """Replace current audio and clear stale results so UI never overlaps old output."""
    st.session_state.audio_bytes = audio_bytes
    st.session_state.audio_id = hashlib.md5(audio_bytes).hexdigest()
    st.session_state.result = None
    st.session_state.severity = None
    st.session_state.pending_analyze = analyze


def run_analysis(predictor: ProjectKaanPredictor, audio_bytes: bytes) -> None:
    result = predictor.predict(audio_bytes)
    st.session_state.result = result
    if result.get("confident") and result.get("class") != "clean":
        st.session_state.severity = estimate_severity(io.BytesIO(audio_bytes))
    else:
        st.session_state.severity = None
    st.session_state.pending_analyze = False


def render_result_block(lang: str, result: dict) -> None:
    pest_name = get_pest_name(lang, result["class"])
    is_pest = result["class"] != "clean"

    if result["confident"]:
        if is_pest:
            st.markdown(
                f'<div class="result-pest"><h2>⚠️ {_no_emdash(get_ui(lang, "result_pest"))}</h2>'
                f"<h3>{pest_name}</h3></div>",
                unsafe_allow_html=True,
            )
        else:
            st.markdown(
                f'<div class="result-clean"><h2>✅ {_no_emdash(get_ui(lang, "result_clean"))}</h2>'
                f"<h3>{pest_name}</h3></div>",
                unsafe_allow_html=True,
            )
    else:
        st.markdown(
            f'<div class="result-low"><h2>🔶 {_no_emdash(get_ui(lang, "low_confidence_title"))}</h2>'
            f'<p>{_no_emdash(get_ui(lang, "low_confidence_msg"))}</p>'
            f'<p><b>{pest_name}</b> ({result["confidence"]:.0%})</p></div>',
            unsafe_allow_html=True,
        )
    st.caption("Result indicated by both colour and symbol for accessibility.")

    if st.session_state.severity is not None:
        render_severity(lang, st.session_state.severity)

    st.markdown(f"**{_no_emdash(get_ui(lang, 'confidence_label'))}:** {result['confidence']:.1%}")
    st.progress(result["confidence"])

    st.markdown(f"### {_no_emdash(get_ui(lang, 'all_scores_title'))}")
    scores_df = pd.DataFrame(
        {
            "Class": [get_pest_name(lang, k) for k in result["all_scores"]],
            "Probability": list(result["all_scores"].values()),
        }
    ).set_index("Class")
    st.bar_chart(scores_df, use_container_width=True, color=CHART)

    st.markdown(f"### {_no_emdash(get_ui(lang, 'advisory_title'))}")
    st.markdown(
        f'<div class="advisory">{_no_emdash(get_advisory(lang, result["class"]))}</div>',
        unsafe_allow_html=True,
    )
    st.caption(_no_emdash(get_ui(lang, "disclaimer")))


def render_intro():
    st.markdown(page_chrome_css("intro"), unsafe_allow_html=True)
    lang = st.session_state.lang

    st.markdown('<div class="kaan-intro">', unsafe_allow_html=True)
    st.markdown(
        '<p class="kaan-name">Kaan</p>'
        '<p class="kaan-devanagari">कान</p>'
        f'<p class="kaan-tagline">{TAGLINE}</p>',
        unsafe_allow_html=True,
    )

    st.markdown('<div class="lang-row"></div>', unsafe_allow_html=True)
    cols = st.columns(len(LANG_BUTTONS))
    for col, (code, label) in zip(cols, LANG_BUTTONS):
        with col:
            is_active = lang == code
            if st.button(
                label,
                key=f"lang_{code}",
                use_container_width=True,
                type="primary" if is_active else "secondary",
            ):
                st.session_state.lang = code
                st.rerun()

    manifesto = _no_emdash(MANIFESTO.get(lang, MANIFESTO["en"]))
    st.markdown(f'<p class="kaan-manifesto">{manifesto}</p>', unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)
    if st.button(
        CTA_LISTEN.get(lang, CTA_LISTEN["en"]),
        type="primary",
        use_container_width=True,
        key="cta_listen",
    ):
        st.session_state.page = "main"
        st.rerun()

    st.markdown("</div>", unsafe_allow_html=True)


def render_sidebar(lang: str):
    st.sidebar.markdown('<p class="sidebar-logo">Kaan</p>', unsafe_allow_html=True)
    st.sidebar.caption(_no_emdash(get_ui(lang, "app_tagline")).replace("Project Kaan", "Kaan"))

    labels = [label for _, label in LANG_BUTTONS]
    codes = [code for code, _ in LANG_BUTTONS]
    current_idx = codes.index(lang) if lang in codes else 0
    selected = st.sidebar.selectbox(
        _no_emdash(get_ui(lang, "language_label")),
        options=labels,
        index=current_idx,
    )
    lang = codes[labels.index(selected)]
    st.session_state.lang = lang

    st.sidebar.markdown(
        f'<div class="sidebar-card">'
        f'<strong style="color:#000;">About</strong><br><br>'
        f'{_no_emdash(get_ui(lang, "sidebar_about_1")).replace("Project Kaan", "Kaan")}<br><br>'
        f'{_no_emdash(get_ui(lang, "sidebar_about_2"))}<br><br>'
        f'{_no_emdash(get_ui(lang, "sidebar_about_3"))}'
        f"</div>",
        unsafe_allow_html=True,
    )
    st.sidebar.caption(_no_emdash(get_ui(lang, "sidebar_citation")))
    st.sidebar.markdown(f"[{_no_emdash(get_ui(lang, 'github_link'))}]({GITHUB_REPO_URL})")

    if st.sidebar.button(BACK_LABEL.get(lang, BACK_LABEL["en"]), use_container_width=True):
        st.session_state.page = "intro"
        st.rerun()

    return lang


def render_severity(lang: str, severity: dict):
    level = severity["level"]
    level_label = SEVERITY_LEVEL_NAMES.get(level, {}).get(lang, level)

    segments = ""
    for seg_level in SEVERITY_ORDER:
        is_active = seg_level == level
        seg_label = SEVERITY_LEVEL_NAMES.get(seg_level, {}).get(lang, seg_level)
        style = f'style="background:{severity["color"]};border-color:{severity["color"]};"' if is_active else ""
        active_class = " active" if is_active else ""
        segments += f'<div class="severity-segment{active_class}" {style}>{seg_label}</div>'

    st.markdown(
        f'<div class="severity-box">'
        f'<span class="panel-label">Severity</span>'
        f'<div class="severity-meter">{segments}</div>'
        f'<p class="severity-level">{severity["symbol"]} {level_label}</p>'
        f'<p>{_no_emdash(severity["message"])}</p>'
        f'<p><b>Action:</b> {_no_emdash(severity["action"])}</p>'
        f"</div>",
        unsafe_allow_html=True,
    )
    st.caption(
        "Severity estimated from acoustic signal density (RMS energy and impulse rate). "
        "Methodology based on Balingbing et al., Computers and Electronics in Agriculture, 2024."
    )


def render_detect(lang: str, predictor: ProjectKaanPredictor):
    if predictor.demo_mode:
        st.markdown(
            f'<div class="demo-banner">{_no_emdash(get_ui(lang, "demo_banner")).replace("Project Kaan", "Kaan")}</div>',
            unsafe_allow_html=True,
        )

    input_mode = st.radio(
        _no_emdash(get_ui(lang, "input_mode_label")),
        [
            _no_emdash(get_ui(lang, "mode_upload")),
            _no_emdash(get_ui(lang, "mode_record")),
        ],
        horizontal=True,
        key="input_mode",
    )

    def _ingest_upload(uploaded_file) -> None:
        file_id = f"{uploaded_file.name}:{uploaded_file.size}:{getattr(uploaded_file, 'file_id', '')}"
        if file_id == st.session_state.upload_file_id:
            return
        st.session_state.upload_file_id = file_id
        set_audio(uploaded_file.getvalue(), analyze=True)
        st.rerun()

    if input_mode == _no_emdash(get_ui(lang, "mode_upload")):
        uploaded = st.file_uploader(
            _no_emdash(get_ui(lang, "upload_label")),
            type=["wav", "mp3", "m4a", "ogg"],
            help=_no_emdash(get_ui(lang, "upload_help")),
            key="detect_upload",
        )
        if uploaded is not None:
            _ingest_upload(uploaded)
    else:
        st.markdown(f"### {_no_emdash(get_ui(lang, 'record_title'))}")
        for key in ("record_step_1", "record_step_2", "record_step_3", "record_step_4"):
            st.markdown(_no_emdash(get_ui(lang, key)))
        diagram = (
            "PHONE  -->  GRAIN BAG\n"
            "Press the phone firmly against the bag surface.\n"
            "Record 30 seconds in a quiet environment,\n"
            "then upload the file below."
        )
        st.markdown(f'<div class="record-box">{diagram}</div>', unsafe_allow_html=True)
        uploaded = st.file_uploader(
            _no_emdash(get_ui(lang, "upload_label")),
            type=["wav", "mp3", "m4a", "ogg"],
            key="record_upload",
        )
        if uploaded is not None:
            _ingest_upload(uploaded)

    demo_col, clear_col = st.columns([2, 1])
    with demo_col:
        if st.button(_no_emdash(get_ui(lang, "try_demo_btn")), use_container_width=True, key="try_demo"):
            set_audio(generate_demo_audio(), analyze=True)
            st.rerun()
    with clear_col:
        if st.session_state.audio_bytes and st.button("Clear", use_container_width=True, key="clear_audio"):
            st.session_state.audio_bytes = None
            st.session_state.audio_id = None
            st.session_state.upload_file_id = None
            st.session_state.result = None
            st.session_state.severity = None
            st.session_state.pending_analyze = False
            st.rerun()

    if st.session_state.audio_bytes:
        audio_bytes = st.session_state.audio_bytes

        if st.session_state.pending_analyze:
            with st.spinner(_no_emdash(get_ui(lang, "analyze_btn")) + "…"):
                run_analysis(predictor, audio_bytes)
            st.rerun()

        col1, col2 = st.columns(2)
        with col1:
            st.markdown(
                f'<div class="panel"><span class="panel-label">{_no_emdash(get_ui(lang, "waveform_title"))}</span></div>',
                unsafe_allow_html=True,
            )
            st.image(cached_waveform_png(audio_bytes), use_container_width=True)
        with col2:
            st.markdown(
                f'<div class="panel"><span class="panel-label">{_no_emdash(get_ui(lang, "spectrogram_title"))}</span></div>',
                unsafe_allow_html=True,
            )
            st.image(cached_spectrogram_png(audio_bytes), use_container_width=True)

        st.audio(audio_bytes)

        if st.button(
            _no_emdash(get_ui(lang, "analyze_btn")),
            type="primary",
            use_container_width=True,
            key="analyze_btn",
        ):
            st.session_state.pending_analyze = True
            st.rerun()

        if st.session_state.result is not None:
            render_result_block(lang, st.session_state.result)
    else:
        st.markdown(
            f'<div class="panel"><span class="panel-label">Input</span>'
            f'<p style="margin:0.4rem 0 0;color:{MUTED};">{_no_emdash(get_ui(lang, "no_file_msg"))}</p></div>',
            unsafe_allow_html=True,
        )


def render_how(lang: str):
    st.markdown(f"## {_no_emdash(get_ui(lang, 'how_title')).replace('Project Kaan', 'Kaan')}")
    steps = [
        ("01", "Record", _no_emdash(get_ui(lang, "how_step_1"))),
        ("02", "Transform", _no_emdash(get_ui(lang, "how_step_2"))),
        ("03", "Classify", _no_emdash(get_ui(lang, "how_step_3"))),
        ("04", "Advise", _no_emdash(get_ui(lang, "how_step_4"))),
    ]
    items = ""
    for num, title, text in steps:
        items += (
            f'<div class="pi-timeline-item">'
            f'<div class="pi-timeline-row">'
            f'<span class="pi-timeline-title">{title}</span>'
            f'<span class="pi-timeline-meta">{num}</span>'
            f"</div>"
            f'<p class="pi-timeline-body">{text}</p>'
            f"</div>"
        )
    st.markdown(f'<div class="pi-timeline">{items}</div>', unsafe_allow_html=True)
    st.markdown("---")
    st.markdown(f"### {_no_emdash(get_ui(lang, 'how_mel_title'))}")
    st.markdown(_no_emdash(get_ui(lang, "how_mel_desc")))
    try:
        st.image(cached_spectrogram_png(generate_demo_audio()), use_container_width=True)
    except Exception as exc:
        st.caption(f"Spectrogram preview unavailable: {exc}")
    for key in (
        "how_arch_title",
        "how_arch_desc",
        "how_datasets_title",
        "how_datasets_desc",
        "how_accuracy_title",
        "how_accuracy_desc",
        "how_freq_title",
        "how_freq_desc",
    ):
        if key.endswith("_title"):
            st.markdown(f"### {_no_emdash(get_ui(lang, key))}")
        else:
            st.markdown(_no_emdash(get_ui(lang, key)))


def render_summit():
    """AI Impact Summit recognition brief (English, shared with the public web app)."""
    st.markdown("## Kaan at the AI Impact Summit")
    st.caption("Recognition")
    st.markdown(
        "Kaan is an AI-powered acoustic detector that helps Indian farmers catch "
        "stored-grain insect infestation early using a normal phone, offline, in their language."
    )
    blocks = [
        (
            "Problem",
            "India stores over 80 million tonnes of food grain. Insects in storage cause about "
            "1,300 crore rupees in annual loss (IGMRI). Smallholders often find damage only after "
            "10 to 20 percent of the grain is already affected, because hand and smell checks miss "
            "early activity.",
        ),
        (
            "Solution",
            "Farmers record grain sounds by holding their phone against the bag or bin. Kaan converts "
            "the audio to a mel spectrogram and runs a compact CNN that classifies clean grain, rice "
            "weevil, lesser grain borer, and red flour beetle. It returns the pest class, confidence, "
            "a simple advisory, and an accessibility-safe result using both symbol and text.",
        ),
        (
            "Why AI",
            "These pests produce overlapping sounds in the 300 to 4000 Hz range. Rule-based thresholds "
            "cannot separate them reliably. A trained CNN learns the subtle spectral patterns needed "
            "for species-level screening.",
        ),
        (
            "Impact and inclusion",
            "Available in English, Hindi, Marathi, Punjabi, and Telugu. Designed for low-resource "
            "settings using phone mic, offline classification, and a guided tour. Aimed at farmers, "
            "FPOs, and Krishi Vigyan Kendras as a screening aid. Open pipeline so local agencies can "
            "retrain on regional audio.",
        ),
        (
            "Ethics and limits",
            "Reports confidence and can stay uncertain instead of guessing. Phone quality, noise, and "
            "early low-density infestations still need field validation. Pulse beetle and legume "
            "detection are future work.",
        ),
    ]
    items = ""
    for i, (title, body) in enumerate(blocks, start=1):
        items += (
            f'<div class="pi-timeline-item">'
            f'<div class="pi-timeline-row">'
            f'<span class="pi-timeline-title">{title}</span>'
            f'<span class="pi-timeline-meta">{i:02d}</span>'
            f"</div>"
            f'<p class="pi-timeline-body">{body}</p>'
            f"</div>"
        )
    tech = (
        "<ul>"
        "<li>Validation accuracy 97.76 percent, macro F1 0.98, after leakage-aware retraining</li>"
        "<li>INT8 model runs on-device / in the browser with no cloud needed for inference</li>"
        "<li>Built on open IRRI acoustic research data, released under MIT</li>"
        "<li>Live at kaan-web.vercel.app</li>"
        "</ul>"
    )
    items += (
        f'<div class="pi-timeline-item">'
        f'<div class="pi-timeline-row">'
        f'<span class="pi-timeline-title">Technical highlights</span>'
        f'<span class="pi-timeline-meta">{len(blocks)+1:02d}</span>'
        f"</div>"
        f'<div class="pi-timeline-body">{tech}</div>'
        f"</div>"
    )
    st.markdown(f'<div class="pi-timeline">{items}</div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="panel"><span class="panel-label">Summit pitch</span>'
        "<p>Kaan listens to grain so farmers do not have to wait until they can see the damage. "
        "Open, multilingual, offline AI for food security and farmer livelihoods.</p></div>",
        unsafe_allow_html=True,
    )
    st.markdown("---")



def render_about(lang: str):
    render_summit()
    st.markdown(f"## {_no_emdash(get_ui(lang, 'about_title'))}")
    sections = [
        ("Problem", "about_problem"),
        ("Audience", "about_audience"),
        ("SDG", "about_sdg"),
        ("Environment", "about_env"),
        ("Privacy", "about_privacy"),
        ("Limitations", "about_limitations"),
        ("Distribution", "about_gtm"),
        ("Why AI", "about_why_ai"),
        ("Inclusion", "about_inclusion"),
        ("Ethics", "about_ethics"),
        ("Innovation", "about_innovation"),
        ("Tech Stack", "about_techstack"),
        ("Deployment", "about_deployment"),
    ]
    items = ""
    for i, (title, key) in enumerate(sections, start=1):
        text = _no_emdash(get_ui(lang, key)).replace("Project Kaan", "Kaan")
        items += (
            f'<div class="pi-timeline-item">'
            f'<div class="pi-timeline-row">'
            f'<span class="pi-timeline-title">{title}</span>'
            f'<span class="pi-timeline-meta">{i:02d}</span>'
            f"</div>"
            f'<p class="pi-timeline-body">{text}</p>'
            f"</div>"
        )
    st.markdown(f'<div class="pi-timeline">{items}</div>', unsafe_allow_html=True)
    st.markdown(
        f'<div class="panel"><p style="margin:0;color:#000;font-weight:700;">'
        f'{_no_emdash(get_ui(lang, "about_team")).replace("Project Kaan", "Kaan")}</p></div>',
        unsafe_allow_html=True,
    )
    st.markdown(
        '<div class="panel">'
        "<p>Kaan is released as open-source under MIT licence. The full training pipeline is "
        "documented on GitHub to allow state agriculture departments and Krishi Vigyan Kendras "
        "to retrain the model on locally recorded data from their specific region and grain "
        "variety.</p>"
        "<p>The TFLite model uses INT8 quantization to enable deployment on AI-optimised mobile "
        "chipsets including Qualcomm Hexagon NPU and MediaTek APU, which accelerate on-device "
        "inference on mid-range Android phones without requiring cloud compute.</p>"
        "<p>Kaan will never communicate a result through colour alone. Every result includes a "
        "symbol indicator so the app is usable by farmers with colour vision deficiency.</p>"
        "</div>",
        unsafe_allow_html=True,
    )


def render_main():
    st.markdown(page_chrome_css("main"), unsafe_allow_html=True)
    lang = render_sidebar(st.session_state.lang)
    st.session_state.lang = lang
    predictor = load_predictor()

    st.markdown(
        f'<div class="kaan-tool-header">'
        f"<h1>Kaan</h1>"
        f'<p>{_no_emdash(get_ui(lang, "app_tagline")).replace("Project Kaan", "Kaan")}</p>'
        f"</div>",
        unsafe_allow_html=True,
    )

    # Stable keys avoid Streamlit tab remount flicker and survive language switches.
    section_keys = ["detect", "how", "about"]
    section_labels = {
        "detect": _no_emdash(get_ui(lang, "tab_detect")),
        "how": _no_emdash(get_ui(lang, "tab_how")),
        "about": _no_emdash(get_ui(lang, "tab_about")),
    }
    section = st.radio(
        "Section",
        section_keys,
        format_func=lambda k: section_labels[k],
        horizontal=True,
        label_visibility="collapsed",
        key="main_section",
    )
    if section == "detect":
        render_detect(lang, predictor)
    elif section == "how":
        render_how(lang)
    else:
        render_about(lang)


def main():
    # Inject transition CSS before any page body so old/new never animate over each other.
    st.markdown(page_chrome_css(st.session_state.page), unsafe_allow_html=True)
    if st.session_state.page == "intro":
        render_intro()
    else:
        render_main()


if __name__ == "__main__":
    main()
