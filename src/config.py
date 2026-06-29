"""Central configuration – paths, constants, and environment loading."""
import os
from pathlib import Path
from dotenv import load_dotenv

ROOT = Path(__file__).parent.parent
load_dotenv(ROOT / ".env")

# ── Paths ─────────────────────────────────────────────────────────────────────
DATA_RAW = ROOT / "data" / "raw"
DATA_PROC = ROOT / "data" / "processed"
MODELS_DIR = ROOT / "data" / "processed" / "models"
ASSETS_DIR = ROOT / "assets" / "charts"
DELIVERABLES = ROOT / "deliverables"

for _p in (DATA_RAW, DATA_PROC, MODELS_DIR, ASSETS_DIR):
    _p.mkdir(parents=True, exist_ok=True)

# ── LLM keys ──────────────────────────────────────────────────────────────────
GROQ_API_KEY: str | None = os.getenv("GROQ_API_KEY")
GROQ_MODEL: str = os.getenv("GROQ_MODEL", "llama-3.1-8b-instant")
OPENAI_API_KEY: str | None = os.getenv("OPENAI_API_KEY")
ANTHROPIC_API_KEY: str | None = os.getenv("ANTHROPIC_API_KEY")

LLM_AVAILABLE = bool(GROQ_API_KEY or OPENAI_API_KEY or ANTHROPIC_API_KEY)

# ── Kaggle ────────────────────────────────────────────────────────────────────
KAGGLE_USERNAME: str | None = os.getenv("KAGGLE_USERNAME")
KAGGLE_KEY: str | None = os.getenv("KAGGLE_KEY")
KAGGLE_DATASET = "rohitrox/healthcare-provider-fraud-detection-analysis"

# ── Model hyper-parameters (sensible defaults) ────────────────────────────────
CLF_N_ESTIMATORS = 300
CLF_MAX_DEPTH = 6
CLF_RANDOM_STATE = 42

CLUSTER_K = 5
CLUSTER_RANDOM_STATE = 42

ANOMALY_CONTAMINATION = 0.05  # expected fraction of anomalous months
ANOMALY_ZSCORE_THRESHOLD = 2.5

# ── Streamlit ─────────────────────────────────────────────────────────────────
APP_TITLE = "TPO Sentinel – Clinical Decision & Payment Integrity"
APP_ICON = "🏥"
