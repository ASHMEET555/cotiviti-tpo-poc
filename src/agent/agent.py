"""
Chain-of-Thought Investigator Agent.

Orchestrates a 5-step reasoning chain:
  Step 1 – Classify (fraud probability)
  Step 2 – Anomaly check (time-series billing anomalies)
  Step 3 – Peer comparison (cluster + outlier flag)
  Step 4 – Explain (SHAP top features)
  Step 5 – Synthesize (LLM / fallback recommendation)

Usage:
    from src.agent.agent import investigate
    result = investigate("PRV00042")
    print(result["recommendation"])
"""
import sys
from pathlib import Path
from typing import Generator

sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from src.agent.tools import tool_classify, tool_anomalies, tool_cluster, tool_explain
from src.agent.llm import generate


STEP_LABELS = [
    "Step 1 – Fraud Classification",
    "Step 2 – Billing Anomaly Detection",
    "Step 3 – Peer-Group Benchmarking",
    "Step 4 – SHAP Explainability",
    "Step 5 – Synthesizing Recommendation",
]


def _build_synthesis_prompt(provider_id: str, steps: dict) -> str:
    clf = steps.get("classify", {})
    anom = steps.get("anomalies", {})
    cluster = steps.get("cluster", {})
    explain = steps.get("explain", {})

    features_str = ", ".join(
        f"{k} ({v:+.3f})" for k, v in (explain.get("top_features") or {}).items()
    )

    return f"""You are investigating provider {provider_id} for potential fraud, waste, or abuse.

EVIDENCE SUMMARY:
fraud_probability: {clf.get('fraud_probability', 'N/A')}
fraud_flag: {clf.get('fraud_flag', 'N/A')}
anomaly_months: {anom.get('anomaly_months', 0)} out of {anom.get('total_months', 'N/A')} months
anomaly_rate: {anom.get('anomaly_rate', 0):.1%}
cluster: {cluster.get('cluster', 'N/A')}
outlier_flag: {cluster.get('outlier_flag', False)}
distance_to_centroid: {cluster.get('distance_to_centroid', 'N/A')}
top_features: {features_str}

Using this evidence, produce a structured 3-paragraph investigation report:
1. Summary of findings (cite specific numbers).
2. Risk assessment – explain the interplay between the fraud score, billing anomalies, and peer comparison.
3. Recommendation: one of PAY / REVIEW / DENY / AUDIT. Justify your recommendation with specific evidence.
"""


def investigate(provider_id: str) -> dict:
    """
    Run the full 5-step investigation chain synchronously.
    Returns a dict with keys: steps, recommendation, llm_source.
    """
    steps: dict = {}

    # Step 1 – Classify
    steps["classify"] = tool_classify(provider_id)

    # Step 2 – Anomalies
    steps["anomalies"] = tool_anomalies(provider_id)

    # Step 3 – Cluster
    steps["cluster"] = tool_cluster(provider_id)

    # Step 4 – Explain
    steps["explain"] = tool_explain(provider_id)

    # Step 5 – Synthesize
    prompt = _build_synthesis_prompt(provider_id, steps)
    recommendation, llm_source = generate(prompt)

    return {
        "provider_id": provider_id,
        "steps": steps,
        "recommendation": recommendation,
        "llm_source": llm_source,
    }


def investigate_streaming(provider_id: str) -> Generator[dict, None, None]:
    """
    Generator version for Streamlit streaming UI.
    Yields {"step": label, "data": result_dict} for each step,
    then {"step": "done", "recommendation": text, "llm_source": src}.
    """
    steps: dict = {}

    yield {"step": STEP_LABELS[0], "data": None, "status": "running"}
    steps["classify"] = tool_classify(provider_id)
    yield {"step": STEP_LABELS[0], "data": steps["classify"], "status": "done"}

    yield {"step": STEP_LABELS[1], "data": None, "status": "running"}
    steps["anomalies"] = tool_anomalies(provider_id)
    yield {"step": STEP_LABELS[1], "data": steps["anomalies"], "status": "done"}

    yield {"step": STEP_LABELS[2], "data": None, "status": "running"}
    steps["cluster"] = tool_cluster(provider_id)
    yield {"step": STEP_LABELS[2], "data": steps["cluster"], "status": "done"}

    yield {"step": STEP_LABELS[3], "data": None, "status": "running"}
    steps["explain"] = tool_explain(provider_id)
    yield {"step": STEP_LABELS[3], "data": steps["explain"], "status": "done"}

    yield {"step": STEP_LABELS[4], "data": None, "status": "running"}
    prompt = _build_synthesis_prompt(provider_id, steps)
    recommendation, llm_source = generate(prompt)
    yield {
        "step": STEP_LABELS[4],
        "data": {"recommendation": recommendation},
        "status": "done",
    }

    yield {
        "step": "done",
        "steps": steps,
        "recommendation": recommendation,
        "llm_source": llm_source,
    }


if __name__ == "__main__":
    import json
    import sys

    pid = sys.argv[1] if len(sys.argv) > 1 else "PRV00001"
    result = investigate(pid)
    print(f"\n{'='*60}")
    print(f"Investigation: {pid}")
    print(f"{'='*60}")
    print(result["recommendation"])
    print(f"\n[LLM source: {result['llm_source']}]")
