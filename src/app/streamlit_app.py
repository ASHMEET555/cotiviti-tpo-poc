"""
TPO Sentinel – Streamlit Dashboard
====================================
Multi-tab healthcare payment-integrity + clinical-decision copilot.

Tabs:
  🏠 Overview          – project summary, technique-to-TPO map
  💳 Payment Integrity – FWA classifier + SHAP explanations
  🩺 Clinical Risk     – reimbursement prediction + LOS regression
  ⚙️ Operations        – provider clustering + time-series anomalies
  🕵️ Investigator Agent – agentic chain-of-thought investigation
"""
import sys
from pathlib import Path

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from src.config import APP_TITLE, APP_ICON, DATA_PROC, MODELS_DIR, LLM_AVAILABLE, GROQ_API_KEY, GROQ_MODEL

st.set_page_config(page_title=APP_TITLE, page_icon=APP_ICON, layout="wide")

# ── Cache data loaders ─────────────────────────────────────────────────────────

@st.cache_data(show_spinner="Loading provider features…")
def load_features() -> pd.DataFrame:
    path = DATA_PROC / "provider_features.parquet"
    if not path.exists():
        st.error("Provider features not found. Run: `python scripts/train_all.py`")
        st.stop()
    return pd.read_parquet(path)


@st.cache_data(show_spinner="Loading time-series…")
def load_timeseries() -> pd.DataFrame:
    path = DATA_PROC / "timeseries.parquet"
    if not path.exists():
        return pd.DataFrame()
    return pd.read_parquet(path)


@st.cache_data(show_spinner="Loading anomaly results…")
def load_anomalies() -> pd.DataFrame:
    path = DATA_PROC / "provider_anomalies.parquet"
    if not path.exists():
        return pd.DataFrame()
    return pd.read_parquet(path)


@st.cache_data(show_spinner="Loading cluster results…")
def load_clusters() -> pd.DataFrame:
    path = DATA_PROC / "provider_clusters.parquet"
    if not path.exists():
        return pd.DataFrame()
    return pd.read_parquet(path)


@st.cache_data(show_spinner="Loading classifier metrics…")
def load_clf_metrics() -> dict:
    import json
    path = MODELS_DIR / "clf_metrics.json"
    if not path.exists():
        return {}
    with open(path) as f:
        return json.load(f)


@st.cache_data(show_spinner="Loading SHAP importances…")
def load_shap_global() -> dict:
    import json
    path = MODELS_DIR / "shap_global.json"
    if not path.exists():
        return {}
    with open(path) as f:
        return json.load(f)


# ── Sidebar ────────────────────────────────────────────────────────────────────

with st.sidebar:
    st.image("https://upload.wikimedia.org/wikipedia/commons/thumb/a/a7/Camponotus_flavomarginatus_ant.jpg/1px-transparent.png", width=1)
    st.title("🏥 TPO Sentinel")
    st.caption("Cotiviti Intern Assessment – Topic #2")
    st.divider()

    if GROQ_API_KEY:
        llm_status = f"🟢 Groq Active ({GROQ_MODEL})"
    elif LLM_AVAILABLE:
        llm_status = "🟢 Live LLM Active"
    else:
        llm_status = "🟡 Fallback Mode (no API key)"
    st.markdown(f"**Agent Status:** {llm_status}")
    if not LLM_AVAILABLE:
        st.info("Add GROQ_API_KEY (or OPENAI/ANTHROPIC) to .env to enable LLM reasoning.")
    st.divider()
    st.markdown(
        "**Technique → TPO**\n"
        "- Classification → Payment\n"
        "- Inference (SHAP) → Payment\n"
        "- Prediction → Treatment\n"
        "- Clustering → Operations\n"
        "- Anomaly Detection → Operations\n"
        "- Agentic Chain Reasoning → All"
    )


# ── Tabs ───────────────────────────────────────────────────────────────────────

tab_names = [
    "🏠 Overview",
    "💳 Payment Integrity",
    "🩺 Clinical Risk",
    "⚙️ Operations",
    "🕵️ Investigator Agent",
]
tabs = st.tabs(tab_names)


# ══════════════════════════════════════════════════════════════════════════════
# TAB 0 – Overview
# ══════════════════════════════════════════════════════════════════════════════
with tabs[0]:
    st.title("TPO Sentinel")
    st.subheader("Clinical Decision Making & Payment Integrity for Treatment, Payment & Operations")

    col1, col2, col3 = st.columns(3)
    df = load_features()
    with col1:
        st.metric("Providers", f"{len(df):,}")
    with col2:
        fraud_rate = df["FraudLabel"].mean() * 100 if "FraudLabel" in df.columns else 0
        st.metric("Fraud Rate", f"{fraud_rate:.1f}%")
    with col3:
        ts = load_timeseries()
        st.metric("Monthly Records", f"{len(ts):,}" if not ts.empty else "N/A")

    st.divider()
    st.markdown("""
    ### What is TPO?
    TPO stands for **Treatment, Payment, and Operations** — the three pillars defined by HIPAA
    that govern how protected health information is used in healthcare.

    This demonstrator applies **6 AI/ML techniques** across each pillar to help Cotiviti
    automate payment-integrity decisions, detect fraud, and surface clinical risk.

    | Technique | TPO Pillar | Purpose |
    |---|---|---|
    | **Classification** | Payment | Predict provider fraud probability |
    | **Inference (SHAP)** | Payment | Explain *why* a provider is flagged |
    | **Prediction** | Treatment | Estimate expected reimbursement & clinical risk |
    | **Clustering** | Operations | Peer-group benchmarking of provider billing patterns |
    | **Time-Series Anomaly Detection** | Operations | Detect unusual billing spikes |
    | **Agentic Chain Reasoning** | All | LLM orchestrates a 5-step investigation |

    ### Architecture
    ```
    Raw Claims Data → Feature Engineering → ML Models → Agent Tools → LLM → Recommendation
    ```
    Navigate the tabs above to explore each component.
    """)

    # Fraud distribution chart
    if "FraudLabel" in df.columns:
        fig = px.histogram(df, x="FraudLabel",
                           color="FraudLabel",
                           color_discrete_map={0: "#2ecc71", 1: "#e74c3c"},
                           labels={"FraudLabel": "Fraud Label"},
                           title="Provider Fraud Label Distribution")
        fig.update_xaxes(tickvals=[0, 1], ticktext=["Legitimate", "Fraud"])
        st.plotly_chart(fig, use_container_width=True)


# ══════════════════════════════════════════════════════════════════════════════
# TAB 1 – Payment Integrity
# ══════════════════════════════════════════════════════════════════════════════
with tabs[1]:
    st.header("💳 Payment Integrity – FWA Classification & SHAP Inference")
    df = load_features()
    metrics = load_clf_metrics()
    shap_global = load_shap_global()

    if metrics:
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("ROC-AUC", f"{metrics.get('roc_auc', 0):.3f}")
        c2.metric("Avg Precision", f"{metrics.get('avg_precision', 0):.3f}")
        c3.metric("Train Set", f"{metrics.get('n_train', 0):,}")
        c4.metric("Test Set", f"{metrics.get('n_test', 0):,}")
        st.divider()

    col_a, col_b = st.columns(2)

    with col_a:
        st.subheader("Fraud Probability Distribution")
        if MODELS_DIR.exists() and (MODELS_DIR / "clf_model.joblib").exists():
            from src.models.classification import load_model
            model, scaler, feature_cols = load_model()
            import numpy as np
            X = scaler.transform(df[feature_cols].fillna(0).values)
            probas = model.predict_proba(X)[:, 1]
            fig2 = px.histogram(
                x=probas, nbins=40, color_discrete_sequence=["#e67e22"],
                labels={"x": "Fraud Probability"},
                title="Distribution of P(fraud) across all providers"
            )
            fig2.add_vline(x=0.5, line_dash="dash", line_color="red",
                           annotation_text="Decision threshold")
            st.plotly_chart(fig2, use_container_width=True)
        else:
            st.info("Train the models first: `python scripts/train_all.py`")

    with col_b:
        st.subheader("Top 15 Features by SHAP Importance")
        if shap_global:
            top15 = dict(list(shap_global.items())[:15])
            fig3 = px.bar(
                x=list(top15.values()),
                y=list(top15.keys()),
                orientation="h",
                labels={"x": "Mean |SHAP value|", "y": "Feature"},
                title="Global SHAP Feature Importance",
                color=list(top15.values()),
                color_continuous_scale="Oranges",
            )
            fig3.update_layout(yaxis={"autorange": "reversed"}, showlegend=False)
            st.plotly_chart(fig3, use_container_width=True)
        else:
            st.info("SHAP values not yet computed. Run `python scripts/train_all.py`.")

    st.divider()
    st.subheader("Provider-Level Fraud Score Lookup")
    provider_list = sorted(df["Provider"].unique())
    selected = st.selectbox("Select Provider", provider_list, key="clf_prov")
    if st.button("Run Classification", key="btn_clf"):
        from src.models.classification import predict_provider
        from src.models.inference import explain_provider
        result = predict_provider(selected, df)
        exp = explain_provider(selected, df)
        col1, col2, col3 = st.columns(3)
        col1.metric("Fraud Probability", f"{result.get('fraud_probability', 0):.1%}")
        col2.metric("Flag", "🚨 FRAUD" if result.get('fraud_flag') else "✅ LEGITIMATE")
        if "FraudLabel" in df.columns:
            col3.metric("Ground Truth", "Fraud" if result.get('true_label') else "Legitimate")

        if exp.get("top_features"):
            feat_df = pd.DataFrame(
                {"Feature": list(exp["top_features"].keys()),
                 "SHAP Value": list(exp["top_features"].values())}
            )
            fig4 = px.bar(feat_df, x="SHAP Value", y="Feature", orientation="h",
                          color="SHAP Value",
                          color_continuous_scale="RdBu",
                          title=f"Local SHAP Explanation – {selected}")
            fig4.update_layout(yaxis={"autorange": "reversed"})
            st.plotly_chart(fig4, use_container_width=True)


# ══════════════════════════════════════════════════════════════════════════════
# TAB 2 – Clinical Risk / Prediction
# ══════════════════════════════════════════════════════════════════════════════
with tabs[2]:
    st.header("🩺 Clinical Risk – Prediction (Treatment Pillar)")
    df = load_features()

    col1, col2 = st.columns(2)
    for target_name, label, col in [
        ("reimbursement", "Expected Reimbursement ($)", col1),
        ("los", "Avg Length of Stay (days)", col2),
    ]:
        metrics_path = MODELS_DIR / f"pred_{target_name}_metrics.json"
        if metrics_path.exists():
            import json
            with open(metrics_path) as f:
                m = json.load(f)
            with col:
                st.subheader(label)
                c1, c2, c3 = st.columns(3)
                c1.metric("MAE", f"{m.get('mae', 0):,.2f}")
                c2.metric("RMSE", f"{m.get('rmse', 0):,.2f}")
                c3.metric("R²", f"{m.get('r2', 0):.3f}")
        else:
            col.info(f"'{target_name}' model not trained yet.")

    st.divider()
    st.subheader("Provider Reimbursement Prediction")
    provider_list = sorted(df["Provider"].unique())
    sel_p = st.selectbox("Select Provider", provider_list, key="pred_prov")
    if st.button("Predict Reimbursement & LOS", key="btn_pred"):
        from src.models.prediction import predict_provider as pred_prov
        r1 = pred_prov(sel_p, "reimbursement", df)
        r2 = pred_prov(sel_p, "los", df)
        c1, c2 = st.columns(2)
        c1.metric("Predicted Total Reimbursement",
                  f"${r1.get('predicted_value', 0):,.0f}")
        c2.metric("Predicted Avg LOS",
                  f"{r2.get('predicted_value', 0):.1f} days")

    st.divider()
    st.subheader("Actual Reimbursement Distribution")
    if "IP_TotalReimbursed" in df.columns:
        fig5 = px.box(df, y="IP_TotalReimbursed", color="PotentialFraud",
                      color_discrete_map={"Yes": "#e74c3c", "No": "#2ecc71"},
                      labels={"IP_TotalReimbursed": "Total IP Reimbursement"},
                      title="Inpatient Reimbursement by Fraud Status")
        st.plotly_chart(fig5, use_container_width=True)


# ══════════════════════════════════════════════════════════════════════════════
# TAB 3 – Operations
# ══════════════════════════════════════════════════════════════════════════════
with tabs[3]:
    st.header("⚙️ Operations – Clustering & Time-Series Anomaly Detection")
    clusters = load_clusters()
    anomalies = load_anomalies()
    ts_df = load_timeseries()

    # ── Clustering section
    st.subheader("Provider Peer-Group Clustering (KMeans)")
    if not clusters.empty:
        c1, c2 = st.columns(2)
        with c1:
            cluster_summary = clusters.groupby("Cluster").agg(
                Count=("Provider", "count"),
                AvgDistance=("DistanceToCentroid", "mean"),
                Outliers=("OutlierFlag", "sum"),
            ).reset_index()
            st.dataframe(cluster_summary, use_container_width=True)
        with c2:
            fig6 = px.scatter(
                clusters, x="DistanceToCentroid", y="Cluster",
                color="OutlierFlag",
                color_discrete_map={0: "#3498db", 1: "#e74c3c"},
                hover_data=["Provider"],
                title="Provider Distance to Cluster Centroid",
                labels={"DistanceToCentroid": "Distance to Centroid",
                        "OutlierFlag": "Outlier"},
            )
            st.plotly_chart(fig6, use_container_width=True)
    else:
        st.info("Cluster data not found. Run `python scripts/train_all.py`.")

    st.divider()

    # ── Anomaly section
    st.subheader("Time-Series Anomaly Detection")
    if not anomalies.empty:
        n_providers = anomalies["Provider"].nunique()
        n_anom = int(anomalies["AnomalyFlag"].sum())
        c1, c2, c3 = st.columns(3)
        c1.metric("Providers Analyzed", n_providers)
        c2.metric("Anomalous Months", n_anom)
        c3.metric("Anomaly Rate", f"{n_anom / len(anomalies):.1%}")

        # Provider time-series viewer
        provider_list_ops = sorted(anomalies["Provider"].unique())
        sel_ts = st.selectbox("Select Provider for Time-Series View",
                               provider_list_ops, key="ts_prov")
        prov_ts = anomalies[anomalies["Provider"] == sel_ts].sort_values("YearMonth")

        if not prov_ts.empty:
            fig7 = go.Figure()
            fig7.add_trace(go.Scatter(
                x=prov_ts["YearMonth"], y=prov_ts["TotalReimbursed"],
                mode="lines+markers", name="Monthly Reimbursement",
                line=dict(color="#3498db"),
            ))
            anom_pts = prov_ts[prov_ts["AnomalyFlag"] == 1]
            if not anom_pts.empty:
                fig7.add_trace(go.Scatter(
                    x=anom_pts["YearMonth"], y=anom_pts["TotalReimbursed"],
                    mode="markers", name="⚠ Anomaly",
                    marker=dict(color="#e74c3c", size=12, symbol="x"),
                ))
            fig7.update_layout(
                title=f"Monthly Billing Time-Series – {sel_ts}",
                xaxis_title="Month", yaxis_title="Total Reimbursed ($)",
            )
            st.plotly_chart(fig7, use_container_width=True)
    else:
        st.info("Anomaly data not found. Run `python scripts/train_all.py`.")


# ══════════════════════════════════════════════════════════════════════════════
# TAB 4 – Investigator Agent
# ══════════════════════════════════════════════════════════════════════════════
with tabs[4]:
    st.header("🕵️ Investigator Agent – Agentic Chain-of-Thought Reasoning")
    st.markdown(
        "Pick a provider and watch the agent execute a **5-step investigation chain**. "
        "The agent calls ML tools, gathers evidence, and synthesizes a final recommendation "
        f"using {'a live LLM' if LLM_AVAILABLE else 'a deterministic fallback (no API key)'}."
    )

    df_agent = load_features()
    providers_agent = sorted(df_agent["Provider"].unique())
    sel_agent = st.selectbox("Select Provider to Investigate", providers_agent, key="agent_prov")

    col_btn, col_info = st.columns([1, 3])
    with col_btn:
        run_agent = st.button("🔍 Investigate", type="primary", key="btn_agent")
    with col_info:
        if "FraudLabel" in df_agent.columns:
            row = df_agent[df_agent["Provider"] == sel_agent].iloc[0]
            ground_truth = "Fraud" if row["FraudLabel"] == 1 else "Legitimate"
            st.markdown(f"**Ground truth:** `{ground_truth}`  _(only visible in demo mode)_")

    if run_agent:
        from src.agent.agent import investigate_streaming, STEP_LABELS

        st.divider()
        st.subheader("🔗 Investigation Chain")

        progress_bar = st.progress(0)
        step_containers = {}
        for i, label in enumerate(STEP_LABELS):
            step_containers[label] = st.expander(f"⏳ {label}", expanded=False)

        recommendation_placeholder = st.empty()

        steps_completed = 0
        final_result = {}

        for event in investigate_streaming(sel_agent):
            if event["step"] == "done":
                final_result = event
                break

            label = event["step"]
            status = event["status"]
            data = event.get("data")

            if status == "running":
                with step_containers[label]:
                    st.write("⏳ Running…")
            elif status == "done" and data:
                steps_completed += 1
                progress_bar.progress(steps_completed / len(STEP_LABELS))
                with step_containers[label]:
                    st.success("✅ Complete")
                    if isinstance(data, dict):
                        # Pretty-print key results
                        if "fraud_probability" in data:
                            prob = data["fraud_probability"]
                            color = "#e74c3c" if data.get("fraud_flag") else "#2ecc71"
                            st.markdown(
                                f"**Fraud Probability:** "
                                f"<span style='color:{color}; font-size:1.2em'>"
                                f"**{prob:.1%}**</span>",
                                unsafe_allow_html=True,
                            )
                        elif "anomaly_months" in data:
                            st.write(f"Anomalous months: **{data['anomaly_months']}** "
                                     f"/ {data.get('total_months', '?')}")
                        elif "cluster" in data:
                            flag = "🚨 Outlier" if data.get("outlier_flag") else "✅ Within peer group"
                            st.write(f"Cluster: **{data['cluster']}** | {flag}")
                        elif "top_features" in data:
                            feat_items = data.get("top_features", {})
                            if feat_items:
                                feat_df = pd.DataFrame(
                                    {"Feature": list(feat_items.keys()),
                                     "SHAP": list(feat_items.values())}
                                )
                                st.dataframe(feat_df, use_container_width=True)
                        else:
                            for k, v in list(data.items())[:5]:
                                st.write(f"- **{k}:** {v}")

        progress_bar.progress(1.0)

        # Final recommendation
        st.divider()
        st.subheader("📋 Investigation Report")
        rec_text = final_result.get("recommendation", "")
        llm_src = final_result.get("llm_source", "fallback")
        src_badge = {
            "groq": f"🟢 Groq ({GROQ_MODEL})",
            "openai": "🟢 GPT-4o Mini",
            "anthropic": "🟢 Claude Haiku",
            "fallback": "🟡 Rule-based fallback",
        }.get(llm_src, llm_src)

        st.caption(f"Generated by: {src_badge}")
        st.markdown(rec_text)

        # Collapsible raw evidence
        with st.expander("🔬 Raw Evidence (all steps)"):
            steps_data = final_result.get("steps", {})
            for step_name, step_data in steps_data.items():
                st.markdown(f"**{step_name}**")
                st.json(step_data)
