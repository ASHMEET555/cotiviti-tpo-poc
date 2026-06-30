# Eraser.io Architecture Diagram Prompt

Copy everything inside the block below and paste it into Eraser AI diagram generator.

---

```
Create a professional end-to-end system architecture diagram for a healthcare payment integrity platform called "TPO Sentinel" (Treatment, Payment, Operations).

STYLE:
- Clean, modern, presentation-ready (suitable for a corporate intern assessment slide)
- Light background, teal and soft coral accent colors (healthcare / trust aesthetic)
- Left-to-right or top-to-bottom flow with clear layer boxes
- No file names, no code, no repository paths — concepts and capabilities only
- Use icons or simple shapes for layers (database, ML, brain/LLM, dashboard)

TITLE (top): TPO Sentinel — Clinical Decision Making & Payment Integrity Architecture

TOP BANNER — linear pipeline (horizontal arrows):
Raw Claims Data → Feature Engineering → ML Models → Agent Tools → LLM Reasoning → Recommendation (Pay / Review / Deny / Audit)

FIVE LAYERS (stacked or grouped boxes):

1) DATA LAYER
   - Medicare claims data (real, public, labeled)
   - Synthetic fallback for offline demo
   Label: "Data Layer"

2) FEATURE ENGINEERING LAYER
   - Provider-level billing aggregates
   - Monthly time-series per provider
   Label: "Feature Engineering"

3) ML LAYER — map techniques to TPO pillars (show three sub-groups inside this box):

   PAYMENT pillar:
   - Classification: fraud / waste / abuse probability scoring
   - Inference (SHAP): explainable feature attribution — why flagged

   TREATMENT pillar:
   - Prediction: expected reimbursement and clinical risk (length of stay)

   OPERATIONS pillar:
   - Clustering: provider peer-group benchmarking
   - Time-Series Anomaly Detection: unusual billing spikes over time

   Label: "ML Layer — Six TPO Techniques"

4) AGENTIC LAYER
   - ML Tool Registry (wraps all model outputs as callable tools)
   - Chain-of-Thought Orchestrator (5 steps: Classify → Anomalies → Peer Compare → Explain → Synthesize)
   - Generative AI (Groq LLM with rule-based fallback)
   - Output: narrative investigation report with cited evidence
   Label: "Agentic Layer"

5) UI LAYER
   - Interactive Streamlit dashboard
   - Tabs: Overview, Payment Integrity, Clinical Risk, Operations, Investigator Agent
   - Real-time visualization of agent reasoning chain
   Label: "UI Layer"

CONNECTIONS:
- Data Layer feeds Feature Engineering
- Feature Engineering feeds all ML components
- All ML outputs feed Agent Tool Registry
- Agent orchestrator calls LLM for final recommendation
- Dashboard displays ML results and agent narrative

OPTIONAL CALLOUT BOX (side or bottom):
"Built for Cotiviti payment integrity: pre-payment accuracy, explainable audits, peer benchmarking, automated FWA triage"

Keep the diagram readable in one slide. Avoid clutter. Emphasize the journey from raw claims to an explainable AI recommendation.
```

---

## Agent chain (optional second small diagram)

If Eraser asks for more detail on the agent, use:

```
Create a small flowchart for "Investigator Agent — 5-Step Chain of Thought":

1. Classify — fraud probability
2. Anomaly Check — billing time-series spikes
3. Peer Compare — cluster and outlier flag
4. Explain — top SHAP drivers
5. Synthesize — LLM writes recommendation (Pay / Review / Deny / Audit)

Show arrows left to right. Teal accent. No code or file names.
```
