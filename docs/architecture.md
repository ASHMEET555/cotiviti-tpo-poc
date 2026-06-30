# TPO Sentinel – Architecture

## End-to-End Data Flow

```mermaid
flowchart LR
    A[Raw Claims Data] --> B[Feature Engineering]
    B --> C[ML Models]
    C --> D[Agent Tools]
    D --> E[LLM Reasoning]
    E --> F[Recommendation]
```

## Layered Architecture

```mermaid
flowchart TB
    subgraph dataLayer ["Data Layer"]
        RD["Medicare Claims Dataset"]
        SF["Synthetic Fallback"]
    end

    subgraph featureLayer ["Feature Engineering"]
        FE["Provider Aggregates"]
        TS["Monthly Billing Time-Series"]
    end

    subgraph mlLayer ["ML Layer — TPO Techniques"]
        direction TB
        subgraph paymentML ["Payment"]
            CLF["Classification — Fraud Probability"]
            SHAP["Inference — SHAP Explainability"]
        end
        subgraph treatmentML ["Treatment"]
            REG["Prediction — Reimbursement & Clinical Risk"]
        end
        subgraph opsML ["Operations"]
            CLU["Clustering — Peer Groups"]
            ANO["Time-Series Anomaly Detection"]
        end
    end

    subgraph agentLayer ["Agentic Layer"]
        TOOLS["ML Tool Registry"]
        COT["Chain-of-Thought Orchestrator"]
        LLM["Generative AI — Groq / Fallback"]
        TOOLS --> COT --> LLM
    end

    subgraph uiLayer ["UI Layer"]
        UI["Streamlit Dashboard — 5 Tabs + Live Agent Chain"]
    end

    RD --> FE
    SF --> FE
    RD --> TS
    SF --> TS

    FE --> CLF
    FE --> REG
    FE --> CLU
    TS --> ANO
    CLF --> SHAP

    CLF --> TOOLS
    SHAP --> TOOLS
    REG --> TOOLS
    CLU --> TOOLS
    ANO --> TOOLS

    LLM --> UI
    CLF -.-> UI
    SHAP -.-> UI
    REG -.-> UI
    CLU -.-> UI
    ANO -.-> UI
```

## Technique → TPO Mapping

| Technique | TPO Pillar | Description |
|---|---|---|
| Classification | Payment | XGBoost trained on provider/claim features; outputs P(fraud) |
| Inference (SHAP) | Payment | Explains *why* a claim is flagged with feature contribution values |
| Prediction (regression) | Treatment | Predicts expected reimbursement and clinical risk score |
| Clustering | Operations | Groups providers into peer cohorts; flags statistical outliers |
| Time-Series Anomaly | Operations | Detects unusual months in provider billing volume/amounts |
| Agentic Chain Reasoning | All | LLM orchestrates a 5-step investigation and writes a narrative recommendation |

## Agent Chain-of-Thought Steps

1. **Classify** – get fraud probability for provider/claim.
2. **Anomaly check** – query last 24 months for billing spikes.
3. **Peer compare** – locate provider's cluster; measure deviation from centroid.
4. **Explain** – pull top SHAP features driving the score.
5. **Synthesize** – generate final recommendation (Pay / Review / Deny / Audit) with cited evidence.
