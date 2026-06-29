"""
Feature engineering pipeline.

Reads the raw Kaggle / synthetic CSVs and produces:
  - data/processed/provider_features.parquet  (one row per provider, used by
      classification, prediction, clustering, inference)
  - data/processed/timeseries.parquet         (monthly billing series per provider,
      used by anomaly detection)

Run directly:
    python src/data/features.py
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from src.config import DATA_RAW, DATA_PROC

PROVIDER_FEATURES_PATH = DATA_PROC / "provider_features.parquet"
TIMESERIES_PATH = DATA_PROC / "timeseries.parquet"

CHRONIC_COLS = [
    "ChronicCond_Alzheimer", "ChronicCond_Heartfailure",
    "ChronicCond_KidneyDisease", "ChronicCond_Cancer",
    "ChronicCond_ObstrPulmonary", "ChronicCond_Depression",
    "ChronicCond_Diabetes", "ChronicCond_IschemicHeart",
    "ChronicCond_Osteoporasis", "ChronicCond_RheumatoidArthritis",
    "ChronicCond_Stroke",
]

# Real Kaggle CSVs use slightly different column names than our synthetic generator.
COLUMN_ALIASES = {
    "DischargeDt": "ClmDischargeDt",
    "NoOfMonths_PartACov": "NoOfMonths_PartACoverage",
    "NoOfMonths_PartBCov": "NoOfMonths_PartBCoverage",
    "ChronicCond_rheumatoidarthritis": "ChronicCond_RheumatoidArthritis",
    "ChronicCond_stroke": "ChronicCond_Stroke",
}


def _normalize_columns(df: pd.DataFrame) -> pd.DataFrame:
    return df.rename(columns={k: v for k, v in COLUMN_ALIASES.items() if k in df.columns})


def _read_csv(path: Path, date_cols: list[str]) -> pd.DataFrame:
    df = _normalize_columns(pd.read_csv(path, low_memory=False))
    for col in date_cols:
        if col in df.columns:
            df[col] = pd.to_datetime(df[col], errors="coerce")
    return df


def load_raw() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    labels = pd.read_csv(DATA_RAW / "Train-1542865627584.csv")
    bene = _read_csv(DATA_RAW / "Train_Beneficiarydata-1542865627584.csv", ["DOB", "DOD"])
    inp = _read_csv(
        DATA_RAW / "Train_Inpatientdata-1542865627584.csv",
        ["ClaimStartDt", "ClaimEndDt", "AdmissionDt", "ClmDischargeDt"],
    )
    outp = _read_csv(
        DATA_RAW / "Train_Outpatientdata-1542865627584.csv",
        ["ClaimStartDt", "ClaimEndDt"],
    )
    return labels, bene, inp, outp


def _bene_features(bene: pd.DataFrame) -> pd.DataFrame:
    bene = bene.copy()
    bene["Age"] = ((pd.Timestamp("2010-12-31") - bene["DOB"]).dt.days / 365.25).round(1)
    bene["IsDead"] = bene["DOD"].notna().astype(int)
    chronic_present = [(bene[c] == 1).astype(int) for c in CHRONIC_COLS if c in bene.columns]
    bene["ChronicCondCount"] = sum(chronic_present) if chronic_present else 0
    for col in ("IPAnnualReimbursementAmt", "OPAnnualReimbursementAmt"):
        if col not in bene.columns:
            bene[col] = 0
    return bene[["BeneID", "Age", "IsDead", "ChronicCondCount",
                 "IPAnnualReimbursementAmt", "OPAnnualReimbursementAmt"]]


def _claim_agg(df: pd.DataFrame, prefix: str) -> pd.DataFrame:
    """Aggregate one claims table (inpatient or outpatient) by Provider."""
    if "DeductibleAmtPaid" not in df.columns:
        df = df.copy()
        df["DeductibleAmtPaid"] = 0
    agg = df.groupby("Provider").agg(
        **{
            f"{prefix}_ClaimCount": ("ClaimID", "count"),
            f"{prefix}_TotalReimbursed": ("InscClaimAmtReimbursed", "sum"),
            f"{prefix}_AvgReimbursed": ("InscClaimAmtReimbursed", "mean"),
            f"{prefix}_MaxReimbursed": ("InscClaimAmtReimbursed", "max"),
            f"{prefix}_TotalDeductible": ("DeductibleAmtPaid", "sum"),
            f"{prefix}_UniqueBenes": ("BeneID", "nunique"),
        }
    ).reset_index()
    return agg


def _physician_diversity(df: pd.DataFrame) -> pd.DataFrame:
    """Count unique attending physicians per provider."""
    col = "AttendingPhysician"
    if col not in df.columns:
        return pd.DataFrame({"Provider": df["Provider"].unique(),
                             "UniquePhysicians": 0})
    return (
        df.groupby("Provider")[col]
        .nunique()
        .reset_index()
        .rename(columns={col: "UniquePhysicians"})
    )


def _los_features(inp: pd.DataFrame) -> pd.DataFrame:
    """Length-of-stay statistics per provider (inpatient only)."""
    df = inp.copy()
    if "AdmissionDt" in df.columns and "ClmDischargeDt" in df.columns:
        df["LOS"] = (df["ClmDischargeDt"] - df["AdmissionDt"]).dt.days.clip(lower=0)
    else:
        df["LOS"] = (df["ClaimEndDt"] - df["ClaimStartDt"]).dt.days.clip(lower=0)
    return (
        df.groupby("Provider")["LOS"]
        .agg(AvgLOS="mean", MaxLOS="max")
        .reset_index()
    )


def build_provider_features(
    labels: pd.DataFrame,
    bene: pd.DataFrame,
    inp: pd.DataFrame,
    outp: pd.DataFrame,
) -> pd.DataFrame:
    bene_feats = _bene_features(bene)
    inp_agg = _claim_agg(inp, "IP")
    outp_agg = _claim_agg(outp, "OP")
    phys = _physician_diversity(pd.concat([
        inp[["Provider", "AttendingPhysician"]] if "AttendingPhysician" in inp.columns else inp[["Provider"]],
        outp[["Provider", "AttendingPhysician"]] if "AttendingPhysician" in outp.columns else outp[["Provider"]],
    ]))
    los = _los_features(inp)

    # Merge bene info through claims
    inp_bene = inp.merge(bene_feats, on="BeneID", how="left")
    bene_by_provider = inp_bene.groupby("Provider").agg(
        AvgBeneAge=("Age", "mean"),
        AvgChronicCount=("ChronicCondCount", "mean"),
        DeadBeneFrac=("IsDead", "mean"),
    ).reset_index()

    # Join everything
    df = labels.copy()
    df = df.merge(inp_agg, on="Provider", how="left")
    df = df.merge(outp_agg, on="Provider", how="left")
    df = df.merge(phys, on="Provider", how="left")
    df = df.merge(los, on="Provider", how="left")
    df = df.merge(bene_by_provider, on="Provider", how="left")

    # Derived ratios
    df["IP_ReimbPerClaim"] = df["IP_TotalReimbursed"] / (df["IP_ClaimCount"] + 1)
    df["OP_ReimbPerClaim"] = df["OP_TotalReimbursed"] / (df["OP_ClaimCount"] + 1)
    df["IP_ClaimsPerBene"] = df["IP_ClaimCount"] / (df["IP_UniqueBenes"] + 1)
    df["OP_ClaimsPerBene"] = df["OP_ClaimCount"] / (df["OP_UniqueBenes"] + 1)
    df["PhysiciansPerClaim"] = df["UniquePhysicians"] / (
        df["IP_ClaimCount"].fillna(0) + df["OP_ClaimCount"].fillna(0) + 1
    )

    # Binary label
    df["FraudLabel"] = (df["PotentialFraud"] == "Yes").astype(int)

    df = df.fillna(0)
    return df


def build_timeseries(inp: pd.DataFrame, outp: pd.DataFrame) -> pd.DataFrame:
    """Monthly billing aggregates per provider."""
    frames = []
    for df in [inp, outp]:
        tmp = df[["Provider", "ClaimStartDt", "InscClaimAmtReimbursed", "ClaimID"]].copy()
        tmp["YearMonth"] = tmp["ClaimStartDt"].dt.to_period("M")
        frames.append(tmp)

    combined = pd.concat(frames, ignore_index=True)
    ts = (
        combined.groupby(["Provider", "YearMonth"])
        .agg(
            ClaimCount=("ClaimID", "count"),
            TotalReimbursed=("InscClaimAmtReimbursed", "sum"),
        )
        .reset_index()
    )
    ts["YearMonth"] = ts["YearMonth"].dt.to_timestamp()
    return ts.sort_values(["Provider", "YearMonth"]).reset_index(drop=True)


def run() -> None:
    print("Loading raw data …")
    labels, bene, inp, outp = load_raw()
    print(f"  Providers: {len(labels)}, Beneficiaries: {len(bene)}, "
          f"IP claims: {len(inp)}, OP claims: {len(outp)}")

    print("Building provider features …")
    pf = build_provider_features(labels, bene, inp, outp)
    pf.to_parquet(PROVIDER_FEATURES_PATH, index=False)
    print(f"  Saved → {PROVIDER_FEATURES_PATH}  shape={pf.shape}")

    print("Building time-series …")
    ts = build_timeseries(inp, outp)
    ts.to_parquet(TIMESERIES_PATH, index=False)
    print(f"  Saved → {TIMESERIES_PATH}  shape={ts.shape}")


if __name__ == "__main__":
    run()
