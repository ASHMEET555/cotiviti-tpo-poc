"""
Generate a synthetic, schema-matched healthcare claims dataset.

Produces the same four CSV files that the Kaggle dataset contains so that
every downstream module works identically whether real or synthetic data
is loaded. Run directly to (re)create synthetic data:

    python src/data/synthetic.py
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from src.config import DATA_RAW

RNG = np.random.default_rng(42)

N_PROVIDERS = 200
N_BENEFICIARIES = 5_000
N_CLAIMS_IN = 8_000
N_CLAIMS_OUT = 20_000
FRAUD_RATE = 0.25  # 25 % of providers are fraudulent (mirrors real dataset)


# ── Helpers ────────────────────────────────────────────────────────────────────

def rand_dates(start: str, end: str, n: int) -> pd.Series:
    s = pd.Timestamp(start).value
    e = pd.Timestamp(end).value
    return pd.Series(pd.to_datetime(RNG.integers(s, e, n)))


def provider_ids(n: int) -> list[str]:
    return [f"PRV{i:05d}" for i in range(1, n + 1)]


def bene_ids(n: int) -> list[str]:
    return [f"BNE{i:06d}" for i in range(1, n + 1)]


def claim_ids(prefix: str, n: int) -> list[str]:
    return [f"{prefix}{i:08d}" for i in range(1, n + 1)]


# ── 1. Provider fraud labels ───────────────────────────────────────────────────

def make_provider_labels() -> pd.DataFrame:
    pids = provider_ids(N_PROVIDERS)
    fraud = RNG.choice(["Yes", "No"], size=N_PROVIDERS,
                       p=[FRAUD_RATE, 1 - FRAUD_RATE])
    return pd.DataFrame({"Provider": pids, "PotentialFraud": fraud})


# ── 2. Beneficiary data ────────────────────────────────────────────────────────

def make_beneficiary() -> pd.DataFrame:
    bids = bene_ids(N_BENEFICIARIES)
    dob = rand_dates("1930-01-01", "1990-01-01", N_BENEFICIARIES)
    dod = [None] * N_BENEFICIARIES
    # ~5 % deceased
    dead_idx = RNG.choice(N_BENEFICIARIES, size=int(0.05 * N_BENEFICIARIES), replace=False)
    for i in dead_idx:
        dod[i] = rand_dates("2010-01-01", "2010-12-31", 1).iloc[0]

    chronic_cols = [f"ChronicCond_{c}" for c in [
        "Alzheimer", "Heartfailure", "KidneyDisease", "Cancer",
        "ObstrPulmonary", "Depression", "Diabetes", "IschemicHeart",
        "Osteoporasis", "RheumatoidArthritis", "Stroke"
    ]]
    df = pd.DataFrame({
        "BeneID": bids,
        "DOB": dob,
        "DOD": dod,
        "Gender": RNG.choice([1, 2], N_BENEFICIARIES),
        "Race": RNG.choice([1, 2, 3, 5], N_BENEFICIARIES),
        "RenalDiseaseIndicator": RNG.choice(["Y", "0"], N_BENEFICIARIES,
                                             p=[0.08, 0.92]),
        "State": RNG.integers(1, 52, N_BENEFICIARIES),
        "County": RNG.integers(1, 500, N_BENEFICIARIES),
        "NoOfMonths_PartACoverage": RNG.integers(0, 13, N_BENEFICIARIES),
        "NoOfMonths_PartBCoverage": RNG.integers(0, 13, N_BENEFICIARIES),
        "IPAnnualReimbursementAmt": RNG.integers(0, 50000, N_BENEFICIARIES),
        "IPAnnualDeductibleAmt": RNG.integers(0, 5000, N_BENEFICIARIES),
        "OPAnnualReimbursementAmt": RNG.integers(0, 20000, N_BENEFICIARIES),
        "OPAnnualDeductibleAmt": RNG.integers(0, 2000, N_BENEFICIARIES),
    })
    for col in chronic_cols:
        df[col] = RNG.choice([1, 2], N_BENEFICIARIES, p=[0.3, 0.7])
    return df


# ── 3. Inpatient claims ────────────────────────────────────────────────────────

def make_inpatient(providers: list[str], benes: list[str]) -> pd.DataFrame:
    cids = claim_ids("CLM-IN-", N_CLAIMS_IN)
    start = rand_dates("2009-01-01", "2010-12-01", N_CLAIMS_IN)
    end = start + pd.to_timedelta(RNG.integers(1, 30, N_CLAIMS_IN), unit="D")
    admit = start - pd.to_timedelta(RNG.integers(0, 3, N_CLAIMS_IN), unit="D")
    discharge = end

    diag_cols = [f"ClmDiagnosisCode_{i}" for i in range(1, 11)]
    proc_cols = [f"ClmProcedureCode_{i}" for i in range(1, 7)]
    physician_cols = ["AttendingPhysician", "OperatingPhysician", "OtherPhysician"]

    diag_pool = [f"D{i:03d}" for i in range(100, 999)]
    proc_pool = [f"P{i:04d}" for i in range(1000, 9999)]
    phys_pool = [f"PHY{i:05d}" for i in range(1, 5001)]

    df = pd.DataFrame({
        "ClaimID": cids,
        "BeneID": RNG.choice(benes, N_CLAIMS_IN),
        "ClaimStartDt": start,
        "ClaimEndDt": end,
        "Provider": RNG.choice(providers, N_CLAIMS_IN),
        "InscClaimAmtReimbursed": RNG.integers(500, 80000, N_CLAIMS_IN),
        "DeductibleAmtPaid": RNG.integers(0, 5000, N_CLAIMS_IN),
        "ClmAdmitDiagnosisCode": RNG.choice(diag_pool, N_CLAIMS_IN),
        "AdmissionDt": admit,
        "ClmDischargeDt": discharge,
        "DiagnosisGroupCode": RNG.integers(100, 999, N_CLAIMS_IN),
    })
    for col in diag_cols:
        vals = RNG.choice(diag_pool, N_CLAIMS_IN).astype(object)
        vals[RNG.random(N_CLAIMS_IN) < 0.15] = None
        df[col] = vals
    for col in proc_cols:
        vals = RNG.choice(proc_pool, N_CLAIMS_IN).astype(object)
        vals[RNG.random(N_CLAIMS_IN) < 0.40] = None
        df[col] = vals
    for col in physician_cols:
        vals = RNG.choice(phys_pool, N_CLAIMS_IN).astype(object)
        vals[RNG.random(N_CLAIMS_IN) < 0.20] = None
        df[col] = vals
    return df


# ── 4. Outpatient claims ───────────────────────────────────────────────────────

def make_outpatient(providers: list[str], benes: list[str]) -> pd.DataFrame:
    cids = claim_ids("CLM-OP-", N_CLAIMS_OUT)
    start = rand_dates("2009-01-01", "2010-12-31", N_CLAIMS_OUT)
    end = start + pd.to_timedelta(RNG.integers(0, 5, N_CLAIMS_OUT), unit="D")

    diag_cols = [f"ClmDiagnosisCode_{i}" for i in range(1, 11)]
    proc_cols = [f"ClmProcedureCode_{i}" for i in range(1, 7)]
    diag_pool = [f"D{i:03d}" for i in range(100, 999)]
    proc_pool = [f"P{i:04d}" for i in range(1000, 9999)]
    phys_pool = [f"PHY{i:05d}" for i in range(1, 5001)]

    df = pd.DataFrame({
        "ClaimID": cids,
        "BeneID": RNG.choice(benes, N_CLAIMS_OUT),
        "ClaimStartDt": start,
        "ClaimEndDt": end,
        "Provider": RNG.choice(providers, N_CLAIMS_OUT),
        "InscClaimAmtReimbursed": RNG.integers(50, 10000, N_CLAIMS_OUT),
        "DeductibleAmtPaid": RNG.integers(0, 500, N_CLAIMS_OUT),
        "ClmAdmitDiagnosisCode": RNG.choice(diag_pool, N_CLAIMS_OUT),
        "AttendingPhysician": RNG.choice(phys_pool, N_CLAIMS_OUT),
    })
    for col in diag_cols:
        vals = RNG.choice(diag_pool, N_CLAIMS_OUT).astype(object)
        vals[RNG.random(N_CLAIMS_OUT) < 0.15] = None
        df[col] = vals
    for col in proc_cols:
        vals = RNG.choice(proc_pool, N_CLAIMS_OUT).astype(object)
        vals[RNG.random(N_CLAIMS_OUT) < 0.40] = None
        df[col] = vals
    return df


# ── Main ───────────────────────────────────────────────────────────────────────

def generate_and_save() -> None:
    print("Generating synthetic dataset ...")
    labels = make_provider_labels()
    pids = labels["Provider"].tolist()
    bene_df = make_beneficiary()
    bids = bene_df["BeneID"].tolist()
    inpatient = make_inpatient(pids, bids)
    outpatient = make_outpatient(pids, bids)

    labels.to_csv(DATA_RAW / "Train-1542865627584.csv", index=False)
    bene_df.to_csv(DATA_RAW / "Train_Beneficiarydata-1542865627584.csv", index=False)
    inpatient.to_csv(DATA_RAW / "Train_Inpatientdata-1542865627584.csv", index=False)
    outpatient.to_csv(DATA_RAW / "Train_Outpatientdata-1542865627584.csv", index=False)
    print(f"Synthetic data written to {DATA_RAW}")


if __name__ == "__main__":
    generate_and_save()
