#!/usr/bin/env python3
"""
Compute data-driven weights for CoLI and BRI composites.

CoLI components:
    - utility_regularit          : 1 - scaled missed-utilities
    - subscription_adherence     : 1 - scaled aggregate subscription misses
    - expense_stability          : 1 - scaled deviation between desired and realised savings rates

BRI components:
    - camels_proxy               : scaled (Adjusted_Annual_Savings / Annual_Income)
    - upi_success                : 1 - scaled total missed payments
    - tier_weight_component      : scaled tier weight (Tier_1 > Tier_2 > Tier_3)

Targets:
    - reliability_target         : Payment_Reliability_Score scaled to [0,1]
    - default_flag               : High-risk threshold based on multiple indicators

Methods implemented:
    - Method A: absolute Pearson correlations -> weights
    - Method C: Ridge regression (positive coefficients, L2) -> weights

Outputs:
    - Weight tables for CoLI and BRI (both methods)
    - Calibrated logistic regression coefficients for mapping composite to default probability
    - Fairness summary (mean/std of composites by City_Tier and Occupation)

"""

import pathlib
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.linear_model import Ridge, LogisticRegression

# ---------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------
def minmax(series: pd.Series) -> pd.Series:
    """Return min-max scaled series (0 if constant)."""
    s = series.astype(float)
    min_, max_ = s.min(), s.max()
    if np.isclose(max_ - min_, 0):
        return pd.Series(np.zeros(len(series)), index=series.index)
    return (s - min_) / (max_ - min_)

def positive_normalised(weights: np.ndarray) -> np.ndarray:
    """Force non-negative, normalise to sum 1 (handles zeros)."""
    w = np.clip(weights, 0, None)
    denom = w.sum()
    return w / denom if denom > 0 else np.full_like(w, 1 / len(w))

def ridge_weights(X: np.ndarray, y: np.ndarray, alpha: float = 1.0) -> np.ndarray:
    """
    Fit Ridge regression with positive constraint; return normalised absolute coefficients.
    """
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    ridge = Ridge(alpha=alpha, fit_intercept=True, positive=True)
    ridge.fit(X_scaled, y)
    coef = np.abs(ridge.coef_)
    return positive_normalised(coef)

def correlation_weights(df_components: pd.DataFrame, target: pd.Series) -> pd.Series:
    """Absolute Pearson correlations → weights."""
    corr = df_components.apply(lambda col: np.abs(col.corr(target)))
    return corr / corr.sum()

def logistic_calibration(feature: pd.Series, target: pd.Series) -> LogisticRegression:
    """
    Fit 1-D logistic regression (with balanced class weight) for calibration.
    Returns fitted model so caller can inspect intercept/coefficient.
    """
    # Check if target has variation
    if target.nunique() < 2:
        print(f"WARNING: Target has only {target.nunique()} class(es). Skipping calibration.")
        return None
    
    model = LogisticRegression(class_weight="balanced", max_iter=1000)
    model.fit(feature.to_numpy().reshape(-1, 1), target.to_numpy())
    return model

def fairness_report(series: pd.Series, group: pd.Series, name: str) -> pd.DataFrame:
    """Return mean/std per group for quick fairness sanity check."""
    stats = (
        pd.DataFrame({name: series, "group": group})
        .groupby("group")[name]
        .agg(["mean", "std", "count"])
        .sort_values("mean", ascending=False)
    )
    return stats

# ---------------------------------------------------------------------
# Load data
# ---------------------------------------------------------------------
DATA_PATH = pathlib.Path("processed_data_with_banks.csv")
df = pd.read_csv(DATA_PATH)
df.columns = df.columns.str.strip()

# Replace +/-inf if any, drop rows where critical fields missing
df = df.replace([np.inf, -np.inf], np.nan)

required_cols = [
    "Missed_Utilities", "Missed_Rent", "Missed_Insurance", "Missed_Loan_Repayment",
    "Desired_Savings_Percentage", "Adjusted_Savings_Rate", "Adjusted_Annual_Savings",
    "Annual_Income", "Total_Missed_Payments", "Payment_Reliability_Score",
    "Missed_Payment_Rate", "City_Tier", "Occupation"
]
missing_cols = set(required_cols) - set(df.columns)
if missing_cols:
    raise ValueError(f"Dataset is missing expected columns: {missing_cols}")

df = df.dropna(subset=required_cols).reset_index(drop=True)

print(f"\n{'='*80}")
print("DATA LOADED & VALIDATED")
print(f"{'='*80}")
print(f"Total records: {len(df):,}")
print(f"Columns: {len(df.columns)}")

# ---------------------------------------------------------------------
# Feature engineering for component metrics
# ---------------------------------------------------------------------
print(f"\n{'='*80}")
print("ENGINEERING COMPONENT FEATURES")
print(f"{'='*80}")

# Utility regularity: fewer missed utility payments is better
df["utility_regularit"] = 1 - minmax(df["Missed_Utilities"])
print("✓ utility_regularit")

# Subscription adherence: aggregated misses across subscription-like items
subscription_miss = df[["Missed_Rent", "Missed_Insurance", "Missed_Loan_Repayment"]].sum(axis=1)
df["subscription_adherence"] = 1 - minmax(subscription_miss)
print("✓ subscription_adherence")

# Expense stability: small gap between planned and realised savings rate is better
savings_gap = (df["Desired_Savings_Percentage"] - df["Adjusted_Savings_Rate"]).abs()
df["expense_stability"] = 1 - minmax(savings_gap)
print("✓ expense_stability")

# CAMELS proxy: capital adequacy-like ratio (Adjusted_Annual_Savings / Annual_Income)
camels_raw = df["Adjusted_Annual_Savings"] / df["Annual_Income"].replace(0, np.nan)
df["camels_proxy"] = minmax(camels_raw.fillna(0)).clip(0, 1)
print("✓ camels_proxy")

# UPI success: fewer missed payments overall
df["upi_success"] = 1 - minmax(df["Total_Missed_Payments"])
print("✓ upi_success")

# Tier weight: higher for Tier_1, scaled to [0,1]
tier_map = {"Tier_1": 1.0, "Tier_2": 0.7, "Tier_3": 0.4}
df["tier_weight_component"] = df["City_Tier"].map(tier_map).fillna(0.55)
df["tier_weight_component"] = minmax(df["tier_weight_component"])
print("✓ tier_weight_component")

# Components dataframes
coli_components = df[["utility_regularit", "subscription_adherence", "expense_stability"]].copy()
bri_components = df[["camels_proxy", "upi_success", "tier_weight_component"]].copy()

# ---------------------------------------------------------------------
# Create better default target with variation
# ---------------------------------------------------------------------
print(f"\n{'='*80}")
print("CREATING DEFAULT TARGET")
print(f"{'='*80}")

# Multi-criteria default definition
# Default = 1 if ANY of these conditions:
#   1. Missed_Payment_Rate > 10%  (high overall miss rate)
#   2. Payment_Reliability_Score < 90  (low reliability)
#   3. Total_Missed_Payments > 75th percentile

missed_rate_threshold = 10.0
reliability_threshold = 90.0
missed_amount_threshold = df["Total_Missed_Payments"].quantile(0.75)

default_flag = (
    (df["Missed_Payment_Rate"] > missed_rate_threshold) |
    (df["Payment_Reliability_Score"] < reliability_threshold) |
    (df["Total_Missed_Payments"] > missed_amount_threshold)
).astype(int)

print(f"Default criteria:")
print(f"  - Missed Payment Rate > {missed_rate_threshold}%")
print(f"  - Payment Reliability Score < {reliability_threshold}")
print(f"  - Total Missed Payments > ₹{missed_amount_threshold:,.2f} (75th percentile)")

print(f"\nDefault distribution:")
print(default_flag.value_counts())
print(f"\nDefault rate: {default_flag.mean()*100:.2f}%")

if default_flag.nunique() < 2:
    print("\n⚠️  WARNING: Still only one class in default_flag. Adjusting thresholds...")
    # More lenient thresholds
    default_flag = (
        (df["Missed_Payment_Rate"] > 5.0) |
        (df["Payment_Reliability_Score"] < 95.0)
    ).astype(int)
    print(f"New default distribution:")
    print(default_flag.value_counts())

# Targets
reliability_target = (df["Payment_Reliability_Score"] / 100).clip(0, 1)

# ---------------------------------------------------------------------
# Method A: Correlation-based weights
# ---------------------------------------------------------------------
print(f"\n{'='*80}")
print("METHOD A: CORRELATION-BASED WEIGHTS")
print(f"{'='*80}")

coli_corr_weights = correlation_weights(coli_components, reliability_target)
bri_corr_weights = correlation_weights(bri_components, reliability_target)

print("\nCoLI correlation weights:")
for comp, weight in coli_corr_weights.items():
    print(f"  {comp:25s}: {weight:.4f}")

print("\nBRI correlation weights:")
for comp, weight in bri_corr_weights.items():
    print(f"  {comp:25s}: {weight:.4f}")

# ---------------------------------------------------------------------
# Method C: Ridge regression-based weights
# ---------------------------------------------------------------------
print(f"\n{'='*80}")
print("METHOD C: RIDGE REGRESSION WEIGHTS")
print(f"{'='*80}")

coli_ridge_weights = ridge_weights(coli_components.values, reliability_target.values, alpha=1.0)
bri_ridge_weights = ridge_weights(bri_components.values, reliability_target.values, alpha=1.0)

coli_ridge_weights = pd.Series(coli_ridge_weights, index=coli_components.columns)
bri_ridge_weights = pd.Series(bri_ridge_weights, index=bri_components.columns)

print("\nCoLI ridge weights:")
for comp, weight in coli_ridge_weights.items():
    print(f"  {comp:25s}: {weight:.4f}")

print("\nBRI ridge weights:")
for comp, weight in bri_ridge_weights.items():
    print(f"  {comp:25s}: {weight:.4f}")

# ---------------------------------------------------------------------
# Build composites (both sets of weights)
# ---------------------------------------------------------------------
coli_corr_score = (coli_components * coli_corr_weights).sum(axis=1)
bri_corr_score = (bri_components * bri_corr_weights).sum(axis=1)

coli_ridge_score = (coli_components * coli_ridge_weights).sum(axis=1)
bri_ridge_score = (bri_components * bri_ridge_weights).sum(axis=1)

# ---------------------------------------------------------------------
# Calibration: map composites -> default probability (via logistic regression)
# ---------------------------------------------------------------------
print(f"\n{'='*80}")
print("LOGISTIC CALIBRATION")
print(f"{'='*80}")

coli_corr_logit = logistic_calibration(coli_corr_score, default_flag)
bri_corr_logit = logistic_calibration(bri_corr_score, default_flag)

coli_ridge_logit = logistic_calibration(coli_ridge_score, default_flag)
bri_ridge_logit = logistic_calibration(bri_ridge_score, default_flag)

def logit_params(model: LogisticRegression) -> dict:
    if model is None:
        return {"intercept": None, "slope": None}
    coef = float(model.coef_.flatten()[0])
    intercept = float(model.intercept_[0])
    return {"intercept": intercept, "slope": coef}

print("\nCalibration parameters (for predicting default probability):")
print(f"\nCoLI_corr:  {logit_params(coli_corr_logit)}")
print(f"BRI_corr:   {logit_params(bri_corr_logit)}")
print(f"CoLI_ridge: {logit_params(coli_ridge_logit)}")
print(f"BRI_ridge:  {logit_params(bri_ridge_logit)}")

# ---------------------------------------------------------------------
# Fairness sanity checks
# ---------------------------------------------------------------------
print(f"\n{'='*80}")
print("FAIRNESS ANALYSIS")
print(f"{'='*80}")

fairness_outputs = {
    "CoLI_corr_city_tier": fairness_report(coli_corr_score, df["City_Tier"], "coli_corr_score"),
    "CoLI_corr_occupation": fairness_report(coli_corr_score, df["Occupation"], "coli_corr_score"),
    "BRI_corr_city_tier": fairness_report(bri_corr_score, df["City_Tier"], "bri_corr_score"),
    "BRI_corr_occupation": fairness_report(bri_corr_score, df["Occupation"], "bri_corr_score"),
}

for name, table in fairness_outputs.items():
    print(f"\n{name}")
    print(table.round(4))

# ---------------------------------------------------------------------
# Summary statistics
# ---------------------------------------------------------------------
print(f"\n{'='*80}")
print("COMPOSITE SCORE STATISTICS")
print(f"{'='*80}")

print("\nCoLI (correlation-based):")
print(coli_corr_score.describe())

print("\nBRI (correlation-based):")
print(bri_corr_score.describe())

print("\nCoLI (ridge-based):")
print(coli_ridge_score.describe())

print("\nBRI (ridge-based):")
print(bri_ridge_score.describe())

# ---------------------------------------------------------------------
# Persist composites & weights
# ---------------------------------------------------------------------
output = df.assign(
    CoLI_corr=coli_corr_score,
    BRI_corr=bri_corr_score,
    CoLI_ridge=coli_ridge_score,
    BRI_ridge=bri_ridge_score,
    reliability_target=reliability_target,
    default_flag=default_flag,
)
output_path = DATA_PATH.with_name("composites_with_weights.csv")
output.to_csv(output_path, index=False)

print(f"\n{'='*80}")
print("OUTPUT SAVED")
print(f"{'='*80}")
print(f"Composite scores saved to: {output_path.resolve()}")
print(f"Total columns in output: {len(output.columns)}")

# Save weights to JSON for easy reference
import json
weights_dict = {
    "CoLI": {
        "correlation_based": coli_corr_weights.to_dict(),
        "ridge_based": coli_ridge_weights.to_dict(),
    },
    "BRI": {
        "correlation_based": bri_corr_weights.to_dict(),
        "ridge_based": bri_ridge_weights.to_dict(),
    },
    "calibration": {
        "CoLI_corr": logit_params(coli_corr_logit),
        "BRI_corr": logit_params(bri_corr_logit),
        "CoLI_ridge": logit_params(coli_ridge_logit),
        "BRI_ridge": logit_params(bri_ridge_logit),
    }
}

weights_path = DATA_PATH.with_name("composite_weights.json")
with open(weights_path, 'w') as f:
    json.dump(weights_dict, f, indent=2)

print(f"Weights saved to: {weights_path.resolve()}")

print(f"\n{'='*80}")
print("✅ COMPOSITE WEIGHT DERIVATION COMPLETE")
print(f"{'='*80}")