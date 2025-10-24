import numpy as np
import pandas as pd
from pathlib import Path
from scipy.special import expit  # logistic / inverse logit

UPDATED_PATH = Path("risk_scored_applicants_updated.csv")
STRICT_PATH = Path("risk_scored_applicants_final1.csv")

BASE_TIER_PRIOR = {
    "Tier_1": 0.055,  # conference prior ~5.5%
    "Tier_2": 0.125,
    "Tier_3": 0.205,
}

OCCUPATION_MULTIPLIER = {
    "Self_Employed": 1.18,
    "Retired": 1.12,
    "Student": 0.82,
    # Professional or any other occupation stays at 1.0
}

PRIOR_STRENGTH = 60
GROUP_RATE_FLOOR = 0.030
GROUP_RATE_CAP = 0.320

BETA_ALPHA = 0.90
BETA_BETA = 1.05
BETA_SHIFT = -0.35

DEFAULT_FLAG_THRESHOLD = {
    "payment_irregularity": 0.80,  # >= 0.80 -> default band
    "coli": 0.35,                  # <= 0.35 -> default band
    "bri": 0.45,                   # <= 0.45 -> default band
    "fri": 0.35,                   # <= 0.35 -> default band
}

SAFE_THRESHOLD = {
    "payment_irregularity": 0.30,  # <= 0.30 -> safe band
    "coli": 0.70,                  # >= 0.70 -> safe band
    "bri": 0.70,                   # >= 0.70 -> safe band
    "fri": 0.70,                   # >= 0.70 -> safe band
}


def beta_calibrate(prob: pd.Series) -> pd.Series:
    """Beta calibration to soften extremes while preserving rank order."""
    eps = 1.0e-6
    prob = prob.clip(eps, 1.0 - eps)
    logits = (
        BETA_ALPHA * np.log(prob)
        - BETA_BETA * np.log1p(-prob)
        + BETA_SHIFT
    )
    return pd.Series(expit(logits), index=prob.index)


def build_composite_risk(df: pd.DataFrame) -> pd.Series:
    """Blend cushion gaps, payment irregularity, and raw model signal."""
    gap_coli = 1.0 - df["CoLI_ridge"].clip(0, 1)
    gap_bri = 1.0 - df["BRI_ridge"].clip(0, 1)
    gap_fri = 1.0 - df["Financial_Resilience_Index"].clip(0, 1)

    pis = df["Payment_Irregularity_Score"].clip(0, 1)
    raw_logit = expit(df["RiskScore_raw"].astype(float))
    # higher raw score is assumed riskier, so use its logistic projection

    # Weighted composite (weights sum to 1):
    composite = (
        0.40 * pis +
        0.20 * gap_coli +
        0.20 * gap_bri +
        0.20 * gap_fri
    )

    # Blend with model raw probability and apply beta calibration
    raw_mix = (0.5 * composite + 0.5 * raw_logit).clip(0, 1)
    return beta_calibrate(raw_mix)


def apply_rule_based_overrides(df: pd.DataFrame) -> pd.Series:
    """Return -1 unresolved, 1 auto default, 0 auto safe labels."""
    risk_flags = pd.DataFrame(
        {
            "low_coli": df["CoLI_ridge"] <= DEFAULT_FLAG_THRESHOLD["coli"],
            "low_bri": df["BRI_ridge"] <= DEFAULT_FLAG_THRESHOLD["bri"],
            "low_fri": (
                df["Financial_Resilience_Index"]
                <= DEFAULT_FLAG_THRESHOLD["fri"]
            ),
        }
    )

    structural_stress = risk_flags.sum(axis=1) >= 2

    default_mask = (
        (df["Payment_Irregularity_Score"]
         >= DEFAULT_FLAG_THRESHOLD["payment_irregularity"])
        | structural_stress
    )

    safe_mask = (
        (df["Payment_Irregularity_Score"]
         <= SAFE_THRESHOLD["payment_irregularity"])
        & (~risk_flags.any(axis=1))
    )

    initial = pd.Series(-1, index=df.index, dtype=int)
    initial.loc[default_mask] = 1
    # keep explicit default triggers dominant
    initial.loc[safe_mask & ~default_mask] = 0

    df["_auto_default"] = default_mask
    df["_auto_safe"] = (initial == 0)

    return initial


def posterior_group_rate(
    expected_defaults: float,
    total: int,
    tier: str,
    occupation: str,
) -> float:
    """Shrink expected defaults toward tier priors with occupation tilt."""
    tier_prior = BASE_TIER_PRIOR.get(
        tier,
        float(np.mean(list(BASE_TIER_PRIOR.values()))),
    )
    adj_prior = np.clip(
        tier_prior * OCCUPATION_MULTIPLIER.get(occupation, 1.0),
        GROUP_RATE_FLOOR,
        GROUP_RATE_CAP,
    )
    alpha0 = PRIOR_STRENGTH * adj_prior
    posterior = (expected_defaults + alpha0) / (total + PRIOR_STRENGTH)
    return float(np.clip(posterior, GROUP_RATE_FLOOR, GROUP_RATE_CAP))


def calibrated_defaults(df: pd.DataFrame) -> pd.DataFrame:
    """Main calibration flow."""
    df = df.copy()
    if not UPDATED_PATH.exists():
        raise FileNotFoundError(f"Cannot find {UPDATED_PATH.resolve()}")

    df["Default_Label_new"] = apply_rule_based_overrides(df)
    df["base_prob"] = build_composite_risk(df)

    # iterate over Tier × Occupation
    group_iter = df.groupby(["City_Tier", "Occupation"]).indices.items()
    for (tier, occ), idx in group_iter:
        group_mask = df.index.isin(idx)
        total = int(group_mask.sum())
        group_df = df.loc[group_mask]

        auto_mask = (
            (group_df["_auto_default"]) & (group_df["Default_Label_new"] == 1)
        )
        auto_defaults = int(auto_mask.sum())
        expected_defaults = float(group_df["base_prob"].sum())

        target_rate = posterior_group_rate(
            expected_defaults=expected_defaults,
            total=total,
            tier=tier,
            occupation=occ,
        )
        target_defaults = int(round(target_rate * total, 0))
        target_defaults = int(np.clip(target_defaults, 0, total))

        if target_defaults < auto_defaults:
            drop = auto_defaults - target_defaults
            drop_idx = (
                group_df.loc[auto_mask]
                .sort_values("base_prob")
                .index[:drop]
            )
            df.loc[drop_idx, "Default_Label_new"] = 0
            df.loc[drop_idx, "_auto_default"] = False

        group_mask = df.index.isin(idx)
        auto_defaults = int(
            ((df["Default_Label_new"] == 1) & group_mask).sum()
        )
        remaining = max(target_defaults - auto_defaults, 0)

        group_unresolved = group_mask & (df["Default_Label_new"] == -1)
        unresolved_idx = df.index[group_unresolved]

        if remaining > 0 and len(unresolved_idx) > 0:
            # rank unresolved within group by base_prob descending
            ranked = (
                df.loc[unresolved_idx]
                .sort_values("base_prob", ascending=False)
            )
            default_candidates = ranked.index[:remaining]
            df.loc[default_candidates, "Default_Label_new"] = 1

        df.loc[
            group_unresolved & (df["Default_Label_new"] == -1),
            "Default_Label_new",
        ] = 0

    # probabilities: enforce sensible floors/ceilings
    df["Default_Prob_Final"] = df["base_prob"]

    # auto default floor
    auto_default_mask = df["Default_Label_new"] == 1
    df.loc[auto_default_mask, "Default_Prob_Final"] = np.clip(
        df.loc[auto_default_mask, "Default_Prob_Final"],
        0.30,
        0.90,
    )

    # safe mask ceiling
    safe_mask = (df["_auto_safe"])
    df.loc[safe_mask & ~auto_default_mask, "Default_Prob_Final"] = np.clip(
        df.loc[
            safe_mask & ~auto_default_mask,
            "Default_Prob_Final",
        ],
        0.005,
        0.15,
    )

    # everyone else: keep within [0.02, 0.75]
    mid_mask = ~(auto_default_mask | safe_mask)
    df.loc[mid_mask, "Default_Prob_Final"] = np.clip(
        df.loc[mid_mask, "Default_Prob_Final"],
        0.02,
        0.60,
    )

    # Mirror Default_Prob column if present
    if "Default_Prob" in df.columns:
        df["Default_Prob"] = df["Default_Prob_Final"]

    df["Default_Label"] = df["Default_Label_new"].astype(int)

    # housekeeping: drop helper columns
    df.drop(
        columns=[
            "Default_Label_new",
            "base_prob",
            "_auto_default",
            "_auto_safe",
        ],
        inplace=True,
        errors="ignore",
    )

    return df


def main() -> None:
    df = pd.read_csv(UPDATED_PATH)
    calibrated = calibrated_defaults(df)

    calibrated.to_csv(STRICT_PATH, index=False)
    print(f"Wrote calibrated labels to {STRICT_PATH.resolve()}")

    overall_rate = calibrated["Default_Label"].mean()
    print(f"Overall default rate: {overall_rate:.2%}")

    tier_rates = (
        calibrated.groupby("City_Tier")["Default_Label"]
        .mean()
        .rename("Default_Rate")
    )
    print("\nDefault rate by City_Tier:")
    print(tier_rates.apply(lambda x: f"{x:.2%}"))

    occ_rates = (
        calibrated.groupby(["City_Tier", "Occupation"])["Default_Label"].mean()
    )
    print("\nDefault rate by City_Tier × Occupation (for spot checks):")
    for (tier, occ), rate in occ_rates.items():
        print(f"  {tier:>5s} | {occ:<13s}: {rate:.2%}")


if __name__ == "__main__":
    main()
