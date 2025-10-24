"""
Realistic Default Flag Calibration
====================================
This script recalibrates default probabilities and flags based on:
1. Ridge regression weights from standardized features
2. Real-world delinquency benchmarks by occupation and city tier
3. Catastrophic override rules for severe cases only
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler

# ============================================================================
# Configuration: Real-world default rate benchmarks
# ============================================================================
DEFAULT_RATE_BENCHMARKS = {
    # Format: (Occupation, City_Tier): target_default_rate
    ('Professional', 'Tier_1'): 0.02,      # 2%
    ('Professional', 'Tier_2'): 0.04,      # 4%
    ('Professional', 'Tier_3'): 0.06,      # 6%
    
    ('Salaried', 'Tier_1'): 0.015,         # 1.5%
    ('Salaried', 'Tier_2'): 0.035,         # 3.5%
    ('Salaried', 'Tier_3'): 0.05,          # 5%
    
    ('Self_Employed', 'Tier_1'): 0.065,    # 6.5%
    ('Self_Employed', 'Tier_2'): 0.125,    # 12.5%
    ('Self_Employed', 'Tier_3'): 0.10,     # 10%
    
    ('Student', 'Tier_1'): 0.25,           # 25%
    ('Student', 'Tier_2'): 0.22,           # 22%
    ('Student', 'Tier_3'): 0.20,           # 20%
    
    ('Retired', 'Tier_1'): 0.18,           # 18%
    ('Retired', 'Tier_2'): 0.20,           # 20%
    ('Retired', 'Tier_3'): 0.15,           # 15%
}

# Ridge regression weights (from your training - original scale)
RIDGE_WEIGHTS = {
    'w1_PIS': 0.01019808,
    'w2_one_minus_CoLI': 0.01173606,
    'w3_one_minus_BRI': 0.03441564,
    'w4_one_minus_FRI': 0.05057033
}

# Catastrophic override thresholds (very strict - only for extreme cases)
CATASTROPHIC_THRESHOLDS = {
    'PIS_threshold': 0.90,           # Payment irregularity > 90%
    'cushion_critical': 0.25,        # Any cushion < 25%
    'min_failed_cushions': 1         # At least 1 cushion must fail
}


def load_data(filepath):
    """Load the risk-scored applicants dataset."""
    print(f"Loading data from {filepath}...")
    df = pd.read_csv(filepath)
    print(f"Loaded {len(df)} records.")
    return df


def compute_ridge_probability(df, weights):
    """
    Compute default probability using ridge regression formula with standardization.
    
    p(default) = w1·PIS + w2·(1-CoLI) + w3·(1-BRI) + w4·(1-FRI)
    """
    print("\nComputing ridge-based default probabilities...")
    
    # Prepare features
    features = pd.DataFrame({
        'PIS': df['Payment_Irregularity_Score'],
        'one_minus_CoLI': 1 - df['CoLI_ridge'],
        'one_minus_BRI': 1 - df['BRI_ridge'],
        'one_minus_FRI': 1 - df['Financial_Resilience_Index']
    })
    
    # Standardize features (same as training)
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(features)
    
    # Compute weighted sum using original-scale weights
    # Note: weights were already back-transformed to original scale
    w_orig = np.array([
        weights['w1_PIS'],
        weights['w2_one_minus_CoLI'],
        weights['w3_one_minus_BRI'],
        weights['w4_one_minus_FRI']
    ])
    
    # Divide by scale to get standardized weights, then apply to scaled features
    w_std = w_orig / scaler.scale_
    p_raw = X_scaled @ w_std
    
    # Clip to [0, 1] range
    p_default = np.clip(p_raw, 0, 1)
    
    print(f"  Mean probability: {p_default.mean():.4f}")
    print(f"  Std probability:  {p_default.std():.4f}")
    print(f"  Min probability:  {p_default.min():.4f}")
    print(f"  Max probability:  {p_default.max():.4f}")
    
    return p_default


def assign_group_calibrated_flags(df, prob_col, benchmarks):
    """
    Assign default flags by selecting top-N% within each occupation-tier group,
    where N% matches the real-world benchmark rate.
    """
    print("\nCalibrating default flags by occupation-tier groups...")
    
    df['Default_Label_Calibrated'] = 0
    
    for (occupation, tier), target_rate in benchmarks.items():
        # Filter group
        mask = (df['Occupation'] == occupation) & (df['City_Tier'] == tier)
        n_group = mask.sum()
        
        if n_group == 0:
            continue
        
        # Calculate number of defaults for this group
        n_defaults = int(np.ceil(n_group * target_rate))
        
        # Get indices sorted by probability (descending)
        group_indices = df[mask].index
        group_probs = df.loc[group_indices, prob_col].values
        
        # Select top N as defaults
        if n_defaults > 0:
            threshold_idx = min(n_defaults, len(group_probs))
            top_indices = group_indices[np.argsort(-group_probs)[:threshold_idx]]
            df.loc[top_indices, 'Default_Label_Calibrated'] = 1
        
        actual_rate = df.loc[mask, 'Default_Label_Calibrated'].mean()
        print(f"  {occupation:15s} | {tier:7s} | Target: {target_rate:5.1%} | "
              f"Actual: {actual_rate:5.1%} | n={n_group:5d} | defaults={n_defaults:4d}")
    
    return df


def apply_catastrophic_overrides(df, thresholds):
    """
    Apply deterministic overrides ONLY for catastrophic cases:
    - Payment Irregularity > 90% AND
    - At least one cushion index < 25%
    """
    print("\nApplying catastrophic override rules...")
    
    # Identify catastrophic cases
    high_irregularity = df['Payment_Irregularity_Score'] > thresholds['PIS_threshold']
    
    critical_coli = df['CoLI_ridge'] < thresholds['cushion_critical']
    critical_bri = df['BRI_ridge'] < thresholds['cushion_critical']
    critical_fri = df['Financial_Resilience_Index'] < thresholds['cushion_critical']
    
    n_failed_cushions = critical_coli.astype(int) + critical_bri.astype(int) + critical_fri.astype(int)
    enough_failures = n_failed_cushions >= thresholds['min_failed_cushions']
    
    catastrophic_mask = high_irregularity & enough_failures
    
    # Force these to default
    n_overrides = catastrophic_mask.sum()
    df.loc[catastrophic_mask, 'Default_Label_Calibrated'] = 1
    df.loc[catastrophic_mask, 'Override_Catastrophic'] = 1
    
    print(f"  Catastrophic overrides applied: {n_overrides} ({n_overrides/len(df)*100:.2f}%)")
    
    return df


def compute_summary_statistics(df):
    """Compute and display summary statistics."""
    print("\n" + "="*70)
    print("SUMMARY STATISTICS")
    print("="*70)
    
    # Overall default rate
    overall_rate = df['Default_Label_Calibrated'].mean()
    print(f"\nOverall Default Rate: {overall_rate:.2%} ({df['Default_Label_Calibrated'].sum()} / {len(df)})")
    
    # By City Tier
    print("\n--- By City Tier ---")
    tier_summary = df.groupby('City_Tier')['Default_Label_Calibrated'].agg(['mean', 'sum', 'count'])
    tier_summary.columns = ['Default_Rate', 'Defaults', 'Total']
    tier_summary['Default_Rate'] = tier_summary['Default_Rate'].map('{:.2%}'.format)
    print(tier_summary)
    
    # By Occupation
    print("\n--- By Occupation ---")
    occ_summary = df.groupby('Occupation')['Default_Label_Calibrated'].agg(['mean', 'sum', 'count'])
    occ_summary.columns = ['Default_Rate', 'Defaults', 'Total']
    occ_summary['Default_Rate'] = occ_summary['Default_Rate'].map('{:.2%}'.format)
    print(occ_summary)
    
    # By Occupation x City Tier
    print("\n--- By Occupation x City Tier ---")
    cross_summary = df.groupby(['Occupation', 'City_Tier'])['Default_Label_Calibrated'].agg(['mean', 'sum', 'count'])
    cross_summary.columns = ['Default_Rate', 'Defaults', 'Total']
    cross_summary['Default_Rate'] = cross_summary['Default_Rate'].map('{:.2%}'.format)
    print(cross_summary)
    
    # Catastrophic overrides
    if 'Override_Catastrophic' in df.columns:
        n_overrides = df['Override_Catastrophic'].sum()
        print(f"\n--- Catastrophic Overrides ---")
        print(f"Total overrides: {n_overrides} ({n_overrides/len(df)*100:.2f}%)")
    
    # Probability distribution
    print("\n--- Default Probability Distribution (Ridge) ---")
    print(df['Default_Prob_Ridge'].describe())
    
    print("\n" + "="*70)


def main():
    """Main execution pipeline."""
    print("="*70)
    print("REALISTIC DEFAULT FLAG CALIBRATION")
    print("="*70)
    
    # 1. Load data
    input_file = r"c:\Users\subha\Desktop\risk_credit_final\risk_scored_applicants_updated.csv"
    df = load_data(input_file)
    
    # 2. Compute ridge-based default probability
    df['Default_Prob_Ridge'] = compute_ridge_probability(df, RIDGE_WEIGHTS)
    
    # 3. Initialize override flag
    df['Override_Catastrophic'] = 0
    
    # 4. Assign calibrated flags based on group benchmarks
    df = assign_group_calibrated_flags(df, 'Default_Prob_Ridge', DEFAULT_RATE_BENCHMARKS)
    
    # 5. Apply catastrophic overrides
    df = apply_catastrophic_overrides(df, CATASTROPHIC_THRESHOLDS)
    
    # 6. Compute summary statistics
    compute_summary_statistics(df)
    
    # 7. Save output
    output_file = r"c:\Users\subha\Desktop\risk_credit_final\risk_scored_applicants_realistic_defaults.csv"
    df.to_csv(output_file, index=False)
    print(f"\n✓ Output saved to: {output_file}")
    
    # 8. Save summary report
    report_file = r"c:\Users\subha\Desktop\risk_credit_final\realistic_defaults_summary.txt"
    with open(report_file, 'w') as f:
        f.write("REALISTIC DEFAULT FLAG CALIBRATION REPORT\n")
        f.write("="*70 + "\n\n")
        
        f.write("RIDGE REGRESSION WEIGHTS (Original Scale)\n")
        f.write("-"*70 + "\n")
        for key, val in RIDGE_WEIGHTS.items():
            f.write(f"{key:25s}: {val:.6f}\n")
        
        f.write("\n\nDEFAULT RATE BENCHMARKS\n")
        f.write("-"*70 + "\n")
        for (occ, tier), rate in DEFAULT_RATE_BENCHMARKS.items():
            f.write(f"{occ:15s} | {tier:7s}: {rate:5.1%}\n")
        
        f.write("\n\nCATASTROPHIC OVERRIDE THRESHOLDS\n")
        f.write("-"*70 + "\n")
        for key, val in CATASTROPHIC_THRESHOLDS.items():
            f.write(f"{key:25s}: {val}\n")
        
        f.write("\n\nOVERALL RESULTS\n")
        f.write("-"*70 + "\n")
        overall_rate = df['Default_Label_Calibrated'].mean()
        f.write(f"Overall Default Rate: {overall_rate:.2%}\n")
        f.write(f"Total Defaults: {df['Default_Label_Calibrated'].sum()}\n")
        f.write(f"Total Records: {len(df)}\n")
        
        if 'Override_Catastrophic' in df.columns:
            n_overrides = df['Override_Catastrophic'].sum()
            f.write(f"\nCatastrophic Overrides: {n_overrides} ({n_overrides/len(df)*100:.2f}%)\n")
    
    print(f"✓ Summary report saved to: {report_file}")
    
    print("\n" + "="*70)
    print("CALIBRATION COMPLETE")
    print("="*70)
    
    return df


if __name__ == "__main__":
    df_calibrated = main()
