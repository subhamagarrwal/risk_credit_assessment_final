import numpy as np
import pandas as pd
from scipy.special import expit as logistic, logit
from typing import Dict, Tuple
import warnings
warnings.filterwarnings('ignore')

class TemporalCreditRiskSimulator:
    """
    Implements temporal separation methodology for credit risk simulation.
    Observes months 1-9, predicts outcomes in months 10-12.
    Ensures no data leakage by using only pre-outcome features for training.
    """
    
    def __init__(self, random_seed: int = 12345):
        self.rng = np.random.default_rng(random_seed)
        self.setup_parameters()
        
    def setup_parameters(self):
        """Initialize all model parameters based on research and best practices"""
        
        # Seasonality multipliers (0 = no change, +0.15 = 15% increase)
        self.seasonality = {
            'month_1': {'Eating_Out': 0.0, 'Entertainment': 0.0},
            'month_2': {'Eating_Out': 0.0, 'Entertainment': 0.0},
            'month_3': {'Eating_Out': 0.0, 'Entertainment': 0.0},
            'month_4': {'Eating_Out': 0.05, 'Entertainment': 0.05},
            'month_5': {'Eating_Out': 0.05, 'Entertainment': 0.05},
            'month_6': {'Education': 0.3},
            'month_7': {'Eating_Out': 0.0, 'Entertainment': 0.0},
            'month_8': {'Eating_Out': 0.0, 'Entertainment': 0.0},
            'month_9': {'Eating_Out': 0.0, 'Entertainment': 0.0},
            'month_10': {'Eating_Out': 0.15, 'Entertainment': 0.20},
            'month_11': {'Eating_Out': 0.15, 'Entertainment': 0.20},
            'month_12': {'Eating_Out': 0.10, 'Entertainment': 0.10, 'Education': 0.2}
        }
        
        # Volatility (standard deviation) by category and city tier
        self.sigma = {
            'Groceries': {'Tier_1': 0.08, 'Tier_2': 0.10, 'Tier_3': 0.12},
            'Utilities': {'Tier_1': 0.03, 'Tier_2': 0.04, 'Tier_3': 0.05},
            'Eating_Out': {'Tier_1': 0.20, 'Tier_2': 0.15, 'Tier_3': 0.10},
            'Entertainment': {'Tier_1': 0.20, 'Tier_2': 0.15, 'Tier_3': 0.10},
            'Healthcare': {'Tier_1': 0.12, 'Tier_2': 0.12, 'Tier_3': 0.12},
            'Transport': {'Tier_1': 0.10, 'Tier_2': 0.12, 'Tier_3': 0.15},
            'Rent': {'Tier_1': 0.02, 'Tier_2': 0.02, 'Tier_3': 0.02},
            'Loan_Repayment': {'Tier_1': 0.02, 'Tier_2': 0.02, 'Tier_3': 0.02},
            'Insurance': {'Tier_1': 0.02, 'Tier_2': 0.02, 'Tier_3': 0.02},
            'Education': {'Tier_1': 0.15, 'Tier_2': 0.15, 'Tier_3': 0.15},
            'Miscellaneous': {'Tier_1': 0.25, 'Tier_2': 0.25, 'Tier_3': 0.25}
        }
        
        # BASE ANNUAL DEFAULT RATES by Occupation × City_Tier (UNCHANGED - THESE ARE CORRECT)
        self.base_annual_default_rates = {
            'Professional|Tier_1': 0.015,  # 1.5%
            'Professional|Tier_2': 0.025,  # 2.5%
            'Professional|Tier_3': 0.035,  # 3.5%
            'Self_Employed|Tier_1': 0.030,  # 3.0%
            'Self_Employed|Tier_2': 0.050,  # 5.0%
            'Self_Employed|Tier_3': 0.070,  # 7.0%
            'Retired|Tier_1': 0.025,  # 2.5%
            'Retired|Tier_2': 0.040,  # 4.0%
            'Retired|Tier_3': 0.050,  # 5.0%
            'Student|Tier_1': 0.030,  # 3.0%
            'Student|Tier_2': 0.060,  # 6.0%
            'Student|Tier_3': 0.050   # 5.0%
        }
        
        # Hazard model coefficients (CALIBRATED - these adjust base, not overwrite)
        self.hazard_coefficients = {
            'alpha1': 5.0,  # Debt Service Ratio is a *major* driver
            'alpha2': 3.0,  # Negative Savings is a *major* driver
            'alpha3': 2.0,  # Volatility is a clear signal
            'alpha4': 2.0   # Bank risk (CAELS) is a strong proxy
        }
        print("\n\n--- DEBUG TEST: HAZARD ALPHAS SET TO 5.0, 3.0, 2.0, 2.0 ---\n\n")
        # Default probability model coefficients (CALIBRATED)
        self.default_model = {
            'gamma0': -4.0,  # A new baseline. THIS IS YOUR MAIN TUNING KNOB.
            'gamma1': 2.5,   # KEEP THIS. The link must be strong.
            'gamma2': 0.0,   # Set to 0. It's redundant.
            'gamma3': 0.0    # Set to 0. It's redundant.
        }
        
        # Expense categories
        self.expense_categories = [
            'Rent', 'Loan_Repayment', 'Insurance', 'Groceries', 
            'Transport', 'Eating_Out', 'Entertainment', 'Utilities',
            'Healthcare', 'Education', 'Miscellaneous'
        ]
        
    def annual_to_monthly_hazard(self, annual_rate: float) -> float:
        """Convert annual default probability to monthly hazard rate"""
        return 1 - (1 - annual_rate) ** (1/12)
    
    def simulate_monthly_expenses(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Simulate 12 months of expenses for each applicant
        Formula: Expense_i,t,c = Base_i,c × (1 + S_c,t + ε_i,t,c)
        """
        print("Simulating 12 months of expenses...")
        monthly_rows = []
        
        for idx, row in df.iterrows():
            if idx % 1000 == 0:
                print(f"Processing applicant {idx}/{len(df)}")
            
            for month in range(1, 13):
                monthly_rec = {
                    'applicant_id': idx,
                    'month': month,
                    'Income': row['Income'],
                    'Age': row['Age'],
                    'Dependents': row['Dependents'],
                    'Occupation': row['Occupation'],
                    'City_Tier': row['City_Tier'],
                    'UPI_Remitter_Bank': row['UPI_Remitter_Bank'],
                    'Bank_CAELS_Score': row['Bank_CAELS_Score'],
                    'Bank_Risk_Tier': row['Bank_Risk_Tier']
                }
                
                for category in self.expense_categories:
                    base_expense = row[category]
                    month_key = f'month_{month}'
                    seasonality = self.seasonality.get(month_key, {}).get(category, 0.0)
                    tier_key = row['City_Tier']
                    sigma = self.sigma.get(category, {}).get(tier_key, 0.10)
                    epsilon = self.rng.normal(0, sigma)
                    monthly_expense = base_expense * (1 + seasonality + epsilon)
                    monthly_expense = max(0, monthly_expense)
                    monthly_rec[f'{category}_monthly'] = monthly_expense
                
                total_expense = sum(monthly_rec[f'{cat}_monthly'] for cat in self.expense_categories)
                monthly_rec['Total_Expense'] = total_expense
                monthly_rec['Savings'] = row['Income'] - total_expense
                
                monthly_rows.append(monthly_rec)
        
        df_monthly = pd.DataFrame(monthly_rows)
        print(f"Generated {len(df_monthly)} monthly records")
        return df_monthly
    
    def compute_pre_outcome_features(self, df_monthly: pd.DataFrame) -> pd.DataFrame:
        """
        Compute features from months 1-9 only (pre-outcome period)
        """
        print("Computing pre-outcome features (months 1-9)...")
        pre_outcome = df_monthly[df_monthly['month'] <= 9].copy()
        
        features = []
        for applicant_id in pre_outcome['applicant_id'].unique():
            app_data = pre_outcome[pre_outcome['applicant_id'] == applicant_id]
            
            feat = {
                'applicant_id': applicant_id,
                'Income': app_data['Income'].iloc[0],
                'Age': app_data['Age'].iloc[0],
                'Dependents': app_data['Dependents'].iloc[0],
                'Occupation': app_data['Occupation'].iloc[0],
                'City_Tier': app_data['City_Tier'].iloc[0],
                'UPI_Remitter_Bank': app_data['UPI_Remitter_Bank'].iloc[0],
                'Bank_CAELS_Score': app_data['Bank_CAELS_Score'].iloc[0],
                'Bank_Risk_Tier': app_data['Bank_Risk_Tier'].iloc[0],
                'Avg_Monthly_Income': app_data['Income'].mean(),
                'Avg_Total_Expense': app_data['Total_Expense'].mean(),
                'Avg_Savings': app_data['Savings'].mean(),
                'Savings_SD': app_data['Savings'].std(),
                'Expense_Volatility': app_data['Total_Expense'].std() / (app_data['Total_Expense'].mean() + 1e-6),
                'Debt_Service_Ratio': app_data['Loan_Repayment_monthly'].mean() / (app_data['Income'].mean() + 1e-6),
                'Essential_to_Income': (app_data['Rent_monthly'] + app_data['Groceries_monthly'] + 
                                       app_data['Utilities_monthly']).mean() / (app_data['Income'].mean() + 1e-6),
                'Discretionary_to_Income': (app_data['Eating_Out_monthly'] + 
                                           app_data['Entertainment_monthly']).mean() / (app_data['Income'].mean() + 1e-6),
                'Avg_Rent': app_data['Rent_monthly'].mean(),
                'Avg_Groceries': app_data['Groceries_monthly'].mean(),
                'Avg_Transport': app_data['Transport_monthly'].mean(),
                'Avg_Healthcare': app_data['Healthcare_monthly'].mean()
            }
            features.append(feat)
        
        df_features = pd.DataFrame(features)
        print(f"Computed features for {len(df_features)} applicants")
        return df_features
    
    def compute_dynamic_hazard(self, df_features: pd.DataFrame) -> np.ndarray:
        """
        Compute personalized monthly hazard using LOGISTIC FORMULA
        h_i = σ(logit(p_m_base) + α₁·DSR + α₂·(-SavingsRatio) + α₃·ExpenseVol + α₄·(0.5-CAELS))
        """
        print("Computing dynamic monthly hazard rates...")
        
        # Get base segment hazards
        df_features['segment'] = df_features['Occupation'] + '|' + df_features['City_Tier']
        df_features['base_annual_rate'] = df_features['segment'].map(self.base_annual_default_rates)
        df_features['base_annual_rate'] = df_features['base_annual_rate'].fillna(0.03)
        
        # Convert to monthly base hazard
        base_monthly = df_features['base_annual_rate'].apply(self.annual_to_monthly_hazard).values
        
        # Compute logit of base hazard
        logit_base = np.log(base_monthly / (1 - base_monthly))
        
        # Get coefficients
        alpha1 = self.hazard_coefficients['alpha1']
        alpha2 = self.hazard_coefficients['alpha2']
        alpha3 = self.hazard_coefficients['alpha3']
        alpha4 = self.hazard_coefficients['alpha4']
        
        # Extract features
        dsr = df_features['Debt_Service_Ratio'].values
        savings_ratio = df_features['Avg_Savings'].values / (df_features['Income'].values + 1e-6)
        expense_vol = df_features['Expense_Volatility'].values
        caels = df_features['Bank_CAELS_Score'].values
        
        # Compute logit of adjusted hazard
        logit_h = (logit_base + 
                   alpha1 * dsr + 
                   alpha2 * (-savings_ratio) + 
                   alpha3 * expense_vol + 
                   alpha4 * (0.5 - caels))
        
        # Convert to probability and clip
        hazard = logistic(logit_h)
        hazard = np.clip(hazard, 1e-4, 0.3)
        
        print(f"Hazard statistics: mean={hazard.mean():.4f}, std={hazard.std():.4f}, "
              f"min={hazard.min():.4f}, max={hazard.max():.4f}")
        
        return hazard
    
    def simulate_missed_payments(self, df_features: pd.DataFrame, 
                                 df_monthly: pd.DataFrame) -> pd.DataFrame:
        """
        Simulate missed payments in months 10-12 using dynamic hazard
        """
        print("Simulating missed payments (months 10-12)...")
        
        hazard = self.compute_dynamic_hazard(df_features)
        
        outcomes = []
        for i, row in df_features.iterrows():
            applicant_id = row['applicant_id']
            h = hazard[i]
            
            # Simulate missed payments for months 10, 11, 12
            missed_payments = self.rng.binomial(1, h, size=3)
            total_missed = missed_payments.sum()
            
            # Get outcome period data for deterioration
            outcome_data = df_monthly[(df_monthly['applicant_id'] == applicant_id) & 
                                     (df_monthly['month'] >= 10)]
            pre_data = df_monthly[(df_monthly['applicant_id'] == applicant_id) & 
                                 (df_monthly['month'] <= 9)]
            
            avg_savings_pre = pre_data['Savings'].mean()
            avg_savings_post = outcome_data['Savings'].mean()
            deterioration_score = (avg_savings_pre - avg_savings_post) / (abs(avg_savings_pre) + 1e-6)
            
            outcomes.append({
                'applicant_id': applicant_id,
                'monthly_hazard': h,
                'missed_month_10': missed_payments[0],
                'missed_month_11': missed_payments[1],
                'missed_month_12': missed_payments[2],
                'total_missed': total_missed,
                'deterioration_score': deterioration_score
            })
        
        df_outcomes = pd.DataFrame(outcomes)
        print(f"Simulated outcomes for {len(df_outcomes)} applicants")
        print(f"Missed payment distribution:\n{df_outcomes['total_missed'].value_counts().sort_index()}")
        
        return df_outcomes
    
    def compute_default_probability(self, df_features: pd.DataFrame, 
                                    df_outcomes: pd.DataFrame) -> pd.DataFrame:
        """
        Convert missed payments → default probability using CORRECTED logistic model
        P(default) = σ(γ₀ + γ₁·missed_3mo + γ₂·avg_hazard + γ₃·(0.5-CAELS))
        """
        print("Computing default probabilities...")
        
        df_combined = df_features.merge(df_outcomes, on='applicant_id')
        
        # Get coefficients
        gamma0 = self.default_model['gamma0']
        gamma1 = self.default_model['gamma1']
        gamma2 = self.default_model['gamma2']
        gamma3 = self.default_model['gamma3']
        
        # Compute default probability with CAELS effect
        default_logit = (gamma0 + 
                        gamma1 * df_combined['total_missed'] + 
                        gamma2 * df_combined['monthly_hazard'] + 
                        gamma3 * (0.5 - df_combined['Bank_CAELS_Score']))
        
        default_prob = logistic(default_logit)
        
        # Sample binary default labels
        default_label = self.rng.binomial(1, default_prob)
        
        df_combined['Default_Probability'] = default_prob
        df_combined['Default_Label'] = default_label
        
        print(f"Default rate: {default_label.mean():.3%}")
        print(f"Default probability statistics: mean={default_prob.mean():.3f}, std={default_prob.std():.3f}")
        
        return df_combined
    
    def create_training_dataset(self, df_final: pd.DataFrame) -> Tuple[pd.DataFrame, pd.Series, pd.Series]:
        """
        Create firewall: remove all outcome period data
        """
        print("Creating training dataset (firewall applied)...")
        
        training_features = [
            'Income', 'Age', 'Dependents', 'Occupation', 'City_Tier',
            'Bank_CAELS_Score', 'UPI_Remitter_Bank',
            'Avg_Monthly_Income', 'Avg_Total_Expense', 'Avg_Savings',
            'Savings_SD', 'Expense_Volatility', 'Debt_Service_Ratio',
            'Essential_to_Income', 'Discretionary_to_Income',
            'Avg_Rent', 'Avg_Groceries', 'Avg_Transport', 'Avg_Healthcare'
        ]
        
        X = df_final[training_features].copy()
        y = df_final['Default_Label'].copy()
        probs = df_final['Default_Probability'].copy()
        
        print(f"Training dataset shape: {X.shape}")
        print(f"Target distribution:\n{y.value_counts()}")
        
        return X, y, probs
    
    def run_full_simulation(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.Series, pd.Series, pd.DataFrame]:
        """
        Execute the complete temporal separation simulation pipeline
        """
        print("="*80)
        print("TEMPORAL CREDIT RISK SIMULATION")
        print("="*80)
        
        df_monthly = self.simulate_monthly_expenses(df)
        df_features = self.compute_pre_outcome_features(df_monthly)
        df_outcomes = self.simulate_missed_payments(df_features, df_monthly)
        df_final = self.compute_default_probability(df_features, df_outcomes)
        X, y, probs = self.create_training_dataset(df_final)
        
        print("="*80)
        print("SIMULATION COMPLETE")
        print("="*80)
        
        return X, y, probs, df_final

def main():
    """Main execution function"""
    
    print("Loading data...")
    df = pd.read_csv('../data/risk_scored_applicants_updated.csv')
    print(f"Loaded {len(df)} applicants")
    
    # CAELS DIAGNOSTIC
    print("\n" + "="*80)
    print("CAELS SCORE DIAGNOSTIC")
    print("="*80)
    print("\nCAELS Score distribution by Bank Risk Tier:")
    print(df.groupby('Bank_Risk_Tier')['Bank_CAELS_Score'].agg(['mean', 'std', 'min', 'max', 'count']))
    print("\nExpected: High Risk banks → low CAELS (0.0-0.3)")
    print("          Low Risk banks  → high CAELS (0.8-1.0)")
    print("="*80)
    
    simulator = TemporalCreditRiskSimulator(random_seed=42)
    X_train, y_train, default_probs, df_final = simulator.run_full_simulation(df)
    
    print("\nSaving results...")
    X_train.to_csv('../data/X_train_temporal.csv', index=False)
    y_train.to_csv('../data/y_train_temporal.csv', index=False, header=['Default_Label'])
    df_final.to_csv('../data/full_simulation_results.csv', index=False)
    
    print("\nSummary Statistics:")
    print(f"Training samples: {len(X_train)}")
    print(f"Default rate: {y_train.mean():.3%}")
    print(f"Mean default probability: {default_probs.mean():.3%}")
    
    # VALIDATION CHECKS
    print("\n" + "="*80)
    print("VALIDATION CHECKS")
    print("="*80)
    
    print("\n1. Default rate by occupation:")
    occ_stats = df_final.groupby('Occupation')['Default_Label'].agg(['mean', 'count'])
    print(occ_stats)
    
    print("\n2. Default rate by city tier:")
    tier_stats = df_final.groupby('City_Tier')['Default_Label'].agg(['mean', 'count'])
    print(tier_stats)
    
    print("\n3. Default rate by bank risk tier:")
    bank_stats = df_final.groupby('Bank_Risk_Tier')['Default_Label'].agg(['mean', 'count'])
    print(bank_stats)
    
    print("\n4. Correlation between missed payments and default:")
    corr_missed = df_final[['total_missed', 'Default_Label']].corr().iloc[0,1]
    print(f"Correlation: {corr_missed:.3f} (Expected: > 0.4)")
    
    print("\n5. CAELS Score impact:")
    corr_caels = df_final[['Bank_CAELS_Score', 'Default_Label']].corr().iloc[0,1]
    print(f"Correlation: {corr_caels:.3f} (Expected: negative)")
    
    print("\n6. Feature correlation with default (top 10):")
    numeric_cols = X_train.select_dtypes(include=[np.number]).columns
    corr_with_default = X_train[numeric_cols].corrwith(y_train).sort_values(ascending=False)
    print(corr_with_default.head(10))
    
    # SANITY CHECK SUMMARY
    print("\n" + "="*80)
    print("SANITY CHECK RESULTS")
    print("="*80)
    checks = {
        'Correlation(Default, Missed) > 0.4': corr_missed > 0.4,
        'Correlation(Default, CAELS) < 0': corr_caels < 0,
        'Tier_3 default > Tier_1 default': tier_stats.loc['Tier_3', 'mean'] > tier_stats.loc['Tier_1', 'mean'],
        'Student default > Professional default': occ_stats.loc['Student', 'mean'] > occ_stats.loc['Professional', 'mean'],
        'High Risk bank > Low Risk bank': bank_stats.loc['High Risk', 'mean'] > bank_stats.loc['Low Risk', 'mean']
    }
    
    for check, passed in checks.items():
        status = "✓ PASS" if passed else "✗ FAIL"
        print(f"{status}: {check}")

if __name__ == "__main__":
    main()