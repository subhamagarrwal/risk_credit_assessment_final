"""
Visualization Script for Realistic Default Calibration Results
===============================================================
Creates charts showing the distribution of defaults across segments.
"""

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Set style
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (14, 10)

# Load data
print("Loading calibrated data...")
df = pd.read_csv(r"c:\Users\subha\Desktop\risk_credit_final\risk_scored_applicants_realistic_defaults.csv")
print(f"Loaded {len(df)} records")
print(f"Overall default rate: {df['Default_Label_Calibrated'].mean():.2%}")

# Create figure with subplots
fig, axes = plt.subplots(2, 2, figsize=(16, 12))
fig.suptitle('Realistic Default Flag Distribution Analysis', fontsize=16, fontweight='bold')

# 1. Default Rate by Occupation
ax1 = axes[0, 0]
occ_summary = df.groupby('Occupation')['Default_Label_Calibrated'].agg(['mean', 'sum', 'count'])
occ_summary = occ_summary.sort_values('mean', ascending=False)
colors1 = ['#d62728' if x > 0.15 else '#ff7f0e' if x > 0.08 else '#2ca02c' for x in occ_summary['mean']]
bars1 = ax1.bar(occ_summary.index, occ_summary['mean'] * 100, color=colors1, alpha=0.7, edgecolor='black')
ax1.set_ylabel('Default Rate (%)', fontsize=11)
ax1.set_title('Default Rate by Occupation', fontsize=12, fontweight='bold')
ax1.set_ylim(0, 30)
for i, (idx, row) in enumerate(occ_summary.iterrows()):
    ax1.text(i, row['mean']*100 + 1, f"{row['mean']*100:.1f}%", 
             ha='center', va='bottom', fontsize=10, fontweight='bold')
ax1.grid(axis='y', alpha=0.3)

# 2. Default Rate by City Tier
ax2 = axes[0, 1]
tier_summary = df.groupby('City_Tier')['Default_Label_Calibrated'].agg(['mean', 'sum', 'count'])
tier_summary = tier_summary.reindex(['Tier_1', 'Tier_2', 'Tier_3'])
colors2 = ['#1f77b4', '#aec7e8', '#c7c7c7']
bars2 = ax2.bar(tier_summary.index, tier_summary['mean'] * 100, color=colors2, alpha=0.7, edgecolor='black')
ax2.set_ylabel('Default Rate (%)', fontsize=11)
ax2.set_title('Default Rate by City Tier', fontsize=12, fontweight='bold')
ax2.set_ylim(0, 20)
for i, (idx, row) in enumerate(tier_summary.iterrows()):
    ax2.text(i, row['mean']*100 + 0.5, f"{row['mean']*100:.1f}%", 
             ha='center', va='bottom', fontsize=10, fontweight='bold')
ax2.grid(axis='y', alpha=0.3)

# 3. Heatmap: Occupation × City Tier
ax3 = axes[1, 0]
cross_tab = pd.crosstab(df['Occupation'], df['City_Tier'], 
                         values=df['Default_Label_Calibrated'], aggfunc='mean') * 100
cross_tab = cross_tab.reindex(['Professional', 'Self_Employed', 'Student', 'Retired'])
cross_tab = cross_tab[['Tier_1', 'Tier_2', 'Tier_3']]
sns.heatmap(cross_tab, annot=True, fmt='.1f', cmap='RdYlGn_r', 
            cbar_kws={'label': 'Default Rate (%)'}, ax=ax3,
            vmin=0, vmax=25, linewidths=1, linecolor='black')
ax3.set_title('Default Rate Heatmap: Occupation × City Tier', fontsize=12, fontweight='bold')
ax3.set_xlabel('City Tier', fontsize=11)
ax3.set_ylabel('Occupation', fontsize=11)

# 4. Ridge Probability Distribution
ax4 = axes[1, 1]
ax4.hist(df[df['Default_Label_Calibrated']==0]['Default_Prob_Ridge'], 
         bins=50, alpha=0.6, color='green', label='Non-Default', edgecolor='black')
ax4.hist(df[df['Default_Label_Calibrated']==1]['Default_Prob_Ridge'], 
         bins=50, alpha=0.6, color='red', label='Default', edgecolor='black')
ax4.set_xlabel('Ridge Default Probability', fontsize=11)
ax4.set_ylabel('Count', fontsize=11)
ax4.set_title('Ridge Probability Distribution by Default Label', fontsize=12, fontweight='bold')
ax4.legend(loc='upper right', fontsize=10)
ax4.grid(axis='y', alpha=0.3)

# Adjust layout and save
plt.tight_layout()
output_path = r"c:\Users\subha\Desktop\risk_credit_final\realistic_defaults_visualization.png"
plt.savefig(output_path, dpi=300, bbox_inches='tight')
print(f"\n✓ Visualization saved to: {output_path}")

# Show plot
plt.show()

print("\nVisualization complete!")
