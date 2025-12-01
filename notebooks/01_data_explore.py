import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os

os.makedirs('results', exist_ok=True)

# load
df = pd.read_csv('data/raw/synthetic_projects.csv')
print("Rows,Cols:", df.shape)
print(df.head(10).to_string(index=False))

# summary
desc = df.describe(include='all').transpose()
desc.to_csv('results/data_summary.csv')
print("Saved results/data_summary.csv")

# missing
print("Missing per col:\n", df.isnull().sum())

# hist final cost
plt.figure(figsize=(8,4))
sns.histplot(df['final_cost_millions'], bins=40, kde=True)
plt.title('Final cost distribution')
plt.xlabel('Final cost (millions)')
plt.tight_layout()
plt.savefig('results/figure_final_cost_hist.png')
plt.close()

# boxplot by soil
plt.figure(figsize=(6,4))
sns.boxplot(x='soil_quality', y='final_cost_millions', data=df)
plt.title('Cost by soil quality')
plt.savefig('results/box_cost_soil.png')
plt.close()

# scatter reliability vs cost
plt.figure(figsize=(6,4))
sns.scatterplot(x='contractor_reliability', y='final_cost_millions', hue='project_type', data=df)
plt.title('Reliability vs Final Cost')
plt.savefig('results/scatter_reliability_cost.png')
plt.close()

# correlation heatmap
num_cols = df.select_dtypes(include=[np.number]).columns
plt.figure(figsize=(9,6))
sns.heatmap(df[num_cols].corr(), annot=True, fmt=".2f", cmap='coolwarm')
plt.savefig('results/heatmap_corr.png')
plt.close()

# time series sample (if exists)
if os.path.exists('data/raw/sample_time_series.csv'):
    ts = pd.read_csv('data/raw/sample_time_series.csv')
    for pid in ts['project_id'].unique()[:3]:
        sub = ts[ts['project_id']==pid]
        plt.figure(figsize=(8,4))
        plt.plot(sub['month_index'], sub['cumulative_cost_millions'], marker='o')
        plt.title(f'S-curve: {pid}')
        plt.xlabel('month index')
        plt.ylabel('cumulative cost (millions)')
        plt.tight_layout()
        plt.savefig(f'results/s_curve_{pid}.png')
        plt.close()
else:
    print("No sample_time_series found; skip TS plots.")
print('EDA finished — results/ contains summary and figures.')
