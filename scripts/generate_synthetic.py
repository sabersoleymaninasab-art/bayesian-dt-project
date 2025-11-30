# scripts/generate_synthetic.py  (نسخه ارتقا)
import numpy as np
import pandas as pd
from datetime import datetime
import os

np.random.seed(42)
os.makedirs("data/raw", exist_ok=True)

N = 300
# project types with different scales
proj_types = np.random.choice(['road','bridge','building'], size=N, p=[0.5,0.2,0.3])
# baseline cost ~ LogNormal by type
cost_scale = {'road': 5.0, 'bridge': 6.0, 'building': 5.5}  # log-scale means
baseline_cost = np.array([np.random.lognormal(mean=cost_scale[t], sigma=0.4) for t in proj_types])
# scale to millions roughly
baseline_cost = np.round(baseline_cost * 1.0, 2)  # units: million

baseline_duration = np.array([np.random.randint(12, 48) if t!='bridge' else np.random.randint(18,72) for t in proj_types])

# features
soil_quality = np.random.choice(['good','moderate','poor'], size=N, p=[0.45,0.35,0.2])
contractor_reliability = np.clip(np.random.beta(5,2, size=N), 0.2, 0.99)
supply_delay_rate = np.random.poisson(1.0, size=N)
design_change_rate = np.random.beta(2,6, size=N)
weather_risk_index = np.random.uniform(0,1, size=N)

# effects (intuition: poor soil and low reliability increase cost & duration)
soil_num = np.array([0 if s=='good' else (1 if s=='moderate' else 2) for s in soil_quality])
theta0 = 0.02
theta_soil = 0.15
theta_contractor = -0.9
theta_supply = 0.12
theta_design = 0.8
theta_weather = 0.5

log_mult = (theta0
            + theta_soil * soil_num
            + theta_contractor * (1 - contractor_reliability)
            + theta_supply * supply_delay_rate
            + theta_design * design_change_rate
            + theta_weather * weather_risk_index
           )
noise = np.random.normal(0, 0.25, size=N)
cost_multiplier = np.exp(log_mult + noise)
final_cost = np.round(baseline_cost * cost_multiplier, 2)

duration_multiplier = 1 + 0.1*soil_num + 0.2*(1-contractor_reliability) + 0.08*supply_delay_rate + 0.25*design_change_rate + 0.15*weather_risk_index
final_duration = np.round(baseline_duration * duration_multiplier).astype(int)

# add number of change orders and total change value
change_orders_count = np.random.poisson(lam=design_change_rate*3+0.2, size=N)
change_orders_value = np.round(final_cost * (design_change_rate * 0.3 * np.random.rand(N) * change_orders_count), 2)

# create time-series monthly S-curve for first K projects sample, and a summary for all
# For each project create monthly spend series as planned burn with random delays
sample_ts = []
for i in range(5):  # create detailed TS for 5 sample projects for DT demo
    pid = f"P{1000+i}"
    months = int(final_duration[i])
    planned = np.linspace(0,1,months)
    # S-curve shape by beta CDF
    a,b = 2+np.random.rand(), 2+np.random.rand()
    s_frac = np.array([np.random.beta(a,b) for _ in range(months)])
    s_frac = np.cumsum(s_frac)
    s_frac = s_frac / s_frac[-1]
    monthly_total = final_cost[i] * np.diff(np.concatenate([[0], s_frac]))
    cum = 0.0
    for m in range(months):
        # occasional missing report
        if np.random.rand() < 0.02:
            continue
        monthly = max(0, monthly_total[m] * (1 + np.random.normal(0,0.05)))
        cum += monthly
        sample_ts.append({
            'project_id': pid,
            'month_index': m+1,
            'monthly_spend_millions': round(monthly,2),
            'cumulative_cost_millions': round(cum,2),
            'reported_design_change': int(np.random.rand() < design_change_rate[i])
        })

df = pd.DataFrame({
    'project_id': [f"P{1000+i}" for i in range(N)],
    'project_type': proj_types,
    'baseline_cost_millions': np.round(baseline_cost,2),
    'baseline_duration_months': baseline_duration,
    'soil_quality': soil_quality,
    'contractor_reliability': np.round(contractor_reliability, 3),
    'supply_delay_rate': supply_delay_rate,
    'design_change_rate': np.round(design_change_rate, 3),
    'weather_risk_index': np.round(weather_risk_index, 3),
    'change_orders_count': change_orders_count,
    'change_orders_value_millions': np.round(change_orders_value,2),
    'cost_multiplier': np.round(cost_multiplier, 3),
    'final_cost_millions': final_cost,
    'final_duration_months': final_duration
})

ts_df = pd.DataFrame(sample_ts)
df.to_csv("data/raw/synthetic_projects.csv", index=False)
ts_df.to_csv("data/raw/sample_time_series.csv", index=False)
print("Written synthetic_projects.csv (N={}) and sample_time_series.csv (rows={})".format(len(df), len(ts_df)))
