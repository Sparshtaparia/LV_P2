"""
Retention Strategy Simulator.
Uses uplift modeling + Monte Carlo simulation to estimate churn reduction
per intervention type (discount, support outreach, plan upgrade).
"""
import sys, os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))
import warnings; warnings.filterwarnings("ignore")
import matplotlib; matplotlib.use("Agg")

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import joblib

from config.config import (
    PROC_DIR, CALIBRATED_MODEL, FEATURE_COLS_PATH, PLOTS_DIR, RANDOM_STATE, MODELS_DIR
)

UPLIFT_MODEL_PATH = MODELS_DIR / "uplift_model.pkl"

PLOTS_DIR.mkdir(parents=True, exist_ok=True)
np.random.seed(RANDOM_STATE)

# Intervention effects: uplift factors applied to feature proxies
INTERVENTIONS = {
    "Discount (10% off)": {
        "MonthlyCharges": 0.90,
        "uplift_base": 0.12,
        "description": "Reduce monthly charges by 10%"
    },
    "Support Outreach": {
        "Support_Tickets": 0.5,
        "uplift_base": 0.18,
        "description": "Proactive support contact reduces frustration"
    },
    "Plan Upgrade Offer": {
        "Is_Month_to_Month": 0.0,
        "Is_Two_Year": 1.0,
        "Contract_Month-to-month": 0.0,
        "Contract_Two year": 1.0,
        "uplift_base": 0.25,
        "description": "Convert month-to-month to 2-year contract"
    },
    "Loyalty Reward": {
        "RFM_Score": 1.1,
        "uplift_base": 0.10,
        "description": "Points + perks improve satisfaction"
    },
}

N_SIMULATIONS = 1_000


def load_scored():
    path = PROC_DIR / "customers_scored.csv"
    if not path.exists():
        raise FileNotFoundError("Run segment.py first to generate customers_scored.csv")
    return pd.read_csv(path)


def predict_batch(df_feat):
    model = joblib.load(CALIBRATED_MODEL)
    feat_cols = joblib.load(FEATURE_COLS_PATH)
    available = [c for c in feat_cols if c in df_feat.columns]
    missing   = [c for c in feat_cols if c not in df_feat.columns]
    X = df_feat[available].copy()
    for m in missing:
        X[m] = 0
    X = X[feat_cols].fillna(0)
    return model.predict_proba(X)[:, 1]


def simulate_intervention(df_model, intervention_name, intervention_cfg):
    """Apply intervention to features, re-predict, compute churn reduction."""
    # Check if we have a true CausalML uplift model
    if UPLIFT_MODEL_PATH.exists() and intervention_name == "Plan Upgrade Offer":
        # Example: we tie the trained CausalML model to a specific intervention
        learner, features = joblib.load(UPLIFT_MODEL_PATH)
        df_mod = df_model.copy().fillna(0)
        X = df_mod[features].values
        
        # Predict ITE (Individual Treatment Effect)
        ite = learner.predict(X)
        # We assume negative ITE means reduction in churn prob
        reduction = np.clip(-ite, 0, 1)
        
        simulated_mean = float(reduction.mean())
        simulated_std  = float(reduction.std())
        simulated_p5   = float(np.percentile(reduction, 5))
        simulated_p95  = float(np.percentile(reduction, 95))
        actual_reduction = simulated_mean
        
    else:
        # Fallback to the old mock Monte Carlo approach
        model = joblib.load(CALIBRATED_MODEL)
        feat_cols = joblib.load(FEATURE_COLS_PATH)

        df_mod = df_model.copy().fillna(0)
        df_mod = df_mod.apply(pd.to_numeric, errors='coerce').fillna(0)
        base_risk = df_mod.copy()

        # Apply feature deltas
        for col, factor in intervention_cfg.items():
            if col in ("uplift_base", "description"):
                continue
            if col in df_mod.columns:
                if factor < 1:
                    df_mod[col] = df_mod[col] * factor
                else:
                    df_mod[col] = factor

        base_proba = model.predict_proba(base_risk)[:, 1]
        new_proba  = model.predict_proba(df_mod)[:, 1]

        # Monte Carlo: sample noise around uplift_base
        uplift_noise = np.random.normal(
            loc=intervention_cfg["uplift_base"], scale=0.03, size=N_SIMULATIONS
        )
        mc_reductions = np.clip(uplift_noise, 0.05, 0.50)

        actual_reduction  = float((base_proba - new_proba).mean())
        simulated_mean    = float(mc_reductions.mean())
        simulated_std     = float(mc_reductions.std())
        simulated_p5      = float(np.percentile(mc_reductions, 5))
        simulated_p95     = float(np.percentile(mc_reductions, 95))

    return {
        "intervention":       intervention_name,
        "description":        intervention_cfg["description"],
        "model_reduction":    round(actual_reduction, 4),
        "mc_mean_reduction":  round(simulated_mean, 4),
        "mc_std":             round(simulated_std, 4),
        "mc_p5":              round(simulated_p5, 4),
        "mc_p95":             round(simulated_p95, 4),
        "customers_targeted": len(df_model),
        "expected_saved":     int(len(df_model) * simulated_mean),
    }


def plot_simulation(results):
    names   = [r["intervention"] for r in results]
    means   = [r["mc_mean_reduction"] * 100 for r in results]
    p5s     = [r["mc_p5"] * 100 for r in results]
    p95s    = [r["mc_p95"] * 100 for r in results]
    errors  = [[m - p5 for m, p5 in zip(means, p5s)],
               [p95 - m for m, p95 in zip(means, p95s)]]

    colors = ["#2196F3", "#4CAF50", "#FF9800", "#9C27B0"]
    plt.figure(figsize=(10, 6))
    bars = plt.barh(names, means, xerr=errors, color=colors,
                    height=0.5, capsize=6, edgecolor="black", alpha=0.85)
    plt.xlabel("Estimated Churn Reduction (%)", fontsize=12)
    plt.title("Monte Carlo Retention Simulation\n(with 90% confidence intervals)", fontsize=13)
    for bar, m in zip(bars, means):
        plt.text(bar.get_width() + 0.3, bar.get_y() + bar.get_height()/2,
                 f"{m:.1f}%", va="center", fontsize=10)
    plt.tight_layout()
    plt.savefig(PLOTS_DIR / "retention_simulation.png", dpi=150)
    plt.close()


def run(segment_filter: str = None):
    df_scored = load_scored()

    if segment_filter:
        df_scored = df_scored[df_scored["segment_name"] == segment_filter]

    f_cols  = joblib.load(FEATURE_COLS_PATH)
    available = [c for c in f_cols if c in df_scored.columns]
    missing   = [c for c in f_cols if c not in df_scored.columns]
    df_model  = df_scored[available].copy().fillna(0)
    for m in missing:
        df_model[m] = 0
    df_model = df_model[f_cols]
    df_model = df_model.apply(pd.to_numeric, errors='coerce').fillna(0)

    results = []
    for name, cfg in INTERVENTIONS.items():
        res = simulate_intervention(df_model, name, cfg)
        results.append(res)
        print(f"[{name}]  Churn Reduction: {res['mc_mean_reduction']*100:.1f}%  "
              f"({res['mc_p5']*100:.1f}–{res['mc_p95']*100:.1f}%)  "
              f"Expected saved: {res['expected_saved']} customers")

    plot_simulation(results)
    summary = pd.DataFrame(results)
    summary.to_csv(PLOTS_DIR / "retention_simulation.csv", index=False)
    print(f"\nRetention simulation saved -> {PLOTS_DIR/'retention_simulation.csv'}")
    return summary


if __name__ == "__main__":
    run()
