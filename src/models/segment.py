"""
Customer segmentation using K-Means on RFM + churn risk scores.
Produces segment labels and named cohort profiles.
"""
import sys, os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))
import warnings; warnings.filterwarnings("ignore")
import matplotlib; matplotlib.use("Agg")

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
from sklearn.cluster     import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.metrics     import silhouette_score

from config.config import (
    MODEL_INPUT_CSV, AUGMENTED_CSV, CALIBRATED_MODEL, CLUSTER_MODEL_PATH,
    FEATURE_COLS_PATH, PLOTS_DIR, RANDOM_STATE, N_CLUSTERS, TARGET_COL, ID_COL
)

PLOTS_DIR.mkdir(parents=True, exist_ok=True)


SEGMENT_NAMES = {
    0: "High-Value Loyal",
    1: "At-Risk Decliners",
    2: "New Explorers",
    3: "Price-Sensitive",
    4: "Long-Tenured Stable",
    5: "Churned Likely",
}

SEGMENT_COLORS = [
    "#4CAF50","#F44336","#2196F3","#FF9800","#9C27B0","#00BCD4"
]


def load_features():
    df = pd.read_csv(MODEL_INPUT_CSV)
    raw = pd.read_csv(AUGMENTED_CSV)
    raw["TotalCharges"] = pd.to_numeric(raw["TotalCharges"], errors="coerce").fillna(0)
    # merge original columns for profiling
    feat_cols = joblib.load(FEATURE_COLS_PATH)
    X = df[feat_cols].fillna(0)
    return X, df, raw


def add_churn_risk(X, df):
    model = joblib.load(CALIBRATED_MODEL)
    feat_cols = joblib.load(FEATURE_COLS_PATH)
    proba = model.predict_proba(X[feat_cols])[:,1]
    df = df.copy()
    df["churn_probability"] = proba
    df["churn_label"]       = (proba >= 0.5).astype(int)
    return df


def run_clustering(X, df):
    feat_cols = joblib.load(FEATURE_COLS_PATH)
    # Use RFM + churn risk for clustering
    cluster_features = [c for c in feat_cols if any(
        kw in c for kw in ["RFM","Monetary","Recency","Frequency",
                            "Service","tenure","MonthlyCharges","TotalCharges",
                            "Is_Month","Contract","Has_Fiber"]
    )] + (["churn_probability"] if "churn_probability" in df.columns else [])
    cluster_features = [c for c in cluster_features if c in df.columns]

    scaler = StandardScaler()
    X_cl   = scaler.fit_transform(df[cluster_features].fillna(0))

    # Elbow + Silhouette
    inertias, sil_scores = [], []
    ks = range(3, 10)
    for k in ks:
        km = KMeans(n_clusters=k, random_state=RANDOM_STATE, n_init=10)
        km.fit(X_cl)
        inertias.append(km.inertia_)
        sil_scores.append(silhouette_score(X_cl, km.labels_))

    # Plot elbow
    fig, ax = plt.subplots(1, 2, figsize=(12, 4))
    ax[0].plot(ks, inertias, "bo-"); ax[0].set_title("Elbow Curve"); ax[0].set_xlabel("k")
    ax[1].plot(ks, sil_scores, "rs-"); ax[1].set_title("Silhouette Score"); ax[1].set_xlabel("k")
    plt.tight_layout()
    plt.savefig(PLOTS_DIR/"elbow_silhouette.png", dpi=150); plt.close()

    # Final model
    km_final = KMeans(n_clusters=N_CLUSTERS, random_state=RANDOM_STATE, n_init=20)
    labels   = km_final.fit_predict(X_cl)
    sil      = silhouette_score(X_cl, labels)
    print(f"K-Means k={N_CLUSTERS}  Silhouette={sil:.4f}")

    joblib.dump(km_final, CLUSTER_MODEL_PATH)
    df["segment_id"]   = labels
    df["segment_name"] = df["segment_id"].map(SEGMENT_NAMES)
    return df, km_final, cluster_features, sil


def profile_segments(df_raw, df_model):
    """Build a business-readable profile table per segment."""
    joined = df_raw.copy()
    joined["segment_id"]      = df_model["segment_id"].values
    joined["segment_name"]    = df_model["segment_name"].values
    joined["churn_probability"]= df_model["churn_probability"].values
    joined["Churn_flag"]      = df_model[TARGET_COL].values

    profile = joined.groupby("segment_name").agg(
        Count            = ("segment_id", "count"),
        Avg_Churn_Prob   = ("churn_probability", "mean"),
        Avg_Tenure       = ("tenure", "mean"),
        Avg_Monthly      = ("MonthlyCharges", "mean"),
        Avg_Total        = ("TotalCharges", "mean"),
        Churn_Rate       = ("Churn_flag", "mean"),
    ).round(3).reset_index()

    profile.to_csv(PLOTS_DIR/"segment_profiles.csv", index=False)
    print("\n=== Segment Profiles ===")
    print(profile.to_string(index=False))
    return profile


def plot_segments(df_model):
    plt.figure(figsize=(10, 7))
    for sid, color in enumerate(SEGMENT_COLORS):
        mask = df_model["segment_id"] == sid
        plt.scatter(
            df_model.loc[mask, "tenure"] if "tenure" in df_model else range(mask.sum()),
            df_model.loc[mask, "churn_probability"],
            c=color, alpha=0.5, s=15,
            label=SEGMENT_NAMES.get(sid, f"Seg {sid}")
        )
    plt.xlabel("Tenure (months)"); plt.ylabel("Churn Probability")
    plt.title("Customer Segments – Tenure vs Churn Risk")
    plt.legend(fontsize=9, markerscale=2); plt.tight_layout()
    plt.savefig(PLOTS_DIR/"segment_scatter.png", dpi=150); plt.close()

    # Bar chart: average churn prob per segment
    seg_means = df_model.groupby("segment_name")["churn_probability"].mean().sort_values(ascending=False)
    plt.figure(figsize=(10, 5))
    seg_means.plot(kind="bar", color=SEGMENT_COLORS[:len(seg_means)], edgecolor="black")
    plt.ylabel("Avg Churn Probability"); plt.title("Churn Risk by Segment")
    plt.xticks(rotation=30, ha="right"); plt.tight_layout()
    plt.savefig(PLOTS_DIR/"segment_churn_bar.png", dpi=150); plt.close()


def run():
    X, df_model, df_raw = load_features()
    df_model = add_churn_risk(X, df_model)

    # Bring tenure into df_model if missing
    if "tenure" not in df_model.columns and "tenure" in df_raw.columns:
        df_model["tenure"] = df_raw["tenure"].values

    df_model, km, cluster_feats, sil = run_clustering(X, df_model)
    profile = profile_segments(df_raw, df_model)
    plot_segments(df_model)

    # Save full scored dataset
    out = df_raw.copy()
    out["churn_probability"] = df_model["churn_probability"].values
    out["churn_label"]       = df_model["churn_label"].values
    out["segment_id"]        = df_model["segment_id"].values
    out["segment_name"]      = df_model["segment_name"].values

    from config.config import PROC_DIR
    out.to_csv(PROC_DIR/"customers_scored.csv", index=False)
    print(f"\nScored dataset saved -> {PROC_DIR/'customers_scored.csv'}")
    return out, profile


if __name__ == "__main__":
    run()
