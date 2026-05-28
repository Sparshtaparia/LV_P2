"""
Training pipeline: RF → XGBoost → LightGBM → best model selection + calibration.
"""
import sys, os, warnings
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))
warnings.filterwarnings("ignore")
import matplotlib; matplotlib.use("Agg")

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import joblib, mlflow, mlflow.sklearn, shap, optuna
optuna.logging.set_verbosity(optuna.logging.WARNING)

from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.ensemble        import RandomForestClassifier, VotingClassifier
from sklearn.calibration     import CalibratedClassifierCV
from sklearn.metrics         import (
    roc_auc_score, average_precision_score, f1_score,
    confusion_matrix, precision_recall_curve, roc_curve
)
from imblearn.over_sampling  import SMOTE
import xgboost as xgb
import lightgbm as lgb
import catboost as cb
import logging

from config.config import (
    MODEL_INPUT_CSV, BEST_MODEL_PATH, CALIBRATED_MODEL, FEATURE_COLS_PATH,
    MLFLOW_TRACKING_URI, MLFLOW_EXPERIMENT, PLOTS_DIR,
    RANDOM_STATE, TEST_SIZE, TARGET_COL, ID_COL, SMOTE_RATIO
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger(__name__)
CV  = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
PLOTS_DIR.mkdir(parents=True, exist_ok=True)


def load_data():
    df = pd.read_csv(MODEL_INPUT_CSV)
    drop = [c for c in [ID_COL] if c in df.columns]
    X = df.drop(columns=drop + [TARGET_COL]).fillna(0)
    X = X.select_dtypes(include=[np.number])
    y = df[TARGET_COL].astype(int)
    joblib.dump(list(X.columns), FEATURE_COLS_PATH)
    return X, y


def apply_smote(X_tr, y_tr):
    sm = SMOTE(sampling_strategy=SMOTE_RATIO, random_state=RANDOM_STATE)
    X_r, y_r = sm.fit_resample(X_tr, y_tr)
    log.info(f"SMOTE: {y_tr.sum()}→{y_r.sum()} positives  total={len(y_r)}")
    return X_r, y_r


def precision_top20(y_true, y_score):
    n   = int(len(y_true) * 0.20)
    idx = np.argsort(y_score)[::-1][:n]
    return float(np.array(y_true)[idx].mean())


def metrics(model, X, y):
    p = model.predict_proba(X)[:, 1]
    d = model.predict(X)
    return dict(
        roc_auc     = roc_auc_score(y, p),
        pr_auc      = average_precision_score(y, p),
        f1          = f1_score(y, d),
        precision_top20 = precision_top20(y, p),
    )


def tune_xgb(X, y):
    spw = (y == 0).sum() / (y == 1).sum()
    def obj(t):
        m = xgb.XGBClassifier(
            n_estimators    = t.suggest_int("n_est", 200, 500),
            max_depth       = t.suggest_int("max_d", 3, 8),
            learning_rate   = t.suggest_float("lr", 0.01, 0.2, log=True),
            subsample       = t.suggest_float("ss", 0.6, 1.0),
            colsample_bytree= t.suggest_float("cs", 0.6, 1.0),
            scale_pos_weight= spw,
            eval_metric="logloss", use_label_encoder=False,
            random_state=RANDOM_STATE, n_jobs=-1,
        )
        return cross_val_score(m, X, y, cv=CV, scoring="roc_auc").mean()
    s = optuna.create_study(direction="maximize")
    s.optimize(obj, n_trials=20, show_progress_bar=False)
    bp = s.best_params
    return xgb.XGBClassifier(
        n_estimators=bp["n_est"], max_depth=bp["max_d"],
        learning_rate=bp["lr"], subsample=bp["ss"],
        colsample_bytree=bp["cs"], scale_pos_weight=spw,
        eval_metric="logloss", use_label_encoder=False,
        random_state=RANDOM_STATE, n_jobs=-1,
    )


def tune_lgbm(X, y):
    def obj(t):
        m = lgb.LGBMClassifier(
            n_estimators    = t.suggest_int("n_est", 200, 500),
            max_depth       = t.suggest_int("max_d", 3, 8),
            learning_rate   = t.suggest_float("lr", 0.01, 0.2, log=True),
            num_leaves      = t.suggest_int("nl", 20, 120),
            subsample       = t.suggest_float("ss", 0.6, 1.0),
            colsample_bytree= t.suggest_float("cs", 0.6, 1.0),
            is_unbalance=True, random_state=RANDOM_STATE,
            n_jobs=-1, verbosity=-1,
        )
        return cross_val_score(m, X, y, cv=CV, scoring="roc_auc").mean()
    s = optuna.create_study(direction="maximize")
    s.optimize(obj, n_trials=20, show_progress_bar=False)
    bp = s.best_params
    return lgb.LGBMClassifier(
        n_estimators=bp["n_est"], max_depth=bp["max_d"],
        learning_rate=bp["lr"], num_leaves=bp["nl"],
        subsample=bp["ss"], colsample_bytree=bp["cs"],
        is_unbalance=True, random_state=RANDOM_STATE,
        n_jobs=-1, verbosity=-1,
    )


def tune_cb(X, y):
    spw = (y == 0).sum() / (y == 1).sum()
    def obj(t):
        m = cb.CatBoostClassifier(
            iterations=t.suggest_int("iters", 200, 500),
            depth=t.suggest_int("depth", 3, 8),
            learning_rate=t.suggest_float("lr", 0.01, 0.2, log=True),
            l2_leaf_reg=t.suggest_float("l2", 1, 10),
            scale_pos_weight=spw,
            verbose=False, random_state=RANDOM_STATE
        )
        return cross_val_score(m, X, y, cv=CV, scoring="roc_auc").mean()
    s = optuna.create_study(direction="maximize")
    s.optimize(obj, n_trials=15, show_progress_bar=False)
    bp = s.best_params
    return cb.CatBoostClassifier(
        iterations=bp["iters"], depth=bp["depth"],
        learning_rate=bp["lr"], l2_leaf_reg=bp["l2"],
        scale_pos_weight=spw, verbose=False, random_state=RANDOM_STATE
    )


def optimize_threshold(model, X, y):
    p = model.predict_proba(X)[:, 1]
    prec, rec, thr = precision_recall_curve(y, p)
    f1 = 2 * (prec * rec) / (prec + rec + 1e-9)
    idx = np.argmax(f1)
    idx = min(idx, len(thr)-1)
    return thr[idx], f1[idx]


def plot_roc(preds, y):
    plt.figure(figsize=(8, 6))
    for name, p in preds.items():
        fpr, tpr, _ = roc_curve(y, p)
        plt.plot(fpr, tpr, lw=2, label=f"{name}  AUC={roc_auc_score(y,p):.3f}")
    plt.plot([0,1],[0,1],"k--")
    plt.xlabel("FPR"); plt.ylabel("TPR")
    plt.title("ROC Curves – Model Comparison"); plt.legend()
    plt.tight_layout()
    plt.savefig(PLOTS_DIR/"roc_curves.png", dpi=150); plt.close()


def plot_pr(preds, y):
    plt.figure(figsize=(8, 6))
    for name, p in preds.items():
        prec, rec, _ = precision_recall_curve(y, p)
        plt.plot(rec, prec, lw=2, label=f"{name}  AP={average_precision_score(y,p):.3f}")
    plt.xlabel("Recall"); plt.ylabel("Precision")
    plt.title("Precision-Recall Curves"); plt.legend()
    plt.tight_layout()
    plt.savefig(PLOTS_DIR/"pr_curves.png", dpi=150); plt.close()


def plot_cm(model, X, y, name):
    cm = confusion_matrix(y, model.predict(X))
    plt.figure(figsize=(5, 4))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=["No Churn","Churn"],
                yticklabels=["No Churn","Churn"])
    plt.title(f"Confusion Matrix – {name}"); plt.tight_layout()
    plt.savefig(PLOTS_DIR/f"cm_{name.lower().replace(' ','_')}.png", dpi=150); plt.close()


def plot_shap(model, X_sample, name):
    try:
        expl = shap.TreeExplainer(model)
        sv   = expl.shap_values(X_sample)
        if isinstance(sv, list): sv = sv[1]
        elif sv.ndim == 3:       sv = sv[:,:,1]
        plt.figure()
        shap.summary_plot(sv, X_sample, show=False, max_display=20)
        plt.tight_layout()
        plt.savefig(PLOTS_DIR/f"shap_{name.lower().replace(' ','_')}.png",
                    dpi=150, bbox_inches="tight")
        plt.close()
        log.info(f"SHAP saved for {name}")
    except Exception as e:
        log.warning(f"SHAP skipped: {e}")


def run():
    X, y = load_data()
    X_tr, X_te, y_tr, y_te = train_test_split(
        X, y, test_size=TEST_SIZE, stratify=y, random_state=RANDOM_STATE)
    X_tr_s, X_val, y_tr_s, y_val = train_test_split(
        X_tr, y_tr, test_size=0.125, stratify=y_tr, random_state=RANDOM_STATE)

    X_sm, y_sm = apply_smote(X_tr_s, y_tr_s)

    os.makedirs(str(BEST_MODEL_PATH.parent), exist_ok=True)
    # Ensure mlruns directory exists before setting URI
    from config.config import ROOT_DIR
    mlruns_path = ROOT_DIR / "mlruns"
    mlruns_path.mkdir(parents=True, exist_ok=True)
    mlflow.set_tracking_uri(str(mlruns_path.as_uri()))
    mlflow.set_experiment(MLFLOW_EXPERIMENT)

    results = {}

    # 1. Random Forest
    log.info("Training Random Forest …")
    with mlflow.start_run(run_name="RandomForest"):
        rf = RandomForestClassifier(n_estimators=300, max_depth=10,
                                    min_samples_leaf=5, class_weight="balanced",
                                    random_state=RANDOM_STATE, n_jobs=-1)
        rf.fit(X_sm, y_sm)
        m = metrics(rf, X_te, y_te)
        mlflow.log_metrics({f"test_{k}": v for k,v in m.items()})
        mlflow.sklearn.log_model(rf, "rf_model")
        results["Random Forest"] = (rf, m)
        log.info(f"RF  AUC={m['roc_auc']:.4f}  P@20%={m['precision_top20']:.4f}")

    # 2. XGBoost
    log.info("Tuning XGBoost …")
    with mlflow.start_run(run_name="XGBoost_Optuna"):
        xgm = tune_xgb(X_sm, y_sm)
        xgm.fit(X_sm, y_sm)
        m = metrics(xgm, X_te, y_te)
        mlflow.log_metrics({f"test_{k}": v for k,v in m.items()})
        results["XGBoost"] = (xgm, m)
        log.info(f"XGB AUC={m['roc_auc']:.4f}  P@20%={m['precision_top20']:.4f}")

    # 3. LightGBM
    log.info("Tuning LightGBM …")
    with mlflow.start_run(run_name="LightGBM_Optuna"):
        lgm = tune_lgbm(X_sm, y_sm)
        lgm.fit(X_sm, y_sm)
        m = metrics(lgm, X_te, y_te)
        mlflow.log_metrics({f"test_{k}": v for k,v in m.items()})
        results["LightGBM"] = (lgm, m)
        log.info(f"LGB AUC={m['roc_auc']:.4f}  P@20%={m['precision_top20']:.4f}")

    # 4. CatBoost
    log.info("Tuning CatBoost …")
    with mlflow.start_run(run_name="CatBoost_Optuna"):
        cbm = tune_cb(X_sm, y_sm)
        cbm.fit(X_sm, y_sm)
        m = metrics(cbm, X_te, y_te)
        mlflow.log_metrics({f"test_{k}": v for k,v in m.items()})
        results["CatBoost"] = (cbm, m)
        log.info(f"CB  AUC={m['roc_auc']:.4f}  P@20%={m['precision_top20']:.4f}")

    # 5. Ensemble (Voting)
    log.info("Training Ensemble (XGB + LGB + CB) …")
    with mlflow.start_run(run_name="Voting_Ensemble"):
        voting = VotingClassifier(
            estimators=[("xgb", xgm), ("lgb", lgm), ("cb", cbm)],
            voting="soft"
        )
        voting.fit(X_sm, y_sm)
        m = metrics(voting, X_te, y_te)
        mlflow.log_metrics({f"test_{k}": v for k,v in m.items()})
        results["Voting Ensemble"] = (voting, m)
        log.info(f"ENS AUC={m['roc_auc']:.4f}  P@20%={m['precision_top20']:.4f}")

    # Best model
    best_name  = max(results, key=lambda k: results[k][1]["roc_auc"])
    best_model, best_m = results[best_name]
    log.info(f"\n✅  Best: {best_name}  AUC={best_m['roc_auc']:.4f}")

    # Threshold Tuning
    best_thr, best_f1 = optimize_threshold(best_model, X_val, y_val)
    log.info(f"🏆 Optimized Decision Threshold: {best_thr:.3f} (Validation F1: {best_f1:.3f})")
    with open(BEST_MODEL_PATH.parent / "threshold.json", "w") as f:
        import json
        json.dump({"best_threshold": float(best_thr)}, f)

    cal = CalibratedClassifierCV(best_model, method="isotonic", cv=3)
    cal.fit(X_val, y_val)
    joblib.dump(best_model, BEST_MODEL_PATH)
    joblib.dump(cal,        CALIBRATED_MODEL)
    log.info(f"Saved → {BEST_MODEL_PATH}")

    preds = {n: m.predict_proba(X_te)[:,1] for n,(m,_) in results.items()}
    plot_roc(preds, y_te)
    plot_pr (preds, y_te)
    plot_cm (best_model, X_te, y_te, best_name)
    plot_shap(best_model, X_te.sample(min(300,len(X_te)), random_state=RANDOM_STATE), best_name)

    summary = pd.DataFrame({n: m for n,(_,m) in results.items()}).T.round(4)
    print("\n=== Model Comparison ===\n", summary.to_string())
    summary.to_csv(PLOTS_DIR/"model_comparison.csv")
    return best_model, cal, X_te, y_te


if __name__ == "__main__":
    run()
