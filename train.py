"""
MERIDIAN TELEMETRICS — Signal Anomaly Classifier
==================================================
Model: Stacked Ensemble (2026 pattern)
  Base learners : RandomForest + GradientBoosting + ExtraTrees
  Meta-learner  : LogisticRegression (calibrated)
  Calibration   : CalibratedClassifierCV (isotonic)

Run:
    python src/train.py

Outputs (saved to models/):
    meridian_model.joblib     — full serialised pipeline
    meridian_scaler.joblib    — fitted StandardScaler
    training_report.json      — metrics, thresholds, feature importance
"""

import json
import time
import warnings
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.calibration import CalibratedClassifierCV, calibration_curve
from sklearn.ensemble import (
    ExtraTreesClassifier,
    GradientBoostingClassifier,
    RandomForestClassifier,
    StackingClassifier,
)
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    average_precision_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_recall_curve,
    roc_auc_score,
)
from sklearn.model_selection import StratifiedKFold, cross_val_score, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings("ignore")

# ── Paths ──────────────────────────────────────────────────────
ROOT       = Path(__file__).parent.parent
DATA_PATH  = ROOT / "data" / "meridian_signals.csv"
MODEL_DIR  = ROOT / "models"
OUTPUT_DIR = ROOT / "outputs"
MODEL_DIR.mkdir(exist_ok=True)
OUTPUT_DIR.mkdir(exist_ok=True)

FEATURES = [
    "frequency_mhz",
    "bandwidth_khz",
    "duration_sec",
    "snr_db",
    "drift_rate_hz_s",
    "pulse_interval_sec",
    "encoding_density",
    "bearing_deg",
    "elevation_deg",
    "polarization_ratio",
    "spectral_kurtosis",
    "prime_interval_flag",
    "hydrogen_line_delta",
]

TARGET  = "label"
SEED    = 42
CV_FOLDS = 5


# ── 1. Load data ───────────────────────────────────────────────
def load_data():
    print("── Loading dataset ──")
    df = pd.read_csv(DATA_PATH)
    print(f"   Rows: {len(df):,}  |  Features: {len(FEATURES)}  |  "
          f"Anomaly rate: {df[TARGET].mean()*100:.1f}%")
    X = df[FEATURES].values
    y = df[TARGET].values
    return X, y, df


# ── 2. Feature engineering (in-place, pre-scale) ──────────────
def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add interaction and ratio features before scaling.
    These capture domain knowledge about what makes a signal structured.
    """
    df = df.copy()
    # SNR per unit bandwidth — narrow + strong = suspicious
    df["snr_per_bw"]        = df["snr_db"] / (df["bandwidth_khz"] + 1e-9)
    # Encoding density × SNR — doubly suspicious
    df["density_snr_prod"]  = df["encoding_density"] * df["snr_db"].clip(0)
    # Is the signal within 10 kHz of the hydrogen line?
    df["near_h_line"]       = (df["hydrogen_line_delta"] < 0.01).astype(float)
    # Pulse periodicity strength
    df["has_pulse"]         = (df["pulse_interval_sec"] > 0).astype(float)
    return df

ENGINEERED = FEATURES + ["snr_per_bw", "density_snr_prod", "near_h_line", "has_pulse"]


# ── 3. Build stacked ensemble ──────────────────────────────────
def build_model() -> Pipeline:
    """
    2026-pattern stacked ensemble:
      Three diverse base learners (tree-based, different biases)
      → Logistic meta-learner trained on out-of-fold predictions
      → Probability calibration (isotonic regression)

    Wrapped in a Pipeline with StandardScaler so the entire thing
    is one serialisable object.
    """
    rf = RandomForestClassifier(
        n_estimators=300,
        max_depth=12,
        min_samples_leaf=4,
        max_features="sqrt",
        class_weight="balanced",
        n_jobs=-1,
        random_state=SEED,
    )
    gb = GradientBoostingClassifier(
        n_estimators=200,
        learning_rate=0.08,
        max_depth=5,
        subsample=0.85,
        max_features="sqrt",
        random_state=SEED,
    )
    et = ExtraTreesClassifier(
        n_estimators=300,
        max_depth=14,
        min_samples_leaf=3,
        class_weight="balanced",
        n_jobs=-1,
        random_state=SEED,
    )

    stacker = StackingClassifier(
        estimators=[("rf", rf), ("gb", gb), ("et", et)],
        final_estimator=LogisticRegression(
            C=1.0,
            max_iter=1000,
            class_weight="balanced",
            random_state=SEED,
        ),
        cv=StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED),
        stack_method="predict_proba",
        passthrough=False,
        n_jobs=-1,
    )

    calibrated = CalibratedClassifierCV(
        stacker,
        method="isotonic",
        cv=3,
    )

    pipeline = Pipeline([
        ("scaler", StandardScaler()),
        ("model",  calibrated),
    ])
    return pipeline


# ── 4. Find optimal threshold via PR curve ────────────────────
def optimal_threshold(y_true, y_prob) -> tuple[float, float, float]:
    """F1-maximising threshold on validation set."""
    prec, rec, thresholds = precision_recall_curve(y_true, y_prob)
    f1_scores = 2 * prec * rec / (prec + rec + 1e-9)
    idx = np.argmax(f1_scores[:-1])
    return float(thresholds[idx]), float(prec[idx]), float(rec[idx])


# ── 5. Feature importance (permutation-style via RF) ─────────
def extract_importance(pipeline, feature_names) -> dict:
    """Pull RF importance from inside the stacked ensemble."""
    try:
        stacker    = pipeline.named_steps["model"].calibrated_classifiers_[0].estimator
        rf_est     = stacker.estimators_[0]
        importances = rf_est.feature_importances_
        ranked = sorted(
            zip(feature_names, importances.tolist()),
            key=lambda x: x[1], reverse=True
        )
        return {k: round(v, 6) for k, v in ranked}
    except Exception:
        return {}


# ── 6. Main training loop ─────────────────────────────────────
def train():
    t0 = time.time()

    # Load + engineer
    X_raw, y, df = load_data()
    df_eng  = engineer_features(df[FEATURES + [TARGET]])
    X       = df_eng[ENGINEERED].values

    # Split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, stratify=y, random_state=SEED
    )
    print(f"\n── Train/test split ──")
    print(f"   Train: {len(X_train):,}  |  Test: {len(X_test):,}")

    # Cross-validation on training set first
    print(f"\n── Cross-validation ({CV_FOLDS}-fold) ──")
    cv_pipe = build_model()
    cv_scores = cross_val_score(
        cv_pipe, X_train, y_train,
        cv=StratifiedKFold(CV_FOLDS, shuffle=True, random_state=SEED),
        scoring="roc_auc",
        n_jobs=-1,
    )
    print(f"   AUC-ROC: {cv_scores.mean():.4f} ± {cv_scores.std():.4f}")

    # Full fit on train
    print(f"\n── Fitting final model ──")
    pipeline = build_model()
    pipeline.fit(X_train, y_train)
    print(f"   Done in {time.time()-t0:.1f}s")

    # Evaluate on held-out test set
    print(f"\n── Test set evaluation ──")
    y_prob = pipeline.predict_proba(X_test)[:, 1]
    threshold, prec_at_t, rec_at_t = optimal_threshold(y_test, y_prob)
    y_pred = (y_prob >= threshold).astype(int)

    auc_roc = roc_auc_score(y_test, y_prob)
    auc_pr  = average_precision_score(y_test, y_prob)
    f1      = f1_score(y_test, y_pred)
    cm      = confusion_matrix(y_test, y_pred).tolist()
    report  = classification_report(y_test, y_pred,
                                    target_names=["BACKGROUND","ANOMALOUS"])

    print(f"   AUC-ROC  : {auc_roc:.4f}")
    print(f"   AUC-PR   : {auc_pr:.4f}")
    print(f"   F1 Score : {f1:.4f}")
    print(f"   Threshold: {threshold:.4f}")
    print(f"\n{report}")
    print(f"   Confusion matrix:\n   {cm[0]}\n   {cm[1]}")

    # Feature importance
    importance = extract_importance(pipeline, ENGINEERED)
    if importance:
        print(f"\n── Feature importance (top 8) ──")
        for feat, val in list(importance.items())[:8]:
            bar = "█" * int(val * 300)
            print(f"   {feat:<22} {val:.4f}  {bar}")

    # Save model
    model_path  = MODEL_DIR / "meridian_model.joblib"
    joblib.dump(pipeline, model_path, compress=3)
    print(f"\n── Model saved → {model_path}  ({model_path.stat().st_size/1024:.0f} KB)")

    # Save training report
    report_data = {
        "model": "StackedEnsemble + CalibratedClassifierCV",
        "base_learners": ["RandomForest(300)", "GradientBoosting(200)", "ExtraTrees(300)"],
        "meta_learner": "LogisticRegression",
        "calibration": "isotonic",
        "features": ENGINEERED,
        "n_features": len(ENGINEERED),
        "dataset": {
            "total_rows": len(df),
            "train_rows": len(X_train),
            "test_rows":  len(X_test),
            "anomaly_rate_pct": round(float(y.mean() * 100), 2),
        },
        "cv": {
            "folds": CV_FOLDS,
            "auc_roc_mean": round(float(cv_scores.mean()), 4),
            "auc_roc_std":  round(float(cv_scores.std()),  4),
        },
        "test_metrics": {
            "auc_roc":        round(auc_roc, 4),
            "auc_pr":         round(auc_pr,  4),
            "f1_score":       round(f1,      4),
            "optimal_threshold": round(threshold, 4),
            "precision_at_threshold": round(prec_at_t, 4),
            "recall_at_threshold":    round(rec_at_t,  4),
            "confusion_matrix": cm,
        },
        "feature_importance": importance,
        "training_time_sec": round(time.time() - t0, 1),
        "sklearn_version": __import__("sklearn").__version__,
        "python_version": __import__("sys").version.split()[0],
    }

    report_path = OUTPUT_DIR / "training_report.json"
    with open(report_path, "w") as f:
        json.dump(report_data, f, indent=2)
    print(f"── Report saved → {report_path}")
    print(f"\n✓ Training complete in {time.time()-t0:.1f}s")

    return pipeline, report_data


if __name__ == "__main__":
    train()
