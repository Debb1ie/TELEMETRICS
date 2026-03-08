"""
MERIDIAN TELEMETRICS — Signal Inference
=========================================
Load the trained model and classify new signal events.

Usage:
    # Single prediction (dict)
    from src.predict import predict_signal, load_model
    model = load_model()
    result = predict_signal(model, {
        "frequency_mhz":     1420.405,
        "bandwidth_khz":     0.8,
        "duration_sec":      87.4,
        "snr_db":            18.2,
        "drift_rate_hz_s":   0.12,
        "pulse_interval_sec":7.0,
        "encoding_density":  11.2,
        "bearing_deg":       47.2,
        "elevation_deg":     42.0,
        "polarization_ratio":0.91,
        "spectral_kurtosis": 8.3,
        "prime_interval_flag": 1,
        "hydrogen_line_delta": 0.0,
    })

    # Batch prediction (CSV)
    python src/predict.py --input data/new_signals.csv --output outputs/predictions.csv
"""

import argparse
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

ROOT      = Path(__file__).parent.parent
MODEL_DIR = ROOT / "models"

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

# Must match train.py
THRESHOLD = None   # loaded from training_report.json if available


def load_model(path: Path | None = None) -> object:
    p = path or MODEL_DIR / "meridian_model.joblib"
    if not p.exists():
        raise FileNotFoundError(
            f"Model not found at {p}. Run `python src/train.py` first."
        )
    return joblib.load(p)


def load_threshold() -> float:
    report_path = ROOT / "outputs" / "training_report.json"
    if report_path.exists():
        with open(report_path) as f:
            data = json.load(f)
        return data["test_metrics"]["optimal_threshold"]
    return 0.50  # default fallback


def _engineer(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["snr_per_bw"]       = df["snr_db"] / (df["bandwidth_khz"] + 1e-9)
    df["density_snr_prod"] = df["encoding_density"] * df["snr_db"].clip(0)
    df["near_h_line"]      = (df["hydrogen_line_delta"] < 0.01).astype(float)
    df["has_pulse"]        = (df["pulse_interval_sec"] > 0).astype(float)
    return df

ENGINEERED = FEATURES + ["snr_per_bw", "density_snr_prod", "near_h_line", "has_pulse"]


def predict_signal(model, signal: dict, threshold: float | None = None) -> dict:
    """
    Classify a single signal event.

    Args:
        model     : fitted pipeline from load_model()
        signal    : dict with keys matching FEATURES
        threshold : decision threshold (float 0–1). Loaded from report if None.

    Returns:
        dict with keys:
            classification  : "ANOMALOUS" | "BACKGROUND"
            probability     : float — P(anomalous)
            confidence      : "HIGH" | "MEDIUM" | "LOW"
            threshold_used  : float
    """
    if threshold is None:
        threshold = load_threshold()

    df = pd.DataFrame([signal])

    # Fill missing features with neutral values
    for feat in FEATURES:
        if feat not in df.columns:
            df[feat] = 0.0

    df = _engineer(df)
    X  = df[ENGINEERED].values

    prob = float(model.predict_proba(X)[0, 1])
    label = "ANOMALOUS" if prob >= threshold else "BACKGROUND"

    margin = abs(prob - threshold)
    confidence = "HIGH" if margin > 0.25 else "MEDIUM" if margin > 0.10 else "LOW"

    return {
        "classification":  label,
        "probability":     round(prob, 4),
        "confidence":      confidence,
        "threshold_used":  round(threshold, 4),
    }


def predict_batch(model, df: pd.DataFrame, threshold: float | None = None) -> pd.DataFrame:
    """
    Classify a DataFrame of signal events.

    Returns original df with appended columns:
        p_anomalous, classification, confidence
    """
    if threshold is None:
        threshold = load_threshold()

    df_feat = _engineer(df[FEATURES])
    X       = df_feat[ENGINEERED].values

    probs = model.predict_proba(X)[:, 1]
    labels = np.where(probs >= threshold, "ANOMALOUS", "BACKGROUND")
    margin = np.abs(probs - threshold)
    conf   = np.where(margin > 0.25, "HIGH", np.where(margin > 0.10, "MEDIUM", "LOW"))

    out = df.copy()
    out["p_anomalous"]    = probs.round(4)
    out["classification"] = labels
    out["confidence"]     = conf

    return out


# ── CLI ────────────────────────────────────────────────────────
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="MERIDIAN signal classifier")
    parser.add_argument("--input",     type=str, help="Input CSV path")
    parser.add_argument("--output",    type=str, help="Output CSV path",
                        default="outputs/predictions.csv")
    parser.add_argument("--threshold", type=float, default=None,
                        help="Override decision threshold (0–1)")
    parser.add_argument("--demo",      action="store_true",
                        help="Run on 5 demo signals")
    args = parser.parse_args()

    model = load_model()
    threshold = args.threshold or load_threshold()

    if args.demo or not args.input:
        print("\n── Demo predictions ──\n")
        demo_signals = [
            # Should be ANOMALOUS
            {"frequency_mhz":1420.405,"bandwidth_khz":0.8,"duration_sec":87,"snr_db":18.2,
             "drift_rate_hz_s":0.12,"pulse_interval_sec":7,"encoding_density":11.2,
             "bearing_deg":47.2,"elevation_deg":42,"polarization_ratio":0.91,
             "spectral_kurtosis":8.3,"prime_interval_flag":1,"hydrogen_line_delta":0.0},
            # Should be BACKGROUND
            {"frequency_mhz":1435.1,"bandwidth_khz":420,"duration_sec":8,"snr_db":2.1,
             "drift_rate_hz_s":12.3,"pulse_interval_sec":0,"encoding_density":0.9,
             "bearing_deg":220,"elevation_deg":60,"polarization_ratio":0.45,
             "spectral_kurtosis":2.9,"prime_interval_flag":0,"hydrogen_line_delta":14.7},
            # Edge case — near hydrogen line but weak
            {"frequency_mhz":1420.408,"bandwidth_khz":2.1,"duration_sec":4,"snr_db":5.5,
             "drift_rate_hz_s":0.5,"pulse_interval_sec":0,"encoding_density":2.1,
             "bearing_deg":51,"elevation_deg":38,"polarization_ratio":0.62,
             "spectral_kurtosis":4.1,"prime_interval_flag":0,"hydrogen_line_delta":0.003},
        ]
        for i, sig in enumerate(demo_signals, 1):
            result = predict_signal(model, sig, threshold)
            print(f"  Signal {i}: {result['classification']:<12} "
                  f"p={result['probability']:.3f}  "
                  f"confidence={result['confidence']}")
    else:
        df_in  = pd.read_csv(args.input)
        df_out = predict_batch(model, df_in, threshold)
        out_path = Path(args.output)
        out_path.parent.mkdir(exist_ok=True)
        df_out.to_csv(out_path, index=False)
        print(f"\n── Predictions saved → {out_path}")
        print(f"   Total: {len(df_out):,}")
        print(f"   ANOMALOUS : {(df_out['classification']=='ANOMALOUS').sum():,}")
        print(f"   BACKGROUND: {(df_out['classification']=='BACKGROUND').sum():,}")
