"""
MERIDIAN TELEMETRICS — Dataset Generator
=========================================
Generates synthetic deep-space transmission records.

Each sample represents one intercepted signal event.
Target: classify whether a signal is ANOMALOUS (1) or BACKGROUND NOISE (0).

Features engineered from real radio-astronomy signal characteristics:
  - frequency_mhz       : signal carrier frequency
  - bandwidth_khz       : spectral width of the signal
  - duration_sec        : length of the transmission
  - snr_db              : signal-to-noise ratio in decibels
  - drift_rate_hz_s     : doppler drift rate (Hz/s) — non-zero = moving source
  - pulse_interval_sec  : time between repeating pulses (0 = non-repeating)
  - encoding_density    : information bits per symbol vs baseline
  - bearing_deg         : azimuth of arrival
  - elevation_deg       : elevation angle of arrival
  - polarization_ratio  : ratio of circular to linear polarization
  - spectral_kurtosis   : measure of non-Gaussianity in spectrum
  - prime_interval_flag : whether pulse intervals follow prime spacing (bool)
  - hydrogen_line_delta : absolute deviation from 1420.405 MHz

Label (target):
  0 = BACKGROUND  (natural radio sources, RFI, thermal noise)
  1 = ANOMALOUS   (structured, non-natural, possibly intelligent signal)
"""

import numpy as np
import pandas as pd
from pathlib import Path

RNG = np.random.default_rng(1974)   # Seed: year MERIDIAN was shut down


def generate_background(n: int) -> pd.DataFrame:
    """Natural radio sources, RFI, satellite spillover."""
    return pd.DataFrame({
        "frequency_mhz":     RNG.uniform(1400, 1450, n),
        "bandwidth_khz":     RNG.lognormal(4.0, 1.2, n),
        "duration_sec":      RNG.exponential(30, n),
        "snr_db":            RNG.normal(4.0, 3.5, n),
        "drift_rate_hz_s":   RNG.normal(0, 8.0, n),
        "pulse_interval_sec":np.zeros(n),                   # no periodicity
        "encoding_density":  RNG.normal(1.0, 0.3, n),       # baseline = 1×
        "bearing_deg":       RNG.uniform(0, 360, n),
        "elevation_deg":     RNG.uniform(5, 85, n),
        "polarization_ratio":RNG.beta(2, 2, n),
        "spectral_kurtosis": RNG.normal(3.0, 1.5, n),       # Gaussian = 3
        "prime_interval_flag": np.zeros(n, dtype=int),
        "label": np.zeros(n, dtype=int),
    })


def generate_anomalous(n: int) -> pd.DataFrame:
    """Structured, high-density, narrowband signals — the interesting ones."""
    # Majority cluster tightly around the hydrogen line
    freq = RNG.choice(
        [RNG.normal(1420.405, 0.005, n),   # hydrogen line (majority)
         RNG.uniform(1400, 1450, n)],       # occasional off-line
        p=[0.82, 0.18]
    )
    freq = np.where(
        RNG.random(n) < 0.82,
        RNG.normal(1420.405, 0.005, n),
        RNG.uniform(1400, 1450, n)
    )

    # Prime-spaced pulse intervals for ~60% of anomalous signals
    primes = [2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31]
    pulse = np.where(
        RNG.random(n) < 0.60,
        RNG.choice(primes, n),
        np.zeros(n)
    )
    prime_flag = (pulse > 0).astype(int)

    return pd.DataFrame({
        "frequency_mhz":     freq,
        "bandwidth_khz":     RNG.lognormal(1.5, 0.6, n),    # much narrower
        "duration_sec":      RNG.lognormal(3.5, 1.0, n),
        "snr_db":            RNG.normal(14.0, 4.0, n),       # stronger
        "drift_rate_hz_s":   RNG.normal(0.18, 0.4, n),      # slight drift
        "pulse_interval_sec":pulse,
        "encoding_density":  RNG.normal(9.5, 1.8, n),       # 11× baseline
        "bearing_deg":       RNG.normal(47.2, 2.5, n),      # consistent bearing
        "elevation_deg":     RNG.normal(42.0, 5.0, n),
        "polarization_ratio":RNG.beta(8, 2, n),              # more circular
        "spectral_kurtosis": RNG.normal(7.5, 2.0, n),       # heavy-tailed
        "prime_interval_flag": prime_flag,
        "label": np.ones(n, dtype=int),
    })


def build_dataset(n_total: int = 12_000, anomaly_ratio: float = 0.22) -> pd.DataFrame:
    n_anom = int(n_total * anomaly_ratio)
    n_back = n_total - n_anom

    df = pd.concat([
        generate_background(n_back),
        generate_anomalous(n_anom),
    ], ignore_index=True)

    # Add computed feature
    df["hydrogen_line_delta"] = (df["frequency_mhz"] - 1420.405).abs()

    # Clip physical impossibilities
    df["snr_db"]          = df["snr_db"].clip(lower=-10)
    df["bandwidth_khz"]   = df["bandwidth_khz"].clip(lower=0.1)
    df["duration_sec"]    = df["duration_sec"].clip(lower=0.5)
    df["elevation_deg"]   = df["elevation_deg"].clip(0, 90)
    df["bearing_deg"]     = df["bearing_deg"] % 360
    df["polarization_ratio"] = df["polarization_ratio"].clip(0, 1)

    # Shuffle
    df = df.sample(frac=1, random_state=42).reset_index(drop=True)

    # Assign event IDs
    df.insert(0, "event_id", [f"EVT-{i:06d}" for i in range(len(df))])

    return df


if __name__ == "__main__":
    out_dir = Path(__file__).parent.parent / "data"
    out_dir.mkdir(exist_ok=True)

    df = build_dataset()
    out_path = out_dir / "meridian_signals.csv"
    df.to_csv(out_path, index=False)

    print(f"Dataset saved → {out_path}")
    print(f"  Rows      : {len(df):,}")
    print(f"  Features  : {len(df.columns) - 2}")   # exclude event_id + label
    print(f"  Anomalous : {df['label'].sum():,} ({df['label'].mean()*100:.1f}%)")
    print(f"  Background: {(df['label']==0).sum():,} ({(df['label']==0).mean()*100:.1f}%)")
    print(f"\nSample:\n{df.head(3).to_string()}")
