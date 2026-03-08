"""
MERIDIAN TELEMETRICS — Test Suite
===================================
Run: pytest tests/ -v
"""

import sys
from pathlib import Path
import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.generate_dataset import build_dataset, generate_background, generate_anomalous
from src.predict import _engineer, FEATURES, ENGINEERED


# ── Dataset tests ──────────────────────────────────────────────

class TestDatasetGeneration:

    def test_dataset_shape(self):
        df = build_dataset(n_total=500, anomaly_ratio=0.20)
        assert len(df) == 500

    def test_anomaly_ratio(self):
        df = build_dataset(n_total=1000, anomaly_ratio=0.25)
        ratio = df["label"].mean()
        assert 0.20 <= ratio <= 0.30, f"Anomaly ratio {ratio:.2f} out of expected range"

    def test_required_columns(self):
        df = build_dataset(n_total=100)
        required = set(FEATURES + ["label", "event_id"])
        assert required.issubset(df.columns), f"Missing: {required - set(df.columns)}"

    def test_no_nulls(self):
        df = build_dataset(n_total=200)
        assert df[FEATURES].isnull().sum().sum() == 0

    def test_physical_constraints(self):
        df = build_dataset(n_total=500)
        assert (df["snr_db"] >= -10).all(),              "SNR below floor"
        assert (df["bandwidth_khz"] > 0).all(),           "Non-positive bandwidth"
        assert (df["duration_sec"] >= 0.5).all(),         "Duration too short"
        assert (df["elevation_deg"].between(0, 90)).all(),"Elevation out of range"
        assert (df["polarization_ratio"].between(0, 1)).all()

    def test_event_ids_unique(self):
        df = build_dataset(n_total=200)
        assert df["event_id"].is_unique

    def test_binary_labels(self):
        df = build_dataset(n_total=200)
        assert set(df["label"].unique()).issubset({0, 1})

    def test_prime_flag_is_binary(self):
        df = build_dataset(n_total=300)
        assert set(df["prime_interval_flag"].unique()).issubset({0, 1})

    def test_hydrogen_line_delta_nonneg(self):
        df = build_dataset(n_total=300)
        assert (df["hydrogen_line_delta"] >= 0).all()

    def test_background_has_no_pulse(self):
        bg = generate_background(200)
        assert (bg["pulse_interval_sec"] == 0).all()

    def test_anomalous_higher_snr(self):
        bg   = generate_background(1000)
        anom = generate_anomalous(1000)
        assert anom["snr_db"].mean() > bg["snr_db"].mean()

    def test_anomalous_higher_encoding(self):
        bg   = generate_background(1000)
        anom = generate_anomalous(1000)
        assert anom["encoding_density"].mean() > bg["encoding_density"].mean()


# ── Feature engineering tests ──────────────────────────────────

class TestFeatureEngineering:

    @pytest.fixture
    def sample_df(self):
        df = build_dataset(n_total=100)
        return df[FEATURES + ["label"]]

    def test_engineered_columns_exist(self, sample_df):
        out = _engineer(sample_df)
        for col in ["snr_per_bw", "density_snr_prod", "near_h_line", "has_pulse"]:
            assert col in out.columns, f"Missing engineered column: {col}"

    def test_snr_per_bw_nonneg_where_snr_nonneg(self, sample_df):
        out = _engineer(sample_df)
        mask = sample_df["snr_db"] >= 0
        assert (out.loc[mask, "snr_per_bw"] >= 0).all()

    def test_near_h_line_binary(self, sample_df):
        out = _engineer(sample_df)
        assert set(out["near_h_line"].unique()).issubset({0.0, 1.0})

    def test_has_pulse_binary(self, sample_df):
        out = _engineer(sample_df)
        assert set(out["has_pulse"].unique()).issubset({0.0, 1.0})

    def test_engineered_feature_count(self, sample_df):
        out = _engineer(sample_df)
        for col in ENGINEERED:
            assert col in out.columns

    def test_no_inf_values(self, sample_df):
        out = _engineer(sample_df)
        numeric = out[ENGINEERED].select_dtypes(include=[np.number])
        assert not np.isinf(numeric.values).any(), "Infinite values in engineered features"

    def test_no_nan_values(self, sample_df):
        out = _engineer(sample_df)
        assert out[ENGINEERED].isnull().sum().sum() == 0

    def test_original_df_not_mutated(self, sample_df):
        original_cols = set(sample_df.columns)
        _engineer(sample_df)
        assert set(sample_df.columns) == original_cols, "Input DataFrame was mutated"


# ── Predict module tests (no model required) ───────────────────

class TestPredictModule:

    def test_engineer_preserves_row_count(self):
        df = build_dataset(n_total=50)[FEATURES]
        out = _engineer(df)
        assert len(out) == 50

    def test_engineered_list_length(self):
        assert len(ENGINEERED) == len(FEATURES) + 4

    def test_features_list_has_13_items(self):
        assert len(FEATURES) == 13


# ── Integration: generate → engineer → shape ──────────────────

class TestIntegration:

    def test_pipeline_produces_correct_shape(self):
        df = build_dataset(n_total=200)
        eng = _engineer(df[FEATURES])
        X = eng[ENGINEERED].values
        assert X.shape == (200, len(ENGINEERED))

    def test_no_constant_features_in_background(self):
        bg = generate_background(500)
        for col in FEATURES:
            if col in bg.columns:
                assert bg[col].std() > 0, f"Feature {col} is constant in background"

    def test_label_distribution_reasonable(self):
        df = build_dataset(n_total=2000, anomaly_ratio=0.22)
        ratio = df["label"].mean()
        assert 0.18 <= ratio <= 0.26
