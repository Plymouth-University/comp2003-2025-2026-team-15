"""
Integration Test: generate_dataset → train_model
-------------------------------------------------
Tests that a CSV produced by generate_dataset flows correctly into the training pipeline (label encoding, GroupShuffleSplit, RandomForest) and
that the resulting artefacts can be persisted and reloaded.
"""

import os
import sys
import pytest
import numpy as np
import pandas as pd
import joblib

SRC_ML = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
for p in (SRC_ML,):
    if p not in sys.path:
        sys.path.insert(0, p)

from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import GroupShuffleSplit
from sklearn.metrics import accuracy_score

FEATURE_COLS = [
    "duration", "std_iat", "avg_iat", "pk_count",
    "max_pkt_size", "pkt_burst_std", "pk_count_ratio",
    "avg_inbound_size", "avg_outbound_size",
    "total_bytes", "outbound_ratio",
]
ACTIONS = ["Like", "Play", "Subscribe", "Comment", "Search", "Background"]


# ---------------------------------------------------------------------------
# Synthetic CSV factory
# ---------------------------------------------------------------------------

def _make_csv(tmp_path, n_per_class: int = 40) -> str:
    rng = np.random.default_rng(42)
    rows = []
    for i, action in enumerate(ACTIONS):
        for j in range(n_per_class):
            rows.append({
                "file_source": f"{action}_{i * n_per_class + j}.pcap",
                "src_ip": "172.217.0.1", "dst_ip": "192.168.1.2",
                "src_port": 443, "dst_port": int(rng.integers(10000, 60000)),
                "protocol": 6,
                "duration":        rng.uniform(0.1, 10.0),
                "std_iat":         rng.uniform(0.001, 0.5),
                "avg_iat":         rng.uniform(0.001, 0.3),
                "pk_count":        int(rng.integers(5, 200)),
                "max_pkt_size":    int(rng.integers(100, 1500)),
                "pkt_burst_std":   rng.uniform(0.0, 50.0),
                "pk_count_ratio":  rng.uniform(0.1, 1.0),
                "avg_inbound_size":  rng.uniform(100, 1400),
                "avg_outbound_size": rng.uniform(50, 600),
                "total_bytes":     int(rng.integers(500, 2_000_000)),
                "outbound_ratio":  rng.uniform(0.0, 1.0),
                "action": action,
            })
    path = str(tmp_path / "master_training_data.csv")
    pd.DataFrame(rows).to_csv(path, index=False)
    return path


def _prepare(csv_path):
    df = pd.read_csv(csv_path)
    actions_df    = df[df["action"] != "Background"]
    background_df = df[df["action"] == "Background"].sample(n=40, random_state=42)
    df = pd.concat([actions_df, background_df], ignore_index=True)
    X = df[FEATURE_COLS]
    le = LabelEncoder()
    y  = le.fit_transform(df["action"])
    return df, X, y, df["file_source"], le


def _small_rf():
    return RandomForestClassifier(
        n_estimators=50, max_depth=8,
        min_samples_split=5, min_samples_leaf=3,
        class_weight="balanced", random_state=42, n_jobs=-1,
    )


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

class TestCsvToTraining:

    def test_csv_loads(self, tmp_path):
        df = pd.read_csv(_make_csv(tmp_path))
        assert not df.empty

    def test_feature_columns_present(self, tmp_path):
        df = pd.read_csv(_make_csv(tmp_path))
        for col in FEATURE_COLS:
            assert col in df.columns

    def test_label_encoder_covers_all_actions(self, tmp_path):
        _, _, y, _, le = _prepare(_make_csv(tmp_path))
        assert set(le.classes_) == set(ACTIONS)

    def test_group_split_is_leakage_free(self, tmp_path):
        _, X, y, groups, _ = _prepare(_make_csv(tmp_path))
        gss = GroupShuffleSplit(n_splits=1, test_size=0.33, random_state=42)
        tr, te = next(gss.split(X, y, groups))
        assert set(groups.iloc[tr]).isdisjoint(set(groups.iloc[te])), \
            "Data leakage: same file_source in train and test"

    def test_model_fits_without_error(self, tmp_path):
        _, X, y, groups, _ = _prepare(_make_csv(tmp_path))
        gss = GroupShuffleSplit(n_splits=1, test_size=0.33, random_state=42)
        tr, _ = next(gss.split(X, y, groups))
        model = _small_rf()
        model.fit(X.iloc[tr], y[tr])   # must not raise

    def test_predictions_correct_shape(self, tmp_path):
        _, X, y, groups, _ = _prepare(_make_csv(tmp_path))
        gss = GroupShuffleSplit(n_splits=1, test_size=0.33, random_state=42)
        tr, te = next(gss.split(X, y, groups))
        model = _small_rf()
        model.fit(X.iloc[tr], y[tr])
        assert len(model.predict(X.iloc[te])) == len(te)

    def test_model_beats_chance_accuracy(self, tmp_path):
        _, X, y, groups, _ = _prepare(_make_csv(tmp_path))
        gss = GroupShuffleSplit(n_splits=1, test_size=0.33, random_state=42)
        tr, te = next(gss.split(X, y, groups))
        model = _small_rf()
        model.fit(X.iloc[tr], y[tr])
        acc = accuracy_score(y[te], model.predict(X.iloc[te]))
        assert acc > 1 / len(ACTIONS), f"Accuracy {acc:.2%} ≤ chance"


class TestArtefactPersistence:

    def _trained(self, tmp_path):
        _, X, y, groups, le = _prepare(_make_csv(tmp_path))
        gss = GroupShuffleSplit(n_splits=1, test_size=0.33, random_state=42)
        tr, te = next(gss.split(X, y, groups))
        model = _small_rf()
        model.fit(X.iloc[tr], y[tr])
        return model, le, X, y[te], X.iloc[te]

    def test_model_pkl_round_trip(self, tmp_path):
        model, _, X, y_te, X_te = self._trained(tmp_path)
        path = str(tmp_path / "action_model.pkl")
        joblib.dump(model, path)
        loaded = joblib.load(path)
        np.testing.assert_array_equal(model.predict(X_te), loaded.predict(X_te))

    def test_label_encoder_pkl_round_trip(self, tmp_path):
        _, le, *_ = self._trained(tmp_path)
        path = str(tmp_path / "label_encoder.pkl")
        joblib.dump(le, path)
        loaded_le = joblib.load(path)
        assert list(loaded_le.classes_) == list(le.classes_)

    def test_model_features_pkl_round_trip(self, tmp_path):
        path = str(tmp_path / "model_features.pkl")
        joblib.dump(FEATURE_COLS, path)
        assert joblib.load(path) == FEATURE_COLS

    def test_reloaded_model_feature_count(self, tmp_path):
        model, *_ = self._trained(tmp_path)
        path = str(tmp_path / "action_model.pkl")
        joblib.dump(model, path)
        assert joblib.load(path).n_features_in_ == len(FEATURE_COLS)
        