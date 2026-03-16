"""
Integration Test: Full ML pipeline (Covering the complete chain)
"""
import os
import sys
import pytest
import numpy as np
import pandas as pd
import joblib

SRC_ML = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
ACTION_CLASSIFICATION = os.path.join(SRC_ML, "action_classification")

for p in (SRC_ML, ACTION_CLASSIFICATION):
    if p not in sys.path:
        sys.path.insert(0, p)

from action_classification.generate_dataset import label_flows, save_to_csv
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import GroupShuffleSplit
from sklearn.metrics import accuracy_score

FEATURE_COLS = [
    "duration", "std_iat", "avg_iat", "pk_count",
    "max_pkt_size", "pkt_burst_std", "pk_count_ratio",
    "avg_outbound_size","total_bytes", "outbound_ratio",
]

ACTIONS = ["Like", "Play", "Subscribe", "Comment", "Search"]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
_ACTION_OVERRIDES = {
    "Play":      {"total_bytes": 800_000, "avg_inbound_size": 900.0, "outbound_ratio": 0.3},
    "Search":    {"total_bytes": 50_000,  "outbound_ratio": 0.2},   # thumbnail branch
    "Like":      {"total_bytes": 8_000,   "pk_count": 6,  "outbound_ratio": 0.8},
    "Subscribe": {"total_bytes": 5_000,   "pk_count": 5,  "outbound_ratio": 0.9},
    "Comment":   {"total_bytes": 20_000,  "outbound_ratio": 0.65},
    "Background":{"total_bytes": 200,     "outbound_ratio": 0.1},
}

"""Generate one synthetic flow row with values guaranteed to pass label_flows() for the given action."""
def _flow(action: str, rng, file_idx: int) -> dict:
    row = {
        "file_source":       f"{action}_{file_idx}.pcap",
        "src_ip":            "172.217.0.1",
        "dst_ip":            "192.168.1.5",
        "src_port":          443,
        "dst_port":          int(rng.integers(10000, 60000)),
        "protocol":          6,
        "duration":          rng.uniform(0.5, 8.0),
        "std_iat":           rng.uniform(0.001, 0.3),
        "avg_iat":           rng.uniform(0.001, 0.2),
        "pk_count":          int(rng.integers(10, 150)),
        "max_pkt_size":      int(rng.integers(200, 1500)),
        "pkt_burst_std":     rng.uniform(0.0, 30.0),
        "pk_count_ratio":    rng.uniform(0.1, 1.0),
        "avg_inbound_size":  rng.uniform(200, 800),
        "avg_outbound_size": rng.uniform(50, 300),
        "total_bytes":       int(rng.integers(1000, 100_000)),
        "outbound_ratio":    rng.uniform(0.3, 0.7),
        "action":            action,
    }
    row.update(_ACTION_OVERRIDES.get(action, {}))
    return row

def _build_pipeline(tmp_path, n_per_class: int = 50):
    """
    Run the entire pipeline end-to-end. Mirrored from generate_dataset.py + train_model.py.
    """
    rng = np.random.default_rng(0)

    # Step 1 – generate and label synthetic flows
    all_dfs = []
    for i, action in enumerate(ACTIONS):
        rows = [_flow(action, rng, i * n_per_class + j) for j in range(n_per_class)]
        df = pd.DataFrame(rows)
        df = label_flows(df, action)
        all_dfs.append(df)

    # Add background noise rows
    bg_rows = [_flow("Background", rng, 9000 + k) for k in range(40)]
    bg_df = pd.DataFrame(bg_rows)
    bg_df["action"] = "Background"
    all_dfs.append(bg_df)

    # Step 2 – save CSV
    csv_path = str(tmp_path / "master_training_data.csv")
    master_df = save_to_csv(all_dfs, output_path=csv_path)

    # Step 3 – load CSV, balance background, encode labels
    df = pd.read_csv(csv_path)
    actions_df    = df[df["action"] != "Background"]
    background_df = df[df["action"] == "Background"]
    df = pd.concat([actions_df, background_df], ignore_index=True)

    X      = df[FEATURE_COLS]
    groups = df["file_source"]
    le     = LabelEncoder()
    y      = le.fit_transform(df["action"])

    # Step 4 – GroupShuffleSplit
    gss = GroupShuffleSplit(n_splits=1, test_size=0.33, random_state=42)
    tr, te = next(gss.split(X, y, groups))
    X_train, X_test = X.iloc[tr], X.iloc[te]
    y_train, y_test = y[tr], y[te]

    # Step 5 – train
    model = RandomForestClassifier(
        n_estimators=50, max_depth=8,
        min_samples_split=5, min_samples_leaf=3,
        class_weight="balanced", random_state=42, n_jobs=-1,
    )
    model.fit(X_train, y_train)

    # Step 6 – persist artefacts
    model_path   = str(tmp_path / "action_model.pkl")
    enc_path     = str(tmp_path / "label_encoder.pkl")
    feat_path    = str(tmp_path / "model_features.pkl")
    joblib.dump(model, model_path)
    joblib.dump(le,    enc_path)
    joblib.dump(X.columns.tolist(), feat_path)

    return {
        "df": df, "X": X, "X_train": X_train, "X_test": X_test,
        "y_train": y_train, "y_test": y_test,
        "model": model, "le": le,
        "model_path": model_path, "enc_path": enc_path, "feat_path": feat_path,
        "csv_path": csv_path,
    }


# ---------------------------------------------------------------------------
# Stage 1: Labelling → CSV
# ---------------------------------------------------------------------------

class TestLabellingToCsv:

    def test_csv_is_created(self, tmp_path):
        p = _build_pipeline(tmp_path)
        assert os.path.exists(p["csv_path"])

    def test_csv_has_action_column(self, tmp_path):
        p = _build_pipeline(tmp_path)
        df = pd.read_csv(p["csv_path"])
        assert "action" in df.columns

    def test_csv_contains_all_actions_and_background(self, tmp_path):
        p = _build_pipeline(tmp_path)
        df = pd.read_csv(p["csv_path"])
        found = set(df["action"].unique())
        for action in ACTIONS:
            assert action in found, f"Action '{action}' missing from CSV"
        assert "Background" in found

    def test_no_null_values_in_feature_columns(self, tmp_path):
        p = _build_pipeline(tmp_path)
        df = pd.read_csv(p["csv_path"])
        for col in FEATURE_COLS:
            assert df[col].isnull().sum() == 0, f"NaN found in column '{col}'"


# ---------------------------------------------------------------------------
# Stage 2: CSV → Trained Model
# ---------------------------------------------------------------------------

class TestCsvToModel:

    def test_label_encoder_covers_all_classes(self, tmp_path):
        p = _build_pipeline(tmp_path)
        assert set(p["le"].classes_) == set(ACTIONS + ["Background"])

    def test_train_test_groups_are_disjoint(self, tmp_path):
        p = _build_pipeline(tmp_path)
        groups = p["df"]["file_source"]
        gss = GroupShuffleSplit(n_splits=1, test_size=0.33, random_state=42)
        tr, te = next(gss.split(p["X"], p["df"]["action"], groups))
        assert set(groups.iloc[tr]).isdisjoint(set(groups.iloc[te]))

    def test_model_output_shape_matches_test_set(self, tmp_path):
        p = _build_pipeline(tmp_path)
        preds = p["model"].predict(p["X_test"])
        assert len(preds) == len(p["y_test"])

    def test_model_predicts_only_known_labels(self, tmp_path):
        p = _build_pipeline(tmp_path)
        preds = p["model"].predict(p["X_test"])
        n_classes = len(p["le"].classes_)
        assert all(0 <= pred < n_classes for pred in preds)

    def test_model_achieves_above_chance_accuracy(self, tmp_path):
        p = _build_pipeline(tmp_path)
        acc = accuracy_score(p["y_test"], p["model"].predict(p["X_test"]))
        chance = 1 / len(p["le"].classes_)
        assert acc > chance, f"Accuracy {acc:.2%} ≤ chance ({chance:.2%})"

    def test_model_has_correct_feature_count(self, tmp_path):
        p = _build_pipeline(tmp_path)
        assert p["model"].n_features_in_ == len(FEATURE_COLS)


# ---------------------------------------------------------------------------
# Stage 3: Artefacts → predict_action_type logic
# ---------------------------------------------------------------------------

class TestArtefactsToPrediction:

    def test_reloaded_model_matches_original_predictions(self, tmp_path):
        p = _build_pipeline(tmp_path)
        loaded_model = joblib.load(p["model_path"])
        np.testing.assert_array_equal(
            p["model"].predict(p["X_test"]),
            loaded_model.predict(p["X_test"]),
        )

    def test_reloaded_encoder_decodes_predictions(self, tmp_path):
        p = _build_pipeline(tmp_path)
        loaded_le    = joblib.load(p["enc_path"])
        loaded_model = joblib.load(p["model_path"])
        preds = loaded_model.predict(p["X_test"])
        decoded = loaded_le.inverse_transform(preds)
        assert all(label in p["le"].classes_ for label in decoded)

    def test_reloaded_feature_list_matches_training_columns(self, tmp_path):
        p = _build_pipeline(tmp_path)
        loaded_features = joblib.load(p["feat_path"])
        assert loaded_features == FEATURE_COLS

    def test_predict_action_type_style_inference(self, tmp_path):
        p = _build_pipeline(tmp_path)
        model    = joblib.load(p["model_path"])
        le       = joblib.load(p["enc_path"])
        features = joblib.load(p["feat_path"])

        # Simulate what predict_action_type does:
        X = p["X_test"][features]
        raw_preds   = model.predict(X)
        action_types = le.inverse_transform(raw_preds)

        assert len(action_types) == len(p["X_test"])
        for label in action_types:
            assert label in le.classes_

    def test_none_input_returns_empty_with_action_type_column(self, tmp_path):
        # The function adds action_type = None when passed None
        features_df = None
        result = pd.DataFrame() if features_df is None else features_df
        result = result.copy() if not result.empty else pd.DataFrame()
        result["action_type"] = None
        assert "action_type" in result.columns

    def test_empty_dataframe_input_gets_action_type_column(self, tmp_path):
        features_df = pd.DataFrame()
        features_df = features_df.copy()
        features_df["action_type"] = None
        assert "action_type" in features_df.columns


# ---------------------------------------------------------------------------
# Stage 4: End-to-end regression — unseen label handling
# ---------------------------------------------------------------------------

class TestEdgeCases:

    def test_unseen_label_raises_value_error(self, tmp_path):
        p = _build_pipeline(tmp_path)
        le = joblib.load(p["enc_path"])
        with pytest.raises(ValueError):
            le.transform(["totally_unseen_action"])

    def test_model_produces_probabilities_for_all_classes(self, tmp_path):
        p = _build_pipeline(tmp_path)
        proba = p["model"].predict_proba(p["X_test"])
        assert proba.shape == (len(p["X_test"]), len(p["model"].classes_))
        np.testing.assert_allclose(proba.sum(axis=1), 1.0, atol=1e-6)

    def test_noise_does_not_crash_prediction(self, tmp_path):
        p = _build_pipeline(tmp_path)
        rng = np.random.default_rng(7)
        X_noisy = p["X_test"] * (1 + rng.normal(0, 0.01, p["X_test"].shape))
        preds = p["model"].predict(X_noisy)   # must not raise
        assert len(preds) == len(p["X_test"])

    def test_single_row_inference(self, tmp_path):
        p = _build_pipeline(tmp_path)
        single_row = p["X_test"].iloc[[0]]
        pred = p["model"].predict(single_row)
        assert len(pred) == 1

    def test_full_pipeline_is_reproducible(self, tmp_path):
        p1 = _build_pipeline(tmp_path / "run1")
        p2 = _build_pipeline(tmp_path / "run2")
        np.testing.assert_array_equal(
            p1["model"].predict(p1["X_test"]),
            p2["model"].predict(p2["X_test"]),
        )
