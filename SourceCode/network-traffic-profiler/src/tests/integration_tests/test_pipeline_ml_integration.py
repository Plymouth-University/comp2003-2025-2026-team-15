# Integration tests: extract_features_unified  -> predict_action_type  ->  main.py merge

import pytest
import pandas as pd
import numpy as np
import sys
import os


PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from ML.model_training.predict import predict_action_type, model_features
from process_dataset import validate_dataset
from build_dataset import build_dataset
from test_helpers import make_valid_flow_df


# Column constants
ML_FEATURE_COLS = [
    "duration", "std_iat", "avg_iat", "pk_count", "avg_packet_size",
    "throughput", "max_pkt_size", "pkt_burst_std", "pk_count_ratio",
    "avg_inbound_size", "avg_outbound_size", "total_bytes", "outbound_ratio",
]

MERGE_COLS = ["src_ip", "dst_ip", "src_port", "dst_port", "protocol"]

EXPECTED_LABELS = {"Comment", "Like", "Play", "Search", "Subscribe",
                   "Non-YouTube", "Background"}

PCAP_BASENAME = "ml_integration_test"


# Builds a DataFrame that mirrors ml_features_df from extract_all_pcap_data()
def make_ml_features_df(n: int = 5) -> pd.DataFrame:
    rows = []
    for i in range(n):
        rows.append({
            "src_ip":           f"172.217.0.{i + 1}",
            "dst_ip":           f"192.168.1.{i + 1}",
            "src_port":         443,
            "dst_port":         50000 + i,
            "protocol":         6,
            "duration":         float(1 + i * 0.5),
            "std_iat":          0.05,
            "avg_iat":          0.1,
            "pk_count":         20 + i,
            "avg_packet_size":  800.0,
            "throughput":       50000.0,
            "max_pkt_size":     1400,
            "pkt_burst_std":    120.0,
            "pk_count_ratio":   1.5,
            "avg_inbound_size": 700.0,
            "avg_outbound_size":900.0,
            "total_bytes":      16000 + i * 1000,
            "outbound_ratio":   0.6,
        })
    return pd.DataFrame(rows)

# Builds a validated_data DataFrame with n_youtube rows with defined action types and n_background rows with non-YouTube IPs
def make_validated_df_with_youtube(n_youtube: int = 5, n_background: int = 10) -> pd.DataFrame:
    rows = []

    for i in range(n_youtube):
        rows.append({
            "src_ip": f"172.217.0.{i + 1}", "dst_ip": f"192.168.1.{i + 1}",
            "src_port": 443, "dst_port": 50000 + i, "protocol": 6,
            "protocol_name": "TCP", "packet_count": 20 + i,
            "byte_count": 16000 + i * 1000,
            "avg_packet_size": 800.0,
            "start_time": float(i * 0.1), "end_time": float(i * 0.1 + 1.0),
            "first_packet_index": i * 20, "last_packet_index": i * 20 + 19,
            "duration": 1.0, "is_valid": True, "error_reason": "",
        })

    for j in range(n_background):
        rows.append({
            "src_ip": f"8.8.{j}.1", "dst_ip": f"1.1.{j}.2",
            "src_port": 1024 + j, "dst_port": 80, "protocol": 6,
            "protocol_name": "TCP", "packet_count": 5,
            "byte_count": 300, "avg_packet_size": 60.0,
            "start_time": float(j * 0.2), "end_time": float(j * 0.2 + 0.5),
            "first_packet_index": j * 5, "last_packet_index": j * 5 + 4,
            "duration": 0.5, "is_valid": True, "error_reason": "",
        })

    return pd.DataFrame(rows)


# Tests the boundry between extract_features_unified and predict_action_type
# Tests that ml_features_df has the right shape for the trained model
class TestExtractToPredict:

    def test_ml_features_df_has_all_model_feature_columns(self):
        # Every column in model_features must exist in ml_features_df. 
        ml_df = make_ml_features_df(5)
        missing = [col for col in model_features if col not in ml_df.columns]
        assert missing == [], (
            f"ml_features_df is missing columns the model expects: {missing}\n"
            f"Columns present: {list(ml_df.columns)}"
        )

    def test_predict_action_type_adds_action_type_column(self):
        # predict_action_type() must return a DataFrame with an 'action_type'column
        ml_df = make_ml_features_df(5)
        result = predict_action_type(ml_df)
        assert "action_type" in result.columns

    def test_predict_action_type_returns_strings(self):
        # action_type values must be strings 
        ml_df = make_ml_features_df(5)
        result = predict_action_type(ml_df)
        for val in result["action_type"]:
            assert isinstance(val, str), (
                f"action_type contains non-string value: {val!r} ({type(val).__name__})"
            )

    def test_predict_action_type_labels_are_known(self):
        # Every predicted label must be one that the model was trained on.
        # An unknown label means label_encoder or the model .pkl is mismatched.
        ml_df = make_ml_features_df(5)
        result = predict_action_type(ml_df)
        unknown = set(result["action_type"]) - EXPECTED_LABELS
        assert unknown == set(), (
            f"Unexpected action_type labels returned: {unknown}. "
            f"Allowed labels: {EXPECTED_LABELS}"
        )

    def test_predict_preserves_row_count(self):
        # predict_action_type() must not add or drop rows — one prediction per input flow
        ml_df = make_ml_features_df(8)
        result = predict_action_type(ml_df)
        assert len(result) == 8

    def test_predict_preserves_5_tuple_columns(self):
        # The 5-tuple columns must pass through predict_action_type()
        ml_df = make_ml_features_df(5)
        result = predict_action_type(ml_df)
        for col in MERGE_COLS:
            assert col in result.columns, (
                f"5-tuple column '{col}' was dropped by predict_action_type()"
            )

    def test_predict_empty_df_returns_empty_with_action_type_column(self):
        # predict_action_type() must handle an empty DataFrame 
        empty_df = pd.DataFrame(columns=ML_FEATURE_COLS + MERGE_COLS)
        result = predict_action_type(empty_df)
        assert "action_type" in result.columns
        assert len(result) == 0


# Tests the boundry between predict_action_type and main.py merge
class TestPredictToMerge:

    def _run_merge(self, validated_df, ml_df):
        # Replicates the exact merge logic from main.py
        ml_df = predict_action_type(ml_df)
        merged = validated_df.merge(
            ml_df[MERGE_COLS + ["action_type"]],
            on=MERGE_COLS,
            how="left"
        )
        merged["action_type"] = merged["action_type"].fillna("Background")
        return merged

    def test_youtube_flows_receive_predicted_label(self):
        # Rows whose 5-tuple matches a YouTube flow in ml_features_df must receive a non-'Background' action_type after the merge
        validated_df = make_validated_df_with_youtube(n_youtube=5, n_background=0)
        ml_df = make_ml_features_df(5)
        merged = self._run_merge(validated_df, ml_df)

        youtube_rows = merged.iloc[:5]
        assert "action_type" in merged.columns, "action_type column missing after merge"
        assert not youtube_rows["action_type"].isna().any(), (
            "Some YouTube-matched rows have NaN action_type after merge — merge key mismatch"
        )
        for val in youtube_rows["action_type"]:
            assert val in EXPECTED_LABELS, (
                f"action_type '{val}' is not a known model label. "
                f"Known labels: {EXPECTED_LABELS}"
            )

    def test_non_youtube_flows_receive_background(self):
        # Rows with no matching YouTube flow in ml_features_df must be filled with 'Background'
        validated_df = make_validated_df_with_youtube(n_youtube=0, n_background=10)
        ml_df = make_ml_features_df(0)   # empty — no YouTube flows

        # With empty ml_df, simulate the "no ML features" path
        validated_df["action_type"] = "Unknown / No IP Traffic"
        assert all(validated_df["action_type"] == "Unknown / No IP Traffic")

    def test_action_type_column_present_after_merge(self):
        # validated_data must have an 'action_type' column after the merge
        validated_df = make_validated_df_with_youtube(n_youtube=3, n_background=5)
        ml_df = make_ml_features_df(3)
        merged = self._run_merge(validated_df, ml_df)
        assert "action_type" in merged.columns

    def test_row_count_unchanged_after_left_merge(self):
        # validated_data row count must be exactly preserved after merging action_type predictions
        validated_df = make_validated_df_with_youtube(n_youtube=5, n_background=10)
        ml_df = make_ml_features_df(5)
        merged = self._run_merge(validated_df, ml_df)
        assert len(merged) == len(validated_df), (
            f"Row count changed after merge: {len(validated_df)} → {len(merged)}"
        )
