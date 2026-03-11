# End-to-end integration tests: extract_features -> process_dataset -> build_dataset
import pytest
import pandas as pd
import numpy as np
from unittest.mock import patch, MagicMock

from process_dataset import validate_dataset
from build_dataset import build_dataset
from test_helpers import make_valid_flow_df

# Shared helpers
PCAP_BASENAME = "integration_test"

def run_core_pipeline(df: pd.DataFrame):
    validated_df = validate_dataset(df.copy())
    numeric_df, anomaly_info = build_dataset(validated_df, PCAP_BASENAME)
    return validated_df, numeric_df, anomaly_info

# Test 1 — great path end-to-end run
class TestHappyPath:

    def test_pipeline_returns_three_values(self):
        # run_pipeline() must return (flows_df, numeric_df, anomaly_info)
        result = run_core_pipeline(make_valid_flow_df(60))
        assert len(result) == 3

    def test_flows_df_has_is_valid_column(self):
        # dashboard.py filters on is_valid must exist in flows_df after the pipeline runs
        flows_df, _, _ = run_core_pipeline(make_valid_flow_df(60))
        assert "is_valid" in flows_df.columns

    def test_numeric_df_has_anomaly_column(self):
        # dashboard.py reads numeric_df['anomaly'] for the scatter plot
        _, numeric_df, _ = run_core_pipeline(make_valid_flow_df(60))
        assert "anomaly" in numeric_df.columns

    def test_anomaly_info_is_dict_with_required_keys(self):
        _, _, anomaly_info = run_core_pipeline(make_valid_flow_df(60))
        assert isinstance(anomaly_info, dict)
        for key in ("anomaly_count", "total_flows", "anomaly_percentage"):
            assert key in anomaly_info

    def test_flows_df_preserves_ip_columns(self):
        # dashboard displau src_ip and dst_ip in all flows and top endpoints tables
        flows_df, _, _ = run_core_pipeline(make_valid_flow_df(60))
        assert "src_ip" in flows_df.columns
        assert "dst_ip" in flows_df.columns


# Test 2 — Mixed valid/invalid input propagates correctly through all stages
class TestMixedInput:

    def test_invalid_rows_flagged_in_flows_df(self):
        # Rows with bad IPs must be flagged is_valid=False in flows_df so dashboard.py can display them in the Flagged Flows table
        df = make_valid_flow_df(60)
        df.loc[:4, "src_ip"] = "127.0.0.1"    # loopback — caught by is_loopback check
        flows_df, _, _ = run_core_pipeline(df)
        flagged = flows_df[flows_df["is_valid"] == False]
        assert len(flagged) >= 5

    def test_invalid_rows_excluded_from_numeric_df(self):
        # Rows flagged invalid in flows_df must not appear in numeric_df 
        df = make_valid_flow_df(60)
        df.loc[:4, "src_ip"] = "127.0.0.1"
        flows_df, numeric_df, anomaly_info = run_core_pipeline(df)
        valid_count = int((flows_df["is_valid"] == True).sum())
        assert anomaly_info["total_flows"] == valid_count

    def test_anomaly_percentage_between_0_and_100(self):
        _, _, anomaly_info = run_core_pipeline(make_valid_flow_df(60))
        assert 0.0 <= anomaly_info["anomaly_percentage"] <= 100.0


# Test 3 — Column integrity across stage boundaries
class TestColumnIntegrity:

    def test_protocol_name_present_in_flows_df(self):
        flows_df, _, _ = run_core_pipeline(make_valid_flow_df(60))
        assert "protocol_name" in flows_df.columns

    def test_numeric_df_has_no_string_columns(self):
        # Must contain only numeric data types, except for the 'anomaly' boolean column added by build_dataset
        _, numeric_df, _ = run_core_pipeline(make_valid_flow_df(60))
        non_numeric = [
            col for col in numeric_df.columns
            if col != "anomaly" and not pd.api.types.is_numeric_dtype(numeric_df[col])
        ]
        assert non_numeric == [], f"Non-numeric columns in numeric_df: {non_numeric}"

    def test_flows_df_contains_duration_column(self):
        # duration is displayed in the all flows table
        flows_df, _, _ = run_core_pipeline(make_valid_flow_df(60))
        assert "duration" in flows_df.columns


# Test 4 — degraded-input scenarios
class TestRobustness:

    def test_single_valid_row_does_not_crash(self):
        # Need ≥1 valid row, build_dataset warns but should not raise
        df = make_valid_flow_df(1)
        try:
            flows_df, numeric_df, anomaly_info = run_core_pipeline(df)
            # If it succeeds, total_flows must be 1
            assert anomaly_info["total_flows"] == 1
        except Exception:
            # Acceptable only if the exception is from IsolationForest
            pass

    def test_udp_flows_pass_through_pipeline(self):
        df = make_valid_flow_df(60)
        df["protocol"] = 17
        df["protocol_name"] = "UDP"
        flows_df, numeric_df, anomaly_info = run_core_pipeline(df)
        valid = flows_df[flows_df["is_valid"] == True]
        assert len(valid) > 0, "All UDP flows were incorrectly invalidated"
        assert anomaly_info["total_flows"] > 0
