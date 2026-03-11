# Integration tests: process_dataset  ->  build_dataset  

import pytest
import pandas as pd
import numpy as np

from process_dataset import validate_dataset
from build_dataset import build_dataset
from test_helpers import make_valid_flow_df


# Shared helpers
PCAP_BASENAME = "test_capture"


def validated(n: int = 60) -> pd.DataFrame:
    # returns a fully-validated DataFrame ready for build_dataset()
    return validate_dataset(make_valid_flow_df(n))


# Test 1 — build_dataset() only uses valid rows
class TestValidRowFiltering:

    def test_invalid_rows_excluded_from_ml(self):
        # build_dataset() must filter out rows where is_valid == False before building the numeric/scaled dataset
        df = make_valid_flow_df(60)
        # Corrupt 10 rows so validate_dataset marks them invalid
        df.loc[:9, "src_ip"] = "999.0.0.1"
        validated_df = validate_dataset(df)

        valid_count = int((validated_df["is_valid"] == True).sum())
        numeric_df, anomaly_info = build_dataset(validated_df, PCAP_BASENAME)

        assert anomaly_info["total_flows"] == valid_count, (
            f"build_dataset processed {anomaly_info['total_flows']} flows "
            f"but only {valid_count} were valid."
        )

    def test_all_valid_rows_used(self):
        # When every row is valid, total_flows should equal the DataFrame length
        vdf = validated(60)
        _, anomaly_info = build_dataset(vdf, PCAP_BASENAME)
        assert anomaly_info["total_flows"] == 60


# Test 2 — Return-value structure matches what dashboard.py expects
class TestReturnValueContract:

    def test_returns_two_values(self):
        # build_dataset() must return exactly (numeric_df, anomaly_info)
        result = build_dataset(validated(), PCAP_BASENAME)
        assert len(result) == 2

    def test_numeric_df_is_dataframe(self):
        numeric_df, _ = build_dataset(validated(), PCAP_BASENAME)
        assert isinstance(numeric_df, pd.DataFrame)

    def test_anomaly_info_keys_present(self):
        # dashboard.py reads anomaly_count, total_flows, anomaly_percentage.
        _, anomaly_info = build_dataset(validated(), PCAP_BASENAME)
        for key in ("anomaly_count", "total_flows", "anomaly_percentage"):
            assert key in anomaly_info, f"Missing key '{key}' in anomaly_info"

    def test_anomaly_column_in_numeric_df(self):
        # 'anomaly' boolean column must exist in numeric_df
        numeric_df, _ = build_dataset(validated(), PCAP_BASENAME)
        assert "anomaly" in numeric_df.columns

    def test_anomaly_column_is_boolean(self):
        numeric_df, _ = build_dataset(validated(), PCAP_BASENAME)
        assert numeric_df["anomaly"].dtype == bool

    def test_anomaly_percentage_calculation(self):
        # anomaly_percentage must equal anomaly_count / total_flows * 100.
        _, info = build_dataset(validated(), PCAP_BASENAME)
        expected = info["anomaly_count"] / info["total_flows"] * 100
        assert abs(info["anomaly_percentage"] - expected) < 1e-6


# Test 3 — Non-numeric columns from validate_dataset() are dropped
class TestColumnDropping:

    def test_src_dst_ip_not_in_numeric_df(self):
        numeric_df, _ = build_dataset(validated(), PCAP_BASENAME)
        assert "src_ip" not in numeric_df.columns
        assert "dst_ip" not in numeric_df.columns

    def test_is_valid_and_error_reason_dropped(self):
        numeric_df, _ = build_dataset(validated(), PCAP_BASENAME)
        assert "is_valid" not in numeric_df.columns
        assert "error_reason" not in numeric_df.columns


# Test 4 — Edge cases
class TestEdgeCases:

    def test_all_invalid_rows_raises_or_returns_gracefully(self):
        # build_dataset() should not crash with an unhandeled exception when receiving an empty dataframe after filtering
        df = make_valid_flow_df(5)
        df["src_ip"] = "999.0.0.1"   # force all rows invalid
        vdf = validate_dataset(df)
        try:
            numeric_df, anomaly_info = build_dataset(vdf, PCAP_BASENAME)
            assert anomaly_info["total_flows"] == 0
        except Exception as exc:
            assert isinstance(exc, (ValueError, IndexError, ZeroDivisionError))

    def test_nan_values_filled_before_scaling(self):
        # NaN values in numeric columns must be filled (with median) before IsolationForest is fitted
        df = make_valid_flow_df(60)
        vdf = validate_dataset(df)
        vdf.loc[5, "packet_count"] = np.nan
        # Should not raise
        numeric_df, anomaly_info = build_dataset(vdf, PCAP_BASENAME)
        assert anomaly_info["total_flows"] > 0
