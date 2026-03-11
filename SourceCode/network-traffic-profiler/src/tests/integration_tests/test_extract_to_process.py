# Integration tests: extract_features -> process_dataset

import pytest
import pandas as pd
import numpy as np
from unittest.mock import patch

from process_dataset import validate_dataset, validate_row, FIELD_SCHEMA
from test_helpers import make_valid_flow_df


# Returns a DataFrame that mirrors the real output of extract_flows()
def simulate_extract_flows(n: int = 60) -> pd.DataFrame:
    return make_valid_flow_df(n)


# Test 1 — Output columns of extract_flows() satisfy process_dataset schema
class TestColumnContract:

    def test_all_schema_columns_present(self):
        df = simulate_extract_flows()
        for col in FIELD_SCHEMA.keys():
            assert col in df.columns, (
                f"Column '{col}' required by process_dataset but missing "
                f"from extract_features output. Columns present: {list(df.columns)}"
            )

    def test_no_extra_columns_break_validation(self):
        df = simulate_extract_flows()
        result = validate_dataset(df)
        assert result is not None


# Test 2 — Valid extracted data passes validation without false positives
class TestValidDataPassesThrough:

    def test_all_rows_valid(self):
        df = simulate_extract_flows(60)
        result = validate_dataset(df)
        invalid = result[result["is_valid"] == False]
        assert len(invalid) == 0, (
            f"{len(invalid)} rows were unexpectedly flagged as invalid.\n"
            f"Reasons: {invalid['error_reason'].tolist()}"
        )

    def test_is_valid_and_error_reason_columns_added(self):
        df = simulate_extract_flows()
        result = validate_dataset(df)
        assert "is_valid" in result.columns
        assert "error_reason" in result.columns

    def test_row_count_preserved(self):
        df = simulate_extract_flows(60)
        result = validate_dataset(df)
        assert len(result) == 60


# Test 3 — Corrupted extract data is correctly flagged
class TestCorruptedExtractDataFlagged:

    def test_invalid_ip_flagged(self):
        # a bad IP address must be flagged as invalid, not silently accepted
        df = simulate_extract_flows(5)
        df.loc[0, "src_ip"] = "999.999.999.999"   # malformed IP
        result = validate_dataset(df)
        assert result.loc[0, "is_valid"] == False
        assert "src_ip" in result.loc[0, "error_reason"].lower() or \
               "invalid" in result.loc[0, "error_reason"].lower()

    def test_negative_packet_count_flagged(self):
        # packet_count must be >= 0
        df = simulate_extract_flows(5)
        df.loc[2, "packet_count"] = -1
        result = validate_dataset(df)
        assert result.loc[2, "is_valid"] == False

    def test_mismatched_protocol_name_flagged(self):
        # extract_flows() derives protocol_name from the protocol number
        # If the mapping is wrong (e.g. protocol=6 but name='UDP'), the cross-check in validate_row() must catch it 
        df = simulate_extract_flows(5)
        df.loc[1, "protocol_name"] = "UDP"   # protocol=6 should be TCP
        result = validate_dataset(df)
        assert result.loc[1, "is_valid"] == False
        assert "mismatch" in result.loc[1, "error_reason"].lower()

    def test_byte_count_less_than_packet_count_flagged(self):
        df = simulate_extract_flows(5)
        df.loc[3, "byte_count"] = 2        # 2 bytes across 5 packets
        df.loc[3, "packet_count"] = 5
        result = validate_dataset(df)
        assert result.loc[3, "is_valid"] == False


# Test 4 — Duplicate flows produced by extract_flows() are flagged
class TestDuplicateDetection:

    def test_duplicate_flows_flagged(self):
        """
        extract_flows() aggregates by 5-tuple; a genuine duplicate (same
        5-tuple AND same first_packet_index) should never appear in real
        data but validate_dataset() must catch it if it does.
        """
        df = simulate_extract_flows(5)
        # Create an exact duplicate of row 0
        dup = df.iloc[[0]].copy()
        df = pd.concat([df, dup], ignore_index=True)

        result = validate_dataset(df)
        # Both copies of the duplicated row must be flagged
        flagged = result[result["error_reason"].str.contains("Duplicate", na=False)]
        assert len(flagged) >= 2
