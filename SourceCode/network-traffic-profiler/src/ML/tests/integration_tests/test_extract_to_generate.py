"""
Integration Test: extract_ml_features → generate_dataset
  Checks that the flow features from extract_ml_features work with the label_flows() labeling in generate_dataset
  and that save_to_csv() saves the results correctly
"""

import os
import sys
from unittest import result
import pytest
import pandas as pd

# ---------------------------------------------------------------------------
# Path setup
# ---------------------------------------------------------------------------
SRC_ML = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
ACTION_CLASSIFICATION = os.path.join(SRC_ML, "action_classification")

for p in (SRC_ML, ACTION_CLASSIFICATION):
    if p not in sys.path:
        sys.path.insert(0, p)

from action_classification.generate_dataset import label_flows, save_to_csv


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_flow_df(overrides: dict) -> pd.DataFrame:
    """Return a single-row DataFrame with all columns produced by
    extract_ml_features, merged with caller-supplied field overrides."""
    base = {
        "src_ip": "172.217.0.1",
        "dst_ip": "192.168.1.5",
        "src_port": 443,
        "dst_port": 54321,
        "protocol": 6,
        "duration": 1.5,
        "std_iat": 0.02,
        "avg_iat": 0.05,
        "pk_count": 20,
        "avg_inbound_size": 800.0,
        "avg_outbound_size": 200.0,
        "total_bytes": 10_000,
        "outbound_ratio": 0.4,
    }
    base.update(overrides)
    return pd.DataFrame([base])


def _multi_flow_df(rows: list) -> pd.DataFrame:
    return pd.concat([_make_flow_df(r) for r in rows], ignore_index=True)


# ---------------------------------------------------------------------------
# 1. Schema compatibility
# ---------------------------------------------------------------------------

class TestSchemaCompatibility:
    """label_flows must accept the exact schema produced by extract_ml_features."""

    @pytest.mark.parametrize("action", ["Play", "Search", "Like", "Subscribe", "Comment"])
    def test_action_schema_accepted(self, action):
        df = _make_flow_df({})
        result = label_flows(df.copy(), action)
        assert "action" in result.columns


# ---------------------------------------------------------------------------
# 2. Labelling correctness
# ---------------------------------------------------------------------------

class TestLabellingCorrectness:

    # Play
    def test_play_large_flow_is_labelled_play(self):
        df = _make_flow_df({"total_bytes": 800_000, "avg_inbound_size": 900.0})
        assert label_flows(df.copy(), "Play").loc[0, "action"] == "Play"

    def test_play_small_flow_is_still_labelled_play(self):
        df = _make_flow_df({"total_bytes": 100, "avg_inbound_size": 100.0})
        assert label_flows(df.copy(), "Play").loc[0, "action"] == "Play"

    # Search — API trigger branch
    def test_search_api_trigger_labelled_search(self):
        df = _make_flow_df({"total_bytes": 3_000, "outbound_ratio": 0.75})
        assert label_flows(df.copy(), "Search").loc[0, "action"] == "Search"

    # Search — thumbnail branch
    def test_search_thumbnail_flow_labelled_search(self):
        df = _make_flow_df({"total_bytes": 50_000, "outbound_ratio": 0.2})
        assert label_flows(df.copy(), "Search").loc[0, "action"] == "Search"

    def test_search_low_outbound_falls_back_to_search(self):
        df = _make_flow_df({"total_bytes": 500, "outbound_ratio": 0.1})
        assert label_flows(df.copy(), "Search").loc[0, "action"] == "Search"

    # Like
    def test_like_qualifying_flow_labelled_like(self):
        df = _make_flow_df({"total_bytes": 8_000, "pk_count": 5, "outbound_ratio": 0.8})
        assert label_flows(df.copy(), "Like").loc[0, "action"] == "Like"

    def test_like_large_flow_still_labelled_like(self):
        # label_flows picks largest outbound-heavy row; no upper-bound cap exists
        df = _make_flow_df({"total_bytes": 20_000, "pk_count": 10, "outbound_ratio": 0.8})
        assert label_flows(df.copy(), "Like").loc[0, "action"] == "Like"

    # Subscribe
    def test_subscribe_qualifying_flow_labelled_subscribe(self):
        df = _make_flow_df({"total_bytes": 5_000, "pk_count": 4, "outbound_ratio": 0.9})
        assert label_flows(df.copy(), "Subscribe").loc[0, "action"] == "Subscribe"

    # Comment
    def test_comment_qualifying_flow_labelled_comment(self):
        df = _make_flow_df({"total_bytes": 20_000, "outbound_ratio": 0.65})
        assert label_flows(df.copy(), "Comment").loc[0, "action"] == "Comment"

    def test_comment_large_flow_still_labelled_comment(self):
        df = _make_flow_df({"total_bytes": 50_000, "outbound_ratio": 0.65})
        assert label_flows(df.copy(), "Comment").loc[0, "action"] == "Comment"

# ---------------------------------------------------------------------------
# 3. Multi-flow labelling
# ---------------------------------------------------------------------------

class TestMultiFlowLabelling:

    def test_only_one_row_is_labelled(self):
        # label_flows picks exactly one row (highest total_bytes with outbound_ratio > 0.4)
        # all other rows are dropped (Candidate), not kept as Background
        df = _multi_flow_df([
            {"total_bytes": 8_000,  "pk_count": 5,  "outbound_ratio": 0.8},
            {"total_bytes": 20_000, "pk_count": 10, "outbound_ratio": 0.8},
            {"total_bytes": 8_000,  "pk_count": 2,  "outbound_ratio": 0.8},
        ])
        result = label_flows(df.copy(), "Like").reset_index(drop=True)
        assert len(result) == 1
        assert result.loc[0, "action"] == "Like"

    def test_empty_dataframe_returns_empty(self):
        df = pd.DataFrame(columns=[
            "src_ip", "dst_ip", "src_port", "dst_port", "protocol",
            "duration", "std_iat", "avg_iat", "pk_count",
            "avg_inbound_size", "avg_outbound_size", "total_bytes", "outbound_ratio",
        ])
        result = label_flows(df.copy(), "Play")
        assert result.empty


# ---------------------------------------------------------------------------
# 4. save_to_csv round-trip
# ---------------------------------------------------------------------------

class TestSaveToCsvRoundTrip:

    def test_single_df_written_and_reloaded(self, tmp_path):
        df = label_flows(_make_flow_df({"total_bytes": 8_000, "pk_count": 5,
                                        "outbound_ratio": 0.8}), "Like")
        out = str(tmp_path / "out.csv")
        save_to_csv([df], output_path=out)
        loaded = pd.read_csv(out)
        assert len(loaded) == 1
        assert loaded.loc[0, "action"] == "Like"

    def test_multiple_dfs_concatenated(self, tmp_path):
        dfs = [label_flows(_make_flow_df({"total_bytes": 8_000, "pk_count": 5,
                                          "outbound_ratio": 0.8}), "Like")
               for _ in range(3)]
        out = str(tmp_path / "out.csv")
        save_to_csv(dfs, output_path=out)
        assert len(pd.read_csv(out)) == 3

    def test_only_winning_row_written_to_csv(self, tmp_path):
        # label_flows keeps only the one labelled row; non-matching rows are dropped
        df = label_flows(_multi_flow_df([
            {"total_bytes": 8_000, "pk_count": 5, "outbound_ratio": 0.8},
            {"total_bytes": 1,     "pk_count": 1, "outbound_ratio": 0.1},
        ]), "Like")
        out = str(tmp_path / "out.csv")
        save_to_csv([df], output_path=out)
        loaded = pd.read_csv(out)
        assert "Like" in loaded["action"].values
        assert len(loaded) == 1

    def test_returns_dataframe(self, tmp_path):
        out = str(tmp_path / "out.csv")
        result = save_to_csv([_make_flow_df({})], output_path=out)
        assert isinstance(result, pd.DataFrame)

    def test_creates_nested_directory(self, tmp_path):
        out = str(tmp_path / "a" / "b" / "out.csv")
        save_to_csv([_make_flow_df({})], output_path=out)
        assert os.path.exists(out)

    def test_all_feature_columns_present_in_csv(self, tmp_path):
        out = str(tmp_path / "out.csv")
        save_to_csv([_make_flow_df({})], output_path=out)
        loaded_cols = set(pd.read_csv(out).columns)
        expected = {"src_ip", "dst_ip", "src_port", "dst_port", "protocol",
                    "duration", "std_iat", "avg_iat", "pk_count",
                    "avg_inbound_size", "avg_outbound_size",
                    "total_bytes", "outbound_ratio"}
        assert expected.issubset(loaded_cols)

 
