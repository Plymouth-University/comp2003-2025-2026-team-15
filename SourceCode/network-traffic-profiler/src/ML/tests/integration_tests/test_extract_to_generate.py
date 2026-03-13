"""
Integration Test: extract_ml_features → generate_dataset
  Checks that the flow features from extract_ml_features work with the label_flows() labeling in generate_dataset
  and that save_to_csv() saves the results correctly
"""

import os
import sys
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

    def test_play_small_flow_stays_background(self):
        df = _make_flow_df({"total_bytes": 100, "avg_inbound_size": 100.0})
        assert label_flows(df.copy(), "Play").loc[0, "action"] == "Background"

    # Search — API trigger branch
    def test_search_api_trigger_labelled_search(self):
        df = _make_flow_df({"total_bytes": 3_000, "outbound_ratio": 0.75})
        assert label_flows(df.copy(), "Search").loc[0, "action"] == "Search"

    # Search — thumbnail branch
    def test_search_thumbnail_flow_labelled_search(self):
        df = _make_flow_df({"total_bytes": 50_000, "outbound_ratio": 0.2})
        assert label_flows(df.copy(), "Search").loc[0, "action"] == "Search"

    def test_search_tiny_flow_stays_background(self):
        df = _make_flow_df({"total_bytes": 500, "outbound_ratio": 0.1})
        assert label_flows(df.copy(), "Search").loc[0, "action"] == "Background"

    # Like
    def test_like_qualifying_flow_labelled_like(self):
        df = _make_flow_df({"total_bytes": 8_000, "pk_count": 5, "outbound_ratio": 0.8})
        assert label_flows(df.copy(), "Like").loc[0, "action"] == "Like"

    def test_like_too_large_stays_background(self):
        df = _make_flow_df({"total_bytes": 20_000, "pk_count": 10, "outbound_ratio": 0.8})
        assert label_flows(df.copy(), "Like").loc[0, "action"] == "Background"

    # Subscribe
    def test_subscribe_qualifying_flow_labelled_subscribe(self):
        df = _make_flow_df({"total_bytes": 5_000, "pk_count": 4, "outbound_ratio": 0.9})
        assert label_flows(df.copy(), "Subscribe").loc[0, "action"] == "Subscribe"

    # Comment
    def test_comment_qualifying_flow_labelled_comment(self):
        df = _make_flow_df({"total_bytes": 20_000, "outbound_ratio": 0.65})
        assert label_flows(df.copy(), "Comment").loc[0, "action"] == "Comment"

    def test_comment_too_large_stays_background(self):
        df = _make_flow_df({"total_bytes": 50_000, "outbound_ratio": 0.65})
        assert label_flows(df.copy(), "Comment").loc[0, "action"] == "Background"


# ---------------------------------------------------------------------------
# 3. Multi-flow labelling
# ---------------------------------------------------------------------------

class TestMultiFlowLabelling:

    def test_only_matching_rows_are_labelled(self):
        df = _multi_flow_df([
            {"total_bytes": 8_000, "pk_count": 5, "outbound_ratio": 0.8},   # Like
            {"total_bytes": 20_000, "pk_count": 10, "outbound_ratio": 0.8}, # Background (too large)
            {"total_bytes": 8_000, "pk_count": 2, "outbound_ratio": 0.8},   # Background (too few pkts)
        ])
        result = label_flows(df.copy(), "Like")
        assert result.loc[0, "action"] == "Like"
        assert result.loc[1, "action"] == "Background"
        assert result.loc[2, "action"] == "Background"

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

    def test_background_flows_preserved(self, tmp_path):
        df = label_flows(_multi_flow_df([
            {"total_bytes": 8_000, "pk_count": 5, "outbound_ratio": 0.8},
            {"total_bytes": 1,     "pk_count": 1, "outbound_ratio": 0.1},
        ]), "Like")
        out = str(tmp_path / "out.csv")
        save_to_csv([df], output_path=out)
        loaded = pd.read_csv(out)
        assert "Like"       in loaded["action"].values
        assert "Background" in loaded["action"].values

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
 
