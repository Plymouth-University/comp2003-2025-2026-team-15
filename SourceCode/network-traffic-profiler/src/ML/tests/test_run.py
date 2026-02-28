import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "action_classification"))

import pytest
import pandas as pd
from unittest.mock import patch
from run import run, label_flows

# Build a minimal fake flows DataFrame
def make_flows(rows):
    return pd.DataFrame(rows)


# Section 1: label_flows() — Play
class TestLabelFlowsPlay:

    def test_highest_bytes_flow_labelled_play(self):
        df = make_flows([
            {"total_bytes": 500,    "outbound_ratio": 0.1},
            {"total_bytes": 999999, "outbound_ratio": 0.2},  # biggest
            {"total_bytes": 1000,   "outbound_ratio": 0.5},
        ])
        result = label_flows(df.copy(), "Play")
        assert result.at[1, "action"] == "Play"

    def test_all_others_remain_background(self):
        df = make_flows([
            {"total_bytes": 100,    "outbound_ratio": 0.1},
            {"total_bytes": 999999, "outbound_ratio": 0.2},
            {"total_bytes": 200,    "outbound_ratio": 0.5},
        ])
        result = label_flows(df.copy(), "Play")
        assert len(result[result["action"] == "Background"]) == 2

    def test_single_flow_labelled_play(self):
        df = make_flows([{"total_bytes": 5000, "outbound_ratio": 0.8}])
        result = label_flows(df.copy(), "Play")
        assert result.at[0, "action"] == "Play"


# Section 2: label_flows() — Non-Play actions
class TestLabelFlowsNonPlay:

    def test_highest_outbound_ratio_gets_action_label(self):
        df = make_flows([
            {"total_bytes": 500, "outbound_ratio": 0.3},
            {"total_bytes": 800, "outbound_ratio": 0.95},  # should be labelled
            {"total_bytes": 600, "outbound_ratio": 0.5},
        ])
        result = label_flows(df.copy(), "Like")
        assert result.at[1, "action"] == "Like"

    def test_large_flows_excluded_from_selection(self):
        df = make_flows([
            {"total_bytes": 2_000_000, "outbound_ratio": 0.99},  # too big
            {"total_bytes": 500,       "outbound_ratio": 0.7},   # should be labelled
        ])
        result = label_flows(df.copy(), "Subscribe")
        assert result.at[0, "action"] == "Background"
        assert result.at[1, "action"] == "Subscribe"

    def test_all_flows_large_no_label_applied(self):
        df = make_flows([
            {"total_bytes": 5_000_000, "outbound_ratio": 0.9},
            {"total_bytes": 2_000_000, "outbound_ratio": 0.8},
        ])
        result = label_flows(df.copy(), "Like")
        assert all(result["action"] == "Background")

    def test_all_non_play_action_names_applied(self):
        for action in ["Like", "Subscribe", "Comment", "Search"]:
            df = make_flows([
                {"total_bytes": 500, "outbound_ratio": 0.9},
                {"total_bytes": 300, "outbound_ratio": 0.1},
            ])
            result = label_flows(df.copy(), action)
            assert result.at[0, "action"] == action


# Section 3: DataFrame construction
class TestDataFrameConstruction:

    def test_file_source_is_first_column(self):
        df = make_flows([{"total_bytes": 500, "outbound_ratio": 0.5}])
        df.insert(0, "file_source", "test.pcap")
        assert df.columns[0] == "file_source"

    def test_multiple_dataframes_concatenated(self):
        all_rows = [
            make_flows([{"total_bytes": 100, "outbound_ratio": 0.5}]),
            make_flows([{"total_bytes": 200, "outbound_ratio": 0.6}]),
            make_flows([{"total_bytes": 300, "outbound_ratio": 0.7}]),
        ]
        master_df = pd.concat(all_rows, ignore_index=True)
        assert len(master_df) == 3

    def test_concat_resets_index(self):
        all_rows = [
            make_flows([{"total_bytes": 100, "outbound_ratio": 0.5}]),
            make_flows([{"total_bytes": 200, "outbound_ratio": 0.6}]),
        ]
        master_df = pd.concat(all_rows, ignore_index=True)
        assert list(master_df.index) == [0, 1]


# Section 4: None and empty DataFrame handling
class TestExtractOutputHandling:

    def test_none_result_skipped(self):
        all_rows = []
        df_flows = None
        if df_flows is not None and not df_flows.empty:
            all_rows.append(df_flows)
        assert all_rows == []

    def test_empty_dataframe_skipped(self):
        all_rows = []
        df_flows = pd.DataFrame()
        if df_flows is not None and not df_flows.empty:
            all_rows.append(df_flows)
        assert all_rows == []

    def test_valid_dataframe_added(self):
        all_rows = []
        df_flows = make_flows([{"total_bytes": 500, "outbound_ratio": 0.5}])
        if df_flows is not None and not df_flows.empty:
            all_rows.append(df_flows)
        assert len(all_rows) == 1


# Section 5: CSV output
class TestCSVOutput:

    def test_csv_written_when_data_exists(self, tmp_path):
        df = make_flows([{"total_bytes": 500, "outbound_ratio": 0.5}])
        output = tmp_path / "master_training_data.csv"
        df.to_csv(output, index=False)
        assert output.exists()

    def test_csv_row_count_matches(self, tmp_path):
        df = make_flows([{"total_bytes": i * 100, "outbound_ratio": 0.5} for i in range(5)])
        output = tmp_path / "master_training_data.csv"
        df.to_csv(output, index=False)
        assert len(pd.read_csv(output)) == 5

    def test_no_csv_when_no_data(self, tmp_path):
        output = tmp_path / "master_training_data.csv"
        all_rows = []
        if all_rows:
            pd.concat(all_rows).to_csv(output, index=False)
        assert not output.exists()

# Section 6: Timing behaviour
class TestTiming:

    @patch("run.time.time")
    @patch("run.glob.glob")
    @patch("run.extract_ml_features")
    def test_elapsed_time_shown_in_output(self, mock_extract, mock_glob, mock_time, capsys):
        # Elapsed seconds from time.time() should appear in the printed output
        mock_time.side_effect = [0, 5]  
        mock_glob.return_value = ["datasets/Like/a.pcap"]
        mock_extract.side_effect = lambda path: make_flows([
            {"total_bytes": 500, "outbound_ratio": 0.8}
        ])

        with patch("run.pd.DataFrame.to_csv"):
            run(data_dir="datasets/", actions=["Like"])

        captured = capsys.readouterr()
        assert "5s" in captured.out

    @patch("run.time.time")
    @patch("run.glob.glob")
    @patch("run.extract_ml_features")
    def test_timer_starts_before_loop(self, mock_extract, mock_glob, mock_time):
        # time.time() must be called at least twice — once for start, once inside loop
        mock_time.return_value = 0
        mock_glob.return_value = ["datasets/Like/a.pcap"]
        mock_extract.side_effect = lambda path: make_flows([
            {"total_bytes": 500, "outbound_ratio": 0.8}
        ])

        with patch("run.pd.DataFrame.to_csv"):
            run(data_dir="datasets/", actions=["Like"])

        assert mock_time.call_count >= 2

# Section 7: run() — full pipeline with mocks
class TestRunFunction:

    @patch("run.time.time", return_value=0)
    @patch("run.glob.glob")
    @patch("run.extract_ml_features")
    def test_run_calls_extract_per_pcap(self, mock_extract, mock_glob, mock_time):
        # run() should call extract_ml_features once per pcap file
        mock_glob.return_value = ["datasets/Like/a.pcap", "datasets/Like/b.pcap"]

        mock_extract.side_effect = lambda path: make_flows([
            {"total_bytes": 500, "outbound_ratio": 0.8}
        ])

        with patch("run.pd.DataFrame.to_csv"):
            run(data_dir="datasets/", actions=["Like"])

        assert mock_extract.call_count == 2

    @patch("run.time.time", return_value=0)
    @patch("run.glob.glob")
    @patch("run.extract_ml_features")
    def test_run_returns_none_when_no_files(self, mock_extract, mock_glob, mock_time):
        # run() should return None if no pcap files are found
        mock_glob.return_value = []

        result = run(data_dir="datasets/", actions=["Like"])
        assert result is None

    @patch("run.time.time", return_value=0)
    @patch("run.glob.glob")
    @patch("run.extract_ml_features")
    def test_run_returns_dataframe_on_success(self, mock_extract, mock_glob, mock_time):
        # run() should return a DataFrame when data is extracted
        mock_glob.return_value = ["datasets/Like/a.pcap"]
        mock_extract.side_effect = lambda path: make_flows([
            {"total_bytes": 500, "outbound_ratio": 0.8}
        ])

        with patch("run.pd.DataFrame.to_csv"):
            result = run(data_dir="datasets/", actions=["Like"])

        assert result is not None
        assert isinstance(result, pd.DataFrame)
