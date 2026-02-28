import pytest
import pandas as pd

LABEL_NAMES = {
    "src_ip": "Source Address",
    "dst_ip": "Destination Address",
    "src_port": "Source Port",
    "dst_port": "Destination Port",
    "protocol": "Protocol Number",
    "protocol_name": "Protocol Name",
    "packet_count": "Packet Count",
    "byte_count": "Byte Count",
    "avg_packet_size": "Avg Packet Size",
    "first_packet_index": "First Packet Index",
    "last_packet_index": "Last Packet Index",
    "duration": "Flow Duration (s)",
    "error_reason": "Flagged Reason",
    "anomaly": "Anomaly",
    "conversation": "Conversation Pair",
    "total_bytes": "Total Bytes",
    "total_packets": "Total Packets",
    "flow_count": "Flow Count",
    "action_type": "User Action Type",
}

def rename_cols(df):
    return df.rename(columns=LABEL_NAMES)


def make_flow_df(rows=None):
    # Build a realistic fake flows DataFrame for testing
    if rows is None:
        rows = [
            {
                "src_ip": "192.168.1.1", "dst_ip": "8.8.8.8",
                "src_port": 443, "dst_port": 55000,
                "protocol": 6, "protocol_name": "TCP",
                "packet_count": 10, "byte_count": 5000,
                "avg_packet_size": 500,
                "first_packet_index": 1, "last_packet_index": 10,
                "duration": 1.5, "action_type": "Like",
                "is_valid": True, "anomaly": False,
            },
            {
                "src_ip": "192.168.1.2", "dst_ip": "1.1.1.1",
                "src_port": 80, "dst_port": 60000,
                "protocol": 6, "protocol_name": "TCP",
                "packet_count": 5, "byte_count": 2000,
                "avg_packet_size": 400,
                "first_packet_index": 2, "last_packet_index": 7,
                "duration": 0.8, "action_type": "Background Traffic",
                "is_valid": True, "anomaly": True,
            },
            {
                "src_ip": "10.0.0.1", "dst_ip": "8.8.8.8",
                "src_port": 53, "dst_port": 53,
                "protocol": 17, "protocol_name": "UDP",
                "packet_count": 2, "byte_count": 300,
                "avg_packet_size": 150,
                "first_packet_index": 3, "last_packet_index": 4,
                "duration": 0.1, "action_type": None,
                "is_valid": False, "error_reason": "Missing fields",
                "anomaly": False,
            },
        ]
    return pd.DataFrame(rows)


# Section 1: tests for rename_cols helper
class TestRenameCols:

    def test_known_columns_are_renamed(self):
        df = pd.DataFrame(columns=["src_ip", "dst_ip", "byte_count"])
        renamed = rename_cols(df)
        assert "Source Address" in renamed.columns
        assert "Destination Address" in renamed.columns
        assert "Byte Count" in renamed.columns

    def test_unknown_columns_are_unchanged(self):
        df = pd.DataFrame(columns=["src_ip", "some_custom_column"])
        renamed = rename_cols(df)
        assert "some_custom_column" in renamed.columns

    def test_original_dataframe_not_mutated(self):
        df = pd.DataFrame(columns=["src_ip", "byte_count"])
        rename_cols(df)
        assert "src_ip" in df.columns

    def test_empty_dataframe_rename(self):
        df = pd.DataFrame()
        renamed = rename_cols(df)
        assert renamed.empty


# Section 2: tests for filtering out invalid rows
class TestIsValidFilter:

    def test_invalid_rows_are_removed(self):
        df = make_flow_df()
        valid_df = df[df["is_valid"] == True]
        assert len(valid_df) == 2

    def test_valid_rows_are_kept(self):
        df = make_flow_df()
        valid_df = df[df["is_valid"] == True]
        assert all(valid_df["is_valid"] == True)

    def test_all_valid_rows_unchanged(self):
        rows = [{"is_valid": True, "byte_count": 100}] * 3
        df = pd.DataFrame(rows)
        valid_df = df[df["is_valid"] == True]
        assert len(valid_df) == 3

    def test_all_invalid_returns_empty(self):
        rows = [{"is_valid": False, "byte_count": 100}] * 3
        df = pd.DataFrame(rows)
        valid_df = df[df["is_valid"] == True]
        assert valid_df.empty


# Section 3: tests that each filter correctly narrows down the DataFrame
class TestSidebarFilters:

    def setup_method(self):
        self.df = make_flow_df()
        self.df = self.df[self.df["is_valid"] == True]  # Start with valid only

    def test_protocol_filter(self):
        filtered = self.df[self.df["protocol_name"].isin(["TCP"])]
        assert all(filtered["protocol_name"] == "TCP")

    def test_src_ip_filter(self):
        filtered = self.df[self.df["src_ip"].isin(["192.168.1.1"])]
        assert len(filtered) == 1
        assert filtered.iloc[0]["src_ip"] == "192.168.1.1"

    def test_dst_ip_filter(self):
        filtered = self.df[self.df["dst_ip"].isin(["8.8.8.8"])]
        assert len(filtered) == 1

    def test_src_port_filter(self):
        filtered = self.df[self.df["src_port"].isin([443])]
        assert len(filtered) == 1
        assert filtered.iloc[0]["src_port"] == 443

    def test_dst_port_filter(self):
        filtered = self.df[self.df["dst_port"].isin([55000])]
        assert len(filtered) == 1

    def test_no_filter_returns_all_valid(self):
        # When no filter is selected, the DataFrame is unchanged
        filtered = self.df.copy()
        assert len(filtered) == 2

    def test_filter_with_no_matches_returns_empty(self):
        filtered = self.df[self.df["src_ip"].isin(["99.99.99.99"])]
        assert filtered.empty

    def test_combined_filters(self):
        filtered = self.df[
            self.df["protocol_name"].isin(["TCP"]) &
            self.df["src_ip"].isin(["192.168.1.1"])
        ]
        assert len(filtered) == 1


# Section 4: tests for the top endpoints groupby aggregation
class TestTopEndpoints:

    def test_groups_by_src_and_dst_ip(self):
        df = make_flow_df()
        df = df[df["is_valid"] == True]

        endpoints_df = (
            df.groupby(["src_ip", "dst_ip"])["byte_count"]
            .sum()
            .reset_index()
            .sort_values(by="byte_count", ascending=False)
        )

        assert "src_ip" in endpoints_df.columns
        assert "dst_ip" in endpoints_df.columns
        assert "byte_count" in endpoints_df.columns

    def test_sorted_descending_by_byte_count(self):
        df = make_flow_df()
        df = df[df["is_valid"] == True]

        endpoints_df = (
            df.groupby(["src_ip", "dst_ip"])["byte_count"]
            .sum()
            .reset_index()
            .sort_values(by="byte_count", ascending=False)
        )

        byte_counts = endpoints_df["byte_count"].tolist()
        assert byte_counts == sorted(byte_counts, reverse=True)

    def test_limited_to_20_rows(self):
        # Build 25 unique pairs
        rows = [
            {"src_ip": f"10.0.0.{i}", "dst_ip": "8.8.8.8",
             "byte_count": i * 100, "is_valid": True}
            for i in range(25)
        ]
        df = pd.DataFrame(rows)
        endpoints_df = (
            df.groupby(["src_ip", "dst_ip"])["byte_count"]
            .sum()
            .reset_index()
            .head(20)
        )
        assert len(endpoints_df) == 20


# Section 5: tests for the conversation pair groupby logic
class TestTopConversations:

    def test_conversation_key_is_sorted_tuple(self):
        # Conversation key should be the same regardless of src/dst direction
        df = make_flow_df()
        df["conversation"] = df.apply(
            lambda r: tuple(sorted([r["src_ip"], r["dst_ip"]])), axis=1
        )
        # 192.168.1.1 <-> 8.8.8.8 from row 0
        assert df.at[0, "conversation"] == ("192.168.1.1", "8.8.8.8")

    def test_bidirectional_flows_grouped_together(self):
        rows = [
            {"src_ip": "1.1.1.1", "dst_ip": "2.2.2.2", "byte_count": 100},
            {"src_ip": "2.2.2.2", "dst_ip": "1.1.1.1", "byte_count": 200},
        ]
        df = pd.DataFrame(rows)
        df["conversation"] = df.apply(
            lambda r: tuple(sorted([r["src_ip"], r["dst_ip"]])), axis=1
        )
        assert df.at[0, "conversation"] == df.at[1, "conversation"]

    def test_aggregation_sums_bytes(self):
        rows = [
            {"src_ip": "1.1.1.1", "dst_ip": "2.2.2.2",
             "byte_count": 100, "packet_count": 5, "avg_packet_size": 20},
            {"src_ip": "2.2.2.2", "dst_ip": "1.1.1.1",
             "byte_count": 200, "packet_count": 3, "avg_packet_size": 66},
        ]
        df = pd.DataFrame(rows)
        df["conversation"] = df.apply(
            lambda r: tuple(sorted([r["src_ip"], r["dst_ip"]])), axis=1
        )
        result = (
            df.groupby("conversation")
            .agg(total_bytes=("byte_count", "sum"))
            .reset_index()
        )
        assert result.at[0, "total_bytes"] == 300


# Section 6: tests for the anomaly detection display logic
class TestAnomalousFlows:

    def test_anomalous_rows_filtered_correctly(self):
        df = make_flow_df()
        anomalous = df[df["anomaly"] == True]
        assert len(anomalous) == 1
        assert anomalous.iloc[0]["src_ip"] == "192.168.1.2"

    def test_no_anomalies_returns_empty(self):
        rows = [{"anomaly": False, "byte_count": 100}] * 3
        df = pd.DataFrame(rows)
        anomalous = df[df["anomaly"] == True]
        assert anomalous.empty

    def test_anomaly_percentage_calculation(self):
        total = 100
        anomaly_count = 25
        pct = (anomaly_count / total) * 100
        assert pct == 25.0


# Section 7: tests for flows that failed validation
class TestFlaggedFlows:

    def test_flagged_flows_are_invalid_rows(self):
        df = make_flow_df()
        flagged = df[df["is_valid"] == False]
        assert len(flagged) == 1
        assert flagged.iloc[0]["error_reason"] == "Missing fields"

    def test_no_flagged_flows_returns_empty(self):
        rows = [{"is_valid": True, "byte_count": 100}] * 3
        df = pd.DataFrame(rows)
        flagged = df[df["is_valid"] == False]
        assert flagged.empty

    def test_error_reason_is_first_column_in_display(self):
        df = make_flow_df()
        flagged = df[df["is_valid"] == False].copy()
        flagged["error_reason"] = flagged.get("error_reason", "Unknown")

        cols = ["error_reason"] + [c for c in flagged.columns if c not in ("error_reason", "is_valid")]
        display = flagged[cols]

        assert display.columns[0] == "error_reason"


# Section 8: Pie chart data (Traffic Composition)
class TestPieChartData:

    def test_groups_by_action_type(self):
        df = make_flow_df()
        df = df[df["is_valid"] == True]
        pie_df = df.groupby("action_type", dropna=False)["byte_count"].sum().reset_index()
        assert "action_type" in pie_df.columns
        assert "byte_count" in pie_df.columns

    def test_byte_counts_summed_per_action(self):
        rows = [
            {"action_type": "Like", "byte_count": 1000, "is_valid": True},
            {"action_type": "Like", "byte_count": 2000, "is_valid": True},
            {"action_type": "Play", "byte_count": 5000, "is_valid": True},
        ]
        df = pd.DataFrame(rows)
        pie_df = df.groupby("action_type")["byte_count"].sum().reset_index()
        like_row = pie_df[pie_df["action_type"] == "Like"]
        assert like_row.iloc[0]["byte_count"] == 3000

    def test_empty_filtered_df_results_in_empty_pie_data(self):
        df = pd.DataFrame(columns=["action_type", "byte_count"])
        pie_df = df.groupby("action_type", dropna=False)["byte_count"].sum().reset_index()
        assert pie_df.empty
