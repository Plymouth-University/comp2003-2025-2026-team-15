import pytest
import pandas as pd
import numpy as np


# builds a valid DataFrame that mimics the output of extract_features.extract_flows()
def make_valid_flow_df(n: int = 60) -> pd.DataFrame:
    rows = []
    for i in range(n):
        packet_count = 5
        byte_count = 300           
        avg_packet_size = byte_count / packet_count
        rows.append({
            "src_ip":             f"8.8.{i % 256}.1",
            "dst_ip":             f"1.1.{i % 256}.2",
            "src_port":           1024 + i,
            "dst_port":           80,
            "protocol":           6,
            "protocol_name":      "TCP",
            "packet_count":       packet_count,
            "byte_count":         byte_count,
            "avg_packet_size":    float(avg_packet_size),
            "start_time":         float(i * 0.1),
            "end_time":           float(i * 0.1 + 0.5),
            "first_packet_index": i * 5,
            "last_packet_index":  i * 5 + 4,
            "duration":           0.5,
        })
    return pd.DataFrame(rows)


# a pre-built `valid_flow_df` fixture used by most tests
@pytest.fixture
def valid_flow_df():
    # 60-row DataFrame that passes all validation rules.
    return make_valid_flow_df(60)
