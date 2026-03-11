import pytest
import pandas as pd
from unittest.mock import patch, MagicMock
from scapy.all import IP, TCP, UDP
# Assuming your merged function is in unified_extraction.py
from extract_features_unified import extract_all_pcap_data, is_google_youtube_ip

# --- Shared Helpers ---

# Helper function to build a fake scapy packet
def make_pkt(src, dst, sport, dport, proto, size, ts, index, layer="TCP"):
    pkt = MagicMock()
    pkt.time = ts
    pkt.__len__.return_value = size
    pkt.haslayer.side_effect = lambda x: x == layer or (x == "IP")
    
    # IP layer
    pkt.__contains__.side_effect = lambda x: x in ["IP", layer]
    pkt.__getitem__.side_effect = lambda x: {
        "IP": MagicMock(src=src, dst=dst, proto=proto),
        "TCP": MagicMock(sport=sport, dport=dport),
        "UDP": MagicMock(sport=sport, dport=dport)
    }.get(x)
    
    return pkt

# --- Unified Tests ---

def test_is_google_youtube_ip_logic():
    assert is_google_youtube_ip("172.217.1.1") is True
    assert is_google_youtube_ip("1.1.1.1") is False

@patch("extract_features_unified.os.path.exists", return_value=True)
@patch("extract_features_unified.rdpcap")
# Tests that one pass over packets correctly populates both 
# the standard dashboard flows and the ML features.
def test_unified_extraction_logic(mock_rdpcap, mock_exists):

    # Create mock traffic: 2 packets for a YouTube flow, 1 for a background flow
    yt_ip = "172.217.0.1"
    local_ip = "192.168.1.5"
    
    mock_rdpcap.return_value = [
        # Flow 1: YouTube
        make_pkt(local_ip, yt_ip, 12345, 443, 6, 100, 1000.0, 0),
        make_pkt(local_ip, yt_ip, 12345, 443, 6, 200, 1001.0, 1),
        # Flow 2: Background Traffic
        make_pkt(local_ip, "8.8.8.8", 54321, 53, 17, 50, 1002.0, 2, layer="UDP")
    ]

    df_flows, df_ml = extract_all_pcap_data("dummy.pcap", False)

    # Validate Standard Flow Extraction
    # Should produce 2 flows
    assert len(df_flows) == 2
    yt_row = df_flows[df_flows['dst_ip'] == yt_ip].iloc[0]
    assert yt_row['packet_count'] == 2
    assert yt_row['byte_count'] == 300
    assert yt_row['protocol_name'] == "TCP"

    # Validate ML Feature Extraction
    # Only the YouTube flow should be in the ML dataframe
    assert len(df_ml) == 1
    ml_row = df_ml.iloc[0]
    assert ml_row['src_ip'] == local_ip
    assert ml_row['pk_count'] == 2
    assert ml_row['duration'] == 1.0  # 1001-1000
    assert ml_row['avg_outbound_size'] == 150.0 # (100+200)/2

@patch("extract_features_unified.os.path.exists", return_value=True)
@patch("extract_features_unified.rdpcap")
# Verifies the ml_only parameter skips dashboard flow processing.
def test_ml_only_flag(mock_rdpcap, mock_exists):
    
    mock_rdpcap.return_value = [
        make_pkt("192.168.1.5", "172.217.0.1", 12345, 443, 6, 100, 1000.0, 0)
    ]

    df_flows, df_ml = extract_all_pcap_data("dummy.pcap", ml_only=True)

    assert df_flows.empty  # General extraction skipped
    assert not df_ml.empty # ML extraction performed