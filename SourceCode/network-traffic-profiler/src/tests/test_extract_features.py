import pandas as pd
from unittest.mock import patch, MagicMock
from extract_features import extract_flows

# Helper function to build a fake scapy packet
def make_pkt(src, dst, sport, dport, proto, size, ts, index, layer="TCP"):
    pkt = MagicMock()
    pkt.time = ts
    pkt.__len__.return_value = size

    # IP layer
    pkt.__contains__.side_effect = lambda x: x == "IP"
    pkt.__getitem__.side_effect = lambda x: {
        "IP": MagicMock(src=src, dst=dst, proto=proto),
        "TCP": MagicMock(sport=sport, dport=dport),
        "UDP": MagicMock(sport=sport, dport=dport)
    }[x]

    pkt.haslayer.return_value = (layer in ["TCP", "UDP"])
    return pkt


@patch("extract_features.os.path.exists", return_value=True)
@patch("extract_features.rdpcap")
def test_extract_flows_basic(mock_rdpcap, mock_exists):
    # Fake packets
    mock_rdpcap.return_value = [
        make_pkt("1.1.1.1", "2.2.2.2", 1234, 80, 6, 60, 1.0, 0),
        make_pkt("1.1.1.1", "2.2.2.2", 1234, 80, 6, 40, 2.0, 1),
        make_pkt("5.5.5.5", "8.8.8.8", 5000, 53, 17, 100, 3.0, 2),
    ]

    df = extract_flows("dummy.pcap")

    # Should produce 2 flows
    assert len(df) == 2

    # Check aggregated fields
    tcp_flow = df[(df.src_ip == "1.1.1.1") & (df.dst_ip == "2.2.2.2")].iloc[0]

    assert tcp_flow.packet_count == 2
    assert tcp_flow.byte_count == 100
    assert tcp_flow.avg_packet_size == 50
    assert tcp_flow.duration == 1.0 
    assert tcp_flow.protocol_name == "TCP"


@patch("extract_features.os.path.exists", return_value=True)
@patch("extract_features.rdpcap")
def test_extract_flows_udp(mock_rdpcap, mock_exists):
    mock_rdpcap.return_value = [
        make_pkt("10.0.0.1", "8.8.8.8", 4444, 53, 17, 80, 1.0, 0, layer="UDP")
    ]

    df = extract_flows("dummy.pcap")

    assert len(df) == 1
    udp_flow = df.iloc[0]
    assert udp_flow.protocol_name == "UDP"
    assert udp_flow.src_port == 4444
    assert udp_flow.dst_port == 53
