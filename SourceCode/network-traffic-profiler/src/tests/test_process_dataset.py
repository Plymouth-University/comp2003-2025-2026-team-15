import pandas as pd
from process_dataset import validate_ip, validate_row, validate_dataset

# test for validate_ip()

def test_validate_ip_valid_ipv4():
    assert validate_ip("192.168.1.1") is True

def test_validate_ip_invalid_ipv4():
    assert validate_ip("999.999.999.999") is False

# test for validate_row()

def test_validate_row_valid_flow():
    row = {
        "src_ip": "192.168.1.10",
        "dst_ip": "192.168.1.20",
        "src_port": 443,
        "dst_port": 5000,
        "protocol": 6,
        "packet_count": 10,
        "byte_count": 5000,
        "avg_packet_size": 500.0,
        "first_packet_index": 0,
        "last_packet_index": 9,
        "duration": 1.0,
        "protocol_name": "TCP",
    }
    errors = validate_row(row)
    assert errors == []

# test validate_dataset() duplicate detection

def test_validate_dataset_duplicate_detection():
    # two identical rows for duplicate detection
    df = pd.DataFrame([
        {
            "src_ip": "10.0.0.1",
            "dst_ip": "10.0.0.2",
            "src_port": 80,
            "dst_port": 8080,
            "protocol": 6,
            "packet_count": 2,
            "byte_count": 200,
            "avg_packet_size": 100.0,
            "first_packet_index": 0,
            "last_packet_index": 1,
            "duration": 0.5,
            "protocol_name": "TCP",
        },
        {
            # duplicate row
            "src_ip": "10.0.0.1",
            "dst_ip": "10.0.0.2",
            "src_port": 80,
            "dst_port": 8080,
            "protocol": 6,
            "packet_count": 2,
            "byte_count": 200,
            "avg_packet_size": 100.0,
            "first_packet_index": 0,
            "last_packet_index": 1,
            "duration": 0.5,
            "protocol_name": "TCP",
        }
    ])

    validated = validate_dataset(df, make_csv=False, pcap_basename="testfile")

    # at least one entry must be marked invalid
    assert validated["is_valid"].iloc[0] == False or validated["is_valid"].iloc[1] == False

    # both should have the duplicate error reason
    assert "Duplicate flow detected" in validated["error_reason"].iloc[0]
    assert "Duplicate flow detected" in validated["error_reason"].iloc[1]
