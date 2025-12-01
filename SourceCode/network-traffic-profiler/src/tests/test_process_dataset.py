import pandas as pd
from process_dataset import validate_ip, validate_row, validate_dataset

# test for validate_ip()
def test_validate_ip_valid_ipv4():
    assert validate_ip("192.168.1.1") is True

def test_validate_ip_invalid_ipv4():
    assert validate_ip("999.999.999.999") is False
    
# validate_row tests
def test_validate_row_invalid_ips_and_ports():
    row = {
        "src_ip": "224.0.0.1",       # multicast
        "dst_ip": "127.0.0.1",       # loopback
        "src_port": 1,               # ICMP invalid port
        "dst_port": 1,               # ICMP invalid port
        "protocol": 1,               # ICMP
        "packet_count": 1,
        "byte_count": 64,
        "avg_packet_size": 64,
        "first_packet_index": 0,
        "last_packet_index": 0,
        "duration": 0,
        "protocol_name": "ICMP",
    }
    errors = validate_row(row)
    assert "src_ip is a multicast address" in errors
    assert "dst_ip is a loopback address" in errors
    assert "ICMP should not use ports (should be 0)" in errors

def test_validate_row_avg_packet_size_extremes_and_duration():
    row = {
        "src_ip": "8.8.8.8",
        "dst_ip": "1.1.1.1",
        "src_port": 1234,
        "dst_port": 5678,
        "protocol": 6,
        "packet_count": 10,
        "byte_count": 50000,
        "avg_packet_size": 5000,   # too high
        "first_packet_index": 0,
        "last_packet_index": 9,
        "duration": 4000,           # >1 hour
        "protocol_name": "TCP",
    }
    errors = validate_row(row)
    assert "avg_packet_size outside typical Ethernet range (20-1500 bytes)" in errors
    assert "flow duration unusually long (>1 hour)" in errors

def test_validate_row_dns_https_rules():
    row_dns = {
        "src_ip": "8.8.8.8",
        "dst_ip": "1.1.1.1",
        "src_port": 53,
        "dst_port": 1234,
        "protocol": 17,
        "packet_count": 10,
        "byte_count": 1000,
        "avg_packet_size": 100,
        "first_packet_index": 0,
        "last_packet_index": 9,
        "duration": 15,  # too long
        "protocol_name": "UDP",
    }
    errors = validate_row(row_dns)
    assert "DNS traffic duration unusually long" in errors

    row_https = {
        "src_ip": "8.8.8.8",
        "dst_ip": "1.1.1.1",
        "src_port": 443,
        "dst_port": 5678,
        "protocol": 6,
        "packet_count": 1,  # too few
        "byte_count": 500,
        "avg_packet_size": 500,
        "first_packet_index": 0,
        "last_packet_index": 0,
        "duration": 0.5,
        "protocol_name": "TCP",
    }
    errors = validate_row(row_https)
    assert "HTTPS flow has suspiciously low packet count" in errors

# validate_dataset duplicate detection
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


