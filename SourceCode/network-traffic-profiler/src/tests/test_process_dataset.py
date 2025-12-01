import pandas as pd
from process_dataset import validate_ip, validate_row,validate_dataset

columns = ["src_ip","dst_ip","src_port","dst_port","protocol","packet_count",
           "byte_count","avg_packet_size","first_packet_index","last_packet_index",
           "duration","protocol_name"]

# Helper function to validate a single row
def validate_single_row(row_data):
    df = pd.DataFrame([row_data], columns=columns)
    return validate_dataset(df, make_csv=False, pcap_basename="testfile")

# test for validate_ip()
def test_validate_ip_valid_ipv4():
    assert validate_ip("192.168.1.1") is True

def test_validate_ip_invalid_ipv4():
    assert validate_ip("999.999.999.999") is False

# Tests for validate_row() 
def test_row_multicast_loopback_icmp_ports():
    row = ["224.0.0.1","127.0.0.1",1,1,1,1,64,64,0,0,0.0,"ICMP"]
    result = validate_single_row(row)
    errors = result["error_reason"].iloc[0]
    assert "src_ip is a multicast address" in errors
    assert "dst_ip is a loopback address" in errors
    assert "ICMP should not use ports" in errors

def test_row_avg_packet_size_and_duration_extremes():
    row = ["8.8.8.8","1.1.1.1",1234,5678,6,10,50000,5000,0,9,4000,"TCP"]
    result = validate_single_row(row)
    errors = result["error_reason"].iloc[0]
    assert "avg_packet_size inconsistent" in errors or "avg_packet_size outside typical Ethernet range" in errors
    assert "flow duration unusually long" in errors

def test_row_dns_duration_too_long():
    row = ["8.8.8.8","1.1.1.1",53,1234,17,10,1000,100,0,9,15,"UDP"]
    result = validate_single_row(row)
    assert "DNS traffic duration unusually long" in result["error_reason"].iloc[0]

def test_row_https_low_packet_count():
    row = ["8.8.8.8","1.1.1.1",443,5678,6,1,500,500,0,0,0.5,"TCP"]
    result = validate_single_row(row)
    assert "HTTPS flow has suspiciously low packet count" in result["error_reason"].iloc[0]

# Test functions for each case
def test_row1_duplicate():
    rows = [
        ["142.250.117.119","192.168.0.21",443,49666,17,244,271560,1112.95,159,3563,1.333,"UDP"],
        ["142.250.117.119","192.168.0.21",443,49666,17,244,271560,1112.95,159,3563,1.333,"UDP"]  # duplicate
    ]
    df = pd.DataFrame(rows, columns=columns)
    result = validate_dataset(df, make_csv=False, pcap_basename="testfile")
    # First row will now be marked as duplicate
    assert not result["is_valid"].iloc[0]
    assert "Duplicate flow detected" in result["error_reason"].iloc[0]
    # Second row is also duplicate
    assert not result["is_valid"].iloc[1]
    assert "Duplicate flow detected" in result["error_reason"].iloc[1]
    
def test_row2_invalid_src_ip():
    row = ["999.999.999.999","192.168.0.21",443,49666,17,244,271560,1112.95,159,3563,1.333,"UDP"]
    result = validate_single_row(row)
    assert "src_ip: invalid IPv4/IPv6 address" in result["error_reason"].iloc[0]

def test_row3_src_eq_dst_ip():
    row = ["192.168.0.10","192.168.0.10",443,49666,17,244,271560,1112.95,159,3563,1.333,"UDP"]
    result = validate_single_row(row)
    assert "src_ip and dst_ip must not be identical" in result["error_reason"].iloc[0]

def test_row4_multicast_src_ip():
    row = ["224.0.0.1","192.168.0.21",443,49666,17,244,271560,1112.95,159,3563,1.333,"UDP"]
    result = validate_single_row(row)
    assert "src_ip is a multicast address" in result["error_reason"].iloc[0]

def test_row5_broadcast_dst_ip():
    row = ["142.250.117.119","255.255.255.255",443,49666,17,244,271560,1112.95,159,3563,1.333,"UDP"]
    result = validate_single_row(row)
    assert "dst_ip is a broadcast address" in result["error_reason"].iloc[0]

def test_row6_invalid_protocol():
    row = ["142.250.117.119","192.168.0.21",443,49666,99,244,271560,1112.95,159,3563,1.333,"UNKNOWN"]
    result = validate_single_row(row)
    assert "Unknown protocol number: 99" in result["error_reason"].iloc[0]

def test_row7_protocol_name_mismatch():
    row = ["142.250.117.119","192.168.0.21",443,49666,17,244,271560,1112.95,159,3563,1.333,"TCP"]
    result = validate_single_row(row)
    assert "Protocol mismatch" in result["error_reason"].iloc[0]

def test_row8_icmp_ports():
    row = ["142.250.117.119","192.168.0.21",1,1,1,1,64,64,0,0,0.0,"ICMP"]
    result = validate_single_row(row)
    assert "ICMP should not use ports" in result["error_reason"].iloc[0]

def test_row9_packet_count_zero_byte_count_gt0():
    row = ["142.250.117.119","192.168.0.21",443,49666,17,0,10,0,159,159,0.0,"UDP"]
    result = validate_single_row(row)
    assert "packet_count=0 but byte_count > 0" in result["error_reason"].iloc[0]

def test_row10_byte_count_less_than_packet_count():
    row = ["142.250.117.119","192.168.0.21",443,49666,17,100,50,0.5,159,259,1.0,"UDP"]
    result = validate_single_row(row)
    assert "byte_count < packet_count" in result["error_reason"].iloc[0]

def test_row11_avg_packet_size_inconsistent():
    row = ["142.250.117.119","192.168.0.21",443,49666,17,10,1000,50,159,168,1.0,"UDP"]
    result = validate_single_row(row)
    assert "avg_packet_size inconsistent" in result["error_reason"].iloc[0]

def test_row12_last_packet_index_lt_first():
    row = ["142.250.117.119","192.168.0.21",443,49666,17,10,1000,100,200,150,1.0,"UDP"]
    result = validate_single_row(row)
    assert "last_packet_index < first_packet_index" in result["error_reason"].iloc[0]

def test_row13_single_packet_duration_gt_0():
    row = ["142.250.117.119","192.168.0.21",443,49666,17,1,100,100,159,159,1.0,"UDP"]
    result = validate_single_row(row)
    assert "duration > 0 for packet_count=1" in result["error_reason"].iloc[0]

def test_row14_multiple_packets_duration_zero():
    row = ["142.250.117.119","192.168.0.21",443,49666,17,10,1000,100,159,168,0.0,"UDP"]
    result = validate_single_row(row)
    assert "duration=0 but multiple packets exist" in result["error_reason"].iloc[0]

def test_row15_dns_duration_too_long():
    row = ["142.250.117.119","192.168.0.21",53,12345,17,10,1000,100,159,168,15.0,"UDP"]
    result = validate_single_row(row)
    assert "DNS traffic duration unusually long" in result["error_reason"].iloc[0]

def test_row16_https_low_packet_count():
    row = ["142.250.117.119","192.168.0.21",443,12345,17,1,100,100,159,159,0.5,"UDP"]
    result = validate_single_row(row)
    assert "HTTPS flow has suspiciously low packet count" in result["error_reason"].iloc[0]

def test_row17_multiple_errors():
    row = ["192.168.1.300","192.168.1.300",70000,443,1,0,10,10.0,5,3,0.0,"ICMP"]
    result = validate_single_row(row)
    error_msg = result["error_reason"].iloc[0]
    assert "src_ip: invalid IPv4/IPv6 address" in error_msg
    assert "dst_ip: invalid IPv4/IPv6 address" in error_msg
    assert "src_ip and dst_ip must not be identical" in error_msg
    assert "src_port: 70000 > maximum 65535" in error_msg

