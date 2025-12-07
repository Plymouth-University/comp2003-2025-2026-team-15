# step 2: validate extracted features against rules

import os
import pandas as pd
import ipaddress
import numpy as np

# =========================
# FIELD SCHEMA & IANA_PROTOCOLS
# =========================
# FIELD_SCHEMA (expected structure of the dataset):
# - Required columns
# - Expected data types
# - Minimum and maximum values for numeric fields
FIELD_SCHEMA = {
    "src_ip": {"type": str, "required": True},                            # Source IP must exist and be string
    "dst_ip": {"type": str, "required": True},                            # Destination IP must exist and be string
    "src_port": {"type": int, "min": 0, "max": 65535, "required": True},  # Source port 0-65535
    "dst_port": {"type": int, "min": 0, "max": 65535, "required": True},  # Destination port 0-65535
    "protocol": {"type": int, "min": 0, "max": 255, "required": True},    # Protocol number 0-255
    "packet_count": {"type": int, "min": 0, "required": True},            # Packet count >= 0
    "byte_count": {"type": int, "min": 0, "required": True},              # Byte count >= 0
    "avg_packet_size": {"type": float, "min": 1.0, "max": 65535.0, "required": True},  # Avg packet size 1-65535
    "first_packet_index": {"type": int, "min": 0, "required": True},      # Index of first packet in flow
    "last_packet_index": {"type": int, "min": 0, "required": True},       # Index of last packet in flow
    "duration": {"type": float, "min": 0.0, "required": True},            # Flow duration in seconds
    "protocol_name": {"type": str, "required": True},                     # Protocol name must exist
}

IANA_PROTOCOLS = {
    1: "ICMP",
    6: "TCP",
    17: "UDP",
}

# =========================
# Helper Function: Validate IP
# - Checks if is a valid IPv4 or IPv6 address
# =========================
def validate_ip(ip_str):
    try:
        ipaddress.ip_address(ip_str)
        return True
    except:
        return False

# updated
def is_private_ip(ip_str):
    try:
        ip_obj = ipaddress.ip_address(ip_str)
        return ip_obj.is_private
    except:
        return False
    
# =========================
# Row-level validation
# =========================
def validate_row(row):
    errors = []
    
    # ---------------------------------------------------------
    # (General) Database-level / Type Validation
    #  - Check required fields are not missing
    #  - Check field types match expected (int, float, str)
    #  - Check min/max bounds for numeric fields
    # ---------------------------------------------------------
    # Check each field for required presence, type, and min/max constraints
    for field, rules in FIELD_SCHEMA.items():
        # Required fields check
        if rules.get("required", False) and pd.isna(row.get(field)):
            errors.append(f"{field}: required but missing.")
            continue
        value = row.get(field)
        if pd.isna(value):
            continue # Skip optional or missing fields
        
        expected_type = rules["type"]
        
        # Handle numpy integer types
        if expected_type == int:
            try:
                value = int(value)
            except Exception:
                errors.append(f"{field}: expected int, got {type(value).__name__}")
                continue


        if expected_type == float:
            try:
                value = float(value)
            except Exception:
                errors.append(f"{field}: expected float, got {type(value).__name__}")
            continue
            
        # Type check
        if not isinstance(value, expected_type):
            errors.append(f"{field}: expected {expected_type.__name__}, got {type(value).__name__}")
            continue
        
        # Min/Max checks
        if "min" in rules and value < rules["min"]:
            errors.append(f"{field}: {value} < minimum {rules['min']}")
        if "max" in rules and value > rules["max"]:
            errors.append(f"{field}: {value} > maximum {rules['max']}")   

    # ---------------------------------------------------------
    # IP Address Validation
    #  - src_ip and dst_ip must be valid IPv4/IPv6 addresses
    #  - src_ip ≠ dst_ip (no loopback flows)
    #  - Detect multicast addresses
    #  - Detect broadcast addresses (IPv4 only)
    # ---------------------------------------------------------
    # Ensure source and destination IPs are valid and not identical
    if not validate_ip(row["src_ip"]):
        errors.append("src_ip: invalid IPv4/IPv6 address")
    if not validate_ip(row["dst_ip"]):
        errors.append("dst_ip: invalid IPv4/IPv6 address")
    if row["src_ip"] == row["dst_ip"]:
        errors.append("src_ip and dst_ip must not be identical")
        
    # updated
    if validate_ip(row["src_ip"]):
        ip_obj = ipaddress.ip_address(row["src_ip"])
        if ip_obj.is_multicast:
            errors.append("src_ip is a multicast address")
        if ip_obj.is_loopback:
            errors.append("src_ip is a loopback address")
   #    if ip_obj.is_private:
   #        errors.append("src_ip is a private address")
        if isinstance(ip_obj, ipaddress.IPv4Address) and ip_obj == ipaddress.IPv4Address("255.255.255.255"):
            errors.append("src_ip is a broadcast address")
         
    # updated
    if validate_ip(row["dst_ip"]):
        ip_obj = ipaddress.ip_address(row["dst_ip"])
        if ip_obj.is_multicast:
            errors.append("dst_ip is a multicast address")
        if ip_obj.is_loopback:
            errors.append("dst_ip is a loopback address")
   #    if ip_obj.is_private:
   #        errors.append("dst_ip is a private address")
        if isinstance(ip_obj, ipaddress.IPv4Address) and ip_obj == ipaddress.IPv4Address("255.255.255.255"):
            errors.append("dst_ip is a broadcast address")

    # ---------------------------------------------------------
    # Protocol & Port Validation
    #  - TCP/UDP must have valid ports
    #  - ICMP must have ports=0
    #  - Protocol number must exist in IANA_PROTOCOLS
    #  - protocol_name must match protocol number
    # ---------------------------------------------------------
    # Protocol/port checks
    protocol = row["protocol"]
    src_port = row["src_port"]
    dst_port = row["dst_port"]
    
    # Protocol number must be valid
    if protocol not in IANA_PROTOCOLS:
        errors.append(f"Unknown protocol number: {protocol}")
    # Protocol name must match the protocol number
    if row["protocol_name"].upper() != IANA_PROTOCOLS.get(protocol,"").upper():
        errors.append(f"Protocol mismatch: protocol={protocol} but protocol_name={row['protocol_name']}")
    # TCP/UDP must have valid ports
    if protocol in (6, 17) and (src_port is None or dst_port is None):
        errors.append("TCP/UDP flows must include src_port and dst_port")
    # ICMP should not have ports (must be 0)
    if protocol == 1 and (src_port != 0 or dst_port != 0):
        errors.append("ICMP should not use ports (should be 0)")
    
    # ---------------------------------------------------------
    # Packet, Byte, Average Packet Size Validation
    #  - packet_count >= 0
    #  - byte_count >= packet_count
    #  - If packet_count=0 then byte_count must be 0
    #  - avg_packet_size ≈ byte_count / packet_count
    # ---------------------------------------------------------
    # Packet/byte/duration checks
    if row["packet_count"] == 0 and row["byte_count"] != 0:
        errors.append("packet_count=0 but byte_count > 0")
    if row["byte_count"] < row["packet_count"]:
        errors.append("byte_count < packet_count (impossible)")
    if row["packet_count"] > 0:
        expected_avg = row["byte_count"] / row["packet_count"]
        if abs(row["avg_packet_size"] - expected_avg) > 5:
            errors.append(f"avg_packet_size inconsistent (expected ~{expected_avg:.2f}, got {row['avg_packet_size']})")
            
    # updated: Extreme packet/byte sizes
    if row["avg_packet_size"] > 1500 or row["avg_packet_size"] < 20:
        errors.append("avg_packet_size outside typical Ethernet range (20-1500 bytes)")
    
    # ---------------------------------------------------------
    # Packet Index Validation
    # - first_packet_index ≤ last_packet_index
    # - If indices equal and single packet, duration ≈ 0
    # ---------------------------------------------------------   
    # Ensure first/last packet indices are logical
    if row["last_packet_index"] < row["first_packet_index"]:
        errors.append("last_packet_index < first_packet_index")
   
    # ---------------------------------------------------------
    # Duration Validation
    #  - duration ≥ 0
    #  - Single packet flows: duration ≈ 0
    #  - Multiple packets: duration cannot be 0
    # ---------------------------------------------------------
    if row["duration"] < 0:
            errors.append("duration < 0")
    if row["packet_count"] == 1 and row["duration"] > 0.001:
        errors.append("duration > 0 for packet_count=1")
    if row["packet_count"] > 1 and row["duration"] == 0:
        errors.append("duration=0 but multiple packets exist")
        
    # updated: Outlier duration check: multi-packet flows > 1 hour
    if row["duration"] > 3600:
        errors.append("flow duration unusually long (>1 hour)")
    # ---------------------------------------------------------
    # Protocol-Port Sanity Rules
    #  - DNS (port 53): duration should be short (<10s)
    #  - HTTPS (port 443): should have multiple packets
    # ---------------------------------------------------------
    if row["src_port"] == 53 or row["dst_port"] == 53:
        if row["duration"] > 10:
            errors.append("DNS traffic duration unusually long")
    if row["src_port"] == 443 or row["dst_port"] == 443:
        if row["packet_count"] < 2:
            errors.append("HTTPS flow has suspiciously low packet count")

    return errors

# =====================================================================
# Dataset-level validation
#  - Detects duplicates (based on key columns)
#  - Validates each row using validate_row()
#  - Adds validation columns to csv ('is_valid' and 'error_reason')
# =====================================================================
def validate_dataset(df):
    #df = pd.read_csv(input_file)
    all_errors = {}
    
    # Column existence check
    for field in FIELD_SCHEMA.keys():
        if field not in df.columns:
            raise ValueError(f"Missing required column: {field}")

    # Duplicate detection
    duplicate_keys = ["src_ip","dst_ip","src_port","dst_port","protocol","first_packet_index"]
    dupes = df[df.duplicated(duplicate_keys, keep=False)]
    for idx in dupes.index:
        all_errors[idx] = ["Duplicate flow detected"]

    # Row validation (run all checks for each row)
    for idx, row in df.iterrows():
        row_errors = validate_row(row)
        if row_errors:
            all_errors.setdefault(idx,[]).extend(row_errors)

    # Add validation columns & results to dataset)
    df["is_valid"] = True
    df["error_reason"] = ""
    for idx, errs in all_errors.items():
        df.at[idx, "is_valid"] = False
        df.at[idx, "error_reason"] = "; ".join(errs)
        
    print("Validated dataset")
    
    return df

