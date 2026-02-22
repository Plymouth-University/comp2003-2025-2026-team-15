import pandas as pd
import numpy as np
import os
import ipaddress
from scapy.all import rdpcap

# Check if an IP belongs to known Youtube CIDR blocks
def is_google_youtube_ip(ip_str):
    try:
        ip_obj = ipaddress.ip_address(ip_str)
        known_ranges = [
            "172.217.0.0/16", "142.250.0.0/15", "104.237.160.0/19", 
            "208.117.224.0/19", "64.15.112.0/20", "216.58.192.0/19", "74.125.0.0/16"
        ]
        for network in known_ranges:
            if ip_obj in ipaddress.ip_network(network):
                return True
        return False
    except ValueError:
        return False

# Process a PCAP file and produce a dataframe where each row represents a flow
def extract_ml_features(pcap_file, label=None):
    # End early if file doesn't exist
    if not os.path.exists(pcap_file):
        return None

    # Read file
    packets = rdpcap(pcap_file)
    flow_data = {}

    for pkt in packets:
        if "IP" in pkt and (pkt.haslayer("TCP") or pkt.haslayer("UDP")):
            # Extract basic stats
            protocol_num = pkt["IP"].proto # 6 for TCP, 17 for UDP
            src_ip = pkt["IP"].src
            dst_ip = pkt["IP"].dst
            sport = pkt[protocol_num == 6 and "TCP" or "UDP"].sport
            dport = pkt[protocol_num == 6 and "TCP" or "UDP"].dport
            
            # Create a 5 tuple key for flow analysis
            flow_key = (src_ip, dst_ip, sport, dport, protocol_num)

            # Filter out flows that aren't associated with Google/YouTube IPs
            if not (is_google_youtube_ip(src_ip) or is_google_youtube_ip(dst_ip)):
                continue

            if flow_key not in flow_data:
                flow_data[flow_key] = []

            size = len(pkt)
            ts = float(pkt.time)
            is_outbound = 1 if dport in [443, 80, 4433] else 0
            flow_data[flow_key].append({'ts': ts, 'size': size, 'is_outbound': is_outbound})

    all_flow_features = []
    for i, (flow_key, packet_list) in enumerate(flow_data.items()):
        df = pd.DataFrame(packet_list)
        df['iat'] = df['ts'].diff().fillna(0)
        

        # keep only features that are used for model training
        features = {
            "flow_id": i, # unique identifier for each flow
            "duration": df['ts'].max() - df['ts'].min(),
            "std_iat": df['iat'].std() if len(df) > 1 else 0,
            "avg_iat": df['iat'].mean(),
            "pk_count": len(df),
            "avg_inbound_size": df[df['is_outbound'] == 0]['size'].mean() if not df[df['is_outbound'] == 0].empty else 0,
            "avg_outbound_size": df[df['is_outbound'] == 1]['size'].mean() if not df[df['is_outbound'] == 1].empty else 0,
            "total_bytes": df['size'].sum(),
            "outbound_ratio": df['is_outbound'].mean(),
        }
        if label: features['action'] = label
        all_flow_features.append(features)

    return pd.DataFrame(all_flow_features)