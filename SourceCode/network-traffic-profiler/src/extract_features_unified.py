import pandas as pd
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
def extract_all_pcap_data(pcap_file, ml_only=False, label=None):
    # End early if file doesn't exist
    if not os.path.exists(pcap_file):
        return None, None
 
    packets = rdpcap(pcap_file)
    
    # Containers for the two different aggregation types
    standard_rows = []
    ml_flow_map = {}

    for i, pkt in enumerate(packets):
        if "IP" in pkt:
            # Basic packet extraction for analysis
            src_ip = pkt["IP"].src
            dst_ip = pkt["IP"].dst
            protocol_num = pkt["IP"].proto
            size = len(pkt)
            ts = float(pkt.time)

            if pkt.haslayer("TCP"):
                sport, dport = pkt["TCP"].sport, pkt["TCP"].dport
            elif pkt.haslayer("UDP"):
                sport, dport = pkt["UDP"].sport, pkt["UDP"].dport
            else:
                sport, dport = None, None

            # Add to general packet list
            if not ml_only:
                standard_rows.append([src_ip, dst_ip, sport, dport, protocol_num, size, ts, i])

            # ML feature extraction
            # Filter for YouTube IPs and valid ports
            if sport is not None and (is_google_youtube_ip(src_ip) or is_google_youtube_ip(dst_ip)):
                flow_key = (src_ip, dst_ip, sport, dport, protocol_num)
                
                if flow_key not in ml_flow_map:
                    ml_flow_map[flow_key] = []
                
                is_outbound = 1 if dport in [443, 80, 4433] else 0
                ml_flow_map[flow_key].append({'ts': ts, 'size': size, 'is_outbound': is_outbound})

    # --- Processing General Flows ---
    df_packets = pd.DataFrame(standard_rows, columns=[
        "src_ip", "dst_ip", "src_port", "dst_port", "protocol", "packet_size", "timestamp", "packet_index"
    ])
    
    # Aggregate standard flow metrics
    flow_key_cols = ["src_ip", "dst_ip", "src_port", "dst_port", "protocol"]
    df_flows = df_packets.groupby(flow_key_cols).agg(
        packet_count=("packet_size", "count"),
        byte_count=("packet_size", "sum"),
        avg_packet_size=("packet_size", "mean"),
        start_time=("timestamp", "min"),
        end_time=("timestamp", "max"),
        first_packet_index=("packet_index", "min"),
        last_packet_index=("packet_index", "max")
    ).reset_index()

    # Relative time calculation
    p_start = df_flows["start_time"].min()
    df_flows["start_time"] -= p_start
    df_flows["end_time"] -= p_start
    df_flows["duration"] = df_flows["end_time"] - df_flows["start_time"]
    
    # Protocol naming
    df_flows['protocol_name'] = df_flows['protocol'].map({6: 'TCP', 17: 'UDP'}).fillna(df_flows['protocol'])

    # --- Processing ML Features ---
    ml_features = []
    for key, pkt_list in ml_flow_map.items():
        m_df = pd.DataFrame(pkt_list)
        m_df["iat"] = m_df["ts"].diff().fillna(0)
        
        in_pkts = len(m_df[m_df["is_outbound"] == 0])
        out_pkts = len(m_df[m_df["is_outbound"] == 1])
        
        ml_features.append({
            # 5-tuple for merging with validated_data in main.py
            "src_ip": key[0],
            "dst_ip": key[1],
            "src_port": key[2],
            "dst_port": key[3],
            "protocol": key[4],

            # ML features matching training data
            "duration": m_df["ts"].max() - m_df["ts"].min(),
            "std_iat": m_df["iat"].std() if len(m_df) > 1 else 0,
            "avg_iat": m_df["iat"].mean(),
            "pk_count": len(m_df),
            "max_pkt_size": m_df["size"].max(),
            "pkt_burst_std": m_df["size"].std() if len(m_df) > 1 else 0,
            "pk_count_ratio": (out_pkts / in_pkts if in_pkts > 0 else out_pkts),
            "avg_inbound_size": m_df[m_df["is_outbound"] == 0]["size"].mean() if in_pkts > 0 else 0,
            "avg_outbound_size": m_df[m_df["is_outbound"] == 1]["size"].mean() if out_pkts > 0 else 0,
            "total_bytes": m_df["size"].sum(),
            "outbound_ratio": m_df["is_outbound"].mean()
        })
        if label:
            ml_features["action"] = label

    return df_flows, pd.DataFrame(ml_features)