import pandas as pd
import numpy as np
import os
from scapy.all import rdpcap


# Process a PCAP file and produce a single row of results in a dataframe 
def extract_ml_features(pcap_file, label=None):

    # End early if file doesn't exist
    if not os.path.exists(pcap_file):
        return None

    # Read file
    packets = rdpcap(pcap_file)
    packet_data = []

    for pkt in packets:
        if "IP" in pkt:
            # Extract basic stats
            size = len(pkt)
            ts = float(pkt.time)
            
            # Determine directionality by checking if dport exists and falls in range
            is_outbound = 1 if (pkt.haslayer("TCP") and pkt["TCP"].dport in [443, 80]) or (pkt.haslayer("UDP") and pkt["UDP"].dport in [443, 4433]) else 0
            packet_data.append([ts, size, is_outbound])

    df = pd.DataFrame(packet_data, columns=["ts", "size", "is_outbound"])
    
    # Return none if file contaiend no packets
    if df.empty:
        return None

    # Calculate inter-arrival time (IAT)
    df['iat'] = df['ts'].diff().fillna(0)

    # Create collection of all data extracted
    features = {
        # Packet stats
        "pk_count": len(df),
        "total_bytes": df['size'].sum(),
        "avg_pkt_size": df['size'].mean(),
        "std_pkt_size": df['size'].std(),
        "max_pkt_size": df['size'].max(),
        
        # Timing stats
        "avg_iat": df['iat'].mean(),
        "std_iat": df['iat'].std(),
        "duration": df['ts'].max() - df['ts'].min(),
        
        # Directional stats
        "outbound_ratio": df['is_outbound'].mean(),
        "avg_outbound_size": df[df['is_outbound'] == 1]['size'].mean() if not df[df['is_outbound'] == 1].empty else 0,
        "avg_inbound_size": df[df['is_outbound'] == 0]['size'].mean() if not df[df['is_outbound'] == 0].empty else 0,
    }

    # Convert to a single row DataFrame
    feature_df = pd.DataFrame([features])

    # Add action label
    if label:
        feature_df['action'] = label
    return feature_df