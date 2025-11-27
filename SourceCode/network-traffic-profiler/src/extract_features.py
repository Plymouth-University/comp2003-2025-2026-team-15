# step 1: extract features from pcap file

from scapy.all import rdpcap
import pandas as pd
import os


def extract_flows(PCAP_FILE):
    # --- Configuration ---
    #PCAP_FILE = "pcap_data/test.pcap"
    CSV_PATH = "csv_output/"
    # Read the file name
    pcap_basename = os.path.splitext(os.path.basename(PCAP_FILE))[0]
    OUTPUT_CSV = f"{CSV_PATH}{pcap_basename}_flows.csv"

    # --- 1. Load PCAP and Extract Packet-Level Data ---

    if not os.path.exists(PCAP_FILE):
        print(f"Error: PCAP file '{PCAP_FILE}' not found.")
        # Exit or handle the error appropriately
        exit() 

    print(f"Loading packets from {PCAP_FILE}...")
    # Read file into packets
    packets = rdpcap(PCAP_FILE)
    # Initialise an empty container array
    rows = []

    for i, pkt in enumerate(packets):
        # Check if the packet has an IP layer (to filter out non-IP protocols like ARP/LLC)
        if "IP" in pkt:
            src_ip = pkt["IP"].src
            dst_ip = pkt["IP"].dst
            protocol = pkt["IP"].proto
            size = len(pkt)
            ts = pkt.time

            # Extract port information from TCP or UDP layers
            if pkt.haslayer("TCP"):
                src_port = pkt["TCP"].sport
                dst_port = pkt["TCP"].dport
            elif pkt.haslayer("UDP"):
                src_port = pkt["UDP"].sport
                dst_port = pkt["UDP"].dport
            else:
                # Set port to none for non IP protocols
                src_port = None
                dst_port = None

            # Append row of data
            rows.append([src_ip, dst_ip, src_port, dst_port, protocol, size, ts, i])

    # Create DataFrame with packet-level data
    df_packets = pd.DataFrame(rows, columns = [
        "src_ip", "dst_ip", "src_port", "dst_port", "protocol", 
        "packet_size", "timestamp", "packet_index"
    ])

    print(f"Successfully extracted {len(df_packets)} IP packets.")

    # --- Aggregate to Flow-Level Features ---

    # 5-tuple that forms a unique key for each network flow
    flow_key = ["src_ip", "dst_ip", "src_port", "dst_port", "protocol"]

    print("Aggregating packets into network flows...")
    # Group packets that share the same tuple together
    df_flows = df_packets.groupby(flow_key).agg(
        # Core flow metrics
        packet_count=("packet_size", "count"),
        byte_count=("packet_size", "sum"),
        avg_packet_size=("packet_size", "mean"),
        
        # Time metrics
        start_time=("timestamp", "min"),
        end_time=("timestamp", "max"),
        
        # Drill-down link (stores the index range in the original PCAP)
        first_packet_index=("packet_index", "min"),
        last_packet_index=("packet_index", "max")
    ).reset_index()

    # Compute flow duration in seconds
    df_flows["duration"] = df_flows["end_time"] - df_flows["start_time"]

    # Convert protocol number to a readable string
    protocol_map = {6: 'TCP', 17: 'UDP'}
    df_flows['protocol_name'] = df_flows['protocol'].map(protocol_map).fillna(df_flows['protocol'])


    # --- Output Results ---

    # Drop raw timestamps for a cleaner output
    df_flows = df_flows.drop(columns=["start_time", "end_time"])

    print(f"\nFlow extraction complete. Generated {len(df_flows)} unique flows.")
    return df_flows

if __name__ == "__main__":
    extract_flows("pcap_data/test.pcap")
