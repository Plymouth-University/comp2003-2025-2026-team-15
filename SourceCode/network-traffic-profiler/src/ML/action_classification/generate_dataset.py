import pandas as pd
import os
import sys
import glob
import time

# Add parent folder to path
src_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if src_path not in sys.path:
    sys.path.append(src_path)
# Can now import:
from extract_features_unified import extract_all_pcap_data

DATA_DIR = "datasets/"
ACTIONS = ["Like", "Play", "Subscribe", "Comment", "Search"]

# Assign an action label to the most likely flow in a DataFrame.
def label_flows(df_flows, action):
     # Default everything to Background noise initially
    df_flows["action"] = "Background"

    if df_flows.empty:
        return df_flows
    
    if action == "Play":
        #Filter for flows that are at least 500KB
        candidates = df_flows[df_flows["total_bytes"] > 500_000 & (df_flows["avg_packet_size"] > 800)]
        # candidates = df_flows[(df_flows["total_bytes"] > 500_000) & (df_flows["avg_inbound_size"] > 800)]
        
        if not candidates.empty:
            # Label all large flows as Play
            df_flows.loc[candidates.index, "action"] = "Play"

    elif action == "Search":
        # Label flows that look like the API trigger (small, outbound)
        # and flows that look like thumbnails (medium, inbound)
        search_mask = (
            ((df_flows["total_bytes"] > 2000) & (df_flows["outbound_ratio"] > 0.6)) | # API trigger
            ((df_flows["total_bytes"] > 20000) & (df_flows["total_bytes"] < 300_000)) # Resulting thumbnails
        )
        df_flows.loc[search_mask, "action"] = "Search"

    elif action in ["Like", "Subscribe"]:
        # Check for flows < 15KB
        candidates = df_flows[
            (df_flows["total_bytes"] < 15000) & 
            (df_flows["pk_count"] >= 3) & # Must be a handshake + request
            (df_flows["outbound_ratio"] > 0.7)
        ]
        if not candidates.empty:
            # prioritise the highest outbound ratio
            df_flows.loc[candidates.index, "action"] = action

    elif action == "Comment":
        # usually slightly larger than a 'Like' as includes text
        candidates = df_flows[
            (df_flows["total_bytes"] < 30000) & 
            (df_flows["outbound_ratio"] > 0.6)
        ]
        if not candidates.empty:
            df_flows.loc[candidates.index, "action"] = "Comment"

    return df_flows

#  Extract ML features from all PCAP files and produce master_training_data.csv.
def run(data_dir=DATA_DIR, actions=ACTIONS):

    all_rows = []
    test_data = {"pcap": [400, 300]}
    print("Testing CSV output")
    if save_to_csv(test_data) is False:
        return
    
    print("Beginning extraction...")
    start = time.time()
    for i, action in enumerate(actions):
        print(f"\nExtracting features from {action} dataset ({i + 1}/{len(actions)})")
        folder_path = os.path.join(data_dir, action)
        # Searches for files that match the specified pattern
        pcap_files = glob.glob(os.path.join(folder_path, "*.pcap"))
        total_files = len(pcap_files)

        for index, full_path in enumerate(pcap_files):
            filename = os.path.basename(full_path)
            count = index + 1
            elapsed = round(time.time() - start)
            print(f"\r{elapsed}s | Processing: {count}/{total_files} ({filename})".ljust(50), end="")

            # Extract ALL flows from this PCAP into a dataframe
            df_flows = extract_all_pcap_data(full_path, True)[1]

            if df_flows is not None and not df_flows.empty:
                # Attempt to find the most likely flow containing the action that is being trained on.
                df_flows = label_flows(df_flows, action)
                df_flows.insert(0, "file_source", filename)
                all_rows.append(df_flows)

    # Combine everything into one dataframe
    if all_rows:
        master_df = save_to_csv(all_rows)
        output_path="../model_training/master_training_data.csv"
        print(f"\n\nSuccess! Dataset created at {output_path} with {len(master_df)} total flows")
        print("Final Label Counts in CSV:")
        print(master_df["action"].value_counts())
        beep()
        return master_df
    else:
        print("\nNo data was extracted. Check your file paths.")
        return None

def save_to_csv(data, output_path="../model_training/master_training_data.csv"):
    # Ensure the directory exists
    try:
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
        if isinstance(data, list) and len(data) > 0 and isinstance(data[0], pd.DataFrame):
            new_df = pd.concat(data, ignore_index=True)
        else:
            new_df = pd.DataFrame(data)

        new_df.to_csv(output_path, index=False)
        return new_df
    except Exception as e:
        print(f"Error creating CSV: {e}")
        beep()
        return False
    
    print(f"\n\nSuccess! Dataset created at {output_path}")

def beep():
    import winsound
    duration = 500
    freq = 440  
    winsound.Beep(freq, duration)

if __name__ == "__main__":
    run()
