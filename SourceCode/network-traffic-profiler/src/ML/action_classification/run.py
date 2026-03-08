import pandas as pd
import os
import glob
import time
from extract_ml_features import extract_ml_features

DATA_DIR = "datasets/"
ACTIONS = ["Like", "Play", "Subscribe", "Comment", "Search"]

# Assign an action label to the most likely flow in a DataFrame.
def label_flows(df_flows, action):
    df_flows["action"] = "Background"

    # Initially mark everything as background noise
    if action == "Play":
        # Attempt to determine the flow with the most bytes
        action_index = df_flows["total_bytes"].idxmax()
        df_flows.at[action_index, "action"] = "Play"

    else:
        small_flows = df_flows[df_flows["total_bytes"] < 1_000_000]

        if not small_flows.empty:
            # The flow with the highest outbound ratio is likely the action
            action_index = small_flows["outbound_ratio"].idxmax()
            df_flows.at[action_index, "action"] = action

    return df_flows

#  Extract ML features from all PCAP files and produce master_training_data.csv.
def run(data_dir=DATA_DIR, actions=ACTIONS):

    all_rows = []
    test_data = {"pcap": [400, 300]}
    print("Testing CSV output")
    save_to_csv(test_data)
    
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
            df_flows = extract_ml_features(full_path)

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
    
    print(f"\n\nSuccess! Dataset created at {output_path}")

def beep():
    import winsound
    duration = 500
    freq = 440  
    winsound.Beep(freq, duration)

if __name__ == "__main__":
    run()
