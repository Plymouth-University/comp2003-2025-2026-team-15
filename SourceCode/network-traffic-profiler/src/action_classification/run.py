# Entry script that will iterate over the five unique action directories,
# extracting required data from each PCAP file before merging the data into
# one master CSV file for ML training

import pandas as pd
import os
from extract_features import extract_ml_features

data_dir = "datasets/"
actions = ["Like", "Play", "Subscribe", "Comment", "Search"]
all_rows = []

print("Beginning extraction...")
for i,action in enumerate(actions):
    print(f"\nExtracting features from {action} dataset ({i+1}/{len(actions)})")
    folder_path = os.path.join(data_dir, action)
    pcap_files = []
    for file in os.listdir(folder_path):
        if file.endswith('.pcap'):
            pcap_files.append(file)
    total_files = len(pcap_files)
            
    for index,filename in enumerate(pcap_files):
        count = index+1
        # Neatly output status
        status_text = f"\rProcessing: {count}/{total_files} ({filename})".ljust(40)
        print(status_text, end="")
        
        full_path = os.path.join(folder_path, filename)
        # Call the extraction method in extract_features.py
        feature_row = extract_ml_features(full_path, label=action)
        
        if feature_row is not None:
            # Add filename for tracking
            feature_row.insert(0, 'file_source', filename)
            all_rows.append(feature_row)

# Combine everything into one dataframe
master_df = pd.concat(all_rows, ignore_index=True)
master_df.to_csv("master_training_data.csv", index=False)
print("Dataset created with shape:", master_df.shape)