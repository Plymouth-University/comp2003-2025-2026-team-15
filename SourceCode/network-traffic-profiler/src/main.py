# program entry, combine steps into one pipeline
# extract features -> validate -> build dataset -> model training -> dashboard visualisation

from process_dataset import validate_dataset
from build_dataset import build_dataset
import os
import pandas as pd

# Import ML prediction function
from ML.model_training.predict import predict_action_type
# Import ML feature extraction
from extract_features_unified import extract_all_pcap_data


def run_pipeline(pcap_path, status=None):
    pcap_basename = os.path.splitext(os.path.basename(pcap_path))[0]
    if status: status.write("1. Extracting network flows")
    pcap_extraction_results = extract_all_pcap_data(pcap_path)
    # Extract data and identify flows
    flows = pcap_extraction_results[0]
    if status: status.write("2. Validating network flows")
    validated_data = validate_dataset(flows)

    # Assign flow_id to each flow
    
    validated_data = validated_data.reset_index(drop=True)
    validated_data['flow_id'] = validated_data.index

    # Extract ML features for each flow 
    print(f"Extracting ML features for {pcap_basename}...")
    
    # Extract ML features for each flow 
    ml_features_df = pcap_extraction_results[1]

  
    if not ml_features_df.empty:
       
        if status: status.write("3. Predicting user actions")
        print("Predicting actions for flows...")
        try:
            # Predict action type for each flow
            if not ml_features_df.empty:
                ml_features_df = predict_action_type(ml_features_df)
            
            # define columns to be merged
            merge_cols = ["src_ip", "dst_ip", "src_port", "dst_port", "protocol"]

            # merge the data
            validated_data = validated_data.merge(
                ml_features_df[merge_cols + ["action_type"]],
                on=merge_cols,
                how="left"
            )
            
            # Fill non youtube related flows
            validated_data['action_type'] = validated_data['action_type'].fillna('Background')
        except Exception as e:
            print("FULL ERROR:", repr(e)) # print detailed error
            raise

    else:
        # If extraction failed
        print("Warning: No ML features extracted.")
        validated_data['action_type'] = "Unknown / No IP Traffic"

    # Build dataset for ML
    if status: status.write("4. Detecting anomolies in the data")
    numeric_df, anomaly_info = build_dataset(validated_data, pcap_basename)
    # Return to dashboard
    return validated_data, numeric_df, anomaly_info