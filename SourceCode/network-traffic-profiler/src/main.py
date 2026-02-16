# program entry, combine steps into one pipeline
# extract features -> validate -> build dataset -> model training -> dashboard visualisation

from extract_features import extract_flows
from process_dataset import validate_dataset
from build_dataset import build_dataset
import os
import pandas as pd

# Import ML prediction function
from ML.model_training.predict import predict_action_type
# Import ML feature extraction
from ML.action_classification.extract_features import extract_ml_features


def run_pipeline(pcap_path):
    pcap_basename = os.path.splitext(os.path.basename(pcap_path))[0]
    # Extract data and identify flows
    flows = extract_flows(pcap_path)
    # Second param indicates whether to generate a CSV file
    validated_data = validate_dataset(flows)

    # Extract ML features for each flow 
    ml_features_list = []
    for idx, row in validated_data.iterrows():
        features = extract_ml_features(pcap_path)
        if features is not None:
            features["flow_index"] = idx
            ml_features_list.append(features)

    if ml_features_list:
        # Combines all extracted features into one DataFrame for prediction
        ml_features_df = pd.concat(ml_features_list, ignore_index=True)
        # Predict action type
        ml_features_df = predict_action_type(ml_features_df)
        # Merge action_type back to validated_data
        validated_data = validated_data.copy()
        validated_data["action_type"] = None
        for _, feat_row in ml_features_df.iterrows():
            idx = feat_row["flow_index"]
            validated_data.at[idx, "action_type"] = feat_row["action_type"]

    # Build dataset for ML
    numeric_df, anomaly_info = build_dataset(validated_data, pcap_basename)
    # Return to dashboard
    return validated_data, numeric_df, anomaly_info
