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
    print(f"Extracting ML features for {pcap_basename}...")
    
    # Extract ML features for each flow 
    ml_features_df = extract_ml_features(pcap_path)

  
    if not ml_features_df.empty:
        print("Predicting actions...")
        try:
            # Predict action type for each flow
            ml_features_df = predict_action_type(ml_features_df)
            
            # Check if prediction succeeded and column exists
            if 'action_type' in ml_features_df.columns:
                predicted_action = ml_features_df['action_type'].iloc[0]
            else:
                predicted_action = "Classification Failed"
        except Exception as e:
            print(f"Prediction error: {e}")
            predicted_action = "Prediction Error"
    else:
        # If extraction failed
        print("Warning: No ML features extracted.")
        predicted_action = "Unknown / No IP Traffic"

    # Merge action_type back to validated_data
    validated_data = validated_data.copy()
    validated_data["action_type"] = predicted_action

    # Build dataset for ML
    numeric_df, anomaly_info = build_dataset(validated_data, pcap_basename)
    # Return to dashboard
    return validated_data, numeric_df, anomaly_info