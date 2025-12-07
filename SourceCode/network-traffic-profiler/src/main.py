# program entry, combine steps into one pipeline
# extract features -> validate -> build dataset -> model training -> dashboard visualisation

from extract_features import extract_flows
from process_dataset import validate_dataset
from build_dataset import build_dataset
import os


def run_pipeline(pcap_path):
    pcap_basename = os.path.splitext(os.path.basename(pcap_path))[0]
    # Extract data and identify flows
    flows = extract_flows(pcap_path)
    # Validate data
    # Second param indicates whether to generate a CSV file
    validated_data = validate_dataset(flows)
    # Build dataset for ML
    numeric_df, anomaly_info = build_dataset(validated_data, pcap_basename)
    # Return to dashboard
    return validated_data, numeric_df, anomaly_info
