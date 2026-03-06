import os
import sys
import pandas as pd

CURRENT_DIR = os.path.dirname(__file__)
SRC_PATH = os.path.abspath(os.path.join(CURRENT_DIR, "../../"))
ACTION_PATH = os.path.abspath(os.path.join(CURRENT_DIR, "../action_classification"))
MODEL_PATH = os.path.abspath(os.path.join(CURRENT_DIR, "../model_training"))

sys.path.append(SRC_PATH)
sys.path.append(ACTION_PATH)
sys.path.append(MODEL_PATH)

from action_classification.extract_ml_features import extract_ml_features
from action_classification.run import label_flows
from model_training.predict import predict_action_type

# Path to a sample PCAP file used for testing
TEST_PCAP = os.path.abspath(os.path.join(SRC_PATH, "pcap_data", "test.pcap"))

#Tests that a PCAP file can be processed and converted into a DataFrame of ML features.
#Pipeline: PCAP → extract_ml_features → DataFrame
def test_extract_features_to_dataframe():
    df = extract_ml_features(TEST_PCAP)
    assert df is not None
    assert isinstance(df, pd.DataFrame)

# Ensures that feature extraction produces all required ML feature columns.
def test_extract_contains_required_columns():
    df = extract_ml_features(TEST_PCAP)
    if df is None or df.empty:
        return

    required = [ # Required features used by the ML model
        "duration",
        "std_iat",
        "avg_iat",
        "pk_count",
        "avg_inbound_size",
        "avg_outbound_size",
        "total_bytes",
        "outbound_ratio"
    ]

    for col in required:
        assert col in df.columns

# Tests the pipeline from feature extraction to flow labeling.
# Pipeline: PCAP → extract_ml_features → label_flows
def test_label_flows_pipeline():
    df = extract_ml_features(TEST_PCAP)
    if df is None or df.empty:
        return

    labeled = label_flows(df, "Play")  # Label flows using the Play action logic

    assert "action" in labeled.columns # Verify labeling added an action column

# Tests the full ML prediction pipeline.
# Pipeline: PCAP → feature extraction → ML prediction
def test_predict_pipeline():
    df = extract_ml_features(TEST_PCAP)
    if df is None or df.empty:
        return

    predictions = predict_action_type(df)
    assert "action_type" in predictions.columns
