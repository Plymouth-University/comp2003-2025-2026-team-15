# Loads a previously trained ML classification model
# Loads the corresponding label encoder
# Accepts extracted flow features as input
# Predicts the action type for each flow
# Returns the original DataFrame with an added 'action_type' column
import joblib
import os
import pandas as pd

# Load trained model and label encoder
MODEL_PATH = os.path.join(os.path.dirname(__file__), "traffic_model.pkl")
ENCODER_PATH = os.path.join(os.path.dirname(__file__), "label_encoder.pkl")
model = joblib.load(MODEL_PATH)
label_encoder = joblib.load(ENCODER_PATH)

# Predicts action types for each flow in the given feature DataFrame
def predict_action_type(features_df: pd.DataFrame) -> pd.DataFrame:
    # Handles None or empty DataFrame
    if features_df is None or features_df.empty:
        features_df = features_df.copy() if features_df is not None else pd.DataFrame()
        features_df["action_type"] = None
        return features_df

    # Drop non-feature columns if present
    X = features_df.drop(columns=["action", "flow_index"], errors="ignore")
    # Predict encoded class labels (numeric)
    pred = model.predict(X)
    # Convert numeric labels back to original string labels
    action_types = label_encoder.inverse_transform(pred)
    # Attach predictions to original DataFrame
    features_df = features_df.copy()
    features_df["action_type"] = action_types
    return features_df
