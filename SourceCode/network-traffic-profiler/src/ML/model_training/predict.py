# Loads a previously trained ML classification model
# Loads the corresponding label encoder
# Accepts extracted flow features as input
# Predicts the action type for each flow
# Returns the original DataFrame with an added 'action_type' column
import joblib
import os
import pandas as pd
import numpy as np

# Load trained model and label encoder
MODEL_PATH = os.path.join(os.path.dirname(os.path.dirname(__file__)), "action_model.pkl")
ENCODER_PATH = os.path.join(os.path.dirname(os.path.dirname(__file__)), "label_encoder.pkl")
FEATURES_PATH = os.path.join(os.path.dirname(os.path.dirname(__file__)), "model_features.pkl")
model = joblib.load(MODEL_PATH)
label_encoder = joblib.load(ENCODER_PATH)
model_features = joblib.load(FEATURES_PATH)

# Predicts action types for each flow in the given feature DataFrame
def predict_action_type(features_df: pd.DataFrame) -> pd.DataFrame:
    
    # Handles None or empty DataFrame
    if features_df is None or features_df.empty:
        return features_df

    # Preparation
    X = features_df.copy()
    X = X[model_features]
    
    # Get a matrix of probabilities for each action for each flow
    probs = model.predict_proba(X) 
    # Action classes
    classes = label_encoder.classes_
    
    final_results = []
    # Iterate through each flow
    for i in range(len(probs)):
        row_probs = probs[i]
        # Finds the highest predicted action index
        best_idx = np.argmax(row_probs)
        # Action label, eg 'Comment'
        best_label = classes[best_idx]
        # Actual confidence 'eg 0.7'
        best_score = row_probs[best_idx]
        
        flow_size = features_df.iloc[i]['total_bytes']
        # If a large flow is labelled as background, and the runner up prediction
        # is is more than 0.3 then overrule
        if best_label == "Background" and flow_size > 1_000_000:
            # Find the second best prediction
            # Sort indices by probability descending
            sorted_indices = np.argsort(row_probs)[::-1]
            runner_up_idx = sorted_indices[1]
            runner_up_label = classes[runner_up_idx]
            runner_up_score = row_probs[runner_up_idx]

            if runner_up_score > 0.30:
                print(f"  - ACTION: Overruling Background -> {runner_up_label}")
                final_results.append(runner_up_label)
            else:
                final_results.append(best_label)
        else:
            final_results.append(best_label)

    features_df["action_type"] = final_results
    return features_df