# step 3: build (clean up) dataset from validated features

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import IsolationForest
import os

def build_dataset(validated_csv):

    print(f"Loading validated dataset: {validated_csv}") # confirm csv file that is being passed into function
    df = pd.read_csv(validated_csv) # read passed csv file

    # 1. keep only valid rows from process_dataset.py
    df = df[df["is_valid"] == True] # filter out invalid or corrupted flows
    print(f"Valid rows: {len(df)}") # number of usable flows for ML

    # print warning if not enough valid flows
    if len(df) < 5:
        print("WARNING: Very few valid flows - ML may be unreliable")

    # 2. drop/ignore columns not useful for ML (kept for dashboard)
    drop_cols = ["src_ip", "dst_ip", "protocol_name", "error_reason", "is_valid"] # do not contain useful information
    df = df.drop(columns=drop_cols, errors="ignore")

    # 3. select numeric fields only (ML can't understand non-numeric fields)
    numeric_df = df.select_dtypes(include=[np.number])
    # confirm numeric features and their amount
    print(f"Numeric features: {list(numeric_df.columns)}")
    print(f"Count: {numeric_df.shape[1]}")

    # 4. check for zero-variance features (column where every value is the same -> no variation -> no valuable information)
    stds = numeric_df.std() # calculate standard deviation (variation from mean) for each column
    zero_var = stds[stds == 0] # zero-variance feature when standard deviation is 0

    if len(zero_var) > 0: # if there are any zero_vars
        print("Zero-variance features detected and removed:")
        print(zero_var.index.tolist()) # print list of zero-variance values
        numeric_df = numeric_df.drop(columns=zero_var.index) # drops (removes) columns with zero-variation values

    # 5. scale values (normalise feature ranges so the model doesn't treat large-number features as more important)
    scaler = StandardScaler() # in-built scikit-learn feature
    scaled = scaler.fit_transform(numeric_df) # calculates mean and std for each column, then returns the scaled version

    # 6. save scaled data as npy file (binary matrix format), ready to be passed to the model to be trained
    npy_file = validated_csv.replace("_validated.csv", "_scaled.npy") # create new .npy file with same base name as the validated csv file
    np.save(npy_file, scaled) # saves the scaled numpy array to npy file

    print(f"Scaled dataset saved to {npy_file}")
    print(f"Shape: {scaled.shape}") # binary matrix preview

    # 7. simple anomaly preview
    model = IsolationForest().fit(scaled) # using isolationforest algorithm -> unsupervised algorithm -> good for isolating unusual points (anomaly detection)
    preds = model.predict(scaled) # apply model to the scaled data and return predicted 1/-1 values (anormal/normal)
    anomalies = np.sum(preds == -1) # print amount of flows (rows) that are flagged as anomaly

    # print amount of flagged flows, usually between 10-50%
    print(f"Anomaly preview: {anomalies}/{len(preds)} flows flagged")

    if anomalies == 0: # no flows flagged
        print("No anomalies detected - dataset may be too uniform")
    elif anomalies > len(preds) * 0.5: # more than half of the flows were flagged
        print("Many anomalies detected - dataset may be unusual")
    else:
        print("Anomaly level within expected range")

    print("Dataset ready for ML") # cleaned and scaled dataset ready to be passed through model

    return npy_file # build_dataset() returns npy_file

# run with build_dataset.py
if __name__ == "__main__":
    script_dir = os.path.dirname(os.path.abspath(__file__)) # get current folder path
    validated_csv = os.path.join(script_dir, "csv_output", "test_flows_validated.csv") # define input and output csv file
    build_dataset(validated_csv) # run build_dataset function on validated csv file
