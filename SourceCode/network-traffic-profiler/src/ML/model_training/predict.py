# load the trained model and label encoder,
# extract features from a new pcap file,
# and make a prediction on the segmented flows?
# used in the main pipeline

# uncompleted and probably not working (yet) !! just the general idea

import joblib
from extract_features import extract_ml_features

model = joblib.load("model_training/traffic_model.pkl")
label_encoder = joblib.load("model_training/label_encoder.pkl")

pcap_path = "test.pcap" # pcap file from user input

features = extract_ml_features(pcap_path) # extract features

if features is not None:
    # drop non-feature columns if they exist
    X = features.drop(columns=["action"], errors="ignore")

    pred = model.predict(X) # run model to predict action
    action = label_encoder.inverse_transform(pred) # convert predicted labels to string

    print("Predicted action:", action[0]) # print the predicted action per segmented flow?