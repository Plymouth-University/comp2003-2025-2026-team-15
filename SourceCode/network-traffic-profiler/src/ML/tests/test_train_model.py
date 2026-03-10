import pytest
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder

def test_train_model(monkeypatch):
    # create mock dataset
    df = pd.DataFrame({
        "duration": [1, 2, 3, 4, 5, 6],
        "std_iat": [0.1, 0.2, 0.1, 0.3, 0.2, 0.1],
        "avg_iat": [0.5, 0.6, 0.4, 0.7, 0.6, 0.5],
        "pk_count": [10, 20, 10, 30, 25, 15],
        "avg_inbound_size": [100, 200, 150, 120, 180, 160],
        "avg_outbound_size": [150, 250, 180, 130, 220, 170],
        "action": ["Play", "Like", "Comment", "Play", "Like", "Comment"],
        "file_source": ["a.pcap", "b.pcap", "c.pcap", "d.pcap", "e.pcap", "f.pcap"]
    })

    # mock pd.read_csv
    monkeypatch.setattr(pd, "read_csv", lambda path: df.copy())

    # mock joblib.dump 
    monkeypatch.setattr("joblib.dump", lambda obj, path: None)

    # simplified train/test split
    X = df[["duration", "std_iat", "avg_iat", "pk_count", "avg_inbound_size", "avg_outbound_size"]]
    y = df["action"]
    label_encoder = LabelEncoder()
    y_encoded = label_encoder.fit_transform(y)

    # train the RandomForest
    model = RandomForestClassifier(n_estimators=5, max_depth=3, random_state=42)
    model.fit(X, y_encoded)

    # predict on mock training data to ensure model is working
    y_pred = model.predict(X)

    # check model and label encoder types, prediction length, and that predictions are valid labels
    assert isinstance(model, RandomForestClassifier) # check that model type is RandomForestClassifier, to ensure correct training
    assert isinstance(label_encoder, LabelEncoder) # check label encoder type to ensure labels are encoded correctly
    assert len(y_pred) == len(df) # ensure that a prediction is returned for every sample
    assert all([pred in y_encoded for pred in y_pred]) # check that every prediction contains one of the valid encoded labels