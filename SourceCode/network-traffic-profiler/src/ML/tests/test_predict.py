import pandas as pd
import numpy as np
import pytest
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder

def test_predict_action_type(monkeypatch):
    # import module
    import ML.model_training.predict as predict_module

    # create dataset with sample values
    X_train = pd.DataFrame({
        "duration": [1, 2],
        "std_iat": [0.1, 0.2],
        "avg_iat": [0.3, 0.4],
        "pk_count": [10, 20],
        "avg_inbound_size": [100, 200],
        "avg_outbound_size": [150, 250],
    })

    # create labels, no need to include all as it's a mock model
    y_train = ["Comment", "Play"]

    # train a simplified mock model with the sample dataset
    le = LabelEncoder()
    y_enc = le.fit_transform(y_train)
    model = RandomForestClassifier(n_estimators=5, random_state=42)
    model.fit(X_train, y_enc)

    # monkeypatch model, label encoder, and features (mimics the .pkl files)
    monkeypatch.setattr(predict_module, "model", model)
    monkeypatch.setattr(predict_module, "label_encoder", le)
    monkeypatch.setattr(predict_module, "model_features", list(X_train.columns))

    # create test input
    test_df = X_train.copy()

    # run prediction
    result = predict_module.predict_action_type(test_df)

    # ensure action column is present
    assert "action_type" in result.columns

    # ensure predictions are returned
    assert len(result["action_type"]) == len(test_df)
    assert result["action_type"].notnull().all()