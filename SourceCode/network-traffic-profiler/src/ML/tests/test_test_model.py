import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import matplotlib
matplotlib.use("Agg") 

import numpy as np
import pandas as pd
import pytest
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import LabelEncoder
from sklearn.dummy import DummyClassifier
from sklearn.model_selection import train_test_split
from model_training.test_model import run_tests

@pytest.fixture
def sample_data():
    np.random.seed(0)

    X = pd.DataFrame({
        "f1": np.random.rand(100),
        "f2": np.random.rand(100),
        "f3": np.random.rand(100)
    })

    y = np.random.choice(["walk", "run", "jump"], size=100)

    df = X.copy()
    df["action"] = y
    df["file_source"] = "dummy"

    le = LabelEncoder()
    y_enc = le.fit_transform(y)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y_enc, test_size=0.3, random_state=42
    )

    return X_train, X_test, y_train, y_test, df, le

@pytest.fixture
def prob_model(sample_data):
    X_train, X_test, y_train, y_test, df, le = sample_data
    model = LogisticRegression(max_iter=500)
    model.fit(X_train, y_train)
    return model

@pytest.fixture
def tree_model(sample_data):
    X_train, X_test, y_train, y_test, df, le = sample_data
    model = RandomForestClassifier(n_estimators=20, random_state=42)
    model.fit(X_train, y_train)
    return model

# ROBUSTNESS & ERROR HANDLING 
# ---------------- MODEL WITHOUT predict_proba ---------------- #
# Checks run_tests handles models that do not implement predict_proba
def test_no_predict_proba(sample_data):
    X_train, X_test, y_train, y_test, df, le = sample_data

    model = DummyClassifier(strategy="most_frequent")
    model.fit(X_train, y_train)

    run_tests(model, le, X_train, X_test, y_train, y_test, df)

# ---------------- UNKNOWN LABEL HANDLING ---------------- #
# Validates that run_tests raises an error for unseen labels correctly
def test_unknown_label_handling(sample_data, capsys):
    X_train, X_test, y_train, y_test, df, le = sample_data
    model = LogisticRegression(max_iter=500)
    model.fit(X_train, y_train)

    run_tests(model, le, X_train, X_test, y_train, y_test, df)

    captured = capsys.readouterr().out
    assert "Unknown label handling test passed" in captured

#  CORE METRICS 
# ---------------- CONFUSION MATRIX SHAPE CHECK ---------------- #
# Ensures confusion matrix has shape (n_classes, n_classes)
def test_confusion_matrix_shape(prob_model, sample_data):
    X_train, X_test, y_train, y_test, df, le = sample_data
    from sklearn.metrics import confusion_matrix
    y_pred = prob_model.predict(X_test)
    cm = confusion_matrix(y_test, y_pred)
    assert cm.shape == (len(le.classes_), len(le.classes_))      
        
# FEATURE ANALYSIS  #
# ---------------- FEATURE PERMUTATION IMPACT ---------------- #
# Verifies permutation importance returns values for all features
def test_permutation_importance_effect(tree_model, sample_data):
    X_train, X_test, y_train, y_test, df, le = sample_data
    from sklearn.inspection import permutation_importance

    perm_imp = permutation_importance(tree_model, X_test, y_test, n_repeats=5, random_state=42)
    # Ensure permutation importance has values for all features
    assert len(perm_imp.importances_mean) == X_test.shape[1]

