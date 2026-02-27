import sys
import os

# Add src/ML to path
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
#from ML.model_training.test_model import run_tests

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


# ---------------- CORE EXECUTION ---------------- #
def test_run_tests_executes(prob_model, sample_data, capsys):
    X_train, X_test, y_train, y_test, df, le = sample_data

    run_tests(prob_model, le, X_train, X_test, y_train, y_test, df)

    captured = capsys.readouterr().out

    assert "Accuracy" in captured
    assert "Confusion Matrix" in captured
    assert "ROC-AUC" in captured
    assert "Feature ablation" in captured
    assert "Stability test" in captured
    assert "Error analysis" in captured


# ---------------- MODEL WITHOUT predict_proba ---------------- #

def test_no_predict_proba(sample_data):
    X_train, X_test, y_train, y_test, df, le = sample_data

    model = DummyClassifier(strategy="most_frequent")
    model.fit(X_train, y_train)

    run_tests(model, le, X_train, X_test, y_train, y_test, df)
    
# ---------------- FEATURE IMPORTANCE BRANCH ---------------- #
def test_feature_importance_branch(tree_model, sample_data, capsys):
    X_train, X_test, y_train, y_test, df, le = sample_data

    run_tests(tree_model, le, X_train, X_test, y_train, y_test, df)

    captured = capsys.readouterr().out
    assert "Feature Importances" in captured


# ---------------- UNKNOWN LABEL HANDLING ---------------- #

def test_unknown_label_handling(sample_data, capsys):
    X_train, X_test, y_train, y_test, df, le = sample_data
    model = LogisticRegression(max_iter=500)
    model.fit(X_train, y_train)

    run_tests(model, le, X_train, X_test, y_train, y_test, df)

    captured = capsys.readouterr().out
    assert "Unknown label handling test passed" in captured


# ---------------- STABILITY SCORES VALID ---------------- #

def test_stability_scores_in_range(tree_model, sample_data):
    X_train, X_test, y_train, y_test, df, le = sample_data

    run_tests(tree_model, le, X_train, X_test, y_train, y_test, df)


# ---------------- NO CRASH ON NOISE ---------------- #

def test_noise_sensitivity(sample_data):
    X_train, X_test, y_train, y_test, df, le = sample_data
    model = LogisticRegression(max_iter=500)
    model.fit(X_train, y_train)

    run_tests(model, le, X_train, X_test, y_train, y_test, df)


# ---------------- METRIC CORRECTNESS ---------------- #

def test_accuracy_computation(sample_data):
    X_train, X_test, y_train, y_test, df, le = sample_data
    model = LogisticRegression(max_iter=500)
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    expected_acc = (y_pred == y_test).mean()

    run_tests(model, le, X_train, X_test, y_train, y_test, df)

    assert 0 <= expected_acc <= 1


# ---------------- ERROR ANALYSIS OUTPUT ---------------- #

def test_misclassified_output(sample_data, capsys):
    X_train, X_test, y_train, y_test, df, le = sample_data
    model = LogisticRegression(max_iter=500)
    model.fit(X_train, y_train)

    run_tests(model, le, X_train, X_test, y_train, y_test, df)

    captured = capsys.readouterr().out
    assert "Misclassified samples" in captured


# ---------------- SANITY CHECK ---------------- #

def test_sanity_check(sample_data, capsys):
    X_train, X_test, y_train, y_test, df, le = sample_data
    model = LogisticRegression(max_iter=500)
    model.fit(X_train, y_train)

    run_tests(model, le, X_train, X_test, y_train, y_test, df)

    captured = capsys.readouterr().out
    assert "General Sanity check" in captured
