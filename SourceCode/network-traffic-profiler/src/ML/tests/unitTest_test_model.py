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

# ---------------- CLASS IMBALANCE CHECK ---------------- #
# Ensures no class is completely missing in the dataset
def test_class_distribution(sample_data):
    X_train, X_test, y_train, y_test, df, le = sample_data
    class_counts = df["action"].value_counts()
    
    # Check that no class is completely missing
    assert all(class_counts > 0), "Some classes are missing in the dataset"

#  CORE METRICS 
# ---------------- OVERFITTING CHECK ---------------- #
# Checks that the model is not extremely overfitting to training data
def test_train_vs_test_accuracy(prob_model, sample_data):
    X_train, X_test, y_train, y_test, df, le = sample_data
    y_train_pred = prob_model.predict(X_train)
    y_test_pred = prob_model.predict(X_test)

    train_acc = (y_train_pred == y_train).mean()
    test_acc = (y_test_pred == y_test).mean()

    # Check that training accuracy is not extremely higher than test accuracy (overfitting)
    assert train_acc - test_acc < 0.2, f"Potential overfitting: Train {train_acc:.2f}, Test {test_acc:.2f}"

# ---------------- PREDICTION CONFIDENCE DISTRIBUTION ---------------- #
# Ensures predicted probabilities are within 0-1 and some predictions have high confidence
def test_prediction_confidence_distribution(prob_model, sample_data):
    X_train, X_test, y_train, y_test, df, le = sample_data
    if hasattr(prob_model, "predict_proba"):
        probs = prob_model.predict_proba(X_test)
        max_probs = np.max(probs, axis=1)
        # Ensure probabilities are within 0-1
        assert (max_probs >= 0).all() and (max_probs <= 1).all()
        # Ensure some high confidence predictions exist
        assert (max_probs > 0.5).any()

# ---------------- ROC-AUC BRANCH CHECK ---------------- #
# Checks that ROC-AUC score is computed correctly for models with predict_proba
def test_roc_auc_computation(prob_model, sample_data):
    X_train, X_test, y_train, y_test, df, le = sample_data
    if hasattr(prob_model, "predict_proba"):
        from sklearn.preprocessing import label_binarize
        from sklearn.metrics import roc_auc_score
        y_bin = label_binarize(y_test, classes=range(len(le.classes_)))
        y_proba = prob_model.predict_proba(X_test)
        score = roc_auc_score(y_bin, y_proba, multi_class="ovr")
        assert 0 <= score <= 1

# ---------------- CONFUSION MATRIX SHAPE CHECK ---------------- #
# Ensures confusion matrix has shape (n_classes, n_classes)
def test_confusion_matrix_shape(prob_model, sample_data):
    X_train, X_test, y_train, y_test, df, le = sample_data
    from sklearn.metrics import confusion_matrix
    y_pred = prob_model.predict(X_test)
    cm = confusion_matrix(y_test, y_pred)
    assert cm.shape == (len(le.classes_), len(le.classes_))      

# ---------------- PER-CLASS ACCURACY CHECK ---------------- #
# Ensures that per-class accuracy is within valid range [0,1]
def test_per_class_accuracy(tree_model, sample_data):
    X_train, X_test, y_train, y_test, df, le = sample_data
    y_pred = tree_model.predict(X_test)
    for i in range(len(le.classes_)):
        acc = (y_pred[y_test==i] == y_test[y_test==i]).mean()
        assert 0 <= acc <= 1
        
# FEATURE ANALYSIS  #
# ---------------- FEATURE ABLATION EFFECT ---------------- #
# Checks that removing individual features does not completely break model accuracy
def test_feature_ablation_effect(tree_model, sample_data):
    X_train, X_test, y_train, y_test, df, le = sample_data
    from sklearn.base import clone

    base_score = (tree_model.predict(X_test) == y_test).mean()

    for col in X_train.columns:
        X_train_drop = X_train.drop(columns=[col])
        X_test_drop = X_test.drop(columns=[col])
        m = clone(tree_model)
        m.fit(X_train_drop, y_train)
        score = (m.predict(X_test_drop) == y_test).mean()
        # Ablating a feature should not give completely random accuracy
        assert score > 0.3, f"Ablating {col} dropped accuracy too low: {score}"

# ---------------- FEATURE PERMUTATION IMPACT ---------------- #
# Verifies permutation importance returns values for all features
def test_permutation_importance_effect(tree_model, sample_data):
    X_train, X_test, y_train, y_test, df, le = sample_data
    from sklearn.inspection import permutation_importance

    perm_imp = permutation_importance(tree_model, X_test, y_test, n_repeats=5, random_state=42)
    # Ensure permutation importance has values for all features
    assert len(perm_imp.importances_mean) == X_test.shape[1]


# MODEL STABILITY 
# ---------------- STABILITY OVER MULTIPLE SPLITS ---------------- #
# Verifies that repeated train/test splits produce reasonably consistent accuracy
def test_model_stability(tree_model, sample_data):
    X_train, X_test, y_train, y_test, df, le = sample_data
    from sklearn.model_selection import ShuffleSplit
    from sklearn.base import clone

    scores = []
    ss = ShuffleSplit(n_splits=3, test_size=0.3, random_state=42)
    for train_idx, test_idx in ss.split(X_train):
        Xtr, Xte = X_train.iloc[train_idx], X_train.iloc[test_idx]
        ytr, yte = y_train[train_idx], y_train[test_idx]
        m = clone(tree_model)
        m.fit(Xtr, ytr)
        scores.append((m.predict(Xte) == yte).mean())

    # Check that the model is reasonably stable
    assert np.std(scores) < 0.1, f"Model stability issue: std={np.std(scores):.2f}"
