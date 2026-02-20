"""
Tests conducted on the trained model to evaluate its performance and robustness:
Core performance metrics : Accuracy, F1-score, Classification report, Confusion matrix, ROC-AUC, Per-class accuracy
Feature analysis : Feature importance, Permutation importance, Feature ablation/usefulness
Model robustness & reliability : Overfitting check, Class imbalance/distribution, Prediction confidence, Noise sensitivity, Stability test
Error & unseen label analysis : Error analysis: Inspect misclassified samples, Unknown/unseen label handling
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import ShuffleSplit
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, f1_score, roc_auc_score, RocCurveDisplay
from sklearn.base import clone
from sklearn.inspection import permutation_importance
from sklearn.preprocessing import label_binarize
import matplotlib.pyplot as plt

"""
Parameters:
model: trained ML model
label_encoder: label encoder used to encode targets
X_train, X_test, y_train, y_test: train/test splits
df: original DataFrame (needed for some tests)
"""
    
def run_tests(model, label_encoder, X_train, X_test, y_train, y_test, df):

    print("\n--- Running Model Evaluation ---\n")
    
    y_pred = model.predict(X_test) # predict on test set

    print("--- Core performance metrics ---\n")
    # how well the model works overall

    # 1) Accuracy: % of correct predictions (overall model performance)
    acc = accuracy_score(y_test, y_pred)
    print(f"[Test 1] Accuracy: {acc*100:.2f}%\n")

    # 2) F1-score: weighted balance of precision and recall
    f1 = f1_score(y_test, y_pred, average='weighted')
    print(f"[Test 2] Weighted F1-score: {f1*100:.2f}%\n")

    # 3) Classification report # Shows model performance for each action
    # Consist of precision, recall, F1-score, and support for each action type and overall averages(macro and weighted)
    # Confusion matrix and classification report together help identify which action types are predicted well and which the model struggles with
    print("[Test 3] Classification Report:")
    print(classification_report(y_test, y_pred, target_names=label_encoder.classes_))

    # 4) Confusion matrix: To view which classes are misclassified
    cm = confusion_matrix(y_test, y_pred)
    print("[Test 4] Confusion Matrix:")
    print(cm)
    
    # Confusion matrix figure
    plt.figure(figsize=(6,5))
    plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
    plt.title("Confusion Matrix")
    plt.colorbar()
    plt.xticks(range(len(label_encoder.classes_)), label_encoder.classes_, rotation=45)
    plt.yticks(range(len(label_encoder.classes_)), label_encoder.classes_)
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.show()

    # 5) ROC-AUC: How well the model distinguishes between classes using probabilities 
    print("\n[Test 5] ROC-AUC Score:")
    if hasattr(model, "predict_proba"):
        try:
            y_bin = label_binarize(y_test, classes=range(len(label_encoder.classes_)))
            y_proba = model.predict_proba(X_test) # Get predicted probabilities for each class 
        
            # Compute overall ROC-AUC (One vs Rest) 
            roc_auc = roc_auc_score(y_bin, y_proba, multi_class="ovr")
            print(f"ROC-AUC Score (multi-class, OVR): {roc_auc:.4f}")
        
            # ROC curves for all classes
            plt.figure(figsize=(8,6))
            for i, class_name in enumerate(label_encoder.classes_):
                RocCurveDisplay.from_predictions(
                    y_bin[:, i],         # true labels for class i
                    y_proba[:, i],       # predicted probabilities for class i
                    name=class_name,
                    ax=plt.gca()         # plot on same axes
                )
            plt.title("ROC Curves (One-vs-Rest)")
            plt.xlabel("False Positive Rate")
            plt.ylabel("True Positive Rate")
            plt.legend(loc="lower right")
            plt.grid(True)
            plt.show()
        
        except Exception as e:
            print(f"ROC-AUC not computed: {e}")
            
        # uncomment for ROC curve for each class separately 
        '''
            for i, class_name in enumerate(label_encoder.classes_):
                plt.figure()
                RocCurveDisplay.from_predictions(
            y_bin[:, i],
            y_proba[:, i],
            name=class_name
            )
            plt.title(f"ROC Curve: {class_name}")
            plt.show()
        '''
        
    # 6) Per-class accuracy: accuracy for each class (well or poorly predicted: where to improve data collection)
    print("\n[Test 6] Per-class accuracy:")
    for i, label in enumerate(label_encoder.classes_):
        idx = (y_test == i)
        class_acc = accuracy_score(y_test[idx], y_pred[idx])
        print(f"{label}: {class_acc*100:.2f}%")

    print("\n\n--- Feature analysis ---")       
    # Understanding what drives predictions and model behavior
     
    # 7) Feature importance: showing which features are key drivers
    if hasattr(model, "feature_importances_"):
        print("\n[Test 7] Top Feature Importances:")
        feat_importances = pd.Series(model.feature_importances_, index=X_train.columns).sort_values(ascending=False)
        print(feat_importances)

        # Feature importance figure
        plt.figure(figsize=(8,5))
        plt.barh(feat_importances.index, feat_importances.values)
        plt.title("Feature Importance")
        plt.xlabel("Importance")
        plt.show()

    # 8) Permutation importance: # measures what happens to accuracy if we shuffle a feature
    # performance drop when feature is shuffled, confirming critical features
    perm_imp = permutation_importance(model, X_test, y_test, n_repeats=10, random_state=42)
    perm_importances = pd.Series(perm_imp.importances_mean, index=X_test.columns).sort_values(ascending=False)
    print("\n[Test 8] Permutation Importance:")
    print(perm_importances)
    
    # Permutation importance figure
    plt.figure(figsize=(8,5))
    plt.barh(perm_importances.index, perm_importances.values)
    plt.title("Permutation Importance")
    plt.xlabel("Importance")
    plt.show()
    
    # 9) Feature ablation/usefulness: removing each feature and retraining to see its impact on model accuracy
    # may need to update/ optimise if more features are added to the dataset 
    base_score = accuracy_score(y_test, y_pred)
    print("\n[Test 9] Feature ablation results:")
    print(f"Baseline accuracy: {base_score*100:.2f}%")

    for col in X_train.columns:
        X_train_drop = X_train.drop(columns=[col])
        X_test_drop = X_test.drop(columns=[col])

        m = clone(model)
        m.fit(X_train_drop, y_train)
        score = accuracy_score(y_test, m.predict(X_test_drop))
        print(f"Without {col}: {score*100:.2f}%")

    print("\n\n--- Model robustness & reliability ---")       
    # Check stability, overfitting, confidence, noise sensitivity
 
    # 10) Overfitting Check: compare train vs test accuracy
    print("\n[Test 10] Overfitting check:")
    train_acc = accuracy_score(y_train, model.predict(X_train))
    test_acc = accuracy_score(y_test, y_pred)
    print(f"\nTraining Accuracy: {train_acc*100:.2f}%")
    print(f"Testing Accuracy:  {test_acc*100:.2f}%")

    # 11) Class imbalance/distribution: # check if classes are balanced
    print("\n[Test 11] Class distribution:")
    print(f"Total samples: {len(df)}") # Total sameples
    print(df["action"].value_counts()) 
    
    # 12) Prediction confidence (first 10 rows): to show how certain the model is per sample
    if hasattr(model, "predict_proba"):
        probs = model.predict_proba(X_test)
        max_probs = np.max(probs, axis=1)  # confidence per sample
        bins = [0, 0.25, 0.5, 0.75, 1.0] # ranges for low, medium, high confidence
        labels = ["Very Low (0-0.25)", "Low (0.25-0.5)", "Medium (0.5-0.75)", "High (0.75-1.0)"]
        counts, _, _ = plt.hist(max_probs, bins=bins, edgecolor='black', color='skyblue', rwidth=0.9)

        plt.title("Prediction Confidence Distribution")
        plt.xlabel("Prediction Confidence")
        plt.ylabel("Number of Samples")
        plt.xticks([(bins[i] + bins[i+1])/2 for i in range(len(bins)-1)], labels, rotation=45)
    
        for i, count in enumerate(counts):
            plt.text((bins[i]+bins[i+1])/2, count + 1, str(int(count)), ha='center', va='bottom', fontsize=10)
    
        plt.ylim(0, max(counts)*1.2)  # add space for text
        plt.tight_layout()
        plt.show()

        # Print first 10 sample confidences
        print("\n[Test 12]Prediction confidence (first 10 rows):")
        for i in range(min(10, len(max_probs))):
            print(f"Sample {i}: {max_probs[i]:.3f}")
        
    # 13) Noise sensitivity: adding small noise and checking accuracy to measure model robustness / stability to minor variations 
    print("\n[Test 13] Noise sensitivity test:")
    for noise_level in [0.005, 0.01, 0.05]:  # 0.5%, 1%, 5%
        noise = np.random.normal(0, noise_level, X_test.shape)
        X_noisy = X_test * (1 + noise) # scale features slightly
        noisy_acc = accuracy_score(y_test, model.predict(X_noisy)) # Make predictions on noisy data & Compare accuracy to original test set
        print(f"Noise {noise_level*100:.1f}% -> Accuracy: {noisy_acc*100:.2f}%")
    
    # 14) Stability test: repeated splits to check model consistency of predictions
    print("\n[Test 14] Stability test:") 
    scores = []
    ss = ShuffleSplit(n_splits=5, test_size=0.33, random_state=42)
    for train_idx, test_idx in ss.split(df.drop(columns=["action", "file_source"])):
        Xtr, Xte = df.drop(columns=["action", "file_source"]).iloc[train_idx], df.drop(columns=["action", "file_source"]).iloc[test_idx]
        ytr, yte = label_encoder.transform(df["action"].iloc[train_idx]), label_encoder.transform(df["action"].iloc[test_idx])
        m = clone(model)
        m.fit(Xtr, ytr)
        scores.append(accuracy_score(yte, m.predict(Xte)))
    print("\nStability scores:", scores)
    print("Mean stability:", np.mean(scores))
    
    print("\n\n--- Error & unseen label analysis ---")       
    # Identify mistakes and unseen situations
 
    # 15) Error analysis: Inspect misclassified samples
    print("\n[Test 15] Error analysis:")
    errors = X_test.copy()
    errors["true"] = label_encoder.inverse_transform(y_test)
    errors["pred"] = label_encoder.inverse_transform(y_pred)
    misclassified = errors[errors["true"] != errors["pred"]]
    print("\nMisclassified samples:")
    print(misclassified.head(20)) # the first 20 misclassified rows
    # print(misclassified) # all misclassified rows
    print(f"Total misclassified: {len(misclassified)}") # Total number of misclassified samples

    # 16) Unknown/unseen label handling
    print("\n[Test 16] Testing model handling of unseen labels:")
    try:
        label_encoder.transform(["new_action"])
    except ValueError as e:
        print("\nUnknown label handling test passed.\nModel correctly raises error for unseen class:")
        print("Error:", e)
        
    # 17) Quick Sanity check
    print("\n[test 17] General Sanity check (first 5 rows of features):")
    print(X_test.head())
    
    print("\n--- Testing complete ---")