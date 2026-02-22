# train the model using the training data from /action_classification
# from .../src/ML run python .\model_training\train_model.py

# using sklearn metrics this produces an accuracy of around 75% on the test set rn
# model doesn't perform well on like and subscribe

import pandas as pd
import numpy as np
import joblib

# using scikit-learn libraries so we don't have to write our code from scratch
from sklearn.model_selection import GroupKFold
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score
from test_model import run_tests
from sklearn.model_selection import GroupShuffleSplit, cross_val_score

# load dataset created in /action_classification
df = pd.read_csv("master_training_data.csv")

# Strip excess background flows 
# Create dataframe of action flows
actions_df = df[df['action'] != 'Background']
# Create dataframe of background flows
background_df = df[df['action'] == 'Background'].sample(n=150, random_state=42)

# Combine them back
df = pd.concat([actions_df, background_df])
print(f"New balanced dataset shape: {df['action'].value_counts()}")

# define X and y, features with low permutation importance are not included to reduce overfitting
X = df[["duration", "std_iat", "avg_iat", "pk_count", "avg_inbound_size", "avg_outbound_size"]]
y = df["action"]
groups = df["file_source"]

# encode labels
label_encoder = LabelEncoder() # sklearn's label encoder to convert string labels to integers
y_encoded = label_encoder.fit_transform(y)

# train-test split with group shuffle to reduce data leakage and overfitting
groups = df["file_source"]

# use GroupShuffleSplit to create a single train/test split
gss = GroupShuffleSplit(n_splits=1, test_size=0.33, random_state=42)
train_idx, test_idx = next(gss.split(X, y_encoded, groups))

X_train = X.iloc[train_idx]
X_test = X.iloc[test_idx]
y_train = y_encoded[train_idx]
y_test = y_encoded[test_idx]

# instantiate Random Forest model: https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.RandomForestClassifier.html
model = RandomForestClassifier(
    n_estimators=700, # number of trees -> more trees = better performance but longer training time
    max_depth=12, # limit depth to reduce overfitting
    min_samples_split=15, # require more samples to split a node, reduces overfitting
    min_samples_leaf=8, # avoid tiny leaves, reduces overfitting
    class_weight="balanced",
    random_state=42,
    n_jobs=-1
)

# cross validation to reduce overfitting and bias
gkf = GroupKFold(n_splits=5)

cv_scores = cross_val_score(
    model,
    X,
    y_encoded,
    cv=gkf,
    groups=groups,
    scoring="accuracy",
    n_jobs=-1
)

# train model
model.fit(X_train, y_train)

# evaluate the model by making predictions on the test set and comparing to true labels
y_pred = model.predict(X_test)

# print accuracy and classification report, just a quick check in
print("Accuracy: {:.2f}%".format(accuracy_score(y_test, y_pred) * 100))
print("\nClassification Report:\n") # for information see https://scikit-learn.org/stable/modules/generated/sklearn.metrics.classification_report.html
print(classification_report(y_test, y_pred, target_names=label_encoder.classes_))

# save the model, label encoder and model features ensuring features are in correct order
joblib.dump(model, "action_model.pkl")
joblib.dump(label_encoder, "label_encoder.pkl")
joblib.dump(X.columns.tolist(), "model_features.pkl")

print("\nModel saved") # meaning .pkl files have been created and can be used now in predict.py

run_tests(model=model, label_encoder=label_encoder, X_train=X_train, X_test=X_test, y_train=y_train, y_test=y_test, df=df)
