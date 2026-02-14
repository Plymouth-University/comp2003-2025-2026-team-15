# train the model using the training data from /action_classification
# from .../src/ML run python .\model_training\train_model.py

# using sklearn metrics this produces an accuracy of around 80% on the test set rn

import pandas as pd
import numpy as np
import joblib

# using scikit-learn libraries so we don't have to write our code from scratch
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score

# load dataset created in /action_classification
df = pd.read_csv("action_classification/master_training_data.csv")

# define X and y
# drop non-feature columns
X = df.drop(columns=["action", "file_source"])
# variable we want to predict
y = df["action"]

# encode labels
label_encoder = LabelEncoder() # sklearn's label encoder to convert string labels to integers
y_encoded = label_encoder.fit_transform(y)

# train-test split: https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.train_test_split.html
X_train, X_test, y_train, y_test = train_test_split(
    X, y_encoded, test_size=0.33, random_state=42, stratify=y_encoded
)

# instantiate Random Forest model: https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.RandomForestClassifier.html
model = RandomForestClassifier(
    n_estimators=200, # number of trees -> more trees = better performance but longer training time
    max_depth=None,
    random_state=42
)

model.fit(X_train, y_train)

# evaluate the model by making predictions on the test set and comparing to true labels
y_pred = model.predict(X_test)

# print accuracy and classification report, just a quick check in
print("Accuracy: {:.2f}%".format(accuracy_score(y_test, y_pred) * 100))
print("\nClassification Report:\n") # for information see https://scikit-learn.org/stable/modules/generated/sklearn.metrics.classification_report.html
print(classification_report(y_test, y_pred, target_names=label_encoder.classes_))

# save the model and label encoder
joblib.dump(model, "traffic_model.pkl")
joblib.dump(label_encoder, "label_encoder.pkl")

print("\nModel saved") # meaning .pkl files have been created and can be used now in predict.py