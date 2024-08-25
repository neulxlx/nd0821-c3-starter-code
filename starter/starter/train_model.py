"""
Script to train machine learning model

Author: LX
Date: Aug, 2024
""" 

from sklearn.model_selection import train_test_split
import pandas as pd
import pickle as pkl

from ml.data import process_data
from ml.model import train_model, inference, compute_model_metrics
from ml.model_slicing import slice_metrics
# Add the necessary imports for the starter code.

# Add code to load in the data.

data = pd.read_csv('../data/clean_census.csv')
# Optional enhancement, use K-fold cross validation instead of a train-test split.
train, test = train_test_split(data, test_size=0.20)

cat_features = [
    "workclass",
    "education",
    "marital-status",
    "occupation",
    "relationship",
    "race",
    "sex",
    "native-country",
]
X_train, y_train, encoder, lb = process_data(
    train, categorical_features=cat_features, label="salary", training=True
)

# Proces the test data with the process_data function.
X_test, y_test, encoder, lb = process_data(
    test, categorical_features=cat_features, label="salary", training=False, encoder=encoder, lb=lb
)
# Train and save a model.
clf_model = train_model(X_train, y_train)

with open('../model/classifier.pkl', "wb") as file:
    pkl.dump([encoder, lb, clf_model], file)


train_preds = inference(clf_model, X_train)
test_preds = inference(clf_model, X_test)

precision, recall, fbeta = compute_model_metrics(y_test, test_preds)

print(f"Precicion: {precision:.2f}")
print(f"Recall: {recall:.2f}")
print(f"F1: {fbeta:.2f}")

metrics = slice_metrics(clf_model, encoder, lb, test, "salary", 
            cat_features, "../model/slice_metrics.csv",)