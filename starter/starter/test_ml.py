"""
Tests for the model training.

Author: LX
Date: Aug, 2024
"""

import os
import pickle as pkl

import pytest
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.base import BaseEstimator, ClassifierMixin

from starter.ml.data import process_data
from starter.ml.model import train_model, inference, compute_model_metrics


@pytest.fixture()
def input_df():

    df = pd.read_csv('./starter/data/clean_census.csv')
    train, test = train_test_split(df, test_size=0.2)
    return train, test

@pytest.fixture()
def cat_features():
    return [
    "workclass",
    "education",
    "marital-status",
    "occupation",
    "relationship",
    "race",
    "sex",
    "native-country",
    ]


def test_process_data(input_df, cat_features):
    train, test = input_df
    X_train, y_train, encoder, lb = process_data(
       train, categorical_features=cat_features, label="salary", training=True
    )

    # Test the number of rows for training dataset
    assert len(X_train) == len(y_train)

    X_test, y_test, encoder_test, lb_test = process_data(
        train, categorical_features=cat_features, label="salary", training=False, encoder=encoder, lb=lb
    )

    # Test the number of generated features for train and test datasets
    assert X_train.shape[1] == X_test.shape[1]


def test_train_model(input_df, cat_features):
    train, _ = input_df

    X_train, y_train, encoder, lb = process_data(
        X=train, categorical_features=cat_features, label="salary", training=True
    )

    clf_model = train_model(X_train, y_train)
    assert isinstance(clf_model, BaseEstimator) and isinstance(clf_model, ClassifierMixin)


def test_compute_metrics(input_df, cat_features):

    train, test = input_df

    X_train, y_train, encoder, lb = process_data(
        X=train, categorical_features=cat_features, label="salary", training=True
    )

    clf_model = train_model(X_train, y_train)

    preds = inference(clf_model, X_train)

    precision, recall, fbeta = compute_model_metrics(y_train, preds)

    assert precision <= 1.0 and recall <= 1.0 and fbeta <= 1.0
