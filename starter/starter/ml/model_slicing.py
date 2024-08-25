"""
Compute metrics by slice of the data.

Author: LX
Date: Aug, 2024
"""
import pandas as pd
from ml.data import process_data
from ml.model import inference, compute_model_metrics

def slice_metrics(model, encoder, lb, data, target, categorical_features, output_path=None):
    """Compute metrics by slice of the data.
    Output the performance of the model on slices of the data

    Inputs
    ------
    model: Classifier model (scikit-learn compliant)
    encoder: trained one-hot-encoder (output of the data.process_data function on training data)
    lb: trained label binarizer (output of the data.process_data function on training data)
    df: pandas dataframe where to compute metrics by slice
    target: target column in the df input dataframe
    cat_columns: categorical columns
    output_path: output path to write the output dataframe.

    Returns
    -------
    metrics_df: pd.DataFrame
        Predictions by slice on the input data according the categorical columns.
    """
    columns = ["column", "category", "precision", "recall", "f1"]
    metrics_df = pd.DataFrame(columns=columns)
    rows_list = []

    for feature in categorical_features:
        for category in data[feature].unique():
            df_line = {}
            tmp_df = data[data[feature] == category]
            X, y, _, _ = process_data(
                X=tmp_df,
                categorical_features=categorical_features,
                label=target,
                training=False,
                encoder=encoder,
                lb=lb
            )

            preds = inference(model, X)
            precision, recall, f1 = compute_model_metrics(y, preds)

            df_line['column'] = feature
            df_line['category'] = category
            df_line['precision'] = precision
            df_line['recall'] = recall
            df_line['f1'] = f1

            rows_list.append(df_line)

    # Instead of appending each line, create a DataFrame from the list of dictionaries and concat it
    new_metrics_df = pd.DataFrame(rows_list)
    metrics_df = pd.concat([metrics_df, new_metrics_df], ignore_index=True)

    if output_path is not None:
        metrics_df.to_csv(output_path)

    return metrics_df