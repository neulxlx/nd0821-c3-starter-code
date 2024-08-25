"""
Compute metrics by slice of the data.

Author: LX
Date: Aug, 2024
"""
import pandas as pd
from ml.data import process_data
from ml.model import inference, compute_model_metrics

def slice_metrics(model, encoder, lb, data, target, categorical_features, output_path=None):
    """
    Output the performance of the model on slices of the data

    Input
        model: A compatible scikit-learn classifier.
        encoder: A pre-trained one-hot encoder obtained from the process_data function applied to the training dataset.
        lb: A pre-trained label binarizer also derived from the process_data function on training data.
        data: A pandas DataFrame intended for the calculation of metrics across different segments.
        target: The column within the DataFrame that denotes the target variable.
        categorical_features:  List of columns identified as categorical types within the DataFrame.
        output_path: output path to write the output dataframe.

    Return
        metrics_df: pd.DataFrame
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