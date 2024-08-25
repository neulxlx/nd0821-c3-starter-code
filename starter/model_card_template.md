# Model Card

For additional information see the Model Card paper: https://arxiv.org/pdf/1810.03993.pdf

## Model Details
The model in this repository is a Random Forest classifier from the scikit-learn library trained with default hyperparameters.

## Intended Use
This model is fitted to predict whether a person makes over 50K a year from a handful of attributes. 
Users can apply this model to their information about the employees and get the prediction for the salary type, for purely educational purposes.

## Training Data
Publicly available Census Bureau dataset is used for training and evaluating the model.
<br>
(https://archive.ics.uci.edu/ml/datasets/census+income)  

The original data set has 32561 rows, with 14 variables: 6 numerical and 8 categorical.

## Evaluation Data
The original dataset is split into training and evaluation sets with a ratio of 4:1.

## Metrics

**Metrics on test data:**
Precision: 0.70
Recall: 0.64
F1: 0.67

## Ethical Considerations

This model exhibits biases related to sex, race, native country, and age, primarily due to imbalances in the dataset for these variables. Consequently, it is crucial to interpret the model's predictions with caution, especially considering the demographic characteristics of the individuals to whom it is applied. 

## Caveats and Recommendations
The model was trained by default parameters. Hence it can be retrained using more accurate hyperparameters.

