# Put the code for your API here.
import pickle as pkl
from fastapi import FastAPI
from pydantic import BaseModel
from starter.ml.data import process_data
from starter.ml.model import inference
import pandas as pd

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

class InputData(BaseModel):
    ''' 
    Input Data class
    '''
    age: int
    workclass: str
    fnlgt: int
    education: str
    education_num: int
    marital_status: str
    occupation: str
    relationship: str
    race: str
    sex: str
    capital_gain: int
    capital_loss: int
    hours_per_week: int
    native_country: str

    class Config:
        ''' 
        Annotation class for Census
         '''
        schema_extra = {
            "example": {
                'age': 39,
                'workclass': 'State-gov',
                'fnlgt': 77516,
                'education': 'Bachelors',
                'education-num': 13,
                'marital-status': 'Never-married',
                'occupation': 'Adm-clerical',
                'relationship': 'Not-in-family',
                'race': 'White',
                'sex': 'Male',
                'capital-gain': 2174,
                'capital-loss': 0,
                'hours-per-week': 40,
                'native-country': 'United-States'
            }
        }

app = FastAPI()

@app.get("/")
async def welcome():
    return {"message": "Welcome to the API!"}

@app.post("/predict")
async def predict(data: InputData):
    with open("./starter/model/classifier.pkl", 'rb') as f:
        encoder, lb, model = pkl.load(f)

    df = pd.DataFrame(
        {k: v for k, v in data.dict().items()}, index=[0]
    )
    df.columns = [_.replace('_', '-') for _ in df.columns]

    X, _, _, _ = process_data(
        X=df,
        label=None,
        training=False,
        categorical_features=cat_features,
        encoder=encoder,
        lb=lb,
    )
    pred = inference(model, X)
    y = lb.inverse_transform(pred)[0]

    return {"predicted salary": y}