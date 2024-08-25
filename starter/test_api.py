import pytest
from fastapi.testclient import TestClient
from main import app


# Instantiate the testing client
@pytest.fixture
def client():
    api_client = TestClient(app)
    return api_client


# Write tests using the same syntax as with the requests' module.
def test_get(client):
    result = client.get("/")
    assert result.status_code == 200
    assert result.json() == {'message': 'Welcome to the API!'}


def test_api_predict_less_or_equal_50K(client):

    result = client.post("/predict", json={
        "age": 39,
        "fnlgt": 77516,
        "workclass": "State-gov",
        "education": "Bachelors",
        "education_num": 13,
        "marital_status": "Never-married",
        "occupation": "Adm-clerical",
        "relationship": "Not-in-family",
        "race": "White",
        "sex": "Male",
        "capital_gain": 2174,
        "capital_loss": 0,
        "hours_per_week": 40,
        "native_country": "United-States"
    })

    assert result.status_code == 200
    assert result.json() == {"predicted salary": "<=50K"}


def test_api_predict_greater_50K(client):

    r = client.post("/predict", json={
        "age": 52,
        "fnlgt": 287927,
        "workclass": "Self-emp-inc",
        "education": "HS-grad",
        "education_num": 9,
        "marital_status": "Married-civ-spouse",
        "occupation": "Exec-managerial",
        "relationship": "Wife",
        "race": "White",
        "sex": "Female",
        "capital_gain": 15024,
        "capital_loss": 0,
        "hours_per_week": 40,
        "native_country": "United-States"
    })

    assert r.status_code == 200
    assert r.json() == {"predicted salary": ">50K"}
