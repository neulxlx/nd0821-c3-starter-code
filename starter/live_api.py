import json
import requests

app_url = "https://salary-prediction-h105.onrender.com"
# app_url = "http://127.0.0.1:8000"

# Test the GET method
request_get = requests.get(app_url)
assert request_get.status_code == 200
print(request_get.json())

# Test the POST method
data = {
        'age': 39,
        'workclass': 'State-gov',
        'fnlgt': 77516,
        'education': 'Bachelors',
        'education_num': 13,
        'marital_status': 'Never-married',
        'occupation': 'Adm-clerical',
        'relationship': 'Not-in-family',
        'race': 'White',
        'sex': 'Male',
        'capital_gain': 2174,
        'capital_loss': 0,
        'hours_per_week': 40,
        'native_country': 'United-States'
}
request_post = requests.post(f"{app_url}/predict", json=data)

assert request_post.status_code == 200
print(request_post.json())