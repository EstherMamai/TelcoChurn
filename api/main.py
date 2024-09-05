from fastapi import FastAPI
from pydantic import BaseModel
import pandas as pd
import joblib
import numpy as np

app = FastAPI()

# Load the trained model and encoders
model = joblib.load("model\churn_model.pkl")
one_hot_columns = joblib.load('encoder\one_hot_columns.pkl')  # Columns from one-hot encoding
tenure_encoder = joblib.load('encoder\tenure_encoder.pkl')  # Encoder for scaling tenure
label_encoder = joblib.load('encoder\label_encoder.pkl')  # Label encoder for other categorical features

# List of label encoded columns
label_encoded_cols = ['Churn']
one_hot_cols =['gender', 'Partner', 'Dependents', 'PhoneService', 'MultipleLines', 'InternetService', 'OnlineSecurity', 'OnlineBackup', 'DeviceProtection',
                'TechSupport', 'StreamingTV', 'StreamingMovies', 'Contract', 'PaperlessBilling', 'PaymentMethod']  # Columns to one-hot encode

# Define the input data model
class CustomerData(BaseModel):
    customerID: str
    gender: str
    SeniorCitizen: int  # 1 for Yes, 0 for No
    Partner: str        # 'Yes' or 'No'
    Dependents: str     # 'Yes' or 'No'
    tenure: float       # Number of months customer has been with the company
    PhoneService: str   # 'Yes' or 'No'
    MultipleLines: str  # 'Yes', 'No', 'No phone service'
    InternetService: str  # 'DSL', 'Fiber optic', 'No'
    OnlineSecurity: str  # 'Yes', 'No', 'No internet service'
    OnlineBackup: str    # 'Yes', 'No', 'No internet service'
    DeviceProtection: str  # 'Yes', 'No', 'No internet service'
    TechSupport: str     # 'Yes', 'No', 'No internet service'
    StreamingTV: str     # 'Yes', 'No', 'No internet service'
    StreamingMovies: str # 'Yes', 'No', 'No internet service'
    Contract: str        # 'Month-to-month', 'One year', 'Two year'
    PaperlessBilling: str  # 'Yes' or 'No'
    PaymentMethod: str   # 'Electronic check', 'Mailed check', 'Bank transfer (automatic)', 'Credit card (automatic)'
    MonthlyCharges: float # Monthly bill amount
    TotalCharges: float  # Total charges to date

@app.post("/predict")
async def predict(data: CustomerData):
    # Convert the incoming data to a DataFrame
    input_data = pd.DataFrame([data.dict()])

    # Apply label encoding to relevant columns
    for col in label_encoded_cols:
        input_data[col] = label_encoder[col].transform(input_data[col])

    # Scale the tenure feature using the tenure encoder
    input_data['tenure'] = tenure_encoder.transform(input_data[['tenure']])

    # Apply one-hot encoding to the relevant columns
    input_data_encoded = pd.get_dummies(input_data, columns=one_hot_cols, drop_first=True)

    # Align the encoded data with the training data columns
    input_data_encoded = input_data_encoded.reindex(columns=one_hot_columns, fill_value=0)

    # Make the prediction
    prediction = model.predict(input_data_encoded)

    return {"prediction": int(prediction[0])}
