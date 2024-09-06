# Telco Customer Churn Prediction and Retention Strategy
**Objective:**
Develop a machine learning model to predict Telco customer churn (i.e., customers likely to stop using a product or service) and devise strategies to reduce churn rates based on the findings.

**1. Data Collection**
The dataset used was obtained from Kaggle, which included information on demographics, transaction history, customer service interactions, subscription details, and whether the customer churned.
Dataset Link: [Telco Customer Churn on Kaggle](https://www.kaggle.com/datasets/blastchar/telco-customer-churn)
Data columns (total 21 columns):
 #   Column            Non-Null Count  Dtype  
---  ------            --------------  -----  
 0   customerID        7043 non-null   object 
 1   gender            7043 non-null   object 
 2   SeniorCitizen     7043 non-null   int64  
 3   Partner           7043 non-null   object 
 4   Dependents        7043 non-null   object 
 5   tenure            7043 non-null   int64  
 6   PhoneService      7043 non-null   object 
 7   MultipleLines     7043 non-null   object 
 8   InternetService   7043 non-null   object 
 9   OnlineSecurity    7043 non-null   object 
 10  OnlineBackup      7043 non-null   object 
 11  DeviceProtection  7043 non-null   object 
 12  TechSupport       7043 non-null   object 
 13  StreamingTV       7043 non-null   object 
 14  StreamingMovies   7043 non-null   object 
 15  Contract          7043 non-null   object 
 16  PaperlessBilling  7043 non-null   object 
 17  PaymentMethod     7043 non-null   object 
 18  MonthlyCharges    7043 non-null   float64
 19  TotalCharges      7043 non-null   object 
 20  Churn             7043 non-null   object 
dtypes: float64(1), int64(2), object(18)

## 2. Data Preprocessing
**Missing values**
There were missing values in the data. Since the columns were all numerical the missing values were replaced with the median.
**Encoding categorical variables**
The customer id column is a unique identifier and was thus dropped from the dataset as it is not useful in the modelling process.
Two types of encoding were used, one-hot encoding and label encoding. ONe-Hot encoding is normally used where there is no inherent order between values in a category. Label encoding on the other hand was used for the churn column since it is the target variable. "Yes" was encoded as 1 and "No" as 0.

**One-Hot Encoding:** gender, Partner, Dependents, PhoneService, MultipleLines, InternetService, OnlineSecurity, OnlineBackup, DeviceProtection, TechSupport, StreamingTV, StreamingMovies, Contract, PaperlessBilling, PaymentMethod
**Label Encoding:** Churn (as the target variable)
