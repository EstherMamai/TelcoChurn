# Telco Customer Churn Prediction and Retention Strategy

This project focuses on predicting customer churn for a telecommunications company and developing a retention strategy to reduce churn. The goal is to predict which customers are likely to leave (churn) based on their behavior and characteristics, and implement strategies to retain them.

## Project Overview

Churn prediction is essential for telecommunications companies to identify at-risk customers before they decide to leave. By using machine learning models, we can predict churn and develop retention strategies to proactively target high-risk customers. This project covers data preprocessing, model building, evaluation, and the creation of actionable insights for retention strategies.

## Dataset

The dataset used was obtained from Kaggle, which included information on demographics, transaction history, customer service interactions, subscription details, and whether the customer churned.
Dataset Link: [Telco Customer Churn on Kaggle](https://www.kaggle.com/datasets/blastchar/telco-customer-churn) The dataset used for this project includes customer information such as demographics, services used, and account details. The columns in the dataset include:

- **customerID**: Unique identifier for each customer
- **gender**: Gender of the customer
- **SeniorCitizen**: Whether the customer is a senior citizen (1 = Yes, 0 = No)
- **Partner**: Whether the customer has a partner (Yes/No)
- **Dependents**: Whether the customer has dependents (Yes/No)
- **tenure**: Number of months the customer has been with the company
- **PhoneService**: Whether the customer has phone service (Yes/No)
- **MultipleLines**: Whether the customer has multiple lines (Yes/No)
- **InternetService**: The type of internet service (DSL/Fiber optic/No)
- **OnlineSecurity**: Whether the customer has online security service (Yes/No)
- **OnlineBackup**: Whether the customer has online backup service (Yes/No)
- **DeviceProtection**: Whether the customer has device protection service (Yes/No)
- **TechSupport**: Whether the customer has tech support service (Yes/No)
- **StreamingTV**: Whether the customer has streaming TV service (Yes/No)
- **StreamingMovies**: Whether the customer has streaming movies service (Yes/No)
- **Contract**: Type of contract (Month-to-month, One year, Two year)
- **PaperlessBilling**: Whether the customer has paperless billing (Yes/No)
- **PaymentMethod**: Payment method used (Electronic check, Mailed check, Bank transfer, Credit card)
- **MonthlyCharges**: Monthly charges for the customer
- **TotalCharges**: Total charges paid by the customer
- **Churn**: Whether the customer churned (1 = Yes, 0 = No)

## Technologies Used

- **Python**
- **Libraries**:
  - **pandas** (for data manipulation and analysis)
  - **numpy** (for numerical operations)
  - **matplotlib** & **seaborn** (for data visualization)
  - **scikit-learn** (for machine learning models and evaluation)
  - **XGBoost** (for gradient boosting models)
  - **shap** (for model interpretability)
  - **Flask** or **FastAPI** (for deployment)

## How to Run

### 1. Clone the repository

Clone the repository to your local machine:
```bash
git clone https://github.com/your-username/telco_churn.git
cd telco_churn
```

### 2. Install dependencies

Install the required libraries:
```bash
pip install -r requirements.txt
```

### 3. Run the notebook

Open the `churn_prediction.ipynb` notebook in Jupyter or Google Colab and run the cells sequentially. The notebook includes steps for:

1. **Loading the dataset**
2. **Data preprocessing**
3. **Exploratory data analysis (EDA)**
4. **Modeling (Logistic Regression, Random Forest, XGBoost, etc.)**
5. **Model evaluation** (accuracy, precision, recall, F1-score, ROC-AUC)
6. **Retention strategy development** using the insights from model predictions

### 4. Model Deployment (Optional)

To deploy the model as an API, use **Flask** or **FastAPI**. You can integrate the model into a CRM system to automate the retention strategy. The model file `churn_model.pkl` can be loaded and used for predictions.

## Steps Involved

1. **Data Import and Preprocessing**:  
   Load the dataset and handle missing values, categorical variables, and scale the numerical features.

2. **Exploratory Data Analysis (EDA)**:  
   Visualize key patterns and relationships in the data, such as customer churn distribution, tenure, and service usage.

3. **Model Development**:  
   Train multiple machine learning models (e.g., Logistic Regression, Random Forest, XGBoost) to predict churn.

4. **Model Evaluation**:  
   Evaluate model performance using classification metrics such as accuracy, precision, recall, F1-score, and ROC-AUC.

5. **Retention Strategy**:  
   Develop actionable retention strategies based on the model predictions. This may involve targeting high-risk customers with personalized offers or support.

6. **Deployment**:  
   Optionally, deploy the model as a web API using **Flask** or **FastAPI** to integrate with customer relationship management (CRM) tools or marketing platforms for automated retention actions.

## Model Evaluation Results

- **Accuracy**: 0.79
- **Precision**: 0.72
- **Recall**: 0.82
- **F1-score**: 0.77
- **ROC-AUC**: 0.85

## Conclusion

The model accurately predicts customer churn with a relatively high recall, making it suitable for identifying customers at risk of leaving. The next step involves implementing targeted retention strategies to reduce churn and improve customer retention.

## Future Improvements

- **Feature Engineering**: Add more features based on customer behavior, such as call duration, service usage frequency, etc.
- **Model Tuning**: Hyperparameter tuning using GridSearchCV or RandomizedSearchCV to improve model performance.
- **Cross-validation**: Use k-fold cross-validation for more robust model evaluation.
- **Deployment**: Deploy the model as a web service (Flask or FastAPI) to integrate with CRM or marketing platforms for real-time predictions and retention actions.
