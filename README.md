# Telco Customer Churn Prediction and Retention Strategy
#### Objective:
Develop a machine learning model to predict Telco customer churn (i.e., customers likely to stop using a product or service) and devise strategies to reduce churn rates based on the findings.

#### 1. Data Collection
The dataset used was obtained from Kaggle, which included information on demographics, transaction history, customer service interactions, subscription details, and whether the customer churned.
Dataset Link: [Telco Customer Churn on Kaggle](https://www.kaggle.com/datasets/blastchar/telco-customer-churn)
#### 2. Libraries and Data Loading
We begin by importing the necessary Python libraries for data manipulation, visualization, and modeling:
```python
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import scipy.stats as stats
```
Then, the Telco customer dataset is loaded:
```python
df = pd.read_csv('path/to/dataset.csv')
```
## 3. Data Preprocessing
**Missing values**
There were missing values in the data. Since the columns were all numerical the missing values were replaced with the median.
**Encoding categorical variables**
The customer id column is a unique identifier and was thus dropped from the dataset as it is not useful in the modelling process.
Two types of encoding were used, one-hot encoding and label encoding. ONe-Hot encoding is normally used where there is no inherent order between values in a category. Label encoding on the other hand was used for the churn column since it is the target variable. "Yes" was encoded as 1 and "No" as 0.

**One-Hot Encoding:** gender, Partner, Dependents, PhoneService, MultipleLines, InternetService, OnlineSecurity, OnlineBackup, DeviceProtection, TechSupport, StreamingTV, StreamingMovies, Contract, PaperlessBilling, PaymentMethod
**Label Encoding:** Churn (as the target variable)

**Scaling Numerical Features**
Numerical columns such as **tenure**, **MonthlyCharges**, and **TotalCharges** often need to be scaled so that features are on a comparable scale. This is especially important for models that rely on distance calculations, such as Logistic Regression or K-Nearest Neighbors.

In this case, we used **StandardScaler** to standardize the features by removing the mean and scaling to unit variance:

```python
from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()
df[['tenure', 'MonthlyCharges', 'TotalCharges']] = scaler.fit_transform(df[['tenure', 'MonthlyCharges', 'TotalCharges']])
```

**Train-Test Split**

To evaluate the model’s performance, the dataset is split into training and testing sets. We typically use an 80/20 or 70/30 split to train the model on one part of the data and test it on the other:

```python
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X_res, y_res, test_size=0.2, random_state=42)
```

This ensures that the model is trained on a representative sample of the data and tested on unseen data to assess generalization.





## 3. Exploratory Data Analysis (EDA)
EDA was then performed on the data. From the EDA, the folliwng insights were observed:
**Churn** 
0    73.421502
1    26.578498
**Skew**
SeniorCitizen
Skew : 1.83

tenure
Skew : 0.24
MonthlyCharges
Skew : -0.22
TotalCharges
Skew : 0.96
Churn
Skew : 1.06
Tenure_MonthlyCharges
Skew : 0.96
AvgMonthlyCharge
Skew : -0.21
Log_TotalCharges
Skew : -0.74
TenureGroup
Skew : 0.23

**Here's a summary of the key findings:**

**Variable Distributions**

* **SeniorCitizen:** A categorical variable with a majority of individuals not being senior citizens.
* **tenure:** A numerical variable skewed to the right, indicating a larger proportion of customers with shorter tenures.
* **MonthlyCharges:** A numerical variable skewed to the right, indicating a larger proportion of customers with lower monthly charges.
* **TotalCharges:** A numerical variable skewed to the right, indicating a larger proportion of customers with lower total charges.

**General Observations**

* **Outliers:** No outliers were found in any of the variables.
* **Distribution Patterns:** All variables exhibited a right-skewed distribution, suggesting a concentration of values at the lower end of the range.
* **Median Values:** The median values for tenure, monthly charges, and total charges were relatively moderate, suggesting a central tendency towards the middle of the range.
* **Percentile Ranges:** The box plots provided insights into the spread of the data. For example, the box for tenure showed a significant portion of customers with tenures between 15 and 45 months.

**Overall, the analysis indicates that the dataset contains a customer base with a mix of short- and long-tenured customers, with a general tendency towards lower monthly and total charges. The lack of outliers suggests a relatively consistent distribution of these variables.**

A pairplot was used to find the relationships between each of the numerical variables
**Key Observations:**

* **SeniorCitizen:** This variable is categorical, and its distribution is shown on the diagonal. The majority of observations have a value of 0 (not a senior citizen), with a smaller number having a value of 1 (senior citizen).
* **tenure:** This variable is numerical and has a right-skewed distribution, indicating a larger proportion of customers with shorter tenures.
* **MonthlyCharges:** This variable is also numerical and has a right-skewed distribution, suggesting a larger proportion of customers with lower monthly charges.
* **TotalCharges:** This variable is numerical and also has a right-skewed distribution, similar to MonthlyCharges, indicating a larger proportion of customers with lower total charges.

**Relationships Between Variables:**

* **SeniorCitizen vs. other variables:** There is no clear relationship between SeniorCitizen and the other variables. The scatter plots show a random distribution of points.
* **tenure vs. other variables:** There appears to be a positive correlation between tenure and MonthlyCharges. As tenure increases, MonthlyCharges tend to increase as well. Similarly, there is a positive correlation between tenure and TotalCharges, suggesting that customers with longer tenures have higher total charges.
* **MonthlyCharges vs. TotalCharges:** There is a strong positive correlation between MonthlyCharges and TotalCharges, as expected. As MonthlyCharges increase, TotalCharges also increase.

**Overall, the pair plot provides a visual representation of the relationships between the variables. While there is a clear positive correlation between tenure and both MonthlyCharges and TotalCharges, there is no significant relationship between SeniorCitizen and the other variables.**

**Correlation Analysis**
**Key Observations:**

* **SeniorCitizen:** This variable has a weak positive correlation with tenure (0.017) and a very weak positive correlation with MonthlyCharges (0.1) and TotalCharges (0.22). This suggests that being a senior citizen has a minimal impact on tenure, monthly charges, and total charges.
* **tenure:** This variable has a moderate positive correlation with MonthlyCharges (0.25) and a strong positive correlation with TotalCharges (0.83). This indicates that customers with longer tenures tend to have higher monthly charges and total charges.
* **MonthlyCharges:** This variable has a strong positive correlation with TotalCharges (0.65), as expected. Customers with higher monthly charges will also have higher total charges.

**Overall, the correlation matrix confirms the findings from the pair plot:**

* There is a strong positive relationship between tenure and both MonthlyCharges and TotalCharges.
* There is no significant relationship between SeniorCitizen and the other variables.

**In conclusion, the analysis indicates that customer tenure is a strong predictor of both monthly charges and total charges, while being a senior citizen has a minimal impact on these variables.**

**Analyzing Categorical Features**
**Count plots for distribution of categorical features**
From the plots, the following results were obtained:
* There was an alost equal distribution of genders, with slightly more men than women
* There were slightly more people without patners(about 3750), than those with patners(3400)
* Most customers dont have dependants(about 5000 customers)
* Majority of the customers have phone service(>6000)
* Majority of customers with multiple lines have phone service
* About 3000 customers have multiple lines
* About 3000 customers use fibre optic
* About 1500 customers don't have internet service
* About 1500 customers have online security but no internet service
* 2000 customers have online security


## Model Development
We apply machine learning models to predict customer churn:
- **Train-Test Split**: The dataset is divided into training and test sets.
- **Model Selection**: Models like Logistic Regression, Decision Trees, and Random Forests are trained and evaluated.
- **Performance Metrics**: Accuracy, precision, recall, F1-score, and ROC-AUC are calculated to assess the model's effectiveness.

**SHAT Summary Plot**

Here's a breakdown of what the plot shows:

**Features:**

- The vertical axis lists the features in the dataset.

**SHAP Values:**

- The horizontal axis represents the SHAP values, which quantify the impact of each feature on the model's output (prediction).
- A positive SHAP value indicates that the feature increases the model's output, while a negative value indicates it decreases the output.
- The magnitude of the SHAP value represents the importance of the feature.

**Colors:**

- The color of each dot represents the feature value.
- Blue dots indicate low feature values, while red dots indicate high feature values.

**Interpretation:**

- **Feature Importance:** The length of the dots on the horizontal axis indicates the feature's importance. Longer dots suggest that the feature has a greater impact on the model's output.
- **Feature Interactions:** The clustering of dots along the horizontal axis can reveal interactions between features. If dots for two features are clustered together, it might suggest that these features interact to influence the prediction.
- **Direction of Impact:** The color of the dots shows the direction of the feature's impact. Blue dots indicate that the feature decreases the output, while red dots indicate that it increases the output.

**In this specific plot:**

- **AvgMonthlyCharge:** Appears to be the most important feature, with a wide range of SHAP values and a significant impact on the model's output.
- **Log_TotalCharges:** Also shows a considerable impact, with a clustering of dots on the positive side, suggesting it generally increases the output.
- **Contract_Two_year, Contract_One_year, tenure:** These features seem to have a moderate impact, with some clustering of dots on both positive and negative sides.
- **TotalServices:** Has a more limited impact, with most dots clustered around the 0 SHAP value.
- **InternetService_Fiber_optic, Tenure_MonthlyCharges, TotalCharges:** These features show a mix of positive and negative SHAP values, suggesting varying impacts on the model's output.
- **Other Features:** The remaining features exhibit varying levels of importance and may have interactions with each other based on the clustering of dots.

**Overall, this SHAP summary plot provides valuable insights into the relative importance of features and their interactions in influencing the model's predictions.**

**Additional Considerations:**

- **Model Complexity:** For complex models, interpreting SHAP plots can be more challenging.
- **Feature Engineering:** The choice of features can significantly impact the interpretability of the plot.
- **Domain Knowledge:** Combining SHAP insights with domain knowledge can lead to a deeper understanding of the model's behavior.







