# Credit Score Prediction

## Overview
This project aims to develop a **Credit Score Prediction** model using various **machine learning algorithms**. The model assesses an individual's **creditworthiness** based on multiple **financial and personal attributes**. The dataset undergoes **extensive cleaning, preprocessing, and feature engineering** before training predictive models.

---

## Dataset Information
The dataset consists of **100,000+ customer financial records**, including:

- **Customer ID & Personal Details:** Unique ID, Name, SSN, Age, Occupation
- **Financial Details:** Annual Income, Monthly Inhand Salary, Number of Bank Accounts, Number of Credit Cards, Interest Rate, Monthly Balance
- **Loan & Payment Behavior:** Payment Behavior, Type of Loan, Credit Mix, Payment of Minimum Amount, Delay from Due Date
- **Credit Score Labels:** Good, Standard, and Bad

After feature selection, **23 key features** were retained.

---

## Data Cleaning & Preprocessing

### Identified Issues & Fixes:

- **Missing & Invalid Values:**
  - `Name`: 9985 missing values **filled**
  - `SSN`: Placeholder values (`#F%$D@*&8`) **replaced**
  - `Occupation`: 7062 missing values **filled**
  - `Credit Mix`: 20195 missing values **filled**
  - `Payment Behavior`: Invalid values (`!@9#%8`) **replaced**

- **Incorrect Ranges & Outliers:**
  - `Age`: Cleaned from **-500 to 8698** → **14 to 56**
  - `Annual Income`: Cleaned from **7,005 to 24M** → **7,005 to 179K**
  - `Num Bank Accounts`: Cleaned from **-1 to 1798** → **-1 to 10**
  - `Num Credit Cards`: Cleaned from **0 to 1499** → **0 to 11**
  - `Interest Rate`: Cleaned from **1 to 5797** (outliers removed)

After preprocessing:
- **All missing values handled** (Null values = 0)
- **Features transformed** for better model performance

---

## Exploratory Data Analysis (EDA)

### Visualizations:
- **Stacked Bar Charts**:
  - `Month` vs. Credit Score
  - `Occupation` vs. Credit Score
  - `Credit Mix` vs. Credit Score
  - `Payment of Minimum Amount` vs. Credit Score
  - `Payment Behavior` vs. Credit Score

- **Distributions**:
  - Age
  - Annual Income
  - Monthly Inhand Salary
  - Number of Bank Accounts
  - Number of Credit Cards

---

## Model Training & Evaluation

### Machine Learning Models Used:
- **Logistic Regression**
- **Decision Tree**
- **Random Forest**
- **K-Nearest Neighbors (KNN)**
- **Gaussian Naïve Bayes**
- **XGBoost**

### Model Performance:
| Model | Accuracy | Precision | Recall |
|--------|---------|----------|--------|
| **Decision Tree** | 72.30% | 70.64% | 70.54% |
| **Random Forest** | 81.60% | 80.57% | 80.89% |
| **KNN** | 70.25% | 67.52% | 68.46% |
| **Naïve Bayes** | 63.93% | 63.28% | 68.82% |
| **XGBoost** | 77.49% | 75.94% | 76.50% |

### Final Model Performance:
After training on an **expanded dataset** of **127,617 records** (with 31,905 test samples), the **final model** achieved:

- **Overall Accuracy:** 88%
- **Precision:** 88%
- **Recall:** 88%
- **F1-Score:** 88%

### Classification Report:

| Class | Precision | Recall | F1-Score | Support |
|-------|-----------|---------|-----------|---------|
| **Good (0)** | 0.89 | 0.95 | 0.92 | 10,635 |
| **Standard (1)** | 0.87 | 0.91 | 0.89 | 10,635 |
| **Bad (2)** | 0.89 | 0.79 | 0.84 | 10,635 |
| **Overall** | **0.88** | **0.88** | **0.88** | **31,905** |

### Confusion Matrix:
The confusion matrix indicates the **distribution of predictions** across different credit score categories.

---

## Future Improvements
- **Better handling of outliers** using advanced techniques
- **Exploring deep learning models** for credit score prediction
- **Integrating real-time credit score monitoring**

---

## Conclusion
This project demonstrates how **machine learning** can effectively predict **credit scores** based on **financial and behavioral data**. The final model achieved a high **88% accuracy**, making it a reliable tool for assessing **creditworthiness**.

---

