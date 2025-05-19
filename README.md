# 🏠 Loan Eligibility and Amount Prediction System

This project focuses on building two machine learning models for a housing finance company aiming to **automate customer eligibility validation** and **predict loan amounts**. The objective is to help the company make accurate and efficient decisions based on customer application details.

---

## 📦 Datasets

There are two CSV files used in this project:

### 1. `loan_old.csv`
- **Records:** 614 applicants
- **Features:**
  - Loan_ID
  - Gender
  - Marital Status
  - Dependents
  - Education
  - ApplicantIncome
  - CoapplicantIncome
  - Loan_Amount_Term (months)
  - Credit_History
  - Property_Area
- **Targets:**
  - LoanAmount (in thousands)
  - Loan_Status (Accepted or Rejected)

### 2. `loan_new.csv`
- **Records:** 367 applicants
- **Features:** Same 10 as above (no target columns)

---

## 🎯 Project Goals

1. **Build a Linear Regression model** to predict the maximum **loan amount** an applicant can borrow.
2. **Build a Logistic Regression model (from scratch)** to classify **loan eligibility**.
3. **Deploy** both models to make predictions on new applicants.

---

## 📊 Data Analysis and Preprocessing

### Exploratory Analysis
- Checked for missing values
- Identified feature types (categorical vs numerical)
- Verified numerical feature scaling
- Created pairplots for numerical columns

### Preprocessing Steps
- Dropped records with missing values
- Separated features and target variables
- Shuffled and split data into training/testing sets
- Encoded categorical features and targets
- Standardized numerical features

---

## 🧠 Model Building

### 1. Linear Regression (for Loan Amount)
- **Tool:** `LinearRegression` from `sklearn`
- **Target:** `LoanAmount`
- **Evaluation Metric:** R² Score

### 2. Logistic Regression (for Loan Status)
- **Implementation:** From scratch using gradient descent
- **Target:** `Loan_Status` (binary classification)
- **Loss Function:** Binary Cross-Entropy
- **Evaluation Metric:** Accuracy (custom function)

---

## 🔍 Model Evaluation

- **Linear Regression:** Evaluated using the R² Score to assess how well the model explains variance in loan amounts.
- **Logistic Regression:** Evaluated using a custom-built accuracy function to calculate correct classification rate.

---

## 🚀 Prediction on New Applicants

- Loaded and preprocessed `loan_new.csv` (except shuffling/splitting).
- Used both trained models to:
  - Predict maximum loan amount.
  - Predict loan acceptance status.

---
