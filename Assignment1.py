import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score

#A
loan_data_old = pd.read_csv('loan_old.csv')

#B
missing_values = loan_data_old.isnull().sum()

feature_types = loan_data_old.dtypes

numeric_features = loan_data_old.select_dtypes(include=['int64', 'float64'])
numeric_scale_check = numeric_features.describe().transpose()[['mean', 'std']]

sns.pairplot(loan_data_old[numeric_features.columns])
plt.show()
#C
loan_data_old = loan_data_old.dropna()

Features = loan_data_old.drop(['Max_Loan_Amount', 'Loan_Status', 'Loan_ID'], axis=1)
Target_Max = loan_data_old['Max_Loan_Amount']
Target_Status = loan_data_old['Loan_Status']

x_train, x_test, y_status_train, y_status_test, y_max_train, y_max_test = train_test_split(
    Features, Target_Status, Target_Max, test_size=0.2, random_state=40
)

categorical_features = x_train[['Gender', 'Married', 'Education', 'Property_Area', 'Dependents']]
label_encoder = LabelEncoder()
for feature in categorical_features:
    x_train[feature] = label_encoder.fit_transform(x_train[feature])
    x_test[feature] = label_encoder.fit_transform(x_test[feature])

y_status_train = (y_status_train == 'Y').astype(int).values.reshape(-1, 1)
y_status_test = (y_status_test == 'Y').astype(int).values.reshape(-1)

numerical_features = x_train[['Loan_Tenor', 'Coapplicant_Income', 'Income']]
scaler = StandardScaler()
x_train[numerical_features.columns] = scaler.fit_transform(x_train[numerical_features.columns])
x_test[numerical_features.columns] = scaler.transform(x_test[numerical_features.columns])

#D
linear_regression_model = LinearRegression()
linear_regression_model.fit(x_train, y_max_train)

#E
y_prediction = linear_regression_model.predict(x_test)
r2 = r2_score(y_max_test, y_prediction)
print("R2 score:", r2)

#F
def sigmoid(z):
    return 1 / (1 + np.exp(-z))

def logistic_regression(x, y, learning_rate=0.01, num_iterations=10000):
    m = x.shape[0]
    n = x.shape[1]
    theta = np.zeros((n, 1))
    b = 0
    cost_list = []
    for i in range(num_iterations):
        gradient = np.dot(x, theta) + b
        A = sigmoid(gradient)
        cost = -(1 / m) * np.sum(y * np.log(A) + (1 - y) * np.log(1 - A))
        dw = (1 / m) * np.dot(x.T, (A - y))
        db = (1 / m) * np.sum(A - y)
        theta -= learning_rate * dw
        b -= learning_rate * db
        cost_list.append(cost)

        if i % (num_iterations / 10) == 0:
            print("Cost after", i, "iteration is:", cost)

    return theta, b, cost_list

theta_final, bias, costlist = logistic_regression(x_train, y_status_train)

def predict(x, theta, b):
    z = np.dot(x, theta) + b
    A = sigmoid(z)
    return np.round(A).astype(int).reshape(-1)

#G
def accuracy(y_true, y_pred):
    if len(y_true) != len(y_pred):
        raise ValueError("Predictions and actual must have the same length.")
    correct_predictions = sum(true == pred for true, pred in zip(y_true, y_pred))
    accuracy = correct_predictions / len(y_true)
    return accuracy * 100


y_status_predict = predict(x_test, theta_final, bias)
print("Accuracy of logistic regression model:", accuracy(y_status_test, y_status_predict))
plt.figure(figsize=(6,6))
plt.plot(range(len(costlist)), costlist, marker='o', linestyle='-')
plt.title('Error Rate by Iterations')
plt.xlabel('Iterations')
plt.ylabel('cost')
plt.show()


#H
loan_data_new = pd.read_csv('loan_new.csv')

#I
loan_data_new = loan_data_new.dropna()
loan_new = loan_data_new.drop(['Loan_ID'], axis=1)

categorical_features_new = loan_new[['Gender', 'Married', 'Education', 'Property_Area', 'Dependents']]
for feature in categorical_features_new:
    loan_new[feature] = label_encoder.fit_transform(loan_new[feature])

numeric_loan_new = loan_new[['Loan_Tenor', 'Coapplicant_Income', 'Income']]
scaler.fit(loan_new[numeric_loan_new.columns])
loan_new[numeric_loan_new.columns] = scaler.transform(loan_new[numeric_loan_new.columns])

#J
y_status_predict_new = predict(loan_new, theta_final, bias)

y_prediction = linear_regression_model.predict(loan_new)

predicted_loan_status_new = np.where(y_status_predict_new == 1, 'Y', 'N')
print("Predicted labels using logistic regression:",predicted_loan_status_new)
print("Predicted loan amounts using linear regression:",y_prediction)