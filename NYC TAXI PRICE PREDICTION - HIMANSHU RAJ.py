#!/usr/bin/env python
# BY HIMANSHU RAJ
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.svm import SVR
from sklearn.neighbors import KNeighborsRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.ensemble import GradientBoostingRegressor


url="https://raw.githubusercontent.com/Premalatha-success/Datasets/main/TaxiFare.csv"
data = pd.read_csv(url)
print(data.head())

# Data Preprocessing
# data = data.head(20000)
data = data.dropna()
data = data[(data['amount'] > 0) & (data['amount'] < 100)]


# Calculate distance between pickup and dropoff
data['distance'] = np.sqrt((data['longitude_of_dropoff'] - data['longitude_of_pickup'])**2 +
                           (data['latitude_of_dropoff'] - data['latitude_of_pickup'])**2)

# Extract features and target variable
X = data[['distance', 'no_of_passenger']]  
y = data['amount']

# Split data into training and testing sets (80% training, 20% testing)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Visualize data
plt.figure(figsize=(12, 8))
plt.subplot(2, 2, 1)
sns.histplot(data['amount'], kde=True)
plt.title('Fare Amount Distribution')

plt.subplot(2, 2, 2)
sns.scatterplot(x=data['distance'], y=data['amount'])
plt.xlabel('Distance')
plt.ylabel('Fare Amount')
plt.title('Distance vs. Fare Amount')


# Linear Regression Model
# lr_model = LinearRegression()
# lr_model.fit(X_train, y_train)
# lr_predictions = lr_model.predict(X_test)

# Random Forest Model
rf_model = RandomForestRegressor(n_estimators=100, random_state=42)
rf_model.fit(X_train, y_train)
rf_predictions = rf_model.predict(X_test)

# Support Vector Machine (SVM) Model
# svm_model = SVR()
# svm_model.fit(X_train, y_train)
# svm_predictions = svm_model.predict(X_test)


# # k-Nearest Neighbors (KNN) Model
knn_model = KNeighborsRegressor(n_neighbors=5)
knn_model.fit(X_train, y_train)
knn_predictions = knn_model.predict(X_test)

# Gradient Boosting Regressor Model
gb_model = GradientBoostingRegressor(n_estimators=100, random_state=42)
gb_model.fit(X_train, y_train)
gb_predictions = gb_model.predict(X_test)
#Best GBR > KNN > Random Forest 
#Worst SVM 

# Evaluate Models
def evaluate_model(predictions, model_name):
    rmse = np.sqrt(mean_squared_error(y_test, predictions))
    mae = mean_absolute_error(y_test, predictions)
    r2 = r2_score(y_test, predictions)
    print(f"{model_name} RMSE: {rmse:.2f}")
    print(f"{model_name} MAE: {mae:.2f}")
    print(f"{model_name} R-squared: {r2:.2f}")

# # Evaluate Linear Regression Model
# evaluate_model(lr_predictions, "Linear Regression")

# Evaluate Random Forest Model
evaluate_model(rf_predictions, "Random Forest")

# # Evaluate SVM Model
# evaluate_model(svm_predictions, "Support Vector Machine")


# # Evaluate KNN Model
evaluate_model(knn_predictions, "k-Nearest Neighbors")

# Evaluate Gradient Boosting Regressor Model
evaluate_model(gb_predictions, "Gradient Boosting Regressor")

'''
RESULT 50k Data Analysis: 

Linear Regression RMSE: 9.45
Linear Regression MAE: 6.00
Linear Regression R-squared: 0.00 ##Worst

Random Forest RMSE: 5.01 ##3rd
Random Forest MAE: 2.79
Random Forest R-squared: 0.72

Support Vector Machine RMSE: 7.85
Support Vector Machine MAE: 4.24
Support Vector Machine R-squared: 0.31

k-Nearest Neighbors RMSE: 4.74 ##2nd
k-Nearest Neighbors MAE: 2.62
k-Nearest Neighbors R-squared: 0.75

Gradient Boosting Regressor RMSE: 4.40 ##BEST
Gradient Boosting Regressor MAE: 2.36
Gradient Boosting Regressor R-squared: 0.78
''' 

# Visualize predictions vs. actual values for all models
plt.figure(figsize=(16, 10))
plt.subplot(2, 2, 3)
# plt.scatter(y_test, lr_predictions, label='Linear Regression', alpha=0.5)
plt.scatter(y_test, rf_predictions, label='Random Forest', alpha=0.5)
# plt.scatter(y_test, svm_predictions, label='SVM', alpha=0.5)
plt.scatter(y_test, knn_predictions, label='KNN', alpha=0.5)
plt.scatter(y_test, gb_predictions, label='GBR', alpha=0.5)
plt.xlabel('Actual Fare Amount')
plt.ylabel('Predicted Fare Amount')
plt.legend()
plt.title('Predictions vs. Actual Values (All Models)')

plt.tight_layout()
plt.show()
