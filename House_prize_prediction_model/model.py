import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
from sklearn import metrics
from xgboost import XGBRegressor


# importing boston house prediction dataset --->

california_dataset = fetch_california_housing()

california_data_as_dataframe = pd.DataFrame(california_dataset.data,columns = california_dataset.feature_names)
california_data_as_dataframe['Prize'] = california_dataset.target

california_data_as_dataframe.head(3)


california_data_as_dataframe.shape

# Check for the missing values ---->

california_data_as_dataframe.isnull().sum()

# Statistical measures of dataset--->
california_data_as_dataframe.describe()

# Understanding the correlation between various features of dataset -->
# Positive correlation --> if one value increases then other value also increases
# negative correlation --> if one value decreases then other value also decreases


feature_correlation = california_data_as_dataframe.corr()

# construct heatmap to better understand the correlation btw features-->

plt.figure(figsize=(10,10))
sns.heatmap(feature_correlation,annot=True,fmt = '.1f')
plt.show()

# dataset splitting -->
x = california_data_as_dataframe.iloc[:,:-1]
y = california_data_as_dataframe["Prize"]

# splitting data into training and testing data ---->

x_train,x_test,y_train,y_test = train_test_split(x,y,random_state = 45,test_size = 0.2)

# Model training --->

xgboost = XGBRegressor()
xgboost.fit(x_train,y_train)


# Model evaluation --->


# For training data --->
training_data_prediction = xgboost.predict(x_train)

# R squared error --->
score_1 = metrics.r2_score(y_train,training_data_prediction)
# mean absolute error --->

score_2 = metrics.mean_absolute_error(y_train,training_data_prediction)

print("R squared error ---> " , score_1)
print("mean absolute error ---> ",score_2)

# Visualizing btw actual prize and predicted prize --->
plt.figure(figsize=(10,10))
plt.scatter(y_train,training_data_prediction)
plt.xlabel("Actual prize")
plt.ylabel("Predicted prize")
plt.title("Actual prize Vs Predicted prize")
plt.show()


# For test data --->

test_data_prediction = xgboost.predict(x_test)

# R squared error --->
score_11 = metrics.r2_score(y_test,test_data_prediction)

# mean squared error --->
score_22 = metrics.mean_absolute_error(y_test,test_data_prediction)

print("R squared error ---> " , score_11)
print("mean absolute error ---> ",score_22)

