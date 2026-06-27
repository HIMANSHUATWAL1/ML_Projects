# importing Dependencies --->
import pandas as pd
import numpy as np 
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score


# Data Collection --->

diabetes_data = pd.read_csv("D:\ML_DataSet\diabetes_data.csv")
diabetes_data.head()

diabetes_data.shape

diabetes_data.describe()

diabetes_data["Outcome"].value_counts()

diabetes_data.groupby("Outcome").mean()

# x = diabetes_data.drop(column="Outcome",axis =1)  we can also use this axis =1 means column
x = diabetes_data.iloc[:,:-1]
y = diabetes_data["Outcome"]


# Data Standardization  --->
# why we are doing this --->
# Because ranges of every feature's value is different that makes differences to predict the final output so we are using standard scaler .

ss = StandardScaler()
ss.fit(x)

standardized_data = ss.transform(x)

# assigning standard data to x ---->
x = standardized_data
y = diabetes_data["Outcome"]

# Splitting Data ---->
x_train,x_test,y_train,y_test = train_test_split(x,y,random_state =42,test_size=0.2,stratify = y)


# Training The Model with data-->

svm = SVC(kernel='linear')
svm.fit(x_train,y_train)

# Model Evaluation --->
# Accuracy Score --->

train_data_prediction = svm.predict(x_train)
train_data_accuracy = accuracy_score(train_data_prediction,y_train)
print("Accuracy score of training data : ",train_data_accuracy*100)


# test data accuracy score --->

test_data_prediction = svm.predict(x_test)
test_data_accuracy = accuracy_score(test_data_prediction,y_test)
print("Accuracy score of test data : ",test_data_accuracy*100)


# Making a predictive system ---->

input_data=(1	,189,	60,	23,	846,	30.1	,0.398,	59)

numpy_arr = np.asarray(input_data)

reshaped_numpy_arr = numpy_arr.reshape(1,-1)

# standardized through standard scaler ---->

standardized_input_data = ss.transform(reshaped_numpy_arr)

predicted_outcome = svm.predict(standardized_input_data)

if(predicted_outcome[0] == 1 ):
    print("Patient is diabetic")
else:
    print("Patient is not diabetic")



