import pandas as pd
import numpy as np 
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

# reading sonar_data.csv
sonar_data = pd.read_csv("D:\ML_DataSet\sonar_data.csv",header=None)
sonar_data.head(3)

# number of rows and columns -->
sonar_data.shape

# stastical measures of a data --->
sonar_data.describe()

# countning value for different outputs-->
sonar_data[60].value_counts()

# there are critical differences btw rock and mine .
# to get this we have to calculate mean for both rock and mine data
sonar_data.groupby(60).mean()

# seperate input/data and output/labels -->
x = sonar_data.iloc[:,:-1]
y = sonar_data[60]

# Split data into training and test ------>
# The stratify parameter in train_test_split ensures 
# that the train and test sets preserve the same proportion of classes (or categories) as in the original dataset.

x_train,x_test,y_train,y_test = train_test_split(x,y,random_state=45,test_size=0.2,stratify = y)

x.shape , x_train.shape,x_test.shape


# Model Training --->

LR = LogisticRegression()
LR.fit(x_train,y_train)

# Accuracy Evaluation ----->
x_train_prediction = LR.predict(x_train)
training_data_accuracy = accuracy_score(x_train_prediction,y_train)

# accuracy of training data --->
training_data_accuracy*100

# test data accuracy --->
x_test_prediction = LR.predict(x_test)
test_data_accuracy = accuracy_score(x_test_prediction,y_test)

# accuracy of testing data --->
test_data_accuracy *100

# Makinga predictive System ---->

input_data = (0.013,0.0006,0.0088,0.0456,0.0525,0.0778,0.0931,0.0941,0.1711,0.1483,0.1532,0.11,0.089,0.1236,0.1197,0.1145,0.2137,0.2838,0.364,0.543,0.6673,0.7979,0.9273,0.9027,0.9192,1,0.9821,0.9092,0.8184,0.6962,0.59,0.5447,0.5142,0.5389,0.5531,0.5318,0.4826,0.379,0.1831,0.175,0.1679,0.0674,0.0609,0.0375,0.0533,0.0278,0.0179,0.0114,0.0073,0.0116,0.0092,0.0078,0.0041,0.0013,0.0011,0.0045,0.0039,0.0022,0.0023,0.0016
)

# changing input_data to numpy array -->
numpy_arr = np.asarray(input_data)  # shape -->(4,)

# model wants 2D shape of aaray.
# reshaping the numpy_arr as we predicting for one instance -->
# With reshaping: Works perfectly, because the model knows you’re predicting for one sample with multiple features.
numpy_arr_reshaped = numpy_arr.reshape(1,-1)  # shape --> (1 sample,4 features)

prediction = LR.predict(numpy_arr_reshaped)

if (prediction[0] =='R'):
    print("The object is Rock !!")
else:
    print("The object is Mine !!")



