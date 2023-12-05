# Diabetes-data-analysis
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

data=pd.read_csv("/content/archive.zip")
data

data.head()

data.shape

data.info()

data.nunique()

data.describe().T

data.duplicated()

data[data.duplicated()]

data.drop_duplicates()

data.isnull()

p=data.hist(figsize=(8,8))

import matplotlib.pyplot as plt
# Line plot
plt.plot(data['BMI'])
plt.xlabel("BMI")
plt.ylabel("Levels")
plt.title("Line Plot")
plt.show()

fig,(ax1,ax2)=plt.subplots(1,2,figsize=(13,5))
data_len=data[data['Outcome']==1]['BMI'].value_counts()
ax1.hist(data_len,color='red')
ax1.set_title('Having diabetis')

data_len=data[data['Outcome']==0]['BMI'].value_counts()
ax2.hist(data_len,color='green')
ax2.set_title('NOT Having diabetis')

fig.suptitle('diabetis Levels')
plt.show()


data.isnull().sum() #checking for total null values

from sklearn import preprocessing
import pandas as pd

d = preprocessing.normalize(data.iloc[:,1:5], axis=0)
scaled_df = pd.DataFrame(d, columns=["Hemoglobin", "MCH", "MCHC", "MCV"])
scaled_df.head()

from sklearn.model_selection import train_test_split #training and testing data split
from sklearn import metrics #accuracy measure
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score, classification_report #for confusion matrix
from sklearn.linear_model import LogisticRegression,LinearRegression #logistic regression

train,test=train_test_split(data,test_size=0.3,random_state=0,stratify=data['Outcome'])
train_X=train[train.columns[:-1]]
train_Y=train[train.columns[-1:]]
test_X=test[test.columns[:-1]]
test_Y=test[test.columns[-1:]]
X=data[data.columns[:-1]]
Y=data['Outcome']
len(train_X), len(train_Y), len(test_X), len(test_Y)

model = LogisticRegression()
model.fit(train_X,train_Y)
prediction3=model.predict(test_X)
print('The accuracy of the Logistic Regression is',metrics.accuracy_score(prediction3,test_Y))
report = classification_report(test_Y, prediction3)
print("Classification Report:\n", report)

model = LinearRegression()
model.fit(train_X, train_Y)
prediction = model.predict(test_X)
accuracy = accuracy_score(test_Y, prediction.round())

print('The accuracy of Linear Regression is:', accuracy)


