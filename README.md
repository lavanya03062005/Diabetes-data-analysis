This project analyzes a diabetes dataset using Python and various data science libraries.

1. Data Loading & Preprocessing

Key features include Glucose, BMI, Insulin, Blood Pressure, and Outcome (0 = No Diabetes, 1 = Diabetes).
No missing values or duplicates were found.

2. Exploratory Data Analysis (EDA)
   
Histogram plots show feature distributions.
BMI vs. Diabetes Outcome is analyzed using bar charts.

4. Data Normalization & Splitting
Data is normalized using sklearn.preprocessing.
Dataset split into 70% training and 30% testing.


4. Model Training & Evaluation
Logistic Regression:
Accuracy: 77.06%

Precision & Recall: Higher for non-diabetic (Outcome = 0) cases.

Linear Regression:
Accuracy: 77.06% (though not ideal for classification).
