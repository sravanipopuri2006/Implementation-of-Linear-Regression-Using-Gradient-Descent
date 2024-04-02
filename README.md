# Implementation-of-Linear-Regression-Using-Gradient-Descent

## AIM:
To write a program to predict the profit of a city using the linear regression model with gradient descent.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
```
1.Import the standard libraries in python required for finding Gradient Design.

2.Read the dataset file and check any null value using .isnull() method.

3.Declare the default variables with respective values for linear regression.

4.Calculate the loss using Mean Square Error.

5.Predict the value of y.

6.Plot the graph respect to hours and scores using .scatterplot() method for Linear Regression.

7.Plot the graph respect to loss and iterations using .plot() method for Gradient Descent.
```


## Program:
```
/*
Program to implement the linear regression using gradient descent.
Developed by:POPURI SRAVANI 
RegisterNumber:  212223240117
*/
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

def linear_regression(X1, y, learning_rate=0.01, num_iters=1000):
    # Add a column of ones to X for the intercept term
    X= np.c_[np.ones(len(X1)), X1]
    
    # Initialize theta with zeros
    theta= np.zeros(X.shape[1]).reshape(-1,1)
    
    # Perform gradient descent
    for _ in range(num_iters):
        # Calculate predictions
        predictions= (X).dot(theta).reshape(-1,1)
        
        # Calculate errors
        errors= (predictions-y).reshape(-1,1)
        
        # Update theta using gradient descent
        theta-=learning_rate * (1/len(X1)) * X.T.dot(errors)
    return theta

data=pd.read_csv('50_Startups.csv',header=None)
print(data.head())

# Assuming the last column is your target variable 'y' and the preceding columns are
X=(data.iloc[1:,:-2].values)
print(X)

X1= X.astype(float)
scaler= StandardScaler()

y= (data.iloc[1:,-1].values).reshape(-1,1)
print(y)

X1_Scaled= scaler.fit_transform(X1)
Y1_Scaled= scaler.fit_transform(y)
print(X1_Scaled)
print(Y1_Scaled)

# Learn model parameters
theta= linear_regression(X1_Scaled, Y1_Scaled)

# Predict target value for a new data point
new_data= np.array([165349.2, 136897.8, 471784.1]).reshape(-1,1)
new_Scaled= scaler.fit_transform(new_data)
prediction= np.dot(np.append(1, new_Scaled), theta)
prediction= prediction.reshape(-1,1)
pre = scaler.inverse_transform(prediction)
print(f"Predicted value: {pre}")


```

## Output:
## DATASET
![Screenshot 2024-04-02 131107](https://github.com/sravanipopuri2006/Implementation-of-Linear-Regression-Using-Gradient-Descent/assets/139778301/3c47ad9d-e665-4d38-a4d5-4defa8693c96)
## X-VALUES
![Screenshot 2024-04-02 131131](https://github.com/sravanipopuri2006/Implementation-of-Linear-Regression-Using-Gradient-Descent/assets/139778301/37cfa532-f81b-4366-bfd0-7bcf36a0ffd1)
## Y-VALUES
![Screenshot 2024-04-02 131233](https://github.com/sravanipopuri2006/Implementation-of-Linear-Regression-Using-Gradient-Descent/assets/139778301/42061908-99f6-4a8f-8563-08250a2d3005)
## X1-SCALED
![Screenshot 2024-04-02 131301](https://github.com/sravanipopuri2006/Implementation-of-Linear-Regression-Using-Gradient-Descent/assets/139778301/9e6ef5f6-c139-422f-8c86-99d97d92a0e2)
## Y1_SCALED
![Screenshot 2024-04-02 131316](https://github.com/sravanipopuri2006/Implementation-of-Linear-Regression-Using-Gradient-Descent/assets/139778301/2ea3a67b-de8e-48fc-b2db-9dfcebc10693)
## PREDICTED VALUE
![image](https://github.com/sravanipopuri2006/Implementation-of-Linear-Regression-Using-Gradient-Descent/assets/139778301/7a2fe047-1fc1-4ce1-b451-b49dbc6d724e)








## Result:
Thus the program to implement the linear regression using gradient descent is written and verified using python programming.
