import matplotlib.pyplot as plt
import numpy as np
from sklearn import datasets, linear_model
from sklearn.metrics import mean_squared_error, r2_score

# Load the iris dataset
iris = datasets.load_iris()

# Use only one feature
iris_X = iris.data[:, np.newaxis, 2]

# Split the data into training/testing sets
iris_X_train = iris_X[:]
iris_X_test = iris_X[:]

# Split the targets into training/testing sets
iris_y_train = iris.target[:]
iris_y_test = iris.target[:]

# Create linear regression object
regr = linear_model.LinearRegression()

# Train the model using the training sets
regr.fit(iris_X_train, iris_y_train)

# Make predictions using the testing set
iris_y_pred = regr.predict(iris_X_test)

# The coefficients
print('Coefficients: \n', regr.coef_)
# The mean squared error
print("Mean squared error: %.2f"
      % mean_squared_error(iris_y_test, iris_y_pred))
# Explained variance score: 1 is perfect prediction
print('Variance score: %.2f' % r2_score(iris_y_test, iris_y_pred))

# Plot outputs
plt.scatter(iris_X_test, iris_y_test,  color='red')
plt.plot(iris_X_test, iris_y_pred, color='blue', linewidth=3)

plt.xticks(())
plt.yticks(())

plt.show()