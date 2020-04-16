# Import statements 
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
import pandas as pd
import numpy as np

# Read the data.
data = np.asarray(pd.read_csv('data.csv', header=None))
# Assign the features to the variable X, and the labels to the variable y, then split data to testing and training
X_train = data[:70,0:2]
y_train = data[:70,2]

X_test = data[71:97, 0:2]
y_test = data[71:97, 2]

# TODO: Create the model and assign it to the variable model.

# Find the right parameters for this model to achieve 100% accuracy on the dataset.
# we can specify the hyperparameters. As we've seen in this section, the most common ones are the C parameter
# kernel:most common ones are 'linear', 'poly', and 'rbf'.
# degree: If the kernel is polynomial, this is the maximum degree of the monomials in the kernel
# gamma: If the kernel is rbf, this is the gamma parameter.
model = SVC(kernel = 'poly',degree=4, C=0.5)

# TODO: Fit the model.
model.fit(X_train, y_train)
# TODO: Make predictions. Store them in the variable y_pred.
y_pred = model.predict(X_test)

# TODO: Calculate the accuracy and assign it to the variable acc.
acc = accuracy_score(y_test, y_pred)
