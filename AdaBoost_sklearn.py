# Use scikit-learn's AdaBoostClassifier class. This class provides the functions to define and fit the model to your data.
from sklearn.ensemble import AdaBoostClassifier
model = AdaBoostClassifier()
model.fit(x_train, y_train)
model.predict(x_test)
# Hyperparameters: 
# we can specify the hyperparameters. most common are:
# base_estimator: The model utilized for the weak learners 
# (Warning: Don't forget to import the model that you decide to use for the weak learner).
# n_estimators: The maximum number of weak learners used.

 # exemple, we define a model which uses decision trees of max_depth 2 as the weak learners, and it allows a maximum of 4 of them.
from sklearn.tree import DecisionTreeClassifier
model = AdaBoostClassifier(base_estimator = DecisionTreeClassifier(max_depth=2), n_estimators = 4)
