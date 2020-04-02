{
 "cells": [
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "# TODO: Add import statements\n",
    "from sklearn.linear_model import LinearRegression\n",
    "from sklearn.preprocessing import PolynomialFeatures\n",
    "import numpy as np\n",
    "import pandas as pd\n",
    "# Assign the data to predictor and outcome variables\n",
    "# TODO: Load the data\n",
    "train_data = pd.read_csv(\"data.csv\")\n",
    "X = train_data['Var_X'].values.reshape(-1, 1)\n",
    "y = train_data['Var_Y'].values\n",
    "\n",
    "\n",
    "# Create polynomial features\n",
    "# TODO: Create a PolynomialFeatures object, then fit and transform the\n",
    "# predictor feature\n",
    "poly_feat = PolynomialFeatures(degree = 4)\n",
    "X_poly = poly_feat.fit_transform(X)\n",
    "\n",
    "# Make and fit the polynomial regression model\n",
    "# TODO: Create a LinearRegression object and fit it to the polynomial predictor\n",
    "# features\n",
    "poly_model = LinearRegression()\n",
    "poly_model.fit(X_poly, y)\n",
    "\n",
    "# Once you've completed all of the steps, select Test Run to see your model\n",
    "# predictions against the data, or select Submit Answer to check if the degree\n",
    "# of the polynomial features is the same as ours!"
   ]
  }
 ],
 "metadata": {
  "kernelspec": {
   "display_name": "Python 3",
   "language": "python",
   "name": "python3"
  },
  "language_info": {
   "codemirror_mode": {
    "name": "ipython",
    "version": 3
   },
   "file_extension": ".py",
   "mimetype": "text/x-python",
   "name": "python",
   "nbconvert_exporter": "python",
   "pygments_lexer": "ipython3",
   "version": "3.7.4"
  }
 },
 "nbformat": 4,
 "nbformat_minor": 2
}
