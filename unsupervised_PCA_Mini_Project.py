#!/usr/bin/env python
# coding: utf-8

# ### PCA Mini Project
# 
# In the lesson, you saw how you could use PCA to substantially reduce the dimensionality of the handwritten digits.  In this mini-project, you will be using the **cars.csv** file.  
# 
# To begin, run the cell below to read in the necessary libraries and the dataset.  I also read in the helper functions that you used throughout the lesson in case you might find them helpful in completing this project.  Otherwise, you can always create functions of your own!

# In[1]:


import pandas as pd
import numpy as np
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, accuracy_score
from helper_functions import do_pca, scree_plot, plot_components, pca_results
from IPython import display
import test_code2 as t

import matplotlib.pyplot as plt
import seaborn as sns

get_ipython().run_line_magic('matplotlib', 'inline')

df = pd.read_csv('./data/cars.csv')


# `1.` Now your data is stored in **df**.  Use the below cells to take a look your dataset.  At the end of your exploration, use your findings to match the appropriate variable to each key in the dictionary below.  

# In[2]:


#Use this cell for work
df.head()


# In[3]:


# check dataset size
df.shape


# In[4]:


# check feature types
df.dtypes


# In[5]:


# check if missing values
df.isnull().sum()


# In[6]:


# check col headers
df.columns


# In[7]:


# check general stats
df.describe()


# In[8]:


df.Minivan.value_counts()


# In[9]:


21/387


# In[10]:


a = 7
b = 66
c = 387
d = 18
e = 0.23
f = 0.05

solution_1_dict = {
    'The number of cars in the dataset': c, #letter here,
    'The number of car features in the dataset': d, #letter here,
    'The number of dummy variables in the dataset': a,#letter here,
    'The proportion of minivans in the dataset': f, #letter here,
    'The max highway mpg for any car': b #letter here
}


# In[11]:


# Check your solution against ours by running this cell
display.HTML(t.check_question_one(solution_1_dict))


# `2.` There are some particularly nice properties about PCA to keep in mind.  Use the dictionary below to match the correct variable as the key to each statement.  When you are ready, check your solution against ours by running the following cell.

# In[12]:


a = True
b = False

solution_2_dict = {
    'The components span the directions of maximum variability.': a, #letter here,
    'The components are always orthogonal to one another.': a, #letter here,
    'Eigenvalues tell us the amount of information a component holds': a #letter here
}


# In[13]:


# Check your solution against ours by running this cell
t.check_question_two(solution_2_dict)


# `3.` Fit PCA to reduce the current dimensionality of the datset to 3 dimensions.  You can use the helper functions, or perform the steps on your own.  If you fit on your own, be sure to standardize your data.  At the end of this process, you will want an X matrix with the reduced dimensionality to only 3 features.  Additionally, you will want your **pca** object back that has been used to fit and transform your dataset. 

# In[14]:


#Scale your data, fit, and transform using pca
df_ss = StandardScaler().fit_transform(df)


# In[15]:


#Create a dataframe
df_ss = pd.DataFrame(df_ss)
df_ss.columns = ['Sports', 'SUV', 'Wagon', 'Minivan', 'Pickup', 'AWD', 'RWD', 'Retail', 'Dealer', 'Engine', 'Cylinders', 'Horsepower', 'CityMPG', 'HighwayMPG',
'Weight', 'Wheelbase', 'Length', 'Width']


# In[16]:


#Check first few rows
df_ss.head()


# In[17]:


df_ss.describe()


# In[18]:


#Reduce feature down to 3
comp = 3
pca, X_pca = do_pca(comp, df_ss)


# In[19]:


X_pca


# In[23]:


pca


# `4.` Once you have your pca object, you can take a closer look at what comprises each of the principal components.  Use the **pca_results** function from the **helper_functions** module assist with taking a closer look at the results of your analysis.  The function takes two arguments: the full dataset and the pca object you created.

# In[20]:


pca_results(df_ss, pca)


# `5.` Use the results, to match each of the variables as the value to the most appropriate key in the dictionary below.  When you are ready to check your answers, run the following cell to see if your solution matches ours!

# In[21]:


a = 'car weight'
b = 'sports cars'
c = 'gas mileage'
d = 0.4352
e = 0.3061
f = 0.1667
g = 0.7053

solution_5_dict = {
    'The first component positively weights items related to': c, #letter here, 
    'The amount of variability explained by the first component is': d, #letter here,
    'The largest weight of the second component is related to': b, #letter here,
    'The total amount of variability explained by the first three components': g #letter here
}


# In[22]:


# Run this cell to check if your solution matches ours.
t.check_question_five(solution_5_dict)


# `6.` How many components need to be kept to explain at least 85% of the variability in the original dataset?  When you think you have the answer, store it in the variable `num_comps`.  Then run the following cell to see if your solution matches ours!

# In[38]:


#Code to find number of components providing more than 
# 85% of variance explained
score =[]
feature_number = []
for comps in range(2, df_ss.shape[1]):
    pca, X_pca = do_pca(comps, df_ss)
    variability_score = pca_results(df_ss, pca).iloc[:, 0].sum() * 100
    score.append(variability_score)
    feature_number.append(comps)


# In[44]:


# Make a scatter plot to identify num components to explain at least 85% variability
plt.scatter(feature_number, score, c="tomato", alpha=0.8)
plt.xlabel("Feature #")
plt.ylabel("Variability score");
    
num_comps = 6 #num components stored here


# In[43]:


# Now check your answer here to complete this mini project!
display.HTML(t.question_check_six(num_comps))

