import streamlit as st
import pandas as pd
import numpy as np

dataset_url = ('https://raw.githubusercontent.com/rahadis/datamining/main/train.csv')
df = pd.read_csv(dataset_url)



st.markdown("# Home")
st.sidebar.markdown("# Home")
st.dataframe(df)

import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.colors import ListedColormap
from sklearn import neighbors, datasets
#from sklearn.inspection import DecisionBoundaryDisplay
X = df.drop(columns=['price_range'])
st.dataframe(X)
#separate target values
y = df['price_range'].values
#view target values
y[0:5]

from sklearn.model_selection import train_test_split
#split dataset into train and test data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1, stratify=y)

from sklearn.neighbors import KNeighborsClassifier
# Create KNN classifier
knn = KNeighborsClassifier(n_neighbors = 3)
# Fit the classifier to the data
knn.fit(X_train,y_train)

#show first 5 model predictions on the test data
knn.predict(X_test)[0:5]

#check accuracy of our model on the test data
knn.score(X_test, y_test)

from sklearn.model_selection import cross_val_score
import numpy as np
#create a new KNN model
knn_cv = KNeighborsClassifier(n_neighbors=3)
#train model with cv of 5 
cv_scores = cross_val_score(knn_cv, X, y, cv=5)
#print each cv score (accuracy) and average them
st.write(cv_scores)
st.write('cv_scores mean:{}'.format(np.mean(cv_scores)))

from sklearn.model_selection import GridSearchCV
#create new a knn model
knn2 = KNeighborsClassifier()
#create a dictionary of all values we want to test for n_neighbors
param_grid = {'n_neighbors': np.arange(1, 25)}
#use gridsearch to test all values for n_neighbors
knn_gscv = GridSearchCV(knn2, param_grid, cv=5)
#fit model to data
knn_gscv.fit(X, y)
#check top performing n_neighbors value

knn_gscv.best_params_

#check mean score for the top performing value of n_neighbors
knn_gscv.best_score_