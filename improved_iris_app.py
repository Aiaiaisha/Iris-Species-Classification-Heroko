# Importing the necessary libraries.
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import streamlit as st
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC

# Loading the dataset.
iris_df = pd.read_csv("iris-species.csv")

# Adding a column in the Iris DataFrame to resemble the non-numeric 'Species' column as numeric using the 'map()' function.
# Creating the numeric target column 'Label' to 'iris_df' using the 'map()' function.
iris_df['Label'] = iris_df['Species'].map({'Iris-setosa': 0, 'Iris-virginica': 1, 'Iris-versicolor':2})

# Creating a model for Support Vector classification to classify the flower types into labels '0', '1', and '2'.

# Creating features and target DataFrames.
X = iris_df[['SepalLengthCm','SepalWidthCm', 'PetalLengthCm', 'PetalWidthCm']]
y = iris_df['Label']

# Splitting the data into training and testing sets.
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.33, random_state = 42)

# Creating the SVC model and storing the accuracy score in a variable 'score'.
svc_model = SVC(kernel = 'linear')
svc_model.fit(X_train, y_train)
score = svc_model.score(X_train, y_train)

# Creating a logistic regression model
lr = LogisticRegression(n_jobs = -1)
lr.fit(X_train,y_train)

# Creating a random forest classifier

rf_clf = RandomForestClassifier(n_jobs = -1,n_estimators = 100)
rf_clf.fit(X_train,y_train)


@st.cache_data()
def prediction(_model,sep_len,sep_width,pet_len,pet_wid):
	species = svc_model.predict([[sep_len,sep_width,pet_len,pet_wid]])
	species = species[0]
	if species == 0:
		return "Iris Setosa"
	elif species == 1:
		return "Iris-virginica"
	else:
		return "Iris-versicolor"

st.sidebar.title("Iris Flower Species Prediction")
sep_len = st.sidebar.slider("Sepal Length",0.00,10.00)
sep_wid = st.sidebar.slider("Sepal Width" ,0.00,10.00)
pet_len = st.sidebar.slider("Petal Length" ,0.00,10.00)
pet_wid = st.sidebar.slider("Petal Width" ,0.00,10.00)
model = st.sidebar.selectbox("Classifier",("Support Vector Machine","RandomForestClassifier","Logistic Regression"))

if st.sidebar.button("Prediction"):
	if model == "Support Vector Machine":
		species_type = prediction(svc_model,sep_len,sep_wid,pet_len,pet_wid)
		score = svc_model.score(X_train,y_train)
	elif model =="Logistic Regression":
		species_type = prediction(lr,sep_len,sep_wid,pet_len,pet_wid)
		score = lr.score(X_train,y_train)
	else:
		species_type = prediction(rf_clf,sep_len,sep_wid,pet_len,pet_wid)
		score = lr.score(X_train,y_train)

	st.write("Species Predicted ",species_type)
	st.write("Accuracy score of this model is ", score)
