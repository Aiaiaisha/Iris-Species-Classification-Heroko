# Importing the necessary libraries.
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import streamlit as st
from sklearn.model_selection import train_test_split
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

@st.cache_data()
def prediction(sep_len,sep_width,pet_len,pet_wid):
	species = svc_model.predict([[sep_len,sep_width,pet_len,pet_wid]])
	species = species[0]
	if species == 0:
		return "Iris Setosa"
	elif species == 1:
		return "Iris-virginica"
	else:
		return "Iris-versicolor"

st.title("Iris Flower Species Prediction")
sep_len = st.slider("Sepal Length",0.00,10.00)
sep_wid = st.slider("Sepal Width" ,0.00,10.00)
pet_len = st.slider("Petal Length" ,0.00,10.00)
pet_wid = st.slider("Petal Width" ,0.00,10.00)

if st.button("Prediction"):
	species_type = prediction(sep_len,sep_wid,pet_len,pet_wid)
	st.write("Species Predicted ",species_type)
	st.write("Accuracy score of this model is ", score)

