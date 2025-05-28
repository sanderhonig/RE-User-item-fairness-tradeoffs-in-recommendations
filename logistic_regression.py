import pickle
import random
import time
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from utils import extract_authors, preprocess_text
from model_evaluation import rec_cited_author, rec_referenced_by_author
import random

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
import statsmodels.api as sm
import pandas as pd
import numpy as np

np.random.seed(42)

data_1 = pd.read_csv('./Data/average_author_similarity_scores_corrected.csv',  dtype={"id": str})
data_2 = pd.read_csv('./Data/recommendations_authors_method_corrected.csv',  dtype={"id": str})

print(data_1.shape)
print(data_1.columns)

print(data_2.shape)
print(data_2.columns)

data_2['Good_recom'] = data_2['References Author'] | data_2['Also Authored by input author'] | (data_2['Citations Present'] == True)

#print(data_2['authors_x'].head(10))

data_1_dropped = data_1.drop("author", axis=1)

data_1_dropped = data_1_dropped.dropna(subset=[data_1_dropped.columns[1]])

num_users = data_1_dropped.shape[0]
num_items = data_1_dropped.shape[1]

#for each other all the similarity scores
data_for_regression = pd.concat([data_1_dropped.iloc[i].dropna().reset_index(drop=True).to_frame(name="simscore") for i in range(len(data_1_dropped))], ignore_index=True)

data_for_regression['Good_recom'] = data_2['Good_recom']

selected_users = np.random.choice(num_users, size=1128, replace=False)

# Get all rows corresponding to selected users
selected_rows = np.concatenate([np.arange(user * num_items, (user + 1) * num_items) for user in selected_users])

selected_data = data_for_regression.iloc[selected_rows].reset_index(drop=True)

X = selected_data[['simscore']]  # Input: Similarity scores
y = selected_data['Good_recom']           # Output: Boolean values

#Add constant (intercept term) to the features (for statsmodels)
X_const = sm.add_constant(X)

#Fit a logistic regression model using statsmodels
model = sm.Logit(y, X_const)
result = model.fit()

#Get the coefficients, standard errors, and z-values
coefficients = result.params
standard_errors = result.bse
z_values = coefficients / standard_errors

#Print the results
print("Coefficients:")
print(coefficients)
print("Standard Errors:")
print(standard_errors)
print("Z-values:")
print(z_values)
