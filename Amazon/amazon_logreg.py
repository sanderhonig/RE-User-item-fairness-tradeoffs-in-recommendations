import pandas as pd
from datetime import datetime

import pickle
import random
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from utils import preprocess_text
import random

import statsmodels.api as sm

random.seed(42)

# Read the reviews csv file into a DataFrame
df_books_rating = pd.read_csv("./Data/Books_rating.csv")

# Read the books csv file into a DataFrame
df_books_data = pd.read_csv("./Data/books_data.csv")

# drop rows where description is empty: (because description is essential for creating embeddings of books)
df_books_data = df_books_data.dropna(subset=['description'])

df_books_rating = df_books_rating[df_books_rating['review/time'] != -1] # drop unix time = -1

df_books_rating['review/time'] = pd.to_datetime(df_books_rating['review/time'], unit='s')
df_books_rating['review_year'] = pd.to_datetime(df_books_rating['review/time'], errors='coerce').dt.year
df_books_data['published_year'] = pd.to_datetime(df_books_data['publishedDate'], errors='coerce').dt.year

train_data = df_books_data[df_books_data['published_year'] < 2007].reset_index(drop=True)

# List of years we used for the test set:
years = [2007, 2008, 2009, 2010, 2011]

# Filter the DataFrame for the selected years
test_data = df_books_data[df_books_data['published_year'].isin(years)].reset_index(drop=True)

print("train data shape:")
print(train_data.shape[0])
print("test data shape:")
print(test_data.shape[0])

stopwords_file = open("stopwords.txt", "r")
stopwords = stopwords_file.read()
stop = stopwords.replace('\n', ',').split(",")

# We use following features to create embeddings for books
features = ['Title', 'description', 'categories']

for feature in features:
    train_data[f'{feature}_processed'] = train_data[feature].fillna('').apply(lambda x: preprocess_text(x, stop))
    test_data[f'{feature}_processed'] = test_data[feature].fillna('').apply(lambda x: preprocess_text(x, stop))

features_processed = ['Title_processed', 'description_processed', 'categories_processed']
# Combine features into a single string for further processing
train_data['combined_features'] = train_data[features_processed].apply(lambda row: ' '.join(row.values.astype(str)),
                                                                       axis=1)
test_data['combined_features'] = test_data[features_processed].apply(lambda row: ' '.join(row.values.astype(str)),
                                                                     axis=1)
tfidf = TfidfVectorizer(stop_words='english')
tfidf_matrix_train_data = tfidf.fit_transform(train_data['combined_features'])
with open('./Data/tfidf_matrix_train_data.pickle', 'wb') as fin:
    pickle.dump(tfidf_matrix_train_data, fin)

tfidf_matrix_test_data = tfidf.transform(test_data['combined_features'])
with open('./Data/tfidf_matrix_test_data.pickle', 'wb') as fin:
    pickle.dump(tfidf_matrix_test_data, fin)

print('successfully created embeddings for train and test data')

# merge books with reviews for training dataset:
df_reviews_before_2007 = df_books_rating.merge(train_data, on='Title')

unique_reviewers = set(df_reviews_before_2007['User_id'])

#for logistic regression, we take 1128 to replicate the original paper:
sample_reviewers = random.sample(list(unique_reviewers), 1128)

#empty dataframe for recommendations:
recommendations = pd.DataFrame()

for reviewer in sample_reviewers:
    current_reviews = df_reviews_before_2007[df_reviews_before_2007['User_id'] == reviewer]
    titles_books = current_reviews['Title']
    indices_books = train_data[train_data['Title'].isin(titles_books)].index.tolist()
    reviewer_matrix = tfidf_matrix_train_data[indices_books, :] # takes embeddings of all books user has reviewed
    
    cosine_sim = cosine_similarity(reviewer_matrix, tfidf_matrix_test_data) # similarity score
    max_sim_score = cosine_sim.max(axis=0) # take max similarity score
    
    # for evaluation (logistic regression):
    titles_items = test_data[['Title']].copy() # books from 2007
    titles_items['simscore'] = max_sim_score

    reviews_user = df_books_rating[df_books_rating['User_id'] == reviewer]['Title']
    titles_items['title_in_reviews'] = titles_items['Title'].isin(reviews_user) # boolean value
        
    recommendations = pd.concat([recommendations, titles_items], ignore_index=True)

X = recommendations[['simscore']]  # Input: Similarity scores
y = recommendations['title_in_reviews']   # Output: Boolean values

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