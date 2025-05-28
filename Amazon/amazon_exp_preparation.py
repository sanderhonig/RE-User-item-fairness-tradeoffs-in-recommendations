import pandas as pd
from datetime import datetime

import pickle
import random
from sklearn.feature_extraction.text import TfidfVectorizer

import sys  #added to import utils
sys.path.insert(0, '..')  #added to import utils
from utils import preprocess_text
import random

random.seed(42)

# Read the reviews csv file into a DataFrame
df_books_rating = pd.read_csv("./Data/Books_rating.csv")

# Read the book csv file into a DataFrame
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

print("train data size:")
print(train_data.shape[0])
print("test data size:")
print(test_data.shape[0])

stopwords_file = open("../stopwords.txt", "r")
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
# with open('./Data/tfidf_matrix_train_data.pickle', 'wb') as fin:
#     pickle.dump(tfidf_matrix_train_data, fin)

tfidf_matrix_test_data = tfidf.transform(test_data['combined_features'])
# with open('./Data/tfidf_matrix_test_data.pickle', 'wb') as fin:
#     pickle.dump(tfidf_matrix_test_data, fin)

print('successfully created embeddings for train and test data')


# merge books with reviews for training dataset:
df_reviews_before_2007 = df_books_rating.merge(train_data, on='Title')

unique_reviewers = set(df_reviews_before_2007['User_id'])

# for experiments, we take 1000 reviewers to replicate the experiments of the original paper:
num_authors_to_select = min(1000, len(unique_reviewers))
sample_reviewers = random.sample(list(unique_reviewers), num_authors_to_select)

for reviewer in sample_reviewers:
    # take all books user has reviewed before 2007:
    current_reviews = df_reviews_before_2007[df_reviews_before_2007['User_id'] == reviewer]
    titles_books = current_reviews['Title']
    indices_books = train_data[train_data['Title'].isin(titles_books)].index.tolist()
    reviewer_matrix = tfidf_matrix_train_data[indices_books, :] # takes embeddings of all books user has reviewed

    # make embedding for every reviewer to pass into the experiments python files:
    with open(f'./Data/authors/{reviewer}_matrix.pickle', 'wb') as fin:
        pickle.dump(reviewer_matrix, fin)
    print(f'successfully pulled embedding for {reviewer} from train data')