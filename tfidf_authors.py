import pandas as pd
import pickle
import random
import time
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from utils import extract_authors, preprocess_text
from model_evaluation import rec_cited_author, rec_referenced_by_author

start_time = time.perf_counter()

# Import dataset
train_data = pd.read_csv('./Data/train_data.csv', dtype={"id": str})  #changed: added dtype to correctly load leading zeros

#  Test Data
test_data = pd.read_csv('./Data/test_data_details.csv', dtype={"id": str})  #changed: added dtype to correctly load leading zeros

test_data['full_names'] = test_data['authors_parsed'].apply(extract_authors)
all_full_names = sum(test_data['full_names'], [])
unique_full_names = set(all_full_names)

stopwords_file = open("stopwords.txt", "r")
stopwords = stopwords_file.read()
stop = stopwords.replace('\n', ',').split(",")

features = ['title', 'abstract', 'categories']

for feature in features:
    train_data[f'{feature}_processed'] = train_data[feature].fillna('').apply(lambda x: preprocess_text(x, stop))
    test_data[f'{feature}_processed'] = test_data[feature].fillna('').apply(lambda x: preprocess_text(x, stop))

features_processed = ['title_processed', 'abstract_processed', 'categories_processed']
# Combine features into a single string for further processing
train_data['combined_features'] = train_data[features_processed].apply(lambda row: ' '.join(row.values.astype(str)),
                                                                       axis=1)
test_data['combined_features'] = test_data[features_processed].apply(lambda row: ' '.join(row.values.astype(str)),
                                                                     axis=1)
tfidf = TfidfVectorizer(stop_words='english')
tfidf_matrix_train_data = tfidf.fit_transform(train_data['combined_features'])
with open('./Data/tfidf_matrix_train_data.pickle', 'wb') as fin:  #changed: store within directory for organization, although worked w/o
    pickle.dump(tfidf_matrix_train_data, fin)

tfidf_matrix_test_data = tfidf.transform(test_data['combined_features'])
with open('./Data/tfidf_matrix_test_data.pickle', 'wb') as fin:  #changed: store within directory for organization, although worked w/o
    pickle.dump(tfidf_matrix_test_data, fin)

print('successfully created embeddings for train and test data')


def recommend_papers(author, train_data, test_data=None, tfidf_matrix_train_data=None, tfidf_matrix_test_data=None):
    """
    Recommends papers for given author
    :param author: author name
    :param train_data: dataset of papers before 2020
    :param test_data: dataset of papers in 2020
    :param tfidf_matrix_train_data: tfidf matrix of train data
    :param tfidf_matrix_test_data: tfidf matrix of test data
    :return:
    """
    author_df = train_data[train_data['authors'].apply(lambda authors: author in authors)]
    trained_on = (author, len(author_df))

    if not author_df.empty:
        # Creating the TF-IDF Vectorizer and computing the cosine similarity matrix
        author_indices = author_df.index.tolist()
        authors_matrix = tfidf_matrix_train_data[author_indices]
        with open("./Data/authors/" + f'{author}_matrix.pickle', 'wb') as fin:  #added: store within directory for organization, although worked w/o
            pickle.dump(authors_matrix, fin)
        print(f'successfully pulled embedding for {author} from train data')
        cosine_sim = cosine_similarity(authors_matrix, tfidf_matrix_test_data)
        avg_cosine_sim = cosine_sim.max(axis=0)
        scores_per_paper = {test_data.iloc[i]['title']: score for i, score in enumerate(avg_cosine_sim)}

        avg_sim_scores = []

        # Iterate over the author's papers in the cosine_sim matrix
        for jdx, score in enumerate(avg_cosine_sim):
                avg_sim_scores.append((jdx, score))

        # Sort papers based on similarity scores
        avg_sim_scores = sorted(avg_sim_scores, key=lambda x: x[1], reverse=True)

        # Get the indices of the top 10 recommendations (excluding author's own papers)
        # top_paper_indices = [i[0] for i in avg_sim_scores[:10]]  #this comment was left in
        all_paper_indices = [i[0] for i in avg_sim_scores]

        recommended_df = test_data.iloc[all_paper_indices][['id_x', 'title', 'authors_x', 'abstract', 'authors_parsed',  #changed: due to the execution of get_paper_details.py, the "id" and "authors" were renamend to "id_x" and "authors_x" which was not reflected here, resulting in an error 
                                                            's2PaperId', 'corpusId', 'year']]
        recommended_df['recommended_to'] = author
        recommended_df['number of papers trained on'] = trained_on[1]
        return recommended_df, scores_per_paper
    else:
        return pd.DataFrame(columns=['id_x', 'title', 'authors_x', 'abstract', 'authors_parsed']), {}  #changed: due to the execution of get_paper_details.py, the "id" and "authors" were renamend to "id_x" and "authors_x" which was not reflected here, resulting in an error


def recommend_for_authors(authors_list, train_data, test_data=None, tfidf_matrix_train_data=None,
                          tfidf_matrix_test_data=None):
    """
    Function that calls recommendations for given author
    :param authors_list:
    :param train_data:
    :param test_data:
    :param tfidf_matrix_train_data:
    :param tfidf_matrix_test_data:
    :return:
    """
    scores_data = []
    print('authors_list', authors_list)
    all_recommendations = pd.DataFrame()
    for author in authors_list:
        recommended_papers, scores_per_paper = recommend_papers(author, train_data, test_data,
                                                                tfidf_matrix_train_data, tfidf_matrix_test_data)
        
        start_row_current_author = len(all_recommendations)  #added: starting row number where data of current author is stored

        #added: comment: concatenate the recommendations of the current author to the df of recommendations
        if not recommended_papers.empty:
            all_recommendations = pd.concat([all_recommendations, recommended_papers], ignore_index=True)
            scores_data.append({'author': author, **scores_per_paper})
        else:
            # Handle the case where no papers are recommended
            scores_data.append({'author': author})

        if all_recommendations.empty:
            all_recommendations['similarity_score'] = []
        else:
            matched_scores = []
            for index, row in recommended_papers.iterrows():  #changed: iterate over recommended papers of this author instead of all recommended papers of all authors
                title = row['title']
                if title in scores_per_paper:
                    matched_scores.append(scores_per_paper[title])
                else:
                    matched_scores.append(float(0))
            all_recommendations.loc[start_row_current_author:, "similarity_score"] = matched_scores  #changed: append the similarity scores to the "similarity_score" column starting from the correct row, instead of row 0
        print(f'successfully recommended for {author}')

    scores_df = pd.DataFrame(scores_data)
    scores_df = scores_df.set_index('author')

    return all_recommendations, scores_df


    # BEGIN ORIGINAL CODE
    # scores_data = []
    # print('authors_list', authors_list)
    # all_recommendations = pd.DataFrame()
    # for author in authors_list:
    #     recommended_papers, scores_per_paper = recommend_papers(author, train_data, test_data,
    #                                                             tfidf_matrix_train_data, tfidf_matrix_test_data)
    #     if not recommended_papers.empty:
    #         all_recommendations = pd.concat([all_recommendations, recommended_papers], ignore_index=True)
    #         scores_data.append({'author': author, **scores_per_paper})
    #     else:
    #         # Handle the case where no papers are recommended
    #         scores_data.append({'author': author})

    #     if all_recommendations.empty:
    #         all_recommendations['similarity_score'] = []
    #     else:
    #         matched_scores = []
    #         for index, row in all_recommendations.iterrows():
    #             title = row['title']
    #             if title in scores_per_paper:
    #                 matched_scores.append(scores_per_paper[title])
    #             else:
    #                 matched_scores.append(float(0))
    #         all_recommendations['similarity_score'] = matched_scores
    #     print(f'successfully recommended for {author}')
    # scores_df = pd.DataFrame(scores_data)
    # scores_df = scores_df.set_index('author')

    # return all_recommendations, scores_df
    # END ORIGINAL CODE


pulled_references = pd.read_csv('./Data/pulled_references.csv')
pulled_citations = pd.read_csv('./Data/pulled_citations.csv')
references_df = pd.read_csv('./Data/papers_and_authors.csv')
num_authors_to_select = len(unique_full_names)  #changed: to be in line with the newer version of the paper
authors_list = sorted(unique_full_names)  #changed: take all unique names of the test set; to be in line with the newer version of the paper

all_recommendations, scores_df = recommend_for_authors(authors_list, train_data, test_data, tfidf_matrix_train_data,
                                                       tfidf_matrix_test_data)
print('successfully created recommendations')
combined_recommendations = rec_referenced_by_author(all_recommendations, pulled_references, authors_list, references_df)
print('successfully rec_referenced_by_author and co-authored')
combined_recommendations = rec_cited_author(combined_recommendations, pulled_citations, authors_list, references_df)
print('successfully rec_cited_author')

scores_df.to_csv("./Data/average_author_similarity_scores_corrected.csv")  #changed: place in Data folder instead of root of project  &  changed file name w.r.t uncorrected tfidf_authors.py
combined_recommendations.to_csv("./Data/recommendations_authors_method_corrected.csv")  #changed: place in Data folder instead of root of project  &  changed file name w.r.t uncorrected tfidf_authors.py

print(f"Finished all the recommendations for {num_authors_to_select} authors")

end_time = time.perf_counter()
elapsed_time = end_time - start_time
print(f"Elapsed time: {elapsed_time} seconds")
