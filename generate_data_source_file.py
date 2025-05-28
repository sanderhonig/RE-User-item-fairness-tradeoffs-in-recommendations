import pickle
import os
import numpy as np
import argparse

import time

def create_file(dir):
    #obtain paper embeddings
    with open(str(dir) + "/tfidf_matrix_test_data.pickle", "rb") as file:
        test_paper_embeddings = pickle.load(file)

    #obtain author embeddings
    authors = []
    for file in os.listdir(str(dir) + "/authors/"):
        filename = os.fsdecode(file)
        print(filename)
        author_name = filename.split("_")[0]

        print("Processing author: " + author_name)

        #matrix of embeddings across all their written papers
        with open(str(dir) + "/authors/" + filename, "rb") as file:
            author_train_paper_embeddings = pickle.load(file)

        author = {
            "name" : author_name,
            "embedding" : author_train_paper_embeddings
            }

        #append the data of current author to list
        authors.append(author)

    obj = {
        "authors" : authors,
        "papers" : test_paper_embeddings
        }


    #store dictionary as pickle file for use in experiments 2a and 2b
    with open(str(dir) + "/data_source_file_experiments.pickle", "wb") as fin:
        pickle.dump(obj, fin)

    print("successfully created data source file for experiments")


if __name__ == "__main__":
    start_time = time.perf_counter()

    parser = argparse.ArgumentParser()
    parser.add_argument("--df", type=str, help="Data directory", default="./Data")
    dir = parser.parse_args().df
    
    create_file(dir)

    end_time = time.perf_counter()
    elapsed_time = end_time - start_time
    print(f"Elapsed time: {elapsed_time} seconds")