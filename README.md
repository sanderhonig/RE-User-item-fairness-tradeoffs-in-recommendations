# User-Item Fairness Tradeoffs in Recommendations


## Overview
This repository houses the source code for a research project aimed at exploring user-item fairness tradeoffs in recommendation systems. The project encompasses both theoretical frameworks and empirical evaluations to assess and improve fairness in recommendations. 

Specifically, this project serves as a reproducability study of the NeurIPS 2024 paper, [User-item fairness tradeoffs in recommendations](https://neurips.cc/virtual/2024/poster/94638). Our primary goal is to replicate the empirical findings of this paper. 

Beyond replication, the original research is extended in two directions:
1. We verify the generalizability of their findings on a different dataset ([Amazon books reviews](https://www.kaggle.com/datasets/mohamedbakhet/amazon-books-reviews)).
2. We analyze the tradeoffs when recommending multiple items to a user instead of a single item.

The original codebase can be found at [https://github.com/vschiniah/ArXiv_Recommendation_Research](https://github.com/vschiniah/ArXiv_Recommendation_Research). 

For more details about this study, see our paper submission on Openreview: [A reproducibility study of “User-item fairness tradeoffs in recommendations”](https://openreview.net/forum?id=vltzxxhzLU).

## Repository Structure
- **./Amazon/amazon_exp_preparation**: Preparation for experiments, creating pickle files for tfidf matrices and pickle files for reviewers for Amazon Books Reviews dataset.
- **./Amazon/amazon_logreg**: Logistic regression for Amazon Books Reviews dataset.
- **./Amazon/books_data_exploration**: Data exploration of the books of Amazon Books Reviews dataset.
- **create_categories.py**: Generates `categories.csv` for explore_data.py.
- **distribution_plots.py**: Creates barplots of the distribution of research paper publications in the train and test set over time.
- **explore_data.py**: Script for getting main and sub categories information, and separate train and test datasets.
- **generate_data_source_file.py**: Generates a single data file containing the embeddings of all authors and all papers of the test set. This is required input for the experiments.
- **get_authors_papers.py**: Script to fetch all authors and their published papers from Semantic Scholar.
- **get_paper_citations.py**: Retrieves citation data for the recommended papers from Semantic Scholar.
- **get_paper_details.py**: Fetches information about papers such as semantic scholar ID etc for the citation/references.
- **get_references.py**: Collects references for the recommended papers from Semantic Scholar.
- **import_metadata.py**: Script for importing and processing metadata from Kaggle.
- **logistic_regression.py**: Script that performs logistic regression for arXiv dataset.
- **model_evaluation.py**: Contains functions to evaluate the recommendation model.
- **requirements.txt**: Lists all the dependencies required to run the scripts.
- **./Amazon/reviews_data_exploration**: Data exploration of the reviews of Amazon Books Reviews dataset.
- **sample_random_authors.py**: Samples part of the author pickle files.
- **sample_dataset.py**: Samples part of dataset proportionally to category distribution. 
- **sentence_transformer_authors.py**: Get recommendations using Sentence Transformer and cosine similarity.
- **stopwords.txt**: Text file containing stopwords used in text processing.
- **tfidf_authors.py**: Get recommendations using TF-IDF embeddings and cosine similarity.
- **tfidf_authors_corrected.py**: Get recommendations using TF-IDF embeddings and cosine similarity without overwriting previous similarity scores.
- **utils.py**: Utility functions used across the project.

## Installation

### Prerequisites
- Python 3.8 or newer
- pip
- Semantic Scholar API Key

### Setup
Clone the repository and install the required dependencies:
```bash
pip install -r requirements.txt
```

### ArXiv Dataset
The original dataset was sourced from the public ArXiv Dataset available on [Kaggle](https://www.kaggle.com/datasets/Cornell-University/arxiv/data).

#### Step 1
Convert the dataset acquired as JSON into CSV for easier processing.
```bash
python3 import_metadata.py
```
#### Step 2
Generate `categories.csv` for the next script.
```bash
python3 create_categories.py
```
#### Step 3
Select only papers in Computer Science category, and split dataset in train and test set.
```bash
python3 explore_data.py
```
#### Step 4
Acquire additional details for each paper in the test set, necessary for logistic regression.
```bash
python3 get_paper_details.py
```
#### Step 5
*(Optional: sample test set)*
Sample the test (and/or train) set to the desired size.
```bash
python3 sample_dataset.py
```
#### Step 6
Acquire additional details for all authors, external papers that cited a paper of the dataset, and all papers that a paper in the dataset cites.
```bash
python3 get_authors_papers.py
python3 get_paper_citations.py
python3 get_references.py
```
#### Step 7
Generate TF-IDF embeddings for all papers of the train and test set, and all authors that appear in both based on cosine similarity.
```bash
python3 tfidf_authors.py
```
#### Step 8
Randomly sample 1,000 authors for the experiments.
```bash
python3 sample_random_authors.py
```
#### Step 9
Generate a single data file containing the embeddings of all authors and all papers of the test set. This is required input for the experiments.
```bash
python3 generate_data_source_file.py
```
#### Step 10
Perform logistic regression on arXiv dataset.
```bash
python3 logistic_regression.py
```

### Amazon Books Reviews Dataset
The dataset for the extension was sourced from Kaggle, available on [Kaggle](https://www.kaggle.com/datasets/mohamedbakhet/amazon-books-reviews/data)

#### Step 1
Split dataset in train and test set, create embeddings for all books, create embeddings for 1,000 random authors.
```bash
cd ./Amazon
```
```bash
python3 amazon_exp_preparation.py
```
#### Step 2
Generate a single data file containing the embeddings of all authors and all papers of the test set. This is required input for the experiments.
```bash
python3 ../generate_data_source_file.py --df ./Amazon
```
#### Step 3
Perform logistic regression on Amazon dataset.
```bash
python3 amazon_logreg.py
``` 
