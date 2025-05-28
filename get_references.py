import os
import requests
import pandas as pd

import time  #added
start_time = time.perf_counter()  #added

BASE_URL = 'https://api.semanticscholar.org/graph/v1/paper/{paper_id}/references'
api_key = os.environ.get('API_KEY')

HEADERS = {
    'x-api-key': api_key,
    'Content-Type': 'application/json'
}


def fetch_references(paper_id): #removed: , paper_title, references):
    """
    Fetches references for a given paper and title.
    :param paper_id: paper_id
    :param paper_title: paper_title
    :param references: Save in references dict
    :return:
    """
    url = BASE_URL.format(paper_id=paper_id)

    for i in range(5):  #added
        response = requests.get(url, headers=HEADERS, params={'fields': 'paperId,title,year,corpusId'})
        if response.status_code == 200:
            data = response.json().get('data', [])
            flattened_data = [entry['citedPaper'] for entry in data if
                            'citedPaper' in entry and entry['citedPaper'] is not None]

            print(f"Successfully fetched references for paper: {paper_id}")  #added
            return flattened_data
        else:  #added
            print("Attempt " + str(i+1) + f" for paper: {paper_id}")  #added
            time.sleep(1)  #added
    else:
        print(f"Failed to fetch references for paper: {paper_id}")
        print("Status Code:", response.status_code)
        print("Response:", response.text)
        return []


papers_df = pd.read_csv('./Data/test_data_details.csv')

papers_with_references = {}
references = []
print(len(papers_df))
# Loop through paper IDs and fetch references
for index, row in papers_df.iterrows():
    paper_id = f'{row["s2PaperId"]}'
    paper_title = row['title']
    corpus_id = row['corpusId']
    references = fetch_references(paper_id) #removed: , paper_title, references)
    papers_with_references[(paper_id, corpus_id, paper_title)] = references

rows = []

for (paper_id, corpus_id, paper_title), references in papers_with_references.items():
    if not references:  # Check if references list is empty
        # Create a row with empty reference information
        row = {
            'Source Paper ID': paper_id,
            'Source Paper Title': paper_title,
            'Source Paper Corpus ID': corpus_id,
            'Reference Paper ID': '',
            'Reference Title': '',
            'Reference Year': '',
            'Reference Authors': ''
        }
        rows.append(row)
    else:
        for ref in references:
            # If ref is a dictionary and not empty
            if ref and isinstance(ref, dict):
                row = {
                    'Source Paper ID': paper_id,
                    'Source Paper Title': paper_title,
                    'Source Paper Corpus ID': corpus_id,
                    'Reference Paper ID': ref.get('paperId', ''),
                    'Reference Paper Corpus ID': ref.get('corpusId', ''),
                    'Reference Title': ref.get('title', ''),
                    'Reference Year': ref.get('year', '')
                }
            else:
                # Create a row with empty reference information if ref is empty
                row = {
                    'Source Paper ID': paper_id,
                    'Source Paper Title': paper_title,
                    'Source Paper Corpus ID': corpus_id,
                    'Reference Paper ID': '',
                    'Reference Paper Corpus ID': '',
                    'Reference Title': '',
                    'Reference Year': '',
                    'Reference Authors': ''
                }
            rows.append(row)


pulled_references = pd.DataFrame(rows)
output_csv_path = './Data/pulled_references.csv'  #changed ./pulled_references.csv  -> ./Data/pulled_references.csv
pulled_references.to_csv(output_csv_path)

print(f"Saved references to {output_csv_path}")  #added this for consistency with other get files

end_time = time.perf_counter()  #added
elapsed_time = end_time - start_time  #added
print(f"Elapsed time: {elapsed_time} seconds")  #added