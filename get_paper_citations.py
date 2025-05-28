import os
import requests
import pandas as pd

import time  #added

start_time = time.perf_counter()  #added

BASE_URL = 'https://api.semanticscholar.org/graph/v1/paper/{paper_id}/citations'
api_key = os.environ.get('API_KEY')

HEADERS = {
    'x-api-key': api_key,
    'Content-Type': 'application/json'
}


def fetch_citations(paper_id):# removed: unused variables: paper_title, citations):
    url = BASE_URL.format(paper_id=paper_id)

    for i in range(5):  #added
        response = requests.get(url, headers=HEADERS, params={'fields': 'paperId,title,year,corpusId'})
        if response.status_code == 200:
            data = response.json().get('data', [])
            flattened_data = [entry['citingPaper'] for entry in data if
                            'citingPaper' in entry and entry['citingPaper'] is not None]

            print(f"Successfully fetched details for paper: {paper_id}")  #added
            return flattened_data
        else:  #added
            print("Attempt " + str(i+1) + f" for paper: {paper_id}")  #added
            time.sleep(1)  #added
    else:
        print(f"Failed to fetch details for paper: {paper_id}")
        print("Status Code:", response.status_code)
        print("Response:", response.text)
        return []

# Initialize the dictionary to store paper details and their references
papers_citations = {}
citations = []

papers_df = pd.read_csv('./Data/test_data_details.csv')
print(len(papers_df))

# Loop through paper IDs and fetch references
for index, row in papers_df.iterrows():
    paper_id = f'{row["s2PaperId"]}'
    paper_title = row['title']
    corpus_id = row['corpusId']
    citations = fetch_citations(paper_id)#removed: unused variables: paper_title, citations)
    papers_citations[(paper_id, corpus_id, paper_title)] = citations

rows = []

for (paper_id, corpus_id, paper_title), citations in papers_citations.items():
    if citations:
        for citation in citations:
            row = {
                'Source Paper ID': paper_id,
                'Source Paper Title': paper_title,
                'Source Paper CorpusID': corpus_id,
                'Citation Paper ID': citation.get('paperId', 'No Paper ID Provided'),
                'Citation Paper CorpusID': citation.get('corpusId', 'No Paper ID Provided'),
                'Citation Paper Title': citation.get('title', 'No Title Provided'),
                'Citation Year': citation.get('year', 'No Year Provided') }
            rows.append(row)
    else:  # Handle the case where there are no citations
        rows.append({
            'Source Paper ID': paper_id,
            'Source Paper Title': paper_title,
            'Source Paper CorpusID': corpus_id,
            'Citation Paper ID': 'No Citations',
            'Citation Paper Title': 'No Citations',
            'Citation Year': 'No Citations',
        })

pulled_citations = pd.DataFrame(rows)

output_csv_path = './Data/pulled_citations.csv'
pulled_citations.to_csv(output_csv_path, index=False)
print(f"Saved citations to {output_csv_path}")  #added this for consistency with other get_ files

end_time = time.perf_counter()  #added
elapsed_time = end_time - start_time  #added
print(f"Elapsed time: {elapsed_time} seconds")  #added
