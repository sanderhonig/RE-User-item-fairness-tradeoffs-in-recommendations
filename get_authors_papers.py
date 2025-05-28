import os
import requests
import pandas as pd
import json

import time  #added

start_time = time.perf_counter()  #added

df = pd.read_csv('./Data/test_data_details.csv')
df['authors_y'] = df['authors_y'].astype(str)

# Function to parse strings and extract author IDs
def extract_author_details(row):
    try:
        authors_list = json.loads(row.replace("'", '"'))
        return [(author['authorId'], author['name']) for author in authors_list]
    except json.JSONDecodeError:
        return []


# Apply the function to each row in the 'authors_y' column and collect all results in a list
all_author_details = []
df['authors_y'].apply(lambda x: all_author_details.extend(extract_author_details(x)))
print(all_author_details)

BASE_URL = 'https://api.semanticscholar.org/graph/v1/author/{author_id}/papers'
api_key = os.environ.get('API_KEY')

HEADERS = {
    'x-api-key': api_key,
    'Content-Type': 'application/json'
}


def fetch_papers(author_id):
    url = BASE_URL.format(author_id=author_id)
    
    for i in range(5):  #added
        response = requests.get(url, headers=HEADERS, params={'fields': 'paperId,title,year', 'limit': 1000})
        if response.status_code == 200:
            data = response.json().get('data', [])
            flattened_data = [(entry['title'], entry['paperId'], entry['year']) for entry in data if
                            'title' in entry and 'paperId' in entry and 'year' in entry and
                            entry['title'] is not None and entry['paperId'] is not None and entry['year'] is not None]
            
            print(f"Successfully fetched details for author: {author_id}")  #added
            return flattened_data
        else:  #added
            print("Attempt " + str(i+1) + f" for author: {author_id}")  #added
            time.sleep(1)  #added
    else:
        print(f"Failed to fetch details for author: {author_id}")  #changed typo: paper -> author
        print("Status Code:", response.status_code)
        print("Response:", response.text)
        return []


authors_papers = {}

for author, name in all_author_details:
    author_id = f'{author}'
    author_name = f'{name}'
    papers_list = fetch_papers(author_id)
    authors_papers[(author_id, author_name)] = papers_list

rows = []
for (author_id, author_name), papers_list in authors_papers.items():
    for ref in papers_list:
        row = {
            'Author ID': author_id,
            'Author Name': author_name,
            'Paper ID': ref[1],
            'Paper Title': ref[0],
            'Year': ref[2],
        }
        rows.append(row)

df_references = pd.DataFrame(rows)
output_csv_path = './Data/papers_and_authors.csv'
df_references.to_csv(output_csv_path, index=False)

print(f"Saved papers and authors to {output_csv_path}")

end_time = time.perf_counter()  #added
elapsed_time = end_time - start_time  #added
print(f"Elapsed time: {elapsed_time} seconds")  #added
