#
# This file was used to explore characteristics of books in the Amazon Books Reviews dataset
# and create a plot of the number of books per year
#

import pandas as pd
import matplotlib.pyplot as plt

# Read the CSV file into a DataFrame
df = pd.read_csv("./Data/books_data.csv")

# Display the a row of the 'description' column
print(df['description'].iloc[8])

# Count the number of NaN values in the 'description' column
nan_count = df['description'].isna().sum()

# Print the count of NaN values
print(f"Number of NaN values in 'description' column: {nan_count}")

df = df.dropna(subset=['description'])

# Extract the year from 'publishedDate' and create a new 'publishedYear' column
# Handle cases where 'publishedDate' may be NaN or not in a standard format
df['publishedYear'] = pd.to_datetime(df['publishedDate'], errors='coerce').dt.year

# Count the number of books published in 2007
books_published_2007 = df[df['publishedYear'] == 2007].shape[0]
books_published_2006 = df[df['publishedYear'] == 2006].shape[0]
books_published_2005 = df[df['publishedYear'] == 2005].shape[0]

# Print the count
print(f"Number of books published in 2007: {books_published_2007}")
print(f"Number of books published in 2007 6 5: {books_published_2007 + books_published_2006 + books_published_2005}")

# Group by 'publishedYear' and count the number of books for each year
books_per_year = df['publishedYear'].value_counts().sort_index()

# Filter for years 1950 and above
books_per_year = books_per_year[books_per_year.index >= 1950]

# Plot the data as a bar plot
plt.figure(figsize=(12, 6))
plt.bar(books_per_year.index, books_per_year.values, color='skyblue', edgecolor='black')
plt.title('Number of Books Published Per Year', fontsize=14)
plt.xlabel('Year', fontsize=12)
plt.ylabel('Number of Books', fontsize=12)
plt.xticks(rotation=45)
plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.tight_layout()
plt.show()
plt.savefig("./books_per_year.png")

# Print the total number of books
total_books = df.shape[0]
print(f"Total number of books in the dataset: {total_books}")