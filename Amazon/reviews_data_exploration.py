#
# This file was used to explore characteristics of reviews in the Amazon Books Reviews dataset
# and create a plot of the number of reviews per year
#

import pandas as pd
import matplotlib.pyplot as plt

# Read the CSV file into a DataFrame
df_books_rating = pd.read_csv("./Data/Books_rating.csv")

# Read the CSV file into a DataFrame
df_books_data = pd.read_csv("./Data/books_data.csv")

# Display the first few rows of the DataFrame
print(df_books_rating.head())

# Get unique values in the "User_id" column
unique_user_ids = df_books_rating["User_id"].unique()

# Get the count of unique User_ids
print(f"Number of unique User_ids: {len(unique_user_ids)}")

# Get the minimum and maximum values of the 'review/time' column
min_time = df_books_rating['review/time'].min()
max_time = df_books_rating['review/time'].max()

print(f"Minimum value of 'review/time': {min_time}")
print(f"Maximum value of 'review/time': {max_time}")

# Convert 'review/time' from Unix time to human-readable datetime
df_books_rating['review/time'] = pd.to_datetime(df_books_rating['review/time'], unit='s')

# Group by year and count the number of books reviewed
reviews_per_year = df_books_rating['review/time'].dt.to_period('Y').value_counts().sort_index()

# Plot the data
plt.figure(figsize=(10, 6))
reviews_per_year.plot(kind='bar', color='skyblue', edgecolor='black')
plt.title("Reviews per year")
plt.xlabel("Year")
plt.ylabel("Number of reviews")
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()
plt.savefig("./reviews_per_year.png")

# Perform a left outer join on 'Title' column
merged_df = pd.merge(df_books_rating, df_books_data, on='Title', how='left')

# Display the first few rows of the merged DataFrame
print(merged_df.head())

# Count the number of NaN values in the 'description' column
nan_count = merged_df['description'].isna().sum()

# Print the count of NaN values
print(f"Number of NaN values in 'description' column: {nan_count}")

# Drop rows with NaN values in the 'description' column
merged_df_clean = merged_df.dropna(subset=['description'])
merged_df_clean = merged_df_clean[merged_df_clean['review/time'] != -1] # drop unix time = -1

# Convert 'review/time' from Unix time to datetime
merged_df_clean['review/time'] = pd.to_datetime(merged_df_clean['review/time'], unit='s')
merged_df_clean['review_year'] = pd.to_datetime(merged_df_clean['review/time'], errors='coerce').dt.year
merged_df_clean['published_year'] = pd.to_datetime(merged_df_clean['publishedDate'], errors='coerce').dt.year

print(merged_df_clean[['Title', 'review_year', 'published_year']].head(10))