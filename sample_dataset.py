import pandas as pd
import time

# Sample fraction of dataset with a similar distribution of categories
def sample_dataset(df, fraction):
    df["Categories"] = df["Categories"].str.strip("[]").str.replace("'", "").str.split(", ")
    df["Category"] = df["Categories"].apply(lambda x: next(item for item in x if item.startswith("cs")) ) # Select the first CS category as the category to base distribution on
    df["Category2"] = df["Category"]  # Copy category column, as groupby().apply() operation removes the column it operates on

    sampled_df = df.groupby("Category2").apply(lambda x: x.sample(frac=fraction, random_state=42), include_groups=False)  # Out of every category, sample the specified fraction of samples (the use of include_groups=True is depricated, hence we remove the column)

    print("Papers per category before sampling:")
    print(df["Category"].value_counts())

    print("Papers per category after sampling:")
    print(sampled_df["Category"].value_counts())

    return sampled_df

start_time = time.perf_counter()
    
test_data = pd.read_csv("./Data/test_data.csv", dtype={"id": str})
categories = pd.read_csv("./Data/categories.csv")

OUTPUT_SIZE = 14307
# FRACTION = 1 / 12
FRACTION = OUTPUT_SIZE / len(test_data)

sampled_test_data = sample_dataset(test_data, FRACTION )
sampled_test_data.to_csv("./Data/test_data_sampled.csv")
print("successfully sampled " + str(FRACTION) + " of dataset")

end_time = time.perf_counter()
elapsed_time = end_time - start_time
print(f"Elapsed time: {elapsed_time} seconds")

