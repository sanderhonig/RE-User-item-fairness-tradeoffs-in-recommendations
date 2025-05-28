#
# This file samples authors from all available pickle files, which ensures the sampled 
# author exists in both train and test set. No data is removed, only the following is performed:
#  - Pickle files of unused authors are moved to the folder ./Data/authors_unused
#  - The original csv files are renamed, while new csv files are created with only information 
#    of the sampled authors  
#
# This seperate file is created to leave most other code intact and easiliy remove sampling in the future
#
import os
import random
import shutil
import pandas as pd

SAMPLE_SIZE = 1000
SEED = 21


# List all available files
directory = "./Data/"
all_files = [f for f in os.listdir(directory + "authors/") if os.path.isfile(os.path.join(directory + "authors/", f))]

# Sample SAMPLE_SIZE nr of files
random.seed(SEED)
sampled_files = random.sample(all_files, SAMPLE_SIZE)  # An error is automatically raised when num_files > all_files 
sampled_names = [name.split('_')[0] for name in sampled_files]    # Convert sampled pickle file names to author names
print(f"the following {SAMPLE_SIZE} authors were sampled using seed {SEED}:")
print(sampled_names)


# Move all other pickle files to authors_unused folder
os.makedirs(directory + "authors_unused/", exist_ok=True)  # Create folder
for filename in all_files:
    if filename not in sampled_files:
        shutil.move(directory + "authors/" + filename, directory + "authors_unused/" + filename)
print("pickle files selected")
