import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

df_test = pd.read_csv('./Data/test_data.csv', dtype={"id": str})
df_train = pd.read_csv('./Data/train_data.csv', dtype={"id": str})
df_categories = pd.read_csv('./Data/categories.csv')

def plot_distribution(df, df_categories, data_name):
    df['Categories'] = df['Categories'].str.strip("[]").str.replace("'", "").str.split(', ')
    df['Categories'] = df['Categories'].apply(lambda x: [cat for cat in x if cat.startswith('cs.')]) # Filter out non-CS categories
    df = df.explode('Categories').reset_index(drop=True) # Create new rows for each non-CS category
    df = df.merge(df_categories[['ID', 'Name']], left_on='Categories', right_on='ID', how='left')
    if df['ID'].isna().any():
        print('Merge failed')
    df = df['Name'].value_counts() # Count the number of publications in each sub-category
    print(f'Total number of occurences in the categories in the {data_name} dataset: {df.sum()}')
    
    # Create a barplot of the distribution of research paper publications in the test dataset over time
    plt.figure(figsize=(10, 6))
    sns.barplot(x=df.index, y=df.values, hue=df.index, palette='viridis', dodge=False, legend=False)
    plt.title(f'Distribution of research paper publications in the {data_name} dataset over time', fontsize=12)
    plt.xlabel('Computer Science Sub-categories', fontsize=10)
    plt.ylabel('Number of Publications', fontsize=10)
    plt.xticks(rotation=45,ha='right', fontsize=6)
    plt.yticks(fontsize=12)
    plt.tight_layout()
    
    # Save the plot as a .png file
    file_name = f"figures/{data_name}_distribution.png"
    plt.savefig(file_name, dpi=300, bbox_inches='tight')
    print(f"Plot saved in {file_name}")
    
    plt.show()
    
plot_distribution(df_test, df_categories, 'test')
plot_distribution(df_train, df_categories, 'train')