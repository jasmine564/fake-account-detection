import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os

def perform_eda(input_file='cleaned_data.csv'):
    print("Starting Exploratory Data Analysis...")
    try:
        df = pd.read_csv(input_file)
    except FileNotFoundError:
        print(f"Error: {input_file} not found. Run preprocessing.py first.")
        return

    # Create plots directory
    if not os.path.exists('plots'):
        os.makedirs('plots')
    
    # 1. Class Distribution
    plt.figure(figsize=(6, 4))
    sns.countplot(x='is_fake', data=df)
    plt.title('Distribution of Fake vs Real Accounts')
    plt.xlabel('Is Fake (0: Real, 1: Fake)')
    plt.ylabel('Count')
    plt.savefig('plots/class_distribution.png')
    plt.close()
    
    # 2. Correlation Matrix
    plt.figure(figsize=(10, 8))
    numeric_df = df.select_dtypes(include=['float64', 'int64'])
    sns.heatmap(numeric_df.corr(), annot=True, cmap='coolwarm', fmt='.2f')
    plt.title('Feature Correlation Matrix')
    plt.savefig('plots/correlation_matrix.png')
    plt.close()
    
    # 3. Pairplot
    sns.pairplot(df, hue='is_fake', vars=['num_posts', 'num_followers', 'num_following', 'bio_len'])
    plt.savefig('plots/pairplot.png')
    plt.close()

    print("EDA complete. Plots saved in 'plots/' directory.")

if __name__ == "__main__":
    perform_eda()
