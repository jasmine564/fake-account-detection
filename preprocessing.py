import pandas as pd
import nltk
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
import re

# Download NLTK data
try:
    nltk.data.find('corpora/stopwords')
except LookupError:
    nltk.download('stopwords')

def preprocess_text(text):
    if not isinstance(text, str):
        return ""
    
    # Lowercase
    text = text.lower()
    
    # Remove special characters and numbers
    text = re.sub(r'[^a-zA-Z\s]', '', text)
    
    # Tokenize (split by space)
    words = text.split()
    
    # Remove stopwords and stemming
    stop_words = set(stopwords.words('english'))
    stemmer = PorterStemmer()
    
    cleaned_words = [stemmer.stem(word) for word in words if word not in stop_words]
    
    return " ".join(cleaned_words)

def clean_data(input_file='fake_accounts_dataset.csv', output_file='cleaned_data.csv'):
    print("Starting data preprocessing...")
    try:
        df = pd.read_csv(input_file)
    except FileNotFoundError:
        print(f"Error: {input_file} not found. Run data_generation.py first.")
        return

    # Handle missing values
    df.fillna({'text_content': ''}, inplace=True)
    df.fillna(0, inplace=True)
    
    # Remove duplicates
    df.drop_duplicates(inplace=True)
    
    # Text cleaning
    print("Cleaning text data...")
    df['cleaned_text'] = df['text_content'].apply(preprocess_text)
    
    df.to_csv(output_file, index=False)
    print(f"Preprocessing complete. Saved to {output_file}")

if __name__ == "__main__":
    clean_data()
