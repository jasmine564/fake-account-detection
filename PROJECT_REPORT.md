# Fake Account Detection Using Machine Learning and Data Science with NLP Techniques

## Abstract
The rapid growth of social media platforms has led to an increase in fake and malicious accounts, which spread misinformation and facilitate fraud. This project proposes an intelligent system to detect fake accounts using **Machine Learning (ML)** and **Natural Language Processing (NLP)**. By analyzing user profile metadata and textual content, the system leverages algorithms such as **XGBoost, Random Forest, and Support Vector Machines (SVM)** to classify accounts as genuine or fake. XGBoost is utilized as the primary algorithm due to its efficiency and high accuracy. The project demonstrates the effectiveness of combining behavioral attributes with content analysis for robust fraud detection.

## Problem Statement
Social media platforms are plagued by fake accounts used for spamming, phishing, and spreading fake news. Manual detection is inefficient and unscalable. Existing systems often focus on either metadata or text, but not both effectively. The objective is to build an automated classification system that integrates **profile metadata (e.g., follower counts, bio length)** and **textual content (posts, comments)** to accurately identify fake accounts with high precision, minimizing false positives.

## Literature Survey
- **Profile-based Detection**: Previous studies have focused on features like follower-following ratio and account age. While useful, these can be manipulated.
- **Content-based Detection**: NLP techniques like Bag-of-Words and TF-IDF have been used to analyze post patterns. Spam accounts often use repetitive or promotional language.
- **Hybrid Approaches**: Recent research suggests combining metadata and content features yields the best results. Algorithms like Random Forest and SVM are commonly used benchmarks.

## System Architecture
1. **Input**: User data (Profile details + Bio/Post text).
2. **Preprocessing**:
   - Cleaning text (Lowercase, Stemming, Stopwords removal).
   - Handling missing values.
3. **Feature Engineering**:
   - **Metadata Features**: `num_posts`, `num_followers`, `following_ratio`, `has_pic`.
   - **Text Features**: `TF-IDF` vectors from bios and posts.
4. **Modeling**: Train classifiers (XGBoost, RF, LR, SVM) on the combined feature set.
5. **Output**: Classification Label (Fake vs Real) and Probability Score.

## Module Description
### 1. Data Collection
- Simulates specific user behaviors for 'Fake' (low interaction, spam text) and 'Real' (normal distribution, organic text) accounts.
- Generates a dataset `fake_accounts_dataset.csv`.

### 2. Data Preprocessing
- **Cleaning**: Removes special characters and stopwords from text using NLTK.
- **Normalization**: Scales numerical features using Standard Scaler.

### 3. Exploratory Data Analysis (EDA)
- Visualizes class balance.
- Analyzes feature correlations (e.g., Fake accounts often have high following but low followers).

### 4. Feature Extraction & Modeling
- Converts text to numerical vectors using **TF-IDF**.
- Trains **XGBoost** (Gradient Boosting) for high performance.
- Compares with Logistic Regression and Random Forest.

## Algorithm Explanation
- **XGBoost (Extreme Gradient Boosting)**: An ensemble learning method that builds models sequentially. It corrects the errors of previous models, making it highly effective for imbalanced datasets and complex feature interactions.
- **TF-IDF (Term Frequency-Inverse Document Frequency)**: Evaluates how important a word is to a document in a collection. It helps in identifying spam keywords common in fake accounts.

## Conclusion and Future Scope
The project successfully demonstrates that integrating NLP features with profile metadata significantly enhances fake account detection. **XGBoost** outperformed traditional classifiers in precision and recall.
**Future Scope**:
- Integration with Graph Neural Networks (GNN) to analyze user connections.
- Deployment as a real-time Browser Extension or Web API.
- Deep Learning (LSTM/BERT) for advanced text analysis.
