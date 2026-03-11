import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, classification_report
import xgboost as xgb
import warnings

warnings.filterwarnings('ignore')

def train_and_evaluate(input_file='cleaned_data.csv'):
    print("Loading data for model training...")
    try:
        df = pd.read_csv(input_file)
    except FileNotFoundError:
        print(f"Error: {input_file} not found. Run preprocessing.py first.")
        return

    # Fill NaN in text if any remain
    df.fillna({'cleaned_text': ''}, inplace=True)

    # Feature Engineering
    print("Performing Feature Engineering...")
    
    # Text Features (TF-IDF)
    tfidf = TfidfVectorizer(max_features=5000)
    X_text = tfidf.fit_transform(df['cleaned_text']).toarray()
    
    # Numeric Features
    X_numeric = df[['has_profile_pic', 'num_posts', 'num_followers', 'num_following', 'bio_len']].values
    scaler = StandardScaler()
    X_numeric_scaled = scaler.fit_transform(X_numeric)
    
    # Combine Features
    X = np.hstack((X_numeric_scaled, X_text))
    y = df['is_fake'].values
    
    # Train Test Split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Models
    models = {
        'Logistic Regression': LogisticRegression(),
        'Random Forest': RandomForestClassifier(n_estimators=100),
        'SVM': SVC(),
        'XGBoost': xgb.XGBClassifier(use_label_encoder=False, eval_metric='logloss')
    }
    
    results = {}
    
    print("\nTraining and Evaluating Models...")
    print("-" * 60)
    print(f"{'Model':<20} | {'Accuracy':<10} | {'Precision':<10} | {'Recall':<10} | {'F1-Score':<10}")
    print("-" * 60)
    
    best_model_name = ""
    best_f1 = 0
    
    for name, model in models.items():
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        
        acc = accuracy_score(y_test, y_pred)
        prec = precision_score(y_test, y_pred)
        rec = recall_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred)
        
        results[name] = f1
        
        print(f"{name:<20} | {acc:.4f}     | {prec:.4f}      | {rec:.4f}    | {f1:.4f}")
        
        if f1 > best_f1:
            best_f1 = f1
            best_model_name = name

        # Plot Confusion Matrix for XGBoost (or all, but XGBoost required)
        if name == 'XGBoost':
            cm = confusion_matrix(y_test, y_pred)
            plt.figure(figsize=(5, 4))
            sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
            plt.title(f'Confusion Matrix - {name}')
            plt.ylabel('Actual')
            plt.xlabel('Predicted')
            plt.savefig(f'plots/confusion_matrix_{name}.png')
            plt.close()
            
    print("-" * 60)
    print(f"\nBest Performing Model: {best_model_name} with F1-Score: {best_f1:.4f}")
    
    if best_model_name == 'XGBoost':
        print("\nConclusion: XGBoost demonstrated superior performance as expected.")
    else:
        print(f"\nConclusion: {best_model_name} performed best in this run.")

if __name__ == "__main__":
    train_and_evaluate()
