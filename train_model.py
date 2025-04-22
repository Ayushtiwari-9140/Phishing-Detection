import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score
import joblib
import os
from urllib.parse import urlparse

# STEP 1: Load Dataset
def load_data(phishing_path, legitimate_path, sample_size=100000):
    # Load phishing URLs
    phishing_df = pd.read_csv(phishing_path, names=['url'], nrows=sample_size)
    phishing_df['label'] = 1
    
    # Load legitimate URLs
    legitimate_df = pd.read_csv(legitimate_path, names=['url'], nrows=sample_size)
    legitimate_df['label'] = 0
    
    # Combine and shuffle datasets
    df = pd.concat([phishing_df, legitimate_df], axis=0).sample(frac=1, random_state=42).reset_index(drop=True)
    return df

# STEP 2: Preprocess with enhanced features
def preprocess(df):
    # Basic URL cleaning
    df['url'] = df['url'].apply(lambda x: x.strip())
    
    # Extract URL components
    df['domain'] = df['url'].apply(lambda x: urlparse(x).netloc)
    df['path'] = df['url'].apply(lambda x: urlparse(x).path)
    
    # Create combined text features
    df['text_features'] = df['url'] + ' ' + df['domain'] + ' ' + df['path']
    
    # TF-IDF with character n-grams
    vectorizer = TfidfVectorizer(
        analyzer='char',
        ngram_range=(3, 5),
        max_features=10000,
        min_df=0.001,
        max_df=0.7
    )
    
    X = vectorizer.fit_transform(df['text_features'])
    y = df['label']
    
    return X, y, vectorizer

# STEP 3: Train model with improved parameters
def train_model(X, y):
    # Stratified split to maintain class balance
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, 
        test_size=0.3, 
        random_state=42,
        stratify=y
    )
    
    # Optimized Random Forest
    model = RandomForestClassifier(
        n_estimators=200,
        max_depth=25,
        min_samples_split=5,
        class_weight='balanced',
        n_jobs=-1,
        random_state=42
    )
    
    model.fit(X_train, y_train)
    
    # Evaluation
    y_pred = model.predict(X_test)
    print(f"Accuracy: {accuracy_score(y_test, y_pred):.4f}")
    print("\nClassification Report:\n", classification_report(y_test, y_pred))
    
    return model

# STEP 4: Save artifacts
def save_artifacts(model, vectorizer):
    os.makedirs("model", exist_ok=True)
    joblib.dump(model, "model/phishing_model.pkl")
    joblib.dump(vectorizer, "model/vectorizer.pkl")
    print("Artifacts saved in /model/ folder.")

if __name__ == "__main__":
    # Update these paths to match your dataset locations
    PHISHING_PATH = "data/phishing_site_urls.csv"
    LEGITIMATE_PATH = "data/legitimate_urls.csv"
    
    # Load data - adjust sample_size based on your system's memory
    df = load_data(PHISHING_PATH, LEGITIMATE_PATH, sample_size=50000)
    
    # Preprocess and train
    X, y, vectorizer = preprocess(df)
    model = train_model(X, y)
    
    # Save model
    save_artifacts(model, vectorizer)