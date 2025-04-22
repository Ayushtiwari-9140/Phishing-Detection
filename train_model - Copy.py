import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score
import joblib
import os

# STEP 1: Load Dataset
def load_data():
    data = {
        "url": [
            "http://login-facebook.com/update",
            "https://secure.paypal.com/account",
            "http://verifybankofamerica.com",
            "https://www.google.com",
            "https://www.apple.com/support",
            "http://paypal.login.verify-user.com",
            "https://amazon.com",
            "http://secure-update-facebook.account-security.com",
            "https://github.com",
            "http://ebay-login-update.com"
        ],
        "label": [1, 0, 1, 0, 0, 1, 0, 1, 0, 1]
    }
    return pd.DataFrame(data)

# STEP 2: Preprocess
def preprocess(df):
    vectorizer = TfidfVectorizer()
    X = vectorizer.fit_transform(df['url'])
    y = df['label']
    return X, y, vectorizer

# STEP 3: Train model
def train_model(X, y):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    print("Accuracy:", accuracy_score(y_test, y_pred))
    print("\nClassification Report:\n", classification_report(y_test, y_pred))

    return model

# STEP 4: Save model and vectorizer
def save_artifacts(model, vectorizer):
    os.makedirs("model", exist_ok=True)
    joblib.dump(model, "model/phishing_model.pkl")
    joblib.dump(vectorizer, "model/vectorizer.pkl")
    print("Artifacts saved in /model/ folder.")

if __name__ == "__main__":
    df = load_data()
    X, y, vectorizer = preprocess(df)
    model = train_model(X, y)
    save_artifacts(model, vectorizer)
