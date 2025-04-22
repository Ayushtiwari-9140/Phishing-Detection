from flask import Flask, render_template, request
import joblib
from urllib.parse import urlparse  # Add this import
from utils.rag_scraper import fetch_rag_results
import numpy as np

app = Flask(__name__)

# Load model and vectorizer
try:
    model = joblib.load("model/phishing_model.pkl")
    vectorizer = joblib.load("model/vectorizer.pkl")
except Exception as e:
    raise RuntimeError("Failed to load model artifacts") from e

def preprocess_url(url):
    """Replicate the preprocessing used during training"""
    # Clean URL
    clean_url = url.strip()
    
    # Extract components
    parsed = urlparse(clean_url)
    domain = parsed.netloc
    path = parsed.path
    
    # Create text features same as training
    text_features = f"{clean_url} {domain} {path}"
    
    return text_features

@app.route("/", methods=["GET", "POST"])
def home():
    result = None
    rag_data = None
    error = None
    
    if request.method == "POST":
        url = request.form.get("url", "").strip()
        if not url:
            error = "Please enter a URL"
        else:
            try:
                # Replicate training preprocessing
                processed_url = preprocess_url(url)
                
                # Vectorize using the saved vectorizer
                url_vector = vectorizer.transform([processed_url])
                
                # Get prediction probabilities
                proba = model.predict_proba(url_vector)[0]
                prediction = model.predict(url_vector)[0]
                
                result = {
                    "class": "🔴 Phishing Detected!" if prediction == 1 else "🟢 Genuine URL",
                    "confidence": f"{max(proba)*100:.2f}%",
                    "phishing_prob": f"{proba[1]*100:.2f}%",
                    "legit_prob": f"{proba[0]*100:.2f}%"
                }
                
                # Get RAG results (ensure this function is properly implemented)
                rag_data = fetch_rag_results(url) if fetch_rag_results else None

            except Exception as e:
                error = f"Error processing URL: {str(e)}"

    return render_template(
        "index.html", 
        result=result, 
        rag=rag_data,
        error=error
    )

if __name__ == "__main__":
    app.run(debug=False)  # Disable debug in production