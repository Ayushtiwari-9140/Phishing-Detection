from flask import Flask, render_template, request
import joblib
from utils.rag_scraper import fetch_rag_results

app = Flask(__name__)

# Load model
model = joblib.load("model/phishing_model.pkl")
vectorizer = joblib.load("model/vectorizer.pkl")

@app.route("/", methods=["GET", "POST"])
def home():
    result = None
    rag_data = None
    if request.method == "POST":
        url = request.form["url"]
        url_vector = vectorizer.transform([url])
        prediction = model.predict(url_vector)[0]
        result = "🔴 Phishing Detected!" if prediction == 1 else "🟢 Genuine URL"

        rag_data = fetch_rag_results(url)

    return render_template("index.html", result=result, rag=rag_data)

if __name__ == "__main__":
    app.run(debug=True)
