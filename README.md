# Phishing-Detection
# Phishing URL Detection Web App

This project is a Flask-based web application that detects phishing websites using machine learning. The model is trained on URL datasets and provides predictions along with a confidence score and relevant information from web scraping (RAG-based insights).


## Features

- Machine Learning-based phishing URL detection
- Flask web interface to input and analyze URLs
- Displays confidence levels and classification reports
- Web scraping for additional real-time URL insights

## Model

The model is a Random Forest classifier trained using TF-IDF vectorized URL data. It considers components like domain and path during preprocessing.



## Installation

git clone https://github.com/Ayushtiwari-9140/Phishing-Detection.git
cd phishing detector
pip install -r requirements.txt
