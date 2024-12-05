#app/app.py
from flask import Flask, request, jsonify, render_template
import pickle
import re
from io import BytesIO
from sklearn.feature_extraction.text import TfidfVectorizer
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import base64
import nltk

nltk.download('stopwords')
STOPWORDS = set(stopwords.words('english'))
stemmer = PorterStemmer()

app = Flask(__name__)

# Load pre-trained models and vectorizer
tfidf = pickle.load(open('models/tfidf_vectorizer.pkl', 'rb'))
logistic_model = pickle.load(open('models/logistic_model.pkl', 'rb'))
mlp_model = load_model('models/mlp_model.h5')
tokenizer = pickle.load(open('models/tokenizer.pkl', 'rb'))

# Preprocessing function
def preprocess_text(text):
    review = re.sub('[^a-zA-Z]', ' ', text)
    review = review.lower().split()
    review = [stemmer.stem(word) for word in review if word not in STOPWORDS]
    return ' '.join(review)

@app.route("/test", methods=["GET"])
def test():
    return "Test request received successfully. Service is running."
@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Extract input review and helpfulness metrics from the request
        text_input = request.json.get('review', '')
        helpfulness_numerator = request.json.get('helpfulness_numerator', 0)
        helpfulness_denominator = request.json.get('helpfulness_denominator', 1)

        # Preprocess input text
        processed_text = preprocess_text(text_input)

        # Logistic Regression Prediction
        tfidf_features = tfidf.transform([processed_text])
        logistic_prediction = logistic_model.predict(tfidf_features)[0]

        # Preprocess helpfulness metrics (normalize as done in training)
        helpfulness_features = np.array([[helpfulness_numerator, helpfulness_denominator]])

        # MLP Prediction
        sequence = tokenizer.texts_to_sequences([processed_text])
        padded_sequence = pad_sequences(sequence, maxlen=150)
        mlp_prediction = mlp_model.predict([padded_sequence, helpfulness_features])[0][0]

        # Map predictions to labels
        logistic_sentiment = 'Positive' if logistic_prediction == 1 else 'Negative'
        mlp_sentiment = 'Positive' if mlp_prediction > 0.5 else 'Negative'

        response = {
            "Logistic Regression Sentiment": logistic_sentiment,
            "MLP Sentiment": mlp_sentiment
        }
        return jsonify(response)

    except Exception as e:
        return jsonify({'error': str(e)})

if __name__ == '__main__':
    app.run(debug=True)
