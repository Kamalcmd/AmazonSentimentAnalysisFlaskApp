#models/ReviewSentimentAnalysis.py
# Import necessary libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import classification_report, confusion_matrix
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Embedding, LSTM, Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from nltk.corpus import stopwords
from nltk.tokenize import RegexpTokenizer
from nltk.stem import PorterStemmer
import nltk
import os
import warnings

# Disable TensorFlow logs and warnings
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
warnings.filterwarnings('ignore')

# Download NLTK data
nltk.download('stopwords')
stop_words = set(stopwords.words('english'))
stemmer = PorterStemmer()
tokenizer = RegexpTokenizer(r'\w+')

# Load dataset
df = pd.read_csv('./data/Reviews.csv')

# Reduce dataset size for faster processing
df = df.sample(n=50000, random_state=42)

# Drop unnecessary columns and preprocess labels
df = df[['Score', 'Text']].drop_duplicates()
df['Text'] = df['Text'].fillna('')
df['Score'] = df['Score'].apply(lambda x: 1 if x >= 3 else 0)

# Optimized text preprocessing
def clean_text(series):
    return series.str.lower().str.replace(r'[^\w\s]', '', regex=True).str.split().apply(
        lambda x: ' '.join([stemmer.stem(word) for word in x if word not in stop_words])
    )

df['cleaned_text'] = clean_text(df['Text'])

# Split data
X = df['cleaned_text']
y = df['Score']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# TF-IDF Vectorization
tfidf = TfidfVectorizer(max_features=20000, ngram_range=(1, 2), stop_words='english')
X_train_tfidf = tfidf.fit_transform(X_train)
X_test_tfidf = tfidf.transform(X_test)

# Logistic Regression Model
log_model = LogisticRegression(max_iter=300, solver='saga', n_jobs=-1)
log_model.fit(X_train_tfidf, y_train)

# Naive Bayes Model
nb_model = MultinomialNB()
nb_model.fit(X_train_tfidf, y_train)

# Tokenization for LSTM
tokenizer = Tokenizer(num_words=20000)
tokenizer.fit_on_texts(X)
X_seq = tokenizer.texts_to_sequences(X)

# Padding sequences
max_len = 150
X_padded = pad_sequences(X_seq, maxlen=max_len)

# Split padded data
X_train_padded, X_test_padded, y_train, y_test = train_test_split(X_padded, y, test_size=0.2, random_state=42)

# Define LSTM model
model = Sequential([
    Embedding(input_dim=20000, output_dim=128, input_length=max_len),
    LSTM(64, dropout=0.2, recurrent_dropout=0.2),
    Dropout(0.3),
    Dense(1, activation='sigmoid')
])

model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# Early stopping
early_stopping = EarlyStopping(monitor='val_loss', patience=3)
model_checkpoint = ModelCheckpoint('best_model.keras', save_best_only=True, monitor='val_loss', mode='min')

# Train model
history = model.fit(
    X_train_padded, y_train,
    validation_data=(X_test_padded, y_test),
    epochs=3,
    batch_size=64,
    callbacks=[early_stopping, model_checkpoint]
)

# Save the tokenizer for later use
import pickle
with open('../tokenizer.pkl', 'wb') as file:
    pickle.dump(tokenizer, file)

# Save the TF-IDF vectorizer
with open('../tfidf_vectorizer.pkl', 'wb') as file:
    pickle.dump(tfidf, file)

# Evaluate Logistic Regression and Naive Bayes
print("Logistic Regression Test Accuracy:", log_model.score(X_test_tfidf, y_test))
print("Naive Bayes Test Accuracy:", nb_model.score(X_test_tfidf, y_test))

# Evaluate LSTM
loss, accuracy = model.evaluate(X_test_padded, y_test)
print(f"LSTM Test Accuracy: {accuracy * 100:.2f}%")

# Predict sentiment for a custom review
def predict_sentiment(review):
    print("\n--- Sentiment Analysis ---")
    print(f"Review: {review}")

    # Logistic Regression Prediction
    log_tfidf = tfidf.transform([review])
    log_pred = log_model.predict(log_tfidf)[0]
    print(f"Logistic Regression Prediction: {'Positive' if log_pred == 1 else 'Negative'}")

    # Naive Bayes Prediction
    nb_pred = nb_model.predict(log_tfidf)[0]
    print(f"Naive Bayes Prediction: {'Positive' if nb_pred == 1 else 'Negative'}")

    # LSTM Prediction
    seq = tokenizer.texts_to_sequences([review])
    padded_seq = pad_sequences(seq, maxlen=max_len)
    lstm_pred = model.predict(padded_seq)[0][0]
    print(f"LSTM Prediction: {'Positive' if lstm_pred > 0.5 else 'Negative'}")

# Recursive Input for Custom Reviews
while True:
    custom_review = input("\nEnter a custom review (type 'exit' to quit): ")
    if custom_review.lower() == 'exit':
        print("Exiting the program. Goodbye!")
        break
    predict_sentiment(custom_review)
