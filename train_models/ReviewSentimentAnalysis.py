import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Embedding, LSTM, Dense, Dropout, Input, Concatenate
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from sklearn.naive_bayes import MultinomialNB
from sklearn.preprocessing import MinMaxScaler
import nltk
import pickle
import os

# Disable warnings and download stopwords
import warnings
nltk.download('stopwords')
warnings.filterwarnings('ignore')

# Preprocessing setup
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer

stop_words = set(stopwords.words('english'))
stemmer = PorterStemmer()

# Load dataset
data_path = './data/Reviews.csv'
data = pd.read_csv(data_path)
data = data.sample(n=50000, random_state=42)  # Reduce size for faster processing
data['Text'] = data['Text'].fillna('')
data['Score'] = data['Score'].apply(lambda x: 1 if x >= 3 else 0)  # Binary classification

# Normalize helpfulness
scaler = MinMaxScaler()
data['HelpfulnessNumerator_scaled'] = scaler.fit_transform(data[['HelpfulnessNumerator']].fillna(0))
data['HelpfulnessDenominator_scaled'] = scaler.fit_transform(data[['HelpfulnessDenominator']].fillna(0))

# Preprocessing function
def clean_text(series):
    return series.str.lower().str.replace(r'[^\w\s]', '', regex=True).str.split().apply(
        lambda x: ' '.join([stemmer.stem(word) for word in x if word not in stop_words])
    )

data['cleaned_text'] = clean_text(data['Text'])

# Train-test split
X = data[['cleaned_text', 'HelpfulnessNumerator_scaled', 'HelpfulnessDenominator_scaled']]
y = data['Score']

# TF-IDF vectorization for Logistic Regression
tfidf = TfidfVectorizer(max_features=20000, stop_words='english')
X_tfidf = tfidf.fit_transform(X['cleaned_text'])
X_train_tfidf, X_test_tfidf, y_train, y_test = train_test_split(X_tfidf, y, test_size=0.2, random_state=42)

# Logistic Regression Model
log_model = LogisticRegression(max_iter=300)
log_model.fit(X_train_tfidf, y_train)

# Evaluate Logistic Regression
print("Logistic Regression Performance:")
y_pred = log_model.predict(X_test_tfidf)
print("Classification Report:\n", classification_report(y_test, y_pred))
print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))

# Tokenize and pad sequences for MLP
tokenizer = Tokenizer(num_words=20000)
tokenizer.fit_on_texts(X['cleaned_text'])
X_seq = tokenizer.texts_to_sequences(X['cleaned_text'])
X_padded = pad_sequences(X_seq, maxlen=150)
X_train_padded, X_test_padded, y_train, y_test = train_test_split(X_padded, y, test_size=0.2, random_state=42)

# Additional features for MLP
# Train-test split for text and helpfulness features
X_text = X['cleaned_text']
X_helpfulness = X[['HelpfulnessNumerator_scaled', 'HelpfulnessDenominator_scaled']]

X_text_train, X_text_test, X_help_train, X_help_test, y_train, y_test = train_test_split(
    X_text, X_helpfulness, y, test_size=0.2, random_state=42
)

# Tokenize and pad sequences for text data
X_train_padded = pad_sequences(tokenizer.texts_to_sequences(X_text_train), maxlen=150)
X_test_padded = pad_sequences(tokenizer.texts_to_sequences(X_text_test), maxlen=150)

# Convert helpfulness data to NumPy arrays
X_train_helpfulness = X_help_train.values
X_test_helpfulness = X_help_test.values

# Define MLP Model
input_text = Input(shape=(150,), name="text_input")
input_helpfulness = Input(shape=(2,), name="helpfulness_input")

embedding_layer = Embedding(input_dim=20000, output_dim=128, input_length=150)(input_text)
lstm_out = LSTM(64, dropout=0.2)(embedding_layer)
concat_layer = Concatenate()([lstm_out, input_helpfulness])
dropout_layer = Dropout(0.3)(concat_layer)
output_layer = Dense(1, activation='sigmoid')(dropout_layer)

mlp_model = Model(inputs=[input_text, input_helpfulness], outputs=output_layer)
mlp_model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# Train MLP Model
early_stopping = EarlyStopping(monitor='val_loss', patience=3)
model_checkpoint = ModelCheckpoint('models/mlp_model.keras', save_best_only=True, monitor='val_loss')

history = mlp_model.fit(
    [X_train_padded, X_train_helpfulness], y_train,
    validation_data=([X_test_padded, X_test_helpfulness], y_test),
    epochs=3,
    batch_size=64,
    callbacks=[early_stopping, model_checkpoint]
)

# Save models and artifacts
if not os.path.exists('models'):
    os.makedirs('models')

with open('models/tfidf_vectorizer.pkl', 'wb') as file:
    pickle.dump(tfidf, file)

with open('models/tokenizer.pkl', 'wb') as file:
    pickle.dump(tokenizer, file)

with open('models/logistic_model.pkl', 'wb') as file:
    pickle.dump(log_model, file)

print("Training completed. Models saved successfully!")
