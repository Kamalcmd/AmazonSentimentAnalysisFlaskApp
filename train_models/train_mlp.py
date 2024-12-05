from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, LSTM, Dense, Dropout
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from sklearn.model_selection import train_test_split
import pandas as pd
import pickle

# Load and preprocess data
data = pd.read_csv('./data/Reviews.csv')
data['Score'] = data['Score'].apply(lambda x: 1 if x >= 3 else 0)
data['Text'] = data['Text'].fillna('')

# Tokenize and Pad Sequences
tokenizer = Tokenizer(num_words=20000)
tokenizer.fit_on_texts(data['Text'])
X = tokenizer.texts_to_sequences(data['Text'])
X = pad_sequences(X, maxlen=150)
y = data['Score']

# Train-Test Split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# MLP Model
model = Sequential([
    Embedding(input_dim=20000, output_dim=128, input_length=150),
    LSTM(64, dropout=0.2, recurrent_dropout=0.2),
    Dropout(0.3),
    Dense(1, activation='sigmoid')
])
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
model.fit(X_train, y_train, validation_data=(X_test, y_test), epochs=3, batch_size=64)

# Save model and tokenizer
model.save('models/mlp_model.h5')
pickle.dump(tokenizer, open('models/tokenizer.pkl', 'wb'))
