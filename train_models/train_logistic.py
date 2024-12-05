import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix
import pickle
import os

# Ensure the output directory exists
output_dir = "models"
os.makedirs(output_dir, exist_ok=True)

# Load dataset
data_path = './data/Reviews.csv'
data = pd.read_csv(data_path)

# Preprocess data
print("Preprocessing data...")
data['Score'] = data['Score'].apply(lambda x: 1 if x >= 3 else 0)  # Convert to binary sentiment
data['Text'] = data['Text'].fillna('')  # Fill missing values in 'Text' column

# Vectorize text using TF-IDF
tfidf = TfidfVectorizer(max_features=20000, stop_words='english')
X = tfidf.fit_transform(data['Text'])
y = data['Score']

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train logistic regression model
print("Training logistic regression model...")
log_model = LogisticRegression(max_iter=300)
log_model.fit(X_train, y_train)

# Evaluate model
print("Evaluating model...")
y_pred = log_model.predict(X_test)
print("Classification Report:\n", classification_report(y_test, y_pred))
print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))

# Save the model and TF-IDF vectorizer
model_path = os.path.join(output_dir, 'logistic_model.pkl')
vectorizer_path = os.path.join(output_dir, 'tfidf_vectorizer.pkl')

print(f"Saving model to {model_path}...")
with open(model_path, 'wb') as model_file:
    pickle.dump(log_model, model_file)

print(f"Saving vectorizer to {vectorizer_path}...")
with open(vectorizer_path, 'wb') as vectorizer_file:
    pickle.dump(tfidf, vectorizer_file)

print("Training and saving completed successfully!")