
# Amazon Review Sentiment Analysis

## Project Overview
This project analyzes Amazon reviews to predict sentiment (positive or negative) using machine learning models. It provides insights into customer reviews to help businesses improve products and marketing strategies. The project is built using Logistic Regression (as the baseline model) and Multi-Layer Perceptrons (MLP) for improved performance.

The application includes:
- A Flask backend to serve predictions.
- A user-friendly HTML frontend to interact with the model.
- Pre-trained models for text sentiment analysis.

---

## Features
1. **Text Sentiment Prediction**:
   - Enter a review into a text box to predict whether the sentiment is positive or negative.
2. **Pre-Trained Models**:
   - Logistic Regression (baseline model).
   - Multi-Layer Perceptron (MLP) for enhanced performance.
3. **Web Interface**:
   - A simple, browser-based interface using Flask and HTML templates.

---

## Directory Structure

```
ProjectSentimentAnalysis/
├── app/
│   ├── app.py                     # Flask backend application
│   ├── templates/
│   │   ├── index.html             # HTML template for the frontend
├── data/
│   ├── Reviews.csv                # Dataset with Amazon reviews
├── models/
│   ├── logistic_model.pkl         # Trained Logistic Regression model
│   ├── tfidf_vectorizer.pkl       # TF-IDF vectorizer for preprocessing
│   ├── mlp_model.h5               # Trained MLP (LSTM) model
│   ├── tokenizer.pkl              # Tokenizer for LSTM model
├── train_models/
│   ├── train_logistic.py          # Script to train Logistic Regression model
│   ├── train_mlp.py               # Script to train MLP (LSTM) model
├── requirements.txt               # Required Python packages
```

---

## Prerequisites
1. Python 3.8 or higher.
2. [Pip](https://pip.pypa.io/en/stable/) for managing Python packages.

---

## Installation

### Step 1: Clone the Repository
```bash
git clone <repository_url>
cd ProjectSentimentAnalysis
```

### Step 2: Set Up a Virtual Environment
```bash
python -m venv venv
source venv/bin/activate  # On Windows, use: venv\Scripts\activate
```

### Step 3: Install Required Libraries
Install all dependencies listed in `requirements.txt`:
```bash
pip install -r requirements.txt
```

---

## Dataset
The dataset (`Reviews.csv`) should be placed in the `data/` directory. It includes the following fields:
- **Id**: Unique ID for each review.
- **Text**: Review content (used for analysis).
- **Score**: Rating (used to determine sentiment: positive if ≥3, negative otherwise).

---

## Training Models

### Logistic Regression
Train and save the Logistic Regression model:
```bash
python train_models/train_logistic.py
```
This generates:
- `logistic_model.pkl` (model)
- `tfidf_vectorizer.pkl` (vectorizer)

### Multi-Layer Perceptron (MLP)
Train and save the MLP model:
```bash
python train_models/train_mlp.py
```
This generates:
- `mlp_model.h5` (model)
- `tokenizer.pkl` (tokenizer)

---

## Running the Application

### Step 1: Start the Flask App
Launch the Flask backend:
```bash
python app/app.py
```

The application will be available at: `http://127.0.0.1:5000`.

### Step 2: Interact with the Frontend
1. Open your browser and navigate to `http://127.0.0.1:5000`.
2. Enter a review in the text box and click "Predict" to see the sentiment analysis results.

---

## Example Usage

### Input
```plaintext
"I love this product! It works perfectly and is great value for money."
```

### Output
```json
{
    "Logistic Regression Sentiment": "Positive",
    "MLP Sentiment": "Positive"
}
```

---

## Models
1. **Baseline Model (Logistic Regression)**:
   - Simpler and interpretable.
   - Provides a benchmark for comparing other models.
2. **Advanced Model (MLP with LSTM)**:
   - Captures more complex relationships in text data.
   - Improved accuracy for sentiment prediction.

---

## Troubleshooting

### Common Issues
1. **ModuleNotFoundError**:
   - Ensure all required libraries are installed via `pip install -r requirements.txt`.
2. **Flask App Not Running**:
   - Verify Python version (3.8 or higher) and check if the Flask app is correctly launched.
3. **Dataset Issues**:
   - Ensure the `Reviews.csv` file is correctly placed in the `data/` directory.

---

## Contribution
1. Fork the repository.
2. Create a new branch: `git checkout -b feature-name`.
3. Commit changes: `git commit -m 'Add feature'`.
4. Push to branch: `git push origin feature-name`.
5. Submit a pull request.

---

## Authors
- **Pooja Shiwakoti**
- **Kamal Ghimire**

---
