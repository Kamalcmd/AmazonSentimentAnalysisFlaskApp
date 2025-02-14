# Amazon Review Sentiment Analysis

## **Project Overview**
This project analyzes Amazon reviews to predict sentiment (**Positive** or **Negative**) using machine learning models. The application provides insights into customer reviews, helping businesses improve their products and marketing strategies. The project includes:

- A Flask backend for serving predictions.
- A user-friendly HTML frontend for interaction.
- Pre-trained machine learning models for sentiment analysis.

### **Features**
1. **Text Sentiment Prediction**:
   - Enter a review in the text box to predict whether the sentiment is **Positive** or **Negative**.
2. **Pre-Trained Models**:
   - Logistic Regression (baseline model).
   - Multi-Layer Perceptron (MLP with LSTM) for enhanced performance.
3. **Web Interface**:
   - A simple, browser-based interface using Flask and Bootstrap HTML templates.

---

## **Directory Structure**
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
│   ├── mlp_model.keras            # Trained MLP (LSTM) model
│   ├── tokenizer.pkl              # Tokenizer for LSTM model
├── train_models/
│   ├── ReviewSentimentAnalysis.py # Unified script for training models
├── requirements.txt               # Required Python packages
```

---

## **Prerequisites**
1. Python 3.8 or higher.
2. Pip for managing Python packages.

---

## **Installation**

### Step 1: Clone the Repository
```bash
# Clone the repository
#git clone git@github.com:Kamalcmd/AmazonSentimentAnalysisFlaskApp.git
#cd ProjectSentimentAnalysisFlaskApp
```

### Step 2: Set Up a Virtual Environment
```bash
# Create and activate the virtual environment
conda create -n ml_course python=3.10
conda activate ml_course
```

### Step 3: Install Required Libraries
Install all dependencies listed in `requirements.txt`:
```bash
pip install -r requirements.txt
```

---

## **Dataset**
The dataset (`Reviews.csv`) is located in the `data/` directory. It includes:
- **Id**: Unique ID for each review.
- **Text**: Review content (used for analysis).
- **Score**: Rating (used to determine sentiment: positive if ≥3, negative otherwise).
- **HelpfulnessNumerator** and **HelpfulnessDenominator**: Indicate the helpfulness of a review.

---

## **Training Models**

### Logistic Regression
Logistic Regression uses **TF-IDF** vectorized features for sentiment classification.

This generates:
- `logistic_model.pkl` (model)
- `tfidf_vectorizer.pkl` (vectorizer)

### Multi-Layer Perceptron (MLP with LSTM)
The MLP model uses tokenized and padded sequences for classification and incorporates additional features like **HelpfulnessNumerator** and **HelpfulnessDenominator**.

This generates:
- `mlp_model.keras` (model)
- `tokenizer.pkl` (tokenizer)
```bash
# Train and save the MLP model
python train_models/ReviewSentimentAnalysis.py
```
---

## **Running the Application**

### Step 1: Start the Flask App
Launch the Flask backend:
```bash
python app/app.py
```
The application will be available at: `http://127.0.0.1:5000`.

### Step 2: Interact with the Frontend
1. Open your browser and navigate to `http://127.0.0.1:5000`.
2. Enter a review in the text box and click **"Predict Sentiment"**.
3. View the sentiment analysis results.

---

## **Example Usage**

### **Input**
```plaintext
"I love this product! It works perfectly and is great value for money."
```

### **Output**
```json
{
    "Logistic Regression Sentiment": "Positive",
    "MLP Sentiment": "Positive"
}
```
### **Some Sample Review to test**

```agsl
Positive Reviews
"I absolutely love this product! The quality is excellent, and it offers great value for money. Highly recommended!"
"This is the best product I’ve ever purchased on Amazon. The features are exactly as described and exceeded my expectations."

Negative Reviews
"The packaging was damaged, and the product didn’t match the description. Very disappointed."
```

---

## **Models**

### **Baseline Model (Logistic Regression):**
- Simpler and interpretable.
- Uses TF-IDF vectorized features.

### **Advanced Model (MLP with LSTM):**
- Captures more complex relationships in text data.
- Incorporates additional features like review helpfulness scores.
- Improved accuracy for sentiment prediction.

---

## **How Sentiment is Defined**

### Logistic Regression
- Outputs a probability for the positive class.
- Threshold: If probability ≥0.5, classify as **Positive**; otherwise, classify as **Negative**.

### MLP (Multi-Layer Perceptron)
- Processes tokenized text and additional numerical features.
- Outputs a probability for the positive class.
- Threshold: If probability ≥0.5, classify as **Positive**; otherwise, classify as **Negative**.

---

## **Troubleshooting**

### Common Issues
1. **ModuleNotFoundError**:
   - Ensure all required libraries are installed via:
     ```bash
     pip install -r requirements.txt
     ```

2. **Flask App Not Running**:
   - Verify Python version (3.8 or higher) and ensure the Flask app is launched in the correct directory.

3. **Dataset Issues**:
   - Ensure the `Reviews.csv` file is correctly placed in the `data/` directory.

4. **Model Discrepancies**:
   - Logistic Regression and MLP may give different results due to differences in feature extraction and model complexity.

---

## **Authors**
Team Member 1: **Pooja Shiwakoti** (912581587)
- pooja.shiwakoti@uky.edu

Team Member 2: **Kamal Ghimire** (12589739)
- kamal.ghimire@uky.edu
