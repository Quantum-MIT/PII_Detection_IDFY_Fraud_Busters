from flask import Flask, request, jsonify, render_template
import re
import nltk
from nltk import word_tokenize, pos_tag, ne_chunk
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
import pickle
import os

app = Flask(__name__)

pii_patterns = {
    'Name': r'[A-Z][a-z]+\s[A-Z][a-z]+',  # Simplified name pattern
    'Email': r'[a-zA-Z0-9_.+-]+@[a-zA-Z0-9-]+\.[a-zA-Z0-9-.]+',
    'SSN': r'\d{3}-\d{2}-\d{4}',
    'Credit Card Number': r'\b(?:\d[ -]*?){13,16}\b',
    'IP Address': r'\b(?:\d{1,3}\.){3}\d{1,3}\b',
    'Address': r'[0-9]{1,5}\s\w+\s\w+',  # Simplified address pattern
    'Date of Birth': r'\b\d{4}-\d{2}-\d{2}\b'
}

# Sensitivity scores for PII
pii_sensitivity_scores = {
    'Email': 1,
    'Name': 1,
    'Date of Birth': 1.5,
    'Address': 2,
    'IP Address': 2,
    'SSN': 3,
    'Credit Card Number': 3
}

with open('pii_classification_model.pkl', 'rb') as f:
    model = pickle.load(f)

with open('tfidf_vectorizer.pkl', 'rb') as f:
    vectorizer = pickle.load(f)

# Function to identify PII using regex and NLTK for NER
def identify_pii(text):
    detected_pii = []

    for pii_type, pattern in pii_patterns.items():
        matches = re.findall(pattern, text)
        for match in matches:
            detected_pii.append((pii_type, match))

    tokens = word_tokenize(text)
    tagged_tokens = pos_tag(tokens)
    named_entities = ne_chunk(tagged_tokens)

    for subtree in named_entities:
        if isinstance(subtree, nltk.Tree):
            if subtree.label() == 'PERSON':
                person = " ".join([leaf[0] for leaf in subtree.leaves()])
                detected_pii.append(('Name', person))

    return detected_pii

# Function to preprocess text for the ML model
def preprocess_for_ml(text):
    processed_text = vectorizer.transform([text])
    return processed_text


def calculate_risk(pii_entities, text, compliance='GDPR'): # Function to calculate the risk score using both regex and ML
    base_risk = 0
    total_pii_count = len(pii_entities)

    for pii_type, _ in pii_entities:
        sensitivity_score = pii_sensitivity_scores.get(pii_type, 0)
        base_risk += sensitivity_score

    processed_text = preprocess_for_ml(text)
    ml_prediction = model.predict_proba(processed_text)[0][1]  # Get the probability of being sensitive

    total_risk = base_risk * (1 + ml_prediction)  # ML modifies the base risk score

    return total_risk

@app.route('/')
def home():
    return render_template('index2.html')

@app.route('/pii', methods=['POST'])
def analyze_pii():
    text_input = request.form.get('text', '').strip()

    pii_entities = identify_pii(text_input)

    risk_score = calculate_risk(pii_entities, text_input)

    return jsonify({'pii_entities': pii_entities, 'risk_score': risk_score})

if __name__ == '__main__':
    app.run(debug=True)
