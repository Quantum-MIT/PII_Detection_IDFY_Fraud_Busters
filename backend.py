from flask import Flask, request, jsonify, render_template
import os
import re
import boto3
import pandas as pd
import nltk
from nltk import word_tokenize, pos_tag, ne_chunk
from werkzeug.utils import secure_filename
import logging
import joblib
import torch
from transformers import BertTokenizer, BertModel

# Setup basic logging
logging.basicConfig(level=logging.DEBUG)

app = Flask(__name__)

clf = joblib.load('pii_identifier_model.pkl') # Load the saved logistic regression model for BERT-based PII classification

tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
bert_model = BertModel.from_pretrained('bert-base-uncased')

pii_patterns = {
    'Name': r'[A-Z][a-z]+\s[A-Z][a-z]+',  # Simplified name pattern
    'Email': r'[a-zA-Z0-9_.+-]+@[a-zA-Z0-9-]+\.[a-zA-Z0-9-.]+',
    'Phone Number': r'(?:(?:\+?1[\s.-]?)?\(?\d{3}\)?[\s.-]\d{3}[\s.-]\d{4})\b',
    'SSN': r'\d{3}-\d{2}-\d{4}',
    'Credit Card Number': r'\b(?:\d[ -]*?){13,16}\b',
    'IP Address': r'\b(?:\d{1,3}\.){3}\d{1,3}\b',
    'Address': r'[0-9]{1,5}\s\w+\s\w+',  # Simplified address pattern
    'Date of Birth': r'\b\d{4}-\d{2}-\d{2}\b'
}

pii_sensitivity_scores = {
    'Email': 1,
    'Name': 1,
    'Date of Birth': 1.5,
    'Phone Number': 2,
    'Address': 2,
    'IP Address': 2,
    'SSN': 3,
    'Credit Card Number': 3
}

regulatory_compliance_factors = {
    'GDPR': 1.5,  # General Data Protection Regulation
    'HIPAA': 2.0,  # Health Insurance Portability and Accountability Act
    'PCI_DSS': 2.5,  # Payment Card Industry Data Security Standard
}

def get_bert_embedding(text):
    inputs = tokenizer(text, return_tensors='pt', truncation=True, padding=True, max_length=128)
    with torch.no_grad():
        outputs = bert_model(**inputs)
    return outputs.last_hidden_state[:, 0, :].cpu().detach().numpy()  # Use the [CLS] token representation

def identify_pii(text):
    detected_pii = []

    # Regex-based detection
    for pii_type, pattern in pii_patterns.items():
        match = re.search(pattern, text)
        if match:
            detected_pii.append((pii_type, match.group()))

    tokens = word_tokenize(text)
    tagged_tokens = pos_tag(tokens)
    named_entities = ne_chunk(tagged_tokens)

    for subtree in named_entities:
        if isinstance(subtree, nltk.Tree) and subtree.label() == 'PERSON':
            person = " ".join([leaf[0] for leaf in subtree.leaves()])
            detected_pii.append(('Name', person))

    bert_embedding = get_bert_embedding(text)
    bert_prediction = clf.predict(bert_embedding.reshape(1, -1))
    
    if bert_prediction[0] == 1:
        detected_pii.append(('According to BERT_PII_Prediction, the following para contains PII :\n', text))
    return detected_pii

def calculate_risk(pii_entities, compliance='GDPR'):
    base_risk = 0
    total_pii_count = len(pii_entities)

    for pii_type, _ in pii_entities:
        sensitivity_score = pii_sensitivity_scores.get(pii_type, 0)
        base_risk += sensitivity_score

    compliance_multiplier = regulatory_compliance_factors.get(compliance, 1.0)
    total_risk = base_risk * total_pii_count * compliance_multiplier

    return total_risk

@app.route('/')
def home():
    return render_template('index2.html')

@app.route('/pii', methods=['POST'])
def analyze_pii():
    print("Analyze PII route called")

    text_input = request.form.get('text', '').strip()
    aws_region = request.form.get('aws_region', '').strip()
    aws_access_key = request.form.get('aws_access_key', '').strip()
    aws_secret_key = request.form.get('aws_secret_key', '').strip()

    if(aws_region != '' and aws_access_key!='' and aws_secret_key!=''):
        s3 = boto3.resource(
            service_name='s3',
            region_name=aws_region,
            aws_access_key_id=aws_access_key,
            aws_secret_access_key=aws_secret_key
        )
        bucket_name = 'pii2dataset'
        object_key = 'pii_file.txt'
            # object_key = 'pii_data.csv'
        obj = s3.Bucket(bucket_name).Object(object_key).get()

        df = pd.read_csv(obj['Body'])
        text_input = df.to_string()
    print("Data from S3 read successfully. Processing...")

    pii_entities = identify_pii(text_input)
    risk_score = calculate_risk(pii_entities)

    print(f"PII entities identified: {pii_entities}")
    print(f"Risk score calculated: {risk_score}")

    return jsonify({'pii_entities': pii_entities, 'risk_score': risk_score})

if __name__ == '__main__':
    app.run(debug=True)
