import random
import nltk
import torch
import numpy as np
from transformers import BertTokenizer, BertModel
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import LabelEncoder
import joblib
import faker

nltk.download('punkt')

fake = faker.Faker()

def generate_text_data(num_records=1000):
    pii_types = ['name', 'email', 'phone', 'address', 'ssn', 'credit_card']
    non_pii_text = [
        "The weather is great today.",
        "I will attend the meeting at 3 PM.",
        "Artificial Intelligence is evolving rapidly.",
        "The new movie is amazing and worth watching.",
        "I have read a fantastic book recently.",
        "The project deadline is next Friday.",
        "We are planning a trip to New York next month.",
        "Technology is revolutionizing many industries.",
        "Python is widely used for data science and machine learning.",
    ]
    
    data = []
    for _ in range(num_records):
        if random.random() > 0.5:  # Randomly choose between PII and non-PII
            pii_type = random.choice(pii_types)
            if pii_type == 'name':
                data.append(f"Hello, my name is {fake.name()}.")
            elif pii_type == 'email':
                data.append(f"You can contact me at {fake.email()}.")
            elif pii_type == 'phone':
                data.append(f"My phone number is {fake.phone_number()}.")
            elif pii_type == 'address':
                data.append(f"I live at {fake.address()}.")
            elif pii_type == 'ssn':
                data.append(f"My SSN is {fake.ssn()}.")
            elif pii_type == 'credit_card':
                data.append(f"My credit card number is {fake.credit_card_number()}.")
        else:
            data.append(random.choice(non_pii_text))
    
    return data

synthetic_data = generate_text_data(1000)

def label_data(data):
    pii_keywords = ['name', 'email', 'phone', 'address', 'ssn', 'credit card']
    labels = []
    for text in data:
        if any(keyword in text.lower() for keyword in pii_keywords):
            labels.append(1)  # PII
        else:
            labels.append(0)  # Non-PII
    return labels

tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
bert_model = BertModel.from_pretrained('bert-base-uncased')

def get_bert_embedding(text):
    inputs = tokenizer(text, return_tensors='pt', truncation=True, padding=True, max_length=128)
    with torch.no_grad():
        outputs = bert_model(**inputs)
    return outputs.last_hidden_state[:, 0, :].cpu().detach().numpy()  # Use the [CLS] token representation

labels = label_data(synthetic_data)
embeddings = np.array([get_bert_embedding(text) for text in synthetic_data])

X = embeddings.reshape(len(embeddings), -1)
y = np.array(labels)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

clf = LogisticRegression(max_iter=1000)
clf.fit(X_train, y_train)

y_pred = clf.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print(f"Model Accuracy: {accuracy * 100:.2f}%")

# Save the model
joblib.dump(clf, 'pii_identifier_model.pkl')
print("Model saved as 'pii_identifier_model.pkl'")