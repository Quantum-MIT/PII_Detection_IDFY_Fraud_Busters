from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
import pickle
import faker
import random

# Use Faker library to generate fake PII data
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
    labels = []  # We will store the corresponding labels here

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
            labels.append(1)  # Label PII as 1
        else:
            data.append(random.choice(non_pii_text))
            labels.append(0)  # Label non-PII as 0
    
    return data, labels

texts, labels = generate_text_data(1000)

vectorizer = TfidfVectorizer(max_features=500)
X = vectorizer.fit_transform(texts)

model = LogisticRegression()
model.fit(X, labels)

with open('pii_classification_model.pkl', 'wb') as f:
    pickle.dump(model, f)

with open('tfidf_vectorizer.pkl', 'wb') as f:
    pickle.dump(vectorizer, f)

print("Model training and saving completed!")
