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

#1000 records
synthetic_data = generate_text_data(1000)

for i, text in enumerate(synthetic_data[:5], 1):
    print(f"{i}: {text}")