import json

# Load the .jsonl file with UTF-8 encoding
data = []
with open('disease_prediction_model.jsonl', 'r', encoding='utf-8') as f:  # Add encoding='utf-8'
    for line in f:
        data.append(json.loads(line))

# Inspect the first few entries
print(data[:5])