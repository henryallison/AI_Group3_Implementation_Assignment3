from flask import Flask, render_template, request, jsonify
import pandas as pd
import numpy as np
from fuzzywuzzy import process
import joblib

# Initialize Flask app
app = Flask(__name__)

# Load the saved model and preprocessing objects
clf = joblib.load("diagnosis_model.pkl")  # Load the trained model
tfidf = joblib.load("tfidf_vectorizer.pkl")  # Load the TF-IDF vectorizer
label_encoder = joblib.load("label_encoder.pkl")  # Load the label encoder
symptom_keywords = joblib.load("symptom_keywords.pkl")  # Load the symptom keywords

# Function to extract keywords from user input using fuzzy matching
def extract_keywords(user_input, keywords, threshold=70):
    extracted_keywords = []
    for keyword in keywords:
        match, score = process.extractOne(keyword, user_input.split())
        if score >= threshold:
            extracted_keywords.append(keyword)
    return extracted_keywords

# Function to predict diagnosis
def predict_diagnosis(user_input):
    # Extract keywords from user input
    keywords = extract_keywords(user_input, symptom_keywords)
    if not keywords:
        return "No relevant symptoms found. Please provide more details."

    # Convert keywords to a string and vectorize using TF-IDF
    symptoms_str = " ".join(keywords)
    symptoms_tfidf = tfidf.transform([symptoms_str])

    # Make a prediction
    prediction_encoded = clf.predict(symptoms_tfidf)
    prediction = label_encoder.inverse_transform(prediction_encoded)[0]

    # Return the diagnosis
    return f"Medical AI Response:\nDiagnosis: {prediction}"

# Load the dataset (for treatment recommendations)
df = pd.read_csv("disease_dataset.csv")

# Strip extra spaces and convert column names to lowercase
df.columns = df.columns.str.strip().str.lower()

# Route for the home page
@app.route("/")
def home():
    return render_template("index.html")

# Route for the About page
@app.route("/about")
def about():
    return render_template("about.html")

# Route to handle chat messages
@app.route("/chat", methods=["POST"])
def chat():
    user_input = request.json["message"]

    # Check for greetings
    greetings = ["hi", "hello", "hey", "sup", "bro", "big man"]
    farewells = ["bye", "goodbye", "see you"]

    if any(word in user_input.lower() for word in greetings):
        ai_response = "How do you feel today?"
    elif any(word in user_input.lower() for word in farewells):
        ai_response = "Goodbye, come again another time!"
    else:
        # Get AI response for symptoms
        ai_response = predict_diagnosis(user_input)

    return jsonify({"response": ai_response})

# Run the Flask app
if __name__ == "__main__":
    app.run(debug=True)