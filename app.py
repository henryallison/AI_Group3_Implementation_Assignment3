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
    return f"Medical AI Response:\n{prediction}"

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
    greetings = ["hi", "hello", "hey", "sup", "bro", "big man", "dator", "pruh", "yo", "my gee"]
    farewells = ["bye", "goodbye", "see you", "later"]

    if any(word in user_input.lower() for word in greetings):
        ai_response = "How do you feel today?"
    elif any(word in user_input.lower() for word in farewells):
        ai_response = "Goodbye, come again another time!"

    # Responses to specific questions
    elif "what is malarial" in user_input.lower():
        ai_response = "Malaria is a life-threatening disease caused by Plasmodium parasites, transmitted through the bites of infected female Anopheles mosquitoes. Symptoms include fever, chills, and fatigue. It is common in tropical and subtropical regions."

    elif "what is hiv/aids" in user_input.lower():
        ai_response = "HIV (Human Immunodeficiency Virus) is a virus that attacks the immune system, weakening the body's ability to fight infections. AIDS (Acquired Immunodeficiency Syndrome) is the final stage of HIV infection, where the immune system is severely damaged."

    elif "do hiv have a cure" in user_input.lower():
        ai_response = "Currently, there is no cure for HIV, but it can be managed with antiretroviral therapy (ART). ART helps people with HIV live long, healthy lives and reduces the risk of transmitting the virus to others."

    elif "do malarial has a cure" in user_input.lower():
        ai_response = "Yes, malaria is treatable with antimalarial drugs. The type of drug depends on the Plasmodium species and the severity of the disease. Early diagnosis and treatment are crucial to prevent complications."

    elif "what can i do if i have hiv" in user_input.lower():
        ai_response = "If you have HIV, seek medical care immediately. Start antiretroviral therapy (ART) as prescribed, attend regular check-ups, maintain a healthy lifestyle, and practice safe sex to protect yourself and others."

    elif "what can i do if i have malarial" in user_input.lower():
        ai_response = "If you have malaria, seek medical attention immediately. Take prescribed antimalarial medications, rest, stay hydrated, and avoid mosquito bites to prevent spreading the disease to others."

    elif "how can i get malarial" in user_input.lower():
        ai_response = "You can get malaria through the bite of an infected female Anopheles mosquito. Rarely, it can also spread through blood transfusions, organ transplants, or from mother to child during pregnancy or childbirth."

    elif "how can i prevent malarial" in user_input.lower():
        ai_response = "Prevent malaria by using insecticide-treated bed nets, applying mosquito repellent, wearing protective clothing, taking antimalarial prophylaxis if traveling to endemic areas, and eliminating mosquito breeding sites."

    elif "how can i get hiv" in user_input.lower():
        ai_response = "HIV can be transmitted through unprotected sex, sharing needles, blood transfusions with infected blood, or from an HIV-positive mother to her child during pregnancy, childbirth, or breastfeeding."

    elif "how can i prevent hiv" in user_input.lower():
        ai_response = "Prevent HIV by practicing safe sex (using condoms), avoiding sharing needles, getting tested regularly, and taking pre-exposure prophylaxis (PrEP) if at high risk. For HIV-positive mothers, medical care can prevent transmission to the baby."

    else:
        # Get AI response for symptoms
        ai_response = predict_diagnosis(user_input)

    return jsonify({"response": ai_response})

# Run the Flask app
if __name__ == "__main__":
    app.run(debug=True)
