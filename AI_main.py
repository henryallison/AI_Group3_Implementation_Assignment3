import pandas as pd
import numpy as np
from fuzzywuzzy import process
import joblib
import streamlit as st

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

# Debugging: Print column names
print("Columns in the dataset:", df.columns)

# Strip extra spaces and convert column names to lowercase
df.columns = df.columns.str.strip().str.lower()

# Streamlit App
st.title("Medical AI Chatbot")
st.write("Enter your symptoms below to get a diagnosis.")

# Custom CSS for styling
st.markdown(
    """
    <style>
    .chat-container {
        border: 2px solid #e0e0e0;
        border-radius: 10px;
        padding: 10px;
        height: 400px;
        overflow-y: auto;
        background-color: #f9f9f9;
        margin-bottom: 10px;
        display: flex;
        flex-direction: column;
    }
    .chat-box {
        flex: 1;
        overflow-y: auto;
        margin-bottom: 10px;
    }
    .user-message {
        background-color: #0078d4;
        color: white;
        border-radius: 10px;
        padding: 8px;
        margin: 5px 0;
        max-width: 70%;
        margin-left: auto;
        margin-right: 0;
        word-wrap: break-word;
        white-space: pre-wrap;
    }
    .ai-message {
        background-color: #e0e0e0;
        color: black;
        border-radius: 10px;
        padding: 8px;
        margin: 5px 0;
        max-width: 70%;
        margin-left: 0;
        margin-right: auto;
        word-wrap: break-word;
        white-space: pre-wrap;
    }
    .input-box {
        display: flex;
        gap: 10px;
        align-items: center;
    }
    .stTextInput>div>div>input {
        border-radius: 20px;
        padding: 10px;
        flex: 1;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# Initialize session state for message history
if "messages" not in st.session_state:
    st.session_state.messages = [{"role": "ai", "content": "Hi, welcome to Group3 AI chatbot!"}]

# Debugging: Print session state
print("Session State Messages:", st.session_state.messages)

# Display chat history in a rectangular box
with st.container():
    st.markdown('<div class="chat-container">', unsafe_allow_html=True)
    st.markdown('<div class="chat-box">', unsafe_allow_html=True)
    for message in st.session_state.messages:
        if message["role"] == "user":
            st.markdown(
                f'<div class="user-message">{message["content"]}</div>',
                unsafe_allow_html=True,
            )
        else:
            st.markdown(
                f'<div class="ai-message">{message["content"]}</div>',
                unsafe_allow_html=True,
            )
    st.markdown("</div>", unsafe_allow_html=True)

    # Input field and send button inside the chat container
    st.markdown('<div class="input-box">', unsafe_allow_html=True)
    user_input = st.text_input("Enter your symptoms (e.g., fever, headache, fatigue):", key="input", label_visibility="collapsed")
    send_button = st.button("Send")
    st.markdown("</div>", unsafe_allow_html=True)
    st.markdown("</div>", unsafe_allow_html=True)

# Handle user input and AI response
if send_button and user_input:
    # Add user message to chat history
    st.session_state.messages.append({"role": "user", "content": user_input})

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

    # Add AI response to chat history
    st.session_state.messages.append({"role": "ai", "content": ai_response})

    # Debugging: Print updated session state
    print("Updated Session State Messages:", st.session_state.messages)

    # Rerun the app to update the chat history
    st.rerun()

# How to Use Section
st.sidebar.title("How to Use")
st.sidebar.write("""
1. Enter your symptoms in the input box (e.g., 'fever, headache, fatigue').
2. Click the 'Send' button to get a diagnosis.
3. The Medical AI will analyze your symptoms and provide a diagnosis.
4. If no match is found, consult a healthcare professional.
""")