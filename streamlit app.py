import streamlit as st
import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder, StandardScaler
from keras.models import load_model
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch
from sklearn.feature_extraction.text import TfidfVectorizer
import joblib

# Load pre-trained model and tokenizer
tokenizer = AutoTokenizer.from_pretrained("dmis-lab/biobert-base-cased-v1.1")
biobert_model = AutoModelForSequenceClassification.from_pretrained("dmis-lab/biobert-base-cased-v1.1")
model = load_model('study_duration_predictor.h5')

# Load the TF-IDF vectorizer and label encoder
tfidf_vectorizer = joblib.load('tfidf_vectorizer.pkl')  # Ensure you have saved this object
le_conditions = joblib.load('le_conditions.pkl')  # Ensure you have saved this object

def classify_text_with_biobert(text):
    inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=512)
    outputs = biobert_model(**inputs)
    return torch.argmax(outputs.logits, dim=1).item()

def preprocess_input(conditions, primary_outcome, interventions):
    # Encode conditions
    conditions_encoded = le_conditions.transform([conditions])[0]
    
    # Classify primary outcome measure
    primary_outcome_biobert_label = classify_text_with_biobert(primary_outcome)
    
    # Vectorize text data
    text_data = primary_outcome + ' ' + interventions
    tfidf_features = tfidf_vectorizer.transform([text_data]).toarray()
    
    # Combine features
    features = np.hstack((
        [conditions_encoded], 
        [primary_outcome_biobert_label], 
        tfidf_features
    ))
    
    return features

def main():
    st.title('Clinical Trial Study Duration Predictor')

    # User input
    conditions = st.text_input("Conditions")
    primary_outcome = st.text_area("Primary Outcome Measure")
    interventions = st.text_area("Medical Treatment Summary")
    study_size = st.number_input("Study Size", min_value=1, value=1)
    num_participants = st.number_input("Number of Study Participants", min_value=1, value=1)

    if st.button('Predict Study Duration'):
        try:
            # Preprocess input
            features = preprocess_input(conditions, primary_outcome, interventions)
            
            # Predict study duration
            predicted_duration = model.predict(features.reshape(1, -1))[0][0]
            
            st.write(f'Predicted Study Duration (days): {predicted_duration:.2f}')
        except Exception as e:
            st.error(f"An error occurred: {e}")

if __name__ == "__main__":
    main()
