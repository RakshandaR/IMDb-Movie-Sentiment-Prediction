# Step 1: Import necessary libraries
import numpy as np
import tensorflow as tf
from tensorflow.keras.datasets import imdb
from tensorflow.keras.preprocessing import sequence
from tensorflow.keras.models import load_model
import streamlit as st

# Step 2: Load the word index for the IMDB dataset

word_index = imdb.get_word_index()
reverse_word_index = {value: key for key, value in word_index.items()}

# Step 3: Load the model with pretrained ReLU activation

model = load_model('simplernn_imdb_model.h5')

# Step 4: Helper function 
# Function to decode reviews
def decode_review(text):
    return ' '.join([reverse_word_index.get(i - 3, '?') for i in text])

# Function to preprocess user input
def preprocess_review(review, maxlen=500):
    # Tokenize the review
    tokens = review.lower().split()
    # Convert tokens to indices based on the word index
    indices = [word_index.get(word, 2) + 3 for word in tokens]  # 2 is for unknown words
    # Pad the sequence to ensure consistent input length
    padded_sequence = sequence.pad_sequences([indices], maxlen=maxlen)
    return padded_sequence

# Step 5: Predict function
def predict_review_sentiment(review):
    processed_review = preprocess_review(review)
    prediction = model.predict(processed_review)
    sentiment = "Positive" if prediction[0][0] > 0.5 else "Negative"
    return sentiment, prediction[0][0]

# Step 6: Streamlit app for user input
st.title("IMDB Movie Review Sentiment Analysis")
st.write("Enter a movie review to predict its sentiment (positive or negative).")

# User input
user_input = st.text_area("Movie Review")
if st.button("Predict Sentiment"):
    preprocess_input = preprocess_review(review=user_input)
    # Make prediction
    sentiment, score = predict_review_sentiment(user_input)
    # Display result
    st.write(f"Predicted Sentiment: **{sentiment}** (Score: {score:.4f})")

