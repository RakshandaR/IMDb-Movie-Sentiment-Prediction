import numpy as np
import tensorflow as tf
from tensorflow.keras.datasets import imdb
from tensorflow.keras.preprocessing import sequence
from tensorflow.keras.models import load_model
import streamlit as st

# -----------------------------
# Step 1: Load word index
# -----------------------------
max_features = 10000
word_index = imdb.get_word_index()
reverse_word_index = {value: key for key, value in word_index.items()}

# -----------------------------
# Step 2: Load the trained model
# -----------------------------
model = load_model('simplernn_imdb_model.h5')

# -----------------------------
# Step 3: Preprocessing function
# -----------------------------
def preprocess_review(review, maxlen=500):
    tokens = review.lower().split()
    indices = [min(word_index.get(word, 2), max_features - 1) for word in tokens]  # Clip to max_features
    padded_sequence = sequence.pad_sequences([indices], maxlen=maxlen)
    return padded_sequence

# -----------------------------
# Step 4: Prediction function
# -----------------------------
def predict_review_sentiment(review):
    processed_review = preprocess_review(review)
    prediction = model.predict(processed_review, verbose=0)
    sentiment = "Positive" if prediction[0][0] > 0.5 else "Negative"
    return sentiment, prediction[0][0]

# -----------------------------
# Step 5: Streamlit app
# -----------------------------
st.title("IMDB Movie Review Sentiment Analysis")
st.write("Enter a movie review to predict its sentiment (positive or negative).")

user_input = st.text_area("Movie Review")

if st.button("Predict Sentiment") and user_input.strip() != "":
    sentiment, score = predict_review_sentiment(user_input)
    st.write(f"Predicted Sentiment: **{sentiment}** (Score: {score:.4f})")
