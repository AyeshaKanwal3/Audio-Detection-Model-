import streamlit as st
import numpy as np
import librosa
import tensorflow as tf
import os

# Function to preprocess audio
def preprocess_audio(audio_path):
    y, sr = librosa.load(audio_path, sr=None)
    mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=40)
    mfccs = np.mean(mfccs.T, axis=0)
    return mfccs

# Load the trained model with error handling
model_path = 'best_model.h5'  # Ensure this is the correct path to your model file

try:
    model = tf.keras.models.load_model(model_path)
    st.success("Model loaded successfully!")
except Exception as e:
    st.error(f"Error loading model: {e}")

# Streamlit app
st.title("Audio Deepfake Detection")

# File uploader
uploaded_file = st.file_uploader("Choose an audio file", type=["wav"])

if uploaded_file is not None:
    try:
        # Save the uploaded file to a temporary file
        with open("temp_audio.wav", "wb") as f:
            f.write(uploaded_file.getbuffer())
        
        # Preprocess the audio file
        features = preprocess_audio("temp_audio.wav")
        features = np.expand_dims(features, axis=0)
        
        # Make prediction
        prediction = model.predict(features)
        result = "Fake" if prediction[0] > 0.5 else "Real"
        
        # Display result
        st.write(f"Prediction: {result}")
    except Exception as e:
        st.error(f"Error processing audio file: {e}")
