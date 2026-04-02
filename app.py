import streamlit as st
import numpy as np
import librosa
import pickle
import os

# ==============================
# CONFIG
# ==============================
MODEL_PATH = "model.pkl"
SAMPLE_RATE = 16000
N_MFCC = 13

# ==============================
# LOAD MODEL
# ==============================
@st.cache_resource
def load_model():
    with open(MODEL_PATH, "rb") as f:
        return pickle.load(f)

model = load_model()

# ==============================
# FEATURE EXTRACTION
# ==============================
def extract_features(file_path):
    audio, sr = librosa.load(file_path, sr=SAMPLE_RATE)
    mfcc = librosa.feature.mfcc(y=audio, sr=sr, n_mfcc=N_MFCC)
    return np.mean(mfcc.T, axis=0)

# ==============================
# UI
# ==============================
st.title("🧠 Autism Prediction from Speech")
st.write("Upload a .wav file")

uploaded_file = st.file_uploader("Upload Audio", type=["wav"])

if uploaded_file is not None:
    file_path = "temp.wav"

    with open(file_path, "wb") as f:
        f.write(uploaded_file.read())

    st.audio(file_path)

    if st.button("Predict"):
        features = extract_features(file_path)
        features = features.reshape(1, -1)

        prediction = model.predict(features)[0]
        prob = model.predict_proba(features)

        if prediction == 0:
            st.error("Prediction: Autism")
        else:
            st.success("Prediction: Non-Autism")

        st.write("Confidence:", np.max(prob))

        os.remove(file_path)
