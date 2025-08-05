import streamlit as st
import pandas as pd
import tensorflow as tf
import numpy as np
import librosa
import tempfile
import os
from sklearn.preprocessing import LabelEncoder

#Lead model and label encoder
@st.cache_resource
def load_model_and_encoder():
    model = tf.keras.models.load_model('models\Instrument_model.keras')
    label_encoder = LabelEncoder()
    label_encoder.classes_ = np.load('models\Instrument_classes.npy')
    return model, label_encoder

def extract_audio_features(audio_file):
    """Extract audio features from uploaded MP3 file"""
    # Save uploaded file temporarily
    with tempfile.NamedTemporaryFile(delete=False, suffix='.mp3') as tmp_file:
        tmp_file.write(audio_file.read())
        tmp_path = tmp_file.name

    try:
        # Load audio file
        y, sr = librosa.load(tmp_path, sr=22050)

        # Extract features (match your training data format)
        features = {}

        # Basic features
        features['tempo'], _ = librosa.beat.beat_track(y=y, sr=sr)
        features['chroma_stft'] = np.mean(librosa.feature.chroma(y=y, sr=sr))
        features['rmse'] = np.mean(librosa.feature.rms(y=y))
        features['spectral_centroid'] = np.mean(librosa.feature.spectral_centroid(y=y, sr=sr))
        features['spectral_bandwidth'] = np.mean(librosa.feature.spectral_bandwidth(y=y, sr=sr))
        features['rolloff'] = np.mean(librosa.feature.spectral_rolloff(y=y, sr=sr))
        features['zero_crossing_rate'] = np.mean(librosa.feature.zero_crossing_rate(y))

        # MFCCs
        mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=20)
        for i in range(20):
            features[f'mfcc{i+1}'] = np.mean(mfccs[i])

        return pd.DataFrame([features])

    finally:
        # Clean up temporary file
        os.unlink(tmp_path)

model, label_encoder = load_model_and_encoder()

#Streamlit app UI
st.title("Instrument Recognition from Audio")

# Create tabs for different input types
tab1, tab2 = st.tabs(["Upload MP3 File", "Upload CSV Features"])

with tab1:
    st.header("Upload MP3 Audio File")
    audio_file = st.file_uploader("Choose an MP3 file", type=["mp3"])

    if audio_file is not None:
        st.audio(audio_file, format='audio/mp3')

        with st.spinner('Extracting audio features...'):
            # Extract features from MP3
            df = extract_audio_features(audio_file)
            st.write("Extracted Features Preview:", df.head())

            # Make predictions
            predictions = model.predict(df)
            predicted_labels = label_encoder.inverse_transform(np.argmax(predictions, axis=1))
            confidence = np.max(predictions, axis=1)

            # Display results
            st.success(f"Predicted Instrument: **{predicted_labels[0]}**")
            st.write(f"Confidence: {confidence[0]:.2%}")

            # Show feature data
            if st.checkbox("Show extracted features"):
                st.dataframe(df)

with tab2:
    st.header("Upload CSV with Pre-extracted Features")
    uploaded_file = st.file_uploader("Upload a CSV file with audio features", type=["csv"])

    if uploaded_file is not None:
        #load csv file
        df = pd.read_csv(uploaded_file)
        st.write("Uploaded Data Preview:", df.head())

        # Make predictions
        predictions = model.predict(df)
        predicted_labels = label_encoder.inverse_transform(np.argmax(predictions, axis=1))

        # Display results
        st.write("Predicted Instruments:")
        st.write(predicted_labels)

        # Optionally, download predictions
        df['Predicted_Instrument'] = predicted_labels
        csv = df.to_csv(index=False).encode('utf-8')
        st.download_button("Download Predictions", csv, "predictions.csv", "text/csv")