import streamlit as st
import pandas as pd
import tensorflow as tf
import numpy as np
from sklearn.preprocessing import LabelEncoder

#Lead model and label encoder
@st.cache_resource
def load_model_and_encoder():
    model = tf.keras.models.load_model(r'models\Instrument_model.keras')
    label_encoder = LabelEncoder()
    label_encoder.classes_ = np.load(r'models\Instrument_classes.npy')
    return model, label_encoder

model, label_encoder = load_model_and_encoder()

#Streamlit app UI
st.title("Instrument Recognition from Audio Features")

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