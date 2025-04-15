# rul_predictor_app.py

import streamlit as st
import pandas as pd
import numpy as np
import pickle
import tensorflow as tf
from tensorflow.keras.models import load_model

# Load model and scaler
model = load_model("rul_lstm_optimized.h5", custom_objects={"MeanSquaredError": tf.keras.losses.MeanSquaredError()})
with open("scaler.pkl", "rb") as f:
    scaler = pickle.load(f)

# Parameters
sequence_length = 50

# Streamlit App
st.title("🔧 Turbofan Engine RUL Prediction (LSTM)")
st.write("Upload test data (e.g., `test_FD001.txt`) to predict Remaining Useful Life of engines.")

uploaded_file = st.file_uploader("📂 Upload test file", type=["txt"])

def preprocess_data(df):
    df.drop(df.columns[[26, 27]], axis=1, inplace=True)
    columns = ["engine_id", "cycle", "setting1", "setting2", "setting3"] + [f"sensor{i}" for i in range(1, 22)]
    df.columns = columns

    # Add delta features
    for sensor in [f"sensor{i}" for i in range(1, 22)]:
        df[f"{sensor}_delta"] = df.groupby("engine_id")[sensor].diff().fillna(0)

    # Scale features
    feature_cols = [col for col in df.columns if "sensor" in col or "setting" in col]
    df[feature_cols] = scaler.transform(df[feature_cols])
    
    return df, feature_cols

def create_latest_sequences(df, feature_cols):
    X = []
    engine_ids = df["engine_id"].unique()
    for eid in engine_ids:
        engine_df = df[df["engine_id"] == eid].copy()
        if len(engine_df) >= sequence_length:
            seq = engine_df[feature_cols].values[-sequence_length:]
            X.append(seq)
    return np.array(X), engine_ids[:len(X)]

if uploaded_file:
    test_df = pd.read_csv(uploaded_file, sep=" ", header=None)
    test_df, feature_cols = preprocess_data(test_df)
    X_test, engine_ids = create_latest_sequences(test_df, feature_cols)

    if len(X_test) > 0:
        predictions = model.predict(X_test).flatten()
        results = pd.DataFrame({
            "engine_id": engine_ids,
            "Predicted_RUL": predictions
        })

        st.subheader("🔍 Predicted RULs")
        st.dataframe(results)
    else:
        st.warning("❗ Not enough data in some engines to form 50-length sequences.")
