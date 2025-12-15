import streamlit as st
import pandas as pd
import joblib
import os

from src.train_models import load_iteration4_model, load_feature_columns
from src.run_pipeline import align_live


st.set_page_config(page_title="Wired Sharks IDS Dashboard", layout="wide")

st.title("Wired Sharks – Intrusion Detection Dashboard")
st.write("This dashboard analyzes CSV flow files and predicts Normal vs Attack using the trained Random Forest model.")

# Load model
model, feature_cols = load_iteration4_model()

uploaded_file = st.file_uploader("Upload network flow CSV file", type=["csv"])

if uploaded_file:
    df = pd.read_csv(uploaded_file)
    st.subheader("Uploaded Data Preview")
    st.dataframe(df.head())

    st.write("---")

    # Align columns
    X = align_live(df, feature_cols)

    # Predict
    preds = model.predict(X)
    df["Prediction"] = preds
    df["PredictionLabel"] = df["Prediction"].map({0: "Normal", 1: "Attack"})

    # Summary
    st.subheader("Prediction Summary")
    st.write(df["PredictionLabel"].value_counts())

    # Chart
    st.subheader("Traffic Breakdown")
    st.bar_chart(df["PredictionLabel"].value_counts())

    st.subheader("Detailed Predictions")
    st.dataframe(df.tail(50))

    # Download
    st.download_button(
        label="Download Results as CSV",
        data=df.to_csv(index=False),
        file_name="ids_predictions.csv",
        mime="text/csv",
    )
else:
    st.info("Upload a flow CSV file to begin analysis.")