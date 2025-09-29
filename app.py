import streamlit as st
import pandas as pd
import numpy as np
import joblib
import shap
import lime
import lime.lime_tabular
import tensorflow as tf
from tensorflow.keras.models import load_model

# code adapted from: Streamlit Documentation (2023) – Web app structure and UI components
st.set_page_config(page_title="CNN-BiLSTM IDS", layout="wide")
st.title("Intrusion Detection System – CNN-BiLSTM")
# end of adapted code

# --- Load Artifacts ---
# code adapted from: Chollet (2015) – Keras model saving/loading
@st.cache_resource
def load_artifacts():
    model = load_model("cnn_bilstm_model.keras")
    scaler = joblib.load("minmax_scaler.joblib")
    le = joblib.load("label_encoder.joblib")
    feature_names = joblib.load("feature_names.pkl")
    return model, scaler, le, feature_names
# end of adapted code

model, scaler, le, feature_names = load_artifacts()

# --- File Upload ---
# code adapted from: Streamlit Documentation (2023) – File uploader widget
uploaded_file = st.file_uploader("Upload CSV for Prediction", type="csv")
# end of adapted code

if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)
    st.write("Preview of uploaded data:", df.head())

    # code adapted from: Pedregosa et al. (2011) – scikit-learn preprocessing
    X = df[feature_names]
    X_scaled = scaler.transform(X)
    X_scaled = X_scaled.reshape(len(X_scaled), X_scaled.shape[1], 1)
    # end of adapted code

    # --- Prediction ---
    # code adapted from: Chollet (2015) – Keras model prediction
    preds = model.predict(X_scaled)
    y_pred = np.argmax(preds, axis=1)
    labels = le.inverse_transform(y_pred)
    df["Predicted_Label"] = labels
    # end of adapted code

    st.subheader("Predictions")
    st.write(df.head())

    # --- Explainability (SHAP + LIME) ---
    # code adapted from: Lundberg & Lee (2017) – SHAP: Interpretable ML
    explainer = shap.DeepExplainer(model, X_scaled[:100])
    shap_values = explainer.shap_values(X_scaled[:10])
    st.subheader("SHAP Summary Plot")
    st.set_option("deprecation.showPyplotGlobalUse", False)
    shap.summary_plot(shap_values, X_scaled[:10], feature_names=feature_names)
    st.pyplot()
    # end of adapted code

    # code adapted from: Ribeiro et al. (2016) – LIME: Local Interpretable Model-Agnostic Explanations
    lime_explainer = lime.lime_tabular.LimeTabularExplainer(
        training_data=np.array(X_scaled[:100]),
        feature_names=feature_names,
        class_names=le.classes_,
        mode="classification"
    )
    exp = lime_explainer.explain_instance(
        X_scaled[0].ravel(),
        model.predict,
        num_features=10
    )
    st.subheader("LIME Explanation for First Instance")
    st.write(exp.as_list())
    # end of adapted code
