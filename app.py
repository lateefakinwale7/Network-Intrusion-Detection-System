# app.py
import streamlit as st
import pandas as pd
import numpy as np
import pickle, joblib
import shap
from lime.lime_tabular import LimeTabularExplainer
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder
import tensorflow as tf
from io import BytesIO
import datetime, zipfile, os

st.set_page_config(page_title="Intrusion Detection System", layout="wide")

# ----------------------------
# Cache Clearing
# ----------------------------
def clear_cache():
    st.cache_resource.clear()
    st.cache_data.clear()

with st.sidebar:
    st.button("Clear Cache", on_click=clear_cache)
    st.markdown("---")
    st.write("If errors occur after uploading new files, try clearing the cache.")

# ----------------------------
# Load Resources (Flexible)
# ----------------------------
@st.cache_resource(show_spinner="Loading model and preprocessing files...")
def load_resources():
    model, scaler, label_encoder, selected_features = None, None, None, None

    # Priority: load deep model if available
    if os.path.exists("cnn_bilstm_model.h5"):
        model = tf.keras.models.load_model("cnn_bilstm_model.h5")
        with open("scaler.pkl", "rb") as f:
            scaler = pickle.load(f)
        with open("label_encoder.pkl", "rb") as f:
            label_encoder = pickle.load(f)
        with open("selected_features.txt", "r") as f:
            selected_features = [line.strip() for line in f.readlines()]
        model_type = "keras"

    # Fallback: load joblib model
    elif os.path.exists("model.pkl"):
        model = joblib.load("model.pkl")
        # Optional: load preprocessing
        if os.path.exists("scaler.pkl"):
            with open("scaler.pkl", "rb") as f:
                scaler = pickle.load(f)
        if os.path.exists("label_encoder.pkl"):
            with open("label_encoder.pkl", "rb") as f:
                label_encoder = pickle.load(f)
        if os.path.exists("selected_features.txt"):
            with open("selected_features.txt", "r") as f:
                selected_features = [line.strip() for line in f.readlines()]
        model_type = "sklearn"

    else:
        st.error("No model file found. Upload either cnn_bilstm_model.h5 or model.pkl")
        st.stop()

    return model, scaler, label_encoder, selected_features, model_type

model, scaler, label_encoder, selected_features, model_type = load_resources()

# ----------------------------
# Prediction Helper
# ----------------------------
def predict_func(x_2d):
    """Unified predict for both keras and sklearn models."""
    x_arr = np.array(x_2d, dtype=np.float32)
    if model_type == "keras":
        x_reshaped = x_arr.reshape(x_arr.shape[0], x_arr.shape[1], 1)
        preds = model.predict(x_reshaped, verbose=0)
    else:  # sklearn
        preds = model.predict_proba(x_arr)
    return preds

# ----------------------------
# SHAP + LIME Setup
# ----------------------------
@st.cache_resource
def get_shap_explainer(data, labels):
    return shap.KernelExplainer(predict_func, shap.sample(data, 50))

@st.cache_resource
def get_lime_explainer(data, feature_names, class_names):
    return LimeTabularExplainer(
        data, feature_names=feature_names, class_names=class_names, mode="classification"
    )

# ----------------------------
# Plot Helpers
# ----------------------------
def fig_to_bytes(fig):
    buf = BytesIO()
    fig.savefig(buf, format="png", bbox_inches="tight")
    buf.seek(0)
    return buf

def plot_shap_local(shap_values, feature_names, title="SHAP Local Explanation"):
    fig, ax = plt.subplots(figsize=(7, 4))
    order = np.argsort(np.abs(shap_values))[::-1]
    top_n = min(20, len(feature_names))
    ax.barh(np.array(feature_names)[order][:top_n][::-1], shap_values[order][:top_n][::-1])
    ax.set_xlabel("SHAP value (impact)")
    ax.set_title(title)
    plt.tight_layout()
    return fig

# ----------------------------
# UI
# ----------------------------
st.title("🛡️ Network Intrusion Detection System — Unified")
st.markdown("Upload a CSV file with network traffic data to get predictions and explanations.")
st.info(f"Required features: {', '.join(selected_features) if selected_features else 'All columns used'}")

uploaded_file = st.file_uploader("Upload CSV", type=["csv"])
if uploaded_file is None:
    st.stop()

# ----------------------------
# Preprocess & Predict
# ----------------------------
df = pd.read_csv(uploaded_file)
if selected_features:
    missing = [c for c in selected_features if c not in df.columns]
    if missing:
        st.error(f"Missing required columns: {', '.join(missing)}")
        st.stop()
    df_sel = df[selected_features].copy()
else:
    df_sel = df.copy()

df_sel.replace([np.inf, -np.inf], np.nan, inplace=True)
df_sel.dropna(inplace=True)

if scaler is not None:
    X_scaled = scaler.transform(df_sel)
else:
    X_scaled = df_sel.values

probs = predict_func(X_scaled)
preds = np.argmax(probs, axis=1)

if label_encoder is not None:
    preds_labels = label_encoder.inverse_transform(preds)
else:
    preds_labels = preds

results_df = df.loc[df_sel.index].copy()
results_df["Predicted_Label"] = preds_labels
results_df["Predicted_Probability"] = np.max(probs, axis=1)

# ----------------------------
# Tabs
# ----------------------------
tab1, tab2 = st.tabs(["📊 Predictions", "🔎 Explainability"])

with tab1:
    st.subheader("Data Preview")
    st.dataframe(results_df.head())
    st.subheader("Prediction Distribution")
    st.bar_chart(results_df["Predicted_Label"].value_counts())

with tab2:
    shap_explainer = get_shap_explainer(X_scaled, preds)
    lime_explainer = get_lime_explainer(X_scaled, df_sel.columns, np.unique(preds_labels))

    row_id = st.number_input("Pick row index:", 0, len(X_scaled)-1, 0)
    row_data = X_scaled[row_id:row_id+1]

    shap_vals = shap_explainer.shap_values(row_data)
    fig_shap = plot_shap_local(np.array(shap_vals[np.argmax(probs[row_id])])[0], df_sel.columns)
    st.pyplot(fig_shap)

    exp = lime_explainer.explain_instance(X_scaled[row_id], predict_func, num_features=10)
    st.pyplot(exp.as_pyplot_figure())
