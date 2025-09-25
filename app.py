import streamlit as st
import pandas as pd
import numpy as np
import shap
import lime
import lime.lime_tabular
import tensorflow as tf
from sklearn.preprocessing import MinMaxScaler
import os

# ---------------------------
# 1. Load model
# ---------------------------
@st.cache_resource
def load_model():
    return tf.keras.models.load_model("model.h5")

model = load_model()

# ---------------------------
# 2. Chunked CSV loader
# ---------------------------
def load_and_predict_in_chunks(file_path, chunksize=50000):
    scaler = MinMaxScaler()
    all_results = []

    for chunk in pd.read_csv(file_path, chunksize=chunksize):
        # Separate features and label (if label exists)
        if "Label" in chunk.columns:
            X = chunk.drop(columns=["Label"])
            y = chunk["Label"]
        else:
            X = chunk
            y = None

        # Scale
        X_scaled = scaler.fit_transform(X)

        # Reshape for CNN-BiLSTM (samples, timesteps, features=1)
        X_scaled = X_scaled.reshape(X_scaled.shape[0], X_scaled.shape[1], 1)

        # Predict
        preds = model.predict(X_scaled, verbose=0)
        preds_classes = np.argmax(preds, axis=1)

        # Store results
        result_chunk = pd.DataFrame(X, columns=X.columns)
        result_chunk["Prediction"] = preds_classes
        if y is not None:
            result_chunk["True_Label"] = y.values
        all_results.append(result_chunk)

    return pd.concat(all_results, ignore_index=True)

# ---------------------------
# 3. SHAP Explainability
# ---------------------------
def explain_with_shap(X_sample):
    explainer = shap.DeepExplainer(model, X_sample[:50])  # background
    shap_values = explainer.shap_values(X_sample[:1])     # explain first row
    return shap_values

# ---------------------------
# 4. LIME Explainability
# ---------------------------
def explain_with_lime(X_train, X_sample):
    lime_explainer = lime.lime_tabular.LimeTabularExplainer(
        training_data=np.array(X_train),
        mode="classification",
        feature_names=X_train.columns.tolist(),
        class_names=["Normal", "Attack"]
    )
    exp = lime_explainer.explain_instance(
        X_sample.values[0],
        model.predict,
        num_features=10
    )
    return exp

# ---------------------------
# 5. Streamlit UI
# ---------------------------
st.title("CICIDS2017 Intrusion Detection (Chunked)")

csv_file = st.file_uploader("Upload CICIDS2017 CSV", type=["csv"])
if csv_file:
    st.write("⚡ Processing in chunks...")
    results = load_and_predict_in_chunks(csv_file)

    st.write("✅ Prediction Results (sample):")
    st.dataframe(results.head())

    # Select a row for explainability
    row_idx = st.number_input("Pick a row for SHAP/LIME explanation:", 0, len(results)-1, 0)

    if st.button("Explain Row"):
        X = results.drop(columns=["Prediction", "True_Label"], errors="ignore")
        X_scaled = MinMaxScaler().fit_transform(X)
        X_scaled = X_scaled.reshape(X_scaled.shape[0], X_scaled.shape[1], 1)

        shap_vals = explain_with_shap(X_scaled)
        st.write("### SHAP Explanation")
        shap.summary_plot(shap_vals, X, plot_type="bar", show=False)
        st.pyplot(bbox_inches="tight")

        st.write("### LIME Explanation")
        exp = explain_with_lime(X, X.iloc[[row_idx]])
        st.write(exp.as_list())
