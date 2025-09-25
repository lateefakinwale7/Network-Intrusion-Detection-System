# app.py
import streamlit as st
import pandas as pd
import numpy as np
import pickle
import tensorflow as tf
import shap
from lime.lime_tabular import LimeTabularExplainer
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder
import datetime
import zipfile
from io import BytesIO
import os
import time

st.set_page_config(page_title="Intrusion Detection System", layout="wide")

# ----------------------------
# Cache Clearing Function
# ----------------------------
def clear_cache():
    st.cache_resource.clear()
    st.cache_data.clear()

# Add a button to the sidebar to clear the cache
with st.sidebar:
    st.button("Clear Cache", on_click=clear_cache)
    st.markdown("---")
    st.write("If you encounter errors after uploading a new file, try clearing the cache.")

# ----------------------------
# Load resources (model, scaler, encoder, features)
# ----------------------------
@st.cache_resource(show_spinner="Loading essential model files...")
def load_resources():
    try:
        model = tf.keras.models.load_model('cnn_bilstm_model.h5')
        with open('scaler.pkl', 'rb') as f:
            scaler = pickle.load(f)
        with open('label_encoder.pkl', 'rb') as f:
            label_encoder = pickle.load(f)
        with open('selected_features.txt', 'r') as f:
            selected_features = [line.strip() for line in f.readlines()]
        return model, scaler, label_encoder, selected_features
    except FileNotFoundError as e:
        st.error(f"Missing required file: {e}")
        st.stop()

model, scaler, label_encoder, selected_features = load_resources()

# ----------------------------
# Helper for explainers
# ----------------------------
def predict_2d_to_3d(x_2d):
    """
    Helper for LIME / SHAP KernelExplainer predict functions:
    Accepts x_2d: (n_samples, n_features) and returns model probabilities:
    Reshapes to (n_samples, timesteps, 1) and calls model.predict.
    """
    # Ensure numpy array
    x_arr = np.array(x_2d, dtype=np.float32)
    # Reshape to (n_samples, timesteps, 1)
    n_samples = x_arr.shape[0]
    n_features = x_arr.shape[1]
    x_3d = x_arr.reshape(n_samples, n_features, 1)
    preds = model.predict(x_3d, verbose=0)
    return preds

# ----------------------------
# Explainers Caching
# ----------------------------
@st.cache_resource(show_spinner="Preparing SHAP explainer...")
def get_shap_explainer(data, labels):
    """Initializes and caches the SHAP KernelExplainer with a balanced background dataset."""
    unique_labels = np.unique(labels)
    background_samples = []
    
    # Ensure the background set includes at least one example of each class
    for label in unique_labels:
        sample_indices = np.where(labels == label)[0]
        if len(sample_indices) > 0:
            background_samples.append(data[sample_indices[0]])
            
    background = np.array(background_samples)
    if background.shape[0] < 2:
        # Fallback to a small random sample if not enough classes are present
        rng = np.random.default_rng(seed=42)
        bg_idx = rng.choice(data.shape[0], size=min(50, data.shape[0]), replace=False)
        background = data[bg_idx]
        
    return shap.KernelExplainer(predict_2d_to_3d, background)

@st.cache_resource(show_spinner="Preparing LIME explainer...")
def get_lime_explainer(data, feature_names, class_names):
    """Initializes and caches the LIME TabularExplainer."""
    return LimeTabularExplainer(
        data,
        feature_names=feature_names,
        class_names=class_names,
        mode='classification'
    )

# ----------------------------
# Plotting Functions
# ----------------------------
def fig_to_bytes(fig):
    """Converts a matplotlib figure to a PNG byte stream."""
    buf = BytesIO()
    fig.savefig(buf, format='png', bbox_inches='tight')
    buf.seek(0)
    return buf

def plot_shap_local(shap_values, feature_names, title="SHAP Local Explanation"):
    """Generates a SHAP local explanation plot."""
    fig, ax = plt.subplots(figsize=(7, 4))
    feat = np.array(feature_names)
    order = np.argsort(np.abs(shap_values))[::-1]
    top_n = min(len(feat), 20)
    ax.barh(feat[order][:top_n][::-1], shap_values[order][:top_n][::-1])
    ax.set_xlabel("SHAP value (contribution)")
    ax.set_title(title)
    plt.tight_layout()
    return fig

def plot_lime_local(lime_exp, title="LIME Local Explanation"):
    """Generates a LIME local explanation plot."""
    fig = lime_exp.as_pyplot_figure()
    fig.suptitle(title, y=1.02)
    plt.tight_layout()
    return fig

def plot_shap_global(shap_explainer, data, feature_names, n_samples=200):
    """Generates a global SHAP feature importance plot."""
    fig, ax = plt.subplots(figsize=(10, 6))
    subset_n = min(n_samples, data.shape[0])
    rng = np.random.default_rng(seed=42)
    global_idx = rng.choice(data.shape[0], size=subset_n, replace=False)
    shap_vals_global = shap_explainer.shap_values(data[global_idx], nsamples=100)
    
    mean_abs_per_class = np.array([np.mean(np.abs(sv), axis=0) for sv in shap_vals_global])
    mean_abs_across_classes = np.mean(mean_abs_per_class, axis=0)
    
    feat = np.array(feature_names)
    orderg = np.argsort(mean_abs_across_classes)[::-1]
    
    ax.barh(feat[orderg][:30][::-1], mean_abs_across_classes[orderg][:30][::-1])
    ax.set_title("Global Feature Importance (Mean |SHAP| across classes)")
    ax.set_xlabel("Mean |SHAP value|")
    plt.tight_layout()
    return fig


# ----------------------------
# Main App UI
# ----------------------------
st.title("🛡️ Network Intrusion Detection System — with SHAP & LIME")
st.markdown("Upload a CSV file with network traffic data to get predictions and explanations.")
st.markdown("---")
st.info("Required features: " + ", ".join(selected_features))

uploaded_file = st.file_uploader("Upload network traffic CSV", type=["csv"])

if uploaded_file is None:
    st.info("Please upload a CSV file to begin.")
    st.stop()

# ----------------------------
# Read, Preprocess & Predict
# ----------------------------
with st.spinner("Processing file and generating predictions..."):
    try:
        df = pd.read_csv(uploaded_file)
    except Exception as e:
        st.error(f"Failed to read CSV: {e}")
        st.stop()

    missing = [c for c in selected_features if c not in df.columns]
    if missing:
        st.error(f"Uploaded CSV is missing required columns: {', '.join(missing)}")
        st.stop()

    df_sel = df[selected_features].copy()
    df_sel.replace([np.inf, -np.inf], np.nan, inplace=True)
    df_sel.dropna(inplace=True)

    if df_sel.empty:
        st.error("No valid data found in the uploaded CSV after handling missing values.")
        st.stop()

    try:
        X_scaled = scaler.transform(df_sel)
        X_reshaped = X_scaled.reshape(X_scaled.shape[0], X_scaled.shape[1], 1)
        preds_probs = model.predict(X_reshaped, verbose=0)
        preds = np.argmax(preds_probs, axis=1)
        preds_labels = label_encoder.inverse_transform(preds)
        
        results_df = df.loc[df_sel.index].copy()
        results_df['Predicted_Label'] = preds_labels
        results_df['Predicted_Probability'] = np.max(preds_probs, axis=1)
        
    except Exception as e:
        st.error(f"Preprocessing or prediction failed: {e}")
        st.stop()

# Cache the explainers after data is loaded and scaled
shap_explainer = get_shap_explainer(X_scaled, preds)
lime_explainer = get_lime_explainer(X_scaled, selected_features, label_encoder.classes_)

# ----------------------------
# Main Tabs
# ----------------------------
tab1, tab2 = st.tabs(["📊 Predictions", "🔎 Explainability & Export"])

with tab1:
    st.subheader("Data Preview")
    st.dataframe(results_df.head())
    
    st.subheader("Prediction Distribution")
    counts = results_df['Predicted_Label'].value_counts()
    st.bar_chart(counts)
    st.dataframe(counts.to_frame(name='Count'))

with tab2:
    st.header("Model Explainability (SHAP + LIME)")
    st.write("Select a predicted class and a row to view detailed explanations.")

    unique_labels = results_df['Predicted_Label'].unique().tolist()
    selected_class = st.selectbox("Select predicted class to inspect:", unique_labels)
    class_rows = results_df[results_df['Predicted_Label'] == selected_class]

    if class_rows.empty:
        st.warning("No rows for the selected class.")
    else:
        selected_row_index = st.selectbox(
            "Select a row index from this class (DataFrame index):", 
            class_rows.index.tolist()
        )
        numpy_row_idx = np.where(df_sel.index.values == selected_row_index)[0][0]

        st.subheader("Selected Row Details")
        st.dataframe(results_df.loc[[selected_row_index]])

        col_shap, col_lime = st.columns(2)

        # ---------------- SHAP
        with col_shap:
            st.subheader("SHAP (KernelExplainer)")
            with st.spinner("Calculating SHAP values..."):
                shap_vals = shap_explainer.shap_values(X_scaled[numpy_row_idx:numpy_row_idx+1], nsamples=100)
                pred_class_idx = int(preds[numpy_row_idx])

            if pred_class_idx < len(shap_vals):
                shap_for_row = np.array(shap_vals[pred_class_idx])[0]

                fig_local = plot_shap_local(shap_for_row, selected_features, f"Local SHAP for Row {selected_row_index}")
                st.pyplot(fig_local)
                plt.close(fig_local)
            else:
                st.warning("SHAP explanation failed. The predicted class was not found in the explainer's output.")

        # ---------------- LIME
        with col_lime:
            st.subheader("LIME (local)")
            with st.spinner("Calculating LIME explanation..."):
                exp = lime_explainer.explain_instance(
                    X_scaled[numpy_row_idx],
                    predict_2d_to_3d,
                    num_features=min(10, len(selected_features))
                )
            fig_lime = plot_lime_local(exp, f"LIME Explanation for Row {selected_row_index}")
            st.pyplot(fig_lime)
            plt.close(fig_lime)
            
            st.write("Top contributions:")
            st.write(pd.DataFrame(exp.as_list(), columns=["Feature", "Contribution"]))

        st.markdown("---")
        
        # ----------------------------
        # Batch export (multiple rows) with progress
        # ----------------------------
        st.header("📦 Batch Export Explanations")
        row_limit = st.number_input(
            f"How many rows to explain (max: {X_scaled.shape[0]}):",
            min_value=1, 
            max_value=X_scaled.shape[0], 
            value=min(20, X_scaled.shape[0]), 
            step=1
        )

        if st.button("Generate Batch Explanations (ZIP)"):
            with st.spinner("Generating batch explanations..."):
                zip_buf = BytesIO()
                with zipfile.ZipFile(zip_buf, "w", zipfile.ZIP_DEFLATED) as zf:
                    # Write global SHAP summary plot
                    fig_glob = plot_shap_global(shap_explainer, X_scaled, selected_features)
                    zf.writestr("shap_global_summary.png", fig_to_bytes(fig_glob).getvalue())
                    plt.close(fig_glob)

                    all_lime_rows = []
                    progress_bar = st.progress(0)
                    for i in range(row_limit):
                        row_index_in_original_df = df_sel.index[i]
                        
                        # SHAP for row i
                        shap_vals_row = shap_explainer.shap_values(X_scaled[i:i+1], nsamples=100)
                        pred_class_idx = int(preds[i])
                        shap_row_vals = np.array(shap_vals_row[pred_class_idx])[0]
                        fig_s = plot_shap_local(shap_row_vals, selected_features, f"SHAP Local (Row {row_index_in_original_df})")
                        zf.writestr(f"shap_row_{row_index_in_original_df}.png", fig_to_bytes(fig_s).getvalue())
                        plt.close(fig_s)
                        
                        # LIME for row i
                        exp = lime_explainer.explain_instance(X_scaled[i], predict_2d_to_3d, num_features=10)
                        fig_l = exp.as_pyplot_figure()
                        fig_l.suptitle(f"LIME Local (Row {row_index_in_original_df})", y=1.02)
                        plt.tight_layout()
                        zf.writestr(f"lime_row_{row_index_in_original_df}.png", fig_to_bytes(fig_l).getvalue())
                        plt.close(fig_l)

                        # Collect LIME CSV rows
                        for f, c in exp.as_list():
                            all_lime_rows.append({"Row_Index": row_index_in_original_df, "Feature": f, "Contribution": c})
                        
                        progress_bar.progress(int(((i + 1) / row_limit) * 100))

                    lime_df_all = pd.DataFrame(all_lime_rows)
                    zf.writestr("lime_all_contributions.csv", lime_df_all.to_csv(index=False).encode('utf-8'))

                ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
                zipfname = f"explanations_batch_{ts}.zip"
                
                st.download_button(
                    "⬇️ Download explanations (ZIP)",
                    data=zip_buf.getvalue(),
                    file_name=zipfname,
                    mime="application/zip"
                )
                progress_bar.empty()
                st.success("Batch export completed!")

st.markdown("---")
st.caption("Notes: SHAP KernelExplainer is model-agnostic but can be slow. For large datasets, choose a moderate row limit.")
