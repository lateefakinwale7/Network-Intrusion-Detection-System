# app.py
# Imports
# Code adapted from Python official documentation (Van Rossum & Drake, 2009)
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
# End of adapted code
# Configuration
# Code adapted from Streamlit documentation (Streamlit, 2023)
st.set_page_config(page_title="Intrusion Detection System", layout="wide")
# End of adapted code

#Constants
BATCH_SIZE = 100000 
MAX_ROWS_FOR_EXPLANATION = 50 # Hard limit for batch export
#Session State Initialization
# Code adapted from Streamlit documentation (Streamlit, 2023)
if 'batch_offset' not in st.session_state:
    st.session_state.batch_offset = 0
if 'current_scaled_data' not in st.session_state:
    st.session_state.current_scaled_data = None
if 'current_predictions' not in st.session_state:
    st.session_state.current_predictions = None
if 'current_results_df' not in st.session_state:
    st.session_state.current_results_df = None
if 'df_sel_all' not in st.session_state:
    st.session_state.df_sel_all = None
if 'df_full' not in st.session_state:
    st.session_state.df_full = None 
if 'total_valid_rows' not in st.session_state:
    st.session_state.total_valid_rows = 0
# End of adapted code
# Cache Clearing Function
# Code adapted from Streamlit documentation (Streamlit, 2023)
def clear_cache():
    st.cache_resource.clear()
    st.cache_data.clear()
    for key in list(st.session_state.keys()):
        if key not in ['batch_offset']:
             del st.session_state[key]
    st.session_state.batch_offset = 0
# End of adapted code
# Code adapted from Streamlit documentation (Streamlit, 2023)
with st.sidebar:
    st.button("Clear Cache & Reset Data", on_click=clear_cache)
    st.markdown("---")
    st.write("If you encounter errors after uploading a new file, try clearing the cache.")
# End of adapted code
# Load resources (model, scaler, encoder, features)
# Code adapted from TensorFlow documentation (TensorFlow, 2023) 
# and Scikit-learn documentation (Pedregosa et al., 2011)
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
        if not hasattr(label_encoder, 'inverse_transform'):
            raise TypeError("Loaded label_encoder object does not have the 'inverse_transform' method.")
            
        return model, scaler, label_encoder, selected_features
    except FileNotFoundError as e:
        st.error(f"Missing required file: {e}")
        st.stop()
    except TypeError as e:
        st.error(f"Resource Loading Error: {e}")
        st.stop()
# End of adapted code
model, scaler, label_encoder, selected_features = load_resources()
# Helper for explainers
# Code adapted from SHAP documentation (Lundberg & Lee, 2017)
def predict_2d_to_3d(x_2d):
    x_arr = np.array(x_2d, dtype=np.float32)
    n_samples = x_arr.shape[0]
    n_features = x_arr.shape[1]
    x_3d = x_arr.reshape(n_samples, n_features, 1)
    preds = model.predict(x_3d, verbose=0)
    return preds
# End of adapted code
# Explainers Caching
# Code adapted from SHAP documentation (Lundberg & Lee, 2017)
@st.cache_resource(show_spinner="Preparing SHAP explainer...")
def get_shap_explainer(data, labels):
    BACKGROUND_SIZE = 50 
    rng = np.random.default_rng(seed=42)
    if data.shape[0] <= BACKGROUND_SIZE:
        background = data
    else:
        bg_idx = rng.choice(data.shape[0], size=BACKGROUND_SIZE, replace=False)
        background = data[bg_idx]
    st.info(f"Using {background.shape[0]} samples for SHAP KernelExplainer background.")
    return shap.KernelExplainer(predict_2d_to_3d, background)
# End of adapted code
# Code adapted from Ribeiro et al. (2016)
@st.cache_resource(show_spinner="Preparing LIME explainer...")
def get_lime_explainer(data, feature_names, class_names):
    return LimeTabularExplainer(
        data,
        feature_names=feature_names,
        class_names=class_names,
        mode='classification'
    )
# End of adapted code
# Plotting Functions
# Code adapted from Matplotlib documentation (Hunter, 2007)
def fig_to_bytes(fig):
    buf = BytesIO()
    fig.savefig(buf, format='png', bbox_inches='tight')
    buf.seek(0)
    return buf
# End of adapted code
# Code adapted from SHAP documentation (Lundberg & Lee, 2017)
def plot_shap_local(shap_values, feature_names, title="SHAP Local Explanation"):
    fig, ax = plt.subplots(figsize=(7, 4))
    feat = np.array(feature_names)
    order = np.argsort(np.abs(shap_values))[::-1]
    top_n = min(len(feat), 20)
    ax.barh(feat[order][:top_n][::-1], shap_values[order][:top_n][::-1])
    ax.set_xlabel("SHAP value (contribution)")
    ax.set_title(title)
    plt.tight_layout()
    return fig
# End of adapted code
# Code adapted from Ribeiro et al. (2016)
def plot_lime_local(lime_exp, title="LIME Local Explanation"):
    fig = lime_exp.as_pyplot_figure()
    fig.suptitle(title, y=1.02)
    plt.tight_layout()
    return fig
# End of adapted code
# Code adapted from SHAP documentation (Lundberg & Lee, 2017)
def plot_shap_global(shap_explainer, data, feature_names, n_samples=50):
    fig, ax = plt.subplots(figsize=(10, 6))
    subset_n = min(n_samples, data.shape[0])
    if subset_n == 0:
        st.warning("Not enough data to generate global plot.")
        return plt.figure()
    rng = np.random.default_rng(seed=42)
    global_idx = rng.choice(data.shape[0], size=subset_n, replace=False)
    shap_vals_global = shap_explainer.shap_values(data[global_idx], nsamples=50) 
    mean_abs_per_class = np.array([np.mean(np.abs(sv), axis=0) for sv in shap_vals_global])
    mean_abs_across_classes = np.mean(mean_abs_per_class, axis=0)
    feat = np.array(feature_names)
    orderg = np.argsort(mean_abs_across_classes)[::-1]
    ax.barh(feat[orderg][:30][::-1], mean_abs_across_classes[orderg][:30][::-1])
    ax.set_title(f"Global Feature Importance (Mean |SHAP| across {subset_n} samples)")
    ax.set_xlabel("Mean |SHAP value|")
    plt.tight_layout()
    return fig
# End of adapted code
# Data Loading & Preprocessing
# Code adapted from Pandas documentation (Pandas Team, 2023)
@st.cache_data(show_spinner="Reading and cleaning file structure...")
def load_and_clean_data(uploaded_file):
    try:
        df = pd.read_csv(uploaded_file)
    except Exception as e:
        raise Exception(f"Failed to read CSV: {e}")
    missing = [c for c in selected_features if c not in df.columns]
    if missing:
        raise Exception(f"CSV is missing required columns: {', '.join(missing)}")
    df_sel_all = df[selected_features].copy()
    df_sel_all.replace([np.inf, -np.inf], np.nan, inplace=True)
    df_sel_all.dropna(inplace=True)
    if df_sel_all.empty:
        raise Exception("No valid data found after cleaning.")
    st.session_state.total_valid_rows = df_sel_all.shape[0]
    return df, df_sel_all 
# End of adapted code
# Main App UI
# Code adapted from Streamlit documentation (Streamlit, 2023)
st.title("Network Intrusion Detection System — Single-Batch Analysis")
st.markdown("Upload a CSV file and process one batch at a time to manage memory.")
st.markdown("---")
st.info("Required features: " + ", ".join(selected_features))
uploaded_file = st.file_uploader("Upload network traffic CSV", type=["csv"])
# End of adapted code
# Prediction Batch Processing
# Code adapted from Scikit-learn documentation (Pedregosa et al., 2011) 
# and TensorFlow documentation (TensorFlow, 2023)
if uploaded_file is None:
    st.info("Please upload a CSV file to begin.")
    st.session_state.batch_offset = 0
    st.stop()
try:
    if st.session_state.df_sel_all is None or st.session_state.df_full is None:
        df_full, df_sel_all = load_and_clean_data(uploaded_file)
        st.session_state.df_full = df_full
        st.session_state.df_sel_all = df_sel_all
    else:
        df_full = st.session_state.df_full
        df_sel_all = st.session_state.df_sel_all
except Exception as e:
    st.error(str(e))
    st.stop()
start_idx = st.session_state.batch_offset
end_idx = min(start_idx + BATCH_SIZE, st.session_state.total_valid_rows)
if start_idx >= st.session_state.total_valid_rows:
    st.success(" All rows processed.")
    st.session_state.batch_offset = 0
    st.stop()
st.info(f"Processing rows {start_idx} to {end_idx} of {st.session_state.total_valid_rows}")
current_batch = df_sel_all.iloc[start_idx:end_idx]
scaled_data = scaler.transform(current_batch.values)
reshaped_data = scaled_data.reshape(scaled_data.shape[0], scaled_data.shape[1], 1)
predictions = model.predict(reshaped_data, verbose=0)
pred_classes = np.argmax(predictions, axis=1)
pred_labels = label_encoder.inverse_transform(pred_classes)
results_df = df_full.iloc[start_idx:end_idx].copy()
results_df['Prediction'] = pred_labels
results_df['Confidence'] = np.max(predictions, axis=1)
st.session_state.current_scaled_data = scaled_data
st.session_state.current_predictions = predictions
st.session_state.current_results_df = results_df
st.session_state.batch_offset = end_idx
# End of adapted code
# Results Display
# Code adapted from Streamlit documentation (Streamlit, 2023)
st.subheader(" Batch Results")
st.dataframe(results_df.head(20))
st.write(f"Showing {len(results_df)} rows in this batch.")
# End of adapted code
# SHAP and LIME Explanations
# Code adapted from SHAP documentation (Lundberg & Lee, 2017) 
# and Ribeiro et al. (2016)
if len(results_df) > 0:
    st.subheader(" Model Explanations")
    shap_explainer = get_shap_explainer(st.session_state.current_scaled_data, pred_labels)
    lime_explainer = get_lime_explainer(
        st.session_state.current_scaled_data, 
        selected_features, 
        list(label_encoder.classes_)
    )
    instance_idx = st.number_input("Select row index for local explanation", min_value=0, max_value=len(results_df)-1, value=0)
    instance_data = st.session_state.current_scaled_data[instance_idx:instance_idx+1]
    shap_values = shap_explainer.shap_values(instance_data, nsamples=50)
    lime_exp = lime_explainer.explain_instance(
        st.session_state.current_scaled_data[instance_idx],
        predict_2d_to_3d,
        num_features=10
    )
    st.write("SHAP Local Explanation")
    fig_shap = plot_shap_local(shap_values[0][0], selected_features)
    st.pyplot(fig_shap)
    st.write(" LIME Local Explanation")
    fig_lime = plot_lime_local(lime_exp)
    st.pyplot(fig_lime)
    st.write(" SHAP Global Explanation")
    fig_global = plot_shap_global(shap_explainer, st.session_state.current_scaled_data, selected_features)
    st.pyplot(fig_global)
# End of adapted code
# Export Results
# Code adapted from Pandas documentation (Pandas Team, 2023) 
# and Python zipfile module documentation (Python Software Foundation, 2023)
st.subheader(" Export Batch Results")
timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
csv_filename = f"batch_results_{timestamp}.csv"
zip_filename = f"batch_results_{timestamp}.zip"
csv_buffer = BytesIO()
results_df.to_csv(csv_buffer, index=False)
csv_buffer.seek(0)
zip_buffer = BytesIO()
with zipfile.ZipFile(zip_buffer, "w", zipfile.ZIP_DEFLATED) as zf:
    zf.writestr(csv_filename, csv_buffer.getvalue())
zip_buffer.seek(0)

st.download_button(
    label="⬇️ Download Batch Results (ZIP)",
    data=zip_buffer,
    file_name=zip_filename,
    mime="application/zip"
)
# End of adapted code
