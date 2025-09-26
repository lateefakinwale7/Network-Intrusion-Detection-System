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

# --- Configuration ---
st.set_page_config(page_title="Intrusion Detection System", layout="wide")

# --- Constants ---
BATCH_SIZE = 100000 
MAX_ROWS_FOR_EXPLANATION = 50 # Hard limit for batch export

# --- Session State Initialization ---
if 'batch_offset' not in st.session_state:
    st.session_state.batch_offset = 0
if 'all_scaled_data' not in st.session_state:
    st.session_state.all_scaled_data = None
if 'all_predictions' not in st.session_state:
    st.session_state.all_predictions = None
if 'full_results_df' not in st.session_state:
    st.session_state.full_results_df = None
if 'df_sel_all' not in st.session_state:
    st.session_state.df_sel_all = None
if 'df_full' not in st.session_state: # 👈 ADDED: Full DataFrame
    st.session_state.df_full = None 
if 'total_valid_rows' not in st.session_state:
    st.session_state.total_valid_rows = 0

# ----------------------------
# Cache Clearing Function
# ----------------------------
def clear_cache():
    st.cache_resource.clear()
    st.cache_data.clear()
    # Reset session state for a fresh start
    for key in list(st.session_state.keys()):
        if key not in ['batch_offset']: # Don't clear offset yet
             del st.session_state[key]
    st.session_state.batch_offset = 0

# Add a button to the sidebar to clear the cache
with st.sidebar:
    st.button("Clear Cache & Reset Data", on_click=clear_cache)
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
    """Initializes and caches the SHAP KernelExplainer with a fixed, small background dataset."""
    
    # CRITICAL: Use a fixed, small sample for the background.
    BACKGROUND_SIZE = 50 
    
    rng = np.random.default_rng(seed=42)
    
    if data.shape[0] <= BACKGROUND_SIZE:
        background = data
    else:
        # Use a random subset if the data is large
        bg_idx = rng.choice(data.shape[0], size=BACKGROUND_SIZE, replace=False)
        background = data[bg_idx]
        
    st.info(f"Using {background.shape[0]} samples for SHAP KernelExplainer background.")
    
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
# Plotting Functions (Unchanged)
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

def plot_shap_global(shap_explainer, data, feature_names, n_samples=50):
    """Generates a global SHAP feature importance plot on a small, safe sample."""
    fig, ax = plt.subplots(figsize=(10, 6))
    subset_n = min(n_samples, data.shape[0])
    if subset_n == 0:
        st.warning("Not enough data to generate global plot.")
        return plt.figure() # Return empty figure
        
    rng = np.random.default_rng(seed=42)
    global_idx = rng.choice(data.shape[0], size=subset_n, replace=False)
    
    # SHAP calculation is costly, reduced nsamples here too for speed
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


# ----------------------------
# Data Loading & Preprocessing (Cached)
# ----------------------------
@st.cache_data(show_spinner="Reading and cleaning file structure...")
def load_and_clean_data(uploaded_file):
    try:
        df = pd.read_csv(uploaded_file)
    except Exception as e:
        raise Exception(f"Failed to read CSV: {e}")

    missing = [c for c in selected_features if c not in df.columns]
    if missing:
        raise Exception(f"CSV is missing required columns: {', '.join(missing)}")

    # Create a copy with only selected features and handle NaNs/Infs
    df_sel_all = df[selected_features].copy()
    df_sel_all.replace([np.inf, -np.inf], np.nan, inplace=True)
    df_sel_all.dropna(inplace=True)
    
    if df_sel_all.empty:
        raise Exception("No valid data found after cleaning.")
        
    st.session_state.total_valid_rows = df_sel_all.shape[0]
    
    return df, df_sel_all # Return both

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
    st.session_state.batch_offset = 0 # Reset state for new upload
    st.stop()
    
# --- Load Data on Upload ---
try:
    if st.session_state.df_full is None or st.session_state.df_sel_all is None: # Check state
        # Only run load_and_clean_data once per uploaded file
        st.session_state.df_full, st.session_state.df_sel_all = load_and_clean_data(uploaded_file)
        
    # --- Assign local variables from session state ---
    df_full = st.session_state.df_full # 👈 Defined here
    df_sel_all = st.session_state.df_sel_all
    N_ROWS = st.session_state.total_valid_rows
    N_BATCHES = int(np.ceil(N_ROWS / BATCH_SIZE))
    
except Exception as e:
    st.error(f"Data loading failed: {e}")
    st.stop()


# ----------------------------
# Prediction Batch Processing Logic
# ----------------------------
current_offset = st.session_state.batch_offset
rows_processed = current_offset

st.subheader("Batch Prediction Control")
st.metric("Total Valid Rows", N_ROWS)
st.metric("Rows Processed", rows_processed)

if rows_processed < N_ROWS:
    batch_index = int(current_offset / BATCH_SIZE)
    st.info(f"Ready to process **Batch {batch_index + 1} of {N_BATCHES}** (Rows {current_offset + 1} to {min(current_offset + BATCH_SIZE, N_ROWS)})")
    
    if st.button(f"Process Next Batch ({BATCH_SIZE} rows)"):
        with st.spinner(f"Processing Batch {batch_index + 1}..."):
            batch_data = df_sel_all.iloc[current_offset:current_offset + BATCH_SIZE]
            
            try:
                # 1. Scale
                X_scaled_batch = scaler.transform(batch_data)
                
                # 2. Reshape & Predict
                X_reshaped_batch = X_scaled_batch.reshape(X_scaled_batch.shape[0], X_scaled_batch.shape[1], 1)
                preds_probs_batch = model.predict(X_reshaped_batch, verbose=0)
                preds_batch = np.argmax(preds_probs_batch, axis=1)
                preds_labels_batch = label_encoder.inverse_transform(preds_batch)
                
                # 3. Store Results (Appending to Session State)
                batch_df = df_full.loc[batch_data.index].copy()
                batch_df['Predicted_Label'] = preds_labels_batch
                batch_df['Predicted_Probability'] = np.max(preds_probs_batch, axis=1)

                if st.session_state.full_results_df is None:
                    st.session_state.full_results_df = batch_df
                    st.session_state.all_scaled_data = X_scaled_batch
                    st.session_state.all_predictions = preds_batch
                else:
                    st.session_state.full_results_df = pd.concat([st.session_state.full_results_df, batch_df])
                    st.session_state.all_scaled_data = np.concatenate([st.session_state.all_scaled_data, X_scaled_batch])
                    st.session_state.all_predictions = np.concatenate([st.session_state.all_predictions, preds_batch])

                # 4. Update Offset and Rerun
                st.session_state.batch_offset += BATCH_SIZE
                st.success(f"Batch {batch_index + 1} completed!")
                st.rerun() 
                
            except Exception as e:
                st.error(f"Prediction failed for batch starting at row {current_offset}: {e}")
                
elif rows_processed > 0:
    st.success(f"✅ All {N_ROWS} rows processed across {N_BATCHES} batches!")
    
    # Define variables for the rest of the application
    results_df = st.session_state.full_results_df
    X_scaled = st.session_state.all_scaled_data
    preds = st.session_state.all_predictions
    
    # Initialize explainers now that we have all data/predictions
    shap_explainer = get_shap_explainer(X_scaled, preds)
    lime_explainer = get_lime_explainer(X_scaled, selected_features, label_encoder.classes_)

    # ----------------------------
    # Main Tabs (Only show if processing is complete)
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
            
            # Find the numpy index corresponding to the selected row's original index
            numpy_row_idx_array = np.where(df_sel_all.index.values == selected_row_index)[0]
            if len(numpy_row_idx_array) == 0:
                 st.error("Error: Could not find row in processed data.")
                 st.stop()
            numpy_row_idx = numpy_row_idx_array[0] # The index in X_scaled
            

            st.subheader("Selected Row Details")
            st.dataframe(results_df.loc[[selected_row_index]])

            col_shap, col_lime = st.columns(2)

            # ---------------- SHAP (Local)
            with col_shap:
                st.subheader("SHAP (KernelExplainer)")
                with st.spinner("Calculating SHAP values..."):
                    # Use a moderate number of samples for the explanation
                    shap_vals = shap_explainer.shap_values(X_scaled[numpy_row_idx:numpy_row_idx+1], nsamples=50) 
                    pred_class_idx = int(preds[numpy_row_idx])

                if pred_class_idx < len(shap_vals):
                    shap_for_row = np.array(shap_vals[pred_class_idx])[0]
                    predicted_label = results_df.loc[selected_row_index, 'Predicted_Label']
                    fig_local = plot_shap_local(shap_for_row, selected_features, 
                                                f"Local SHAP (Row {selected_row_index} - Class: {predicted_label})")
                    st.pyplot(fig_local)
                    plt.close(fig_local)
                else:
                    st.warning("SHAP explanation failed. The predicted class was not found in the explainer's output.")

            # ---------------- LIME (Local)
            with col_lime:
                st.subheader("LIME (local)")
                with st.spinner("Calculating LIME explanation..."):
                    exp = lime_explainer.explain_instance(
                        X_scaled[numpy_row_idx],
                        predict_2d_to_3d,
                        num_features=min(10, len(selected_features))
                    )
                predicted_label = results_df.loc[selected_row_index, 'Predicted_Label']
                fig_lime = plot_lime_local(exp, f"LIME Explanation (Row {selected_row_index} - Class: {predicted_label})")
                st.pyplot(fig_lime)
                plt.close(fig_lime)
                
                st.write("Top contributions:")
                st.write(pd.DataFrame(exp.as_list(), columns=["Feature", "Contribution"]))

            st.markdown("---")
            
            # ----------------------------
            # Global Explanation (Optional/Controlled)
            # ----------------------------
            st.header("🌎 Global Explanations (Resource Intensive)")
            st.warning("This runs SHAP on a small subset (50 rows). Do not use on low-resource environments.")
            if st.button("Generate Global SHAP Plot"):
                with st.spinner("Calculating global SHAP values across subset..."):
                    fig_glob = plot_shap_global(shap_explainer, X_scaled, selected_features, n_samples=50)
                    st.pyplot(fig_glob)
                    plt.close(fig_glob)

            st.markdown("---")
            
            # ----------------------------
            # Batch export (multiple rows) with progress
            # ----------------------------
            st.header("📦 Batch Export Explanations")
            
            # CRITICAL: Hard limit on rows for export
            row_limit = st.number_input(
                f"How many rows to explain (max: {MAX_ROWS_FOR_EXPLANATION}):",
                min_value=1, 
                max_value=min(MAX_ROWS_FOR_EXPLANATION, X_scaled.shape[0]), 
                value=min(20, X_scaled.shape[0], MAX_ROWS_FOR_EXPLANATION), 
                step=1
            )

            if st.button("Generate Batch Explanations (ZIP)"):
                with st.spinner("Generating batch explanations..."):
                    zip_buf = BytesIO()
                    with zipfile.ZipFile(zip_buf, "w", zipfile.ZIP_DEFLATED) as zf:
                        # Write prediction results CSV
                        zf.writestr("full_prediction_results.csv", results_df.to_csv(index=True).encode('utf-8'))
                        
                        # Write global SHAP summary plot
                        fig_glob = plot_shap_global(shap_explainer, X_scaled, selected_features, n_samples=50)
                        zf.writestr("shap_global_summary_50_rows.png", fig_to_bytes(fig_glob).getvalue())
                        plt.close(fig_glob)

                        all_lime_rows = []
                        progress_bar = st.progress(0, text="Batch Explanation Progress...")
                        
                        # The loop runs only up to row_limit
                        for i in range(row_limit):
                            row_index_in_original_df = df_sel_all.index[i]
                            
                            # SHAP for row i (using lower nsamples for speed)
                            shap_vals_row = shap_explainer.shap_values(X_scaled[i:i+1], nsamples=50) 
                            pred_class_idx = int(preds[i])
                            
                            if pred_class_idx < len(shap_vals_row):
                                pred_class_label = results_df.loc[row_index_in_original_df, 'Predicted_Label']
                                shap_row_vals = np.array(shap_vals_row[pred_class_idx])[0]
                                fig_s = plot_shap_local(shap_row_vals, selected_features, 
                                                        f"SHAP Local (Row {row_index_in_original_df} - Class: {pred_class_label})")
                                zf.writestr(f"shap_row_{row_index_in_original_df}.png", fig_to_bytes(fig_s).getvalue())
                                plt.close(fig_s)
                            
                            # LIME for row i
                            exp = lime_explainer.explain_instance(X_scaled[i], predict_2d_to_3d, num_features=10)
                            
                            # Create LIME plot
                            fig_l = exp.as_pyplot_figure()
                            fig_l.suptitle(f"LIME Local (Row {row_index_in_original_df})", y=1.02)
                            plt.tight_layout()
                            zf.writestr(f"lime_row_{row_index_in_original_df}.png", fig_to_bytes(fig_l).getvalue())
                            plt.close(fig_l)

                            # Collect LIME CSV rows
                            for f, c in exp.as_list():
                                all_lime_rows.append({"Row_Index": row_index_in_original_df, "Feature": f, "Contribution": c})
                            
                            progress_bar.progress(int(((i + 1) / row_limit) * 100), text=f"Batch Explanation Progress: {i+1} of {row_limit} rows")

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
if rows_processed == 0 and uploaded_file:
    st.info("Click 'Process Next Batch' to begin analysis.")
st.caption("Notes: The application relies on cached resources and session state. Use 'Clear Cache & Reset Data' if issues persist.")
