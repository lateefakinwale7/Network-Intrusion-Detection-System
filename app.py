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
import gc
import psutil

st.set_page_config(page_title="Intrusion Detection System", layout="wide")

# ----------------------------
# Memory Management Functions
# ----------------------------
def check_memory_safe(threshold=85):
    """Check if memory usage is safe to proceed."""
    try:
        memory = psutil.virtual_memory()
        return memory.percent < threshold
    except:
        return True  # If we can't check memory, proceed anyway

def clear_memory():
    """Force garbage collection."""
    gc.collect()

# ----------------------------
# Cache Clearing Function
# ----------------------------
def clear_cache():
    st.cache_resource.clear()
    st.cache_data.clear()
    clear_memory()

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
# Explainers with Sampling
# ----------------------------
@st.cache_resource(show_spinner="Preparing SHAP explainer with sampling...")
def get_shap_explainer_sampled(data, labels, max_samples=1000):
    """Initializes SHAP explainer with smart sampling for large datasets."""
    
    # If dataset is large, sample strategically
    if data.shape[0] > max_samples:
        st.info(f"Large dataset detected ({data.shape[0]} rows). Sampling {max_samples} rows for explainability.")
        
        # Stratified sampling to maintain class distribution
        unique_labels, label_counts = np.unique(labels, return_counts=True)
        samples_per_class = max(1, max_samples // len(unique_labels))
        
        sampled_indices = []
        for label in unique_labels:
            label_indices = np.where(labels == label)[0]
            # Ensure we have at least 1 sample per class, but not more than available
            n_samples = min(samples_per_class, len(label_indices))
            if n_samples > 0:
                selected = np.random.choice(label_indices, size=n_samples, replace=False)
                sampled_indices.extend(selected)
        
        # If we need more samples, fill randomly
        if len(sampled_indices) < max_samples:
            remaining = max_samples - len(sampled_indices)
            all_indices = np.arange(data.shape[0])
            unused_indices = np.setdiff1d(all_indices, sampled_indices)
            if len(unused_indices) > 0:
                additional = np.random.choice(unused_indices, 
                                            size=min(remaining, len(unused_indices)), 
                                            replace=False)
                sampled_indices.extend(additional)
        
        data = data[sampled_indices]
        labels = labels[sampled_indices]
    
    # Create balanced background
    unique_labels = np.unique(labels)
    background_samples = []
    
    for label in unique_labels:
        sample_indices = np.where(labels == label)[0]
        if len(sample_indices) > 0:
            # Take up to 5 samples per class for background
            n_background = min(5, len(sample_indices))
            background_samples.extend(
                data[sample_indices[:n_background]]
            )
    
    background = np.array(background_samples)
    
    # Fallback if background is too small
    if background.shape[0] < 2:
        rng = np.random.default_rng(seed=42)
        bg_idx = rng.choice(data.shape[0], size=min(50, data.shape[0]), replace=False)
        background = data[bg_idx]
    
    return shap.KernelExplainer(predict_2d_to_3d, background)

@st.cache_resource(show_spinner="Preparing LIME explainer with sampling...")
def get_lime_explainer_sampled(data, feature_names, class_names, max_samples=5000):
    """Initializes LIME explainer with sampling for large datasets."""
    
    # Sample data for LIME training (LIME uses data for statistics)
    if data.shape[0] > max_samples:
        st.info(f"Sampling {max_samples} rows for LIME explainer initialization.")
        rng = np.random.default_rng(seed=42)
        sample_idx = rng.choice(data.shape[0], size=max_samples, replace=False)
        data = data[sample_idx]
    
    return LimeTabularExplainer(
        data,
        feature_names=feature_names,
        class_names=class_names,
        mode='classification',
        discretize_continuous=False,  # Reduces memory usage
        random_state=42
    )

# ----------------------------
# Batch Processing Functions
# ----------------------------
def process_batch_shap(shap_explainer, data_batch, preds_batch, batch_size=10):
    """Process SHAP explanations in batches to manage memory."""
    shap_results = []
    
    for i in range(0, len(data_batch), batch_size):
        batch_end = min(i + batch_size, len(data_batch))
        batch_data = data_batch[i:batch_end]
        batch_preds = preds_batch[i:batch_end]
        
        # Process current batch
        with st.spinner(f"Processing SHAP batch {i//batch_size + 1}/{(len(data_batch)-1)//batch_size + 1}..."):
            for j in range(len(batch_data)):
                row_idx = i + j
                try:
                    shap_vals = shap_explainer.shap_values(
                        batch_data[j:j+1], 
                        nsamples=50  # Reduced for batch processing
                    )
                    pred_class_idx = int(batch_preds[j])
                    
                    if pred_class_idx < len(shap_vals):
                        shap_for_row = np.array(shap_vals[pred_class_idx])[0]
                        shap_results.append({
                            'row_index': row_idx,
                            'shap_values': shap_for_row,
                            'prediction': pred_class_idx
                        })
                    else:
                        shap_results.append({
                            'row_index': row_idx,
                            'shap_values': None,
                            'prediction': pred_class_idx
                        })
                except Exception as e:
                    st.warning(f"SHAP failed for row {row_idx}: {e}")
                    shap_results.append({
                        'row_index': row_idx,
                        'shap_values': None,
                        'prediction': pred_class_idx
                    })
        
        # Clear memory after each batch
        gc.collect()
        
    return shap_results

def process_batch_lime(lime_explainer, data_batch, preds_batch, batch_size=5):
    """Process LIME explanations in batches (LIME is more memory intensive)."""
    lime_results = []
    
    for i in range(0, len(data_batch), batch_size):
        batch_end = min(i + batch_size, len(data_batch))
        batch_data = data_batch[i:batch_end]
        batch_preds = preds_batch[i:batch_end]
        
        # Process current batch
        with st.spinner(f"Processing LIME batch {i//batch_size + 1}/{(len(data_batch)-1)//batch_size + 1}..."):
            for j in range(len(batch_data)):
                row_idx = i + j
                try:
                    exp = lime_explainer.explain_instance(
                        batch_data[j],
                        predict_2d_to_3d,
                        num_features=8  # Reduced for batch processing
                    )
                    
                    lime_results.append({
                        'row_index': row_idx,
                        'explanation': exp,
                        'feature_contributions': exp.as_list(),
                        'prediction': batch_preds[j]
                    })
                except Exception as e:
                    st.warning(f"LIME failed for row {row_idx}: {e}")
                    lime_results.append({
                        'row_index': row_idx,
                        'explanation': None,
                        'feature_contributions': [],
                        'prediction': batch_preds[j]
                    })
        
        # Clear memory after each batch (LIME is memory intensive)
        gc.collect()
        
    return lime_results

def create_explanations_zip(shap_results, lime_results, feature_names, original_indices):
    """Create ZIP file from batched results."""
    zip_buf = BytesIO()
    
    with zipfile.ZipFile(zip_buf, "w", zipfile.ZIP_DEFLATED) as zf:
        # Add global summary if available
        if len(shap_results) > 0:
            try:
                # Create a simple global importance from batch
                all_shap_vals = [r['shap_values'] for r in shap_results if r['shap_values'] is not None]
                if len(all_shap_vals) > 0:
                    mean_abs_shap = np.mean(np.abs(all_shap_vals), axis=0)
                    fig_glob = plot_shap_global_batch(mean_abs_shap, feature_names)
                    zf.writestr("shap_global_summary.png", fig_to_bytes(fig_glob).getvalue())
                    plt.close(fig_glob)
            except Exception as e:
                st.warning(f"Global SHAP summary failed: {e}")
        
        # Add individual explanation files
        successful_exports = 0
        
        for i, (shap_res, lime_res) in enumerate(zip(shap_results, lime_results)):
            original_idx = original_indices[i]
            
            try:
                # SHAP plot
                if shap_res['shap_values'] is not None:
                    fig_shap = plot_shap_local(
                        shap_res['shap_values'], 
                        feature_names, 
                        f"SHAP Row {original_idx}"
                    )
                    zf.writestr(f"shap_row_{original_idx}.png", fig_to_bytes(fig_shap).getvalue())
                    plt.close(fig_shap)
                
                # LIME plot
                if lime_res['explanation'] is not None:
                    fig_lime = plot_lime_local(
                        lime_res['explanation'], 
                        f"LIME Row {original_idx}"
                    )
                    zf.writestr(f"lime_row_{original_idx}.png", fig_to_bytes(fig_lime).getvalue())
                    plt.close(fig_lime)
                
                successful_exports += 1
                
            except Exception as e:
                st.warning(f"Failed to create plots for row {original_idx}: {e}")
                continue
        
        # Add CSV with all LIME contributions
        lime_data = []
        for res in lime_results:
            if res['feature_contributions']:
                for feature, contribution in res['feature_contributions']:
                    lime_data.append({
                        'Row_Index': original_indices[res['row_index']],
                        'Feature': feature,
                        'Contribution': contribution
                    })
        
        if lime_data:
            lime_df = pd.DataFrame(lime_data)
            zf.writestr("lime_contributions.csv", lime_df.to_csv(index=False).encode('utf-8'))
    
    return zip_buf, successful_exports

def plot_shap_global_batch(mean_abs_shap, feature_names):
    """Create global SHAP plot from batched results."""
    fig, ax = plt.subplots(figsize=(10, 6))
    feat = np.array(feature_names)
    order = np.argsort(mean_abs_shap)[::-1]
    
    ax.barh(feat[order][:20][::-1], mean_abs_shap[order][:20][::-1])
    ax.set_title("Global Feature Importance (Mean |SHAP| from batch)")
    ax.set_xlabel("Mean |SHAP value|")
    plt.tight_layout()
    return fig

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
shap_explainer = get_shap_explainer_sampled(X_scaled, preds, max_samples=1000)
lime_explainer = get_lime_explainer_sampled(X_scaled, selected_features, label_encoder.classes_, max_samples=5000)

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
    st.warning("""
    **Note for large datasets:** Explainability features use sampling to avoid crashes.
    - SHAP: Limited to 1,000 background samples
    - LIME: Limited to 5,000 training samples  
    - Batch export: Limited to 1,000 rows maximum
    """)
    
    # Individual row explanations
    st.subheader("Individual Row Explanations")
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
    # Batch Export with True Batching
    # ----------------------------
    st.header("📦 Batch Export Explanations (Batched Processing)")
    
    # Configuration options
    col1, col2, col3 = st.columns(3)
    with col1:
        batch_size_shap = st.number_input("SHAP batch size", min_value=1, max_value=50, value=10, 
                                         help="Smaller = less memory, larger = faster")
    with col2:
        batch_size_lime = st.number_input("LIME batch size", min_value=1, max_value=20, value=5,
                                         help="LIME is more memory intensive")
    with col3:
        max_rows_export = st.number_input("Max rows to process", 
                                        min_value=1, 
                                        max_value=min(1000, X_scaled.shape[0]), 
                                        value=min(100, X_scaled.shape[0]))

    # Sample selection for export
    if X_scaled.shape[0] > max_rows_export:
        st.info(f"Sampling {max_rows_export} rows from {X_scaled.shape[0]} total rows for export.")
        
        # Stratified sampling to maintain class distribution
        unique_classes, class_counts = np.unique(preds, return_counts=True)
        sample_indices = []
        
        for class_label in unique_classes:
            class_indices = np.where(preds == class_label)[0]
            n_samples_class = max(1, int(max_rows_export * len(class_indices) / len(preds)))
            n_samples_class = min(n_samples_class, len(class_indices))
            
            if n_samples_class > 0:
                selected = np.random.choice(class_indices, size=n_samples_class, replace=False)
                sample_indices.extend(selected)
        
        # Fill remaining slots randomly if needed
        if len(sample_indices) < max_rows_export:
            remaining = max_rows_export - len(sample_indices)
            all_indices = np.arange(len(preds))
            unused_indices = np.setdiff1d(all_indices, sample_indices)
            if len(unused_indices) > 0:
                additional = np.random.choice(unused_indices, 
                                            size=min(remaining, len(unused_indices)), 
                                            replace=False)
                sample_indices.extend(additional)
        
        sample_indices = np.array(sample_indices)
    else:
        sample_indices = np.arange(X_scaled.shape[0])

    # Final sample
    X_sample = X_scaled[sample_indices]
    preds_sample = preds[sample_indices]
    original_indices = df_sel.index.values[sample_indices]

    if st.button("🚀 Start Batched Export", type="primary"):
        
        # Memory check
        if not check_memory_safe():
            st.error("Memory usage too high. Please clear cache or reduce batch size.")
            st.stop()
        
        # Progress tracking
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        try:
            # Phase 1: SHAP explanations
            status_text.text("Phase 1/2: Generating SHAP explanations...")
            shap_results = process_batch_shap(shap_explainer, X_sample, preds_sample, batch_size_shap)
            progress_bar.progress(50)
            
            # Phase 2: LIME explanations  
            status_text.text("Phase 2/2: Generating LIME explanations...")
            lime_results = process_batch_lime(lime_explainer, X_sample, preds_sample, batch_size_lime)
            progress_bar.progress(90)
            
            # Phase 3: Create ZIP
            status_text.text("Finalizing ZIP file...")
            zip_buf, successful_exports = create_explanations_zip(
                shap_results, lime_results, selected_features, original_indices
            )
            progress_bar.progress(100)
            
            # Download ready
            ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
            zip_filename = f"explanations_batch_{ts}_{successful_exports}_rows.zip"
            
            st.success(f"✅ Batch export completed! Successfully processed {successful_exports} rows.")
            
            st.download_button(
                "⬇️ Download Explanations (ZIP)",
                data=zip_buf.getvalue(),
                file_name=zip_filename,
                mime="application/zip",
                key=f"download_{ts}"
            )
            
        except Exception as e:
            st.error(f"Batch processing failed: {e}")
            st.info("Try reducing batch sizes or number of rows.")
        
        finally:
            status_text.empty()
            progress_bar.empty()

st.markdown("---")
st.caption("Notes: Batch processing allows handling large datasets by processing in manageable chunks with memory monitoring.")
