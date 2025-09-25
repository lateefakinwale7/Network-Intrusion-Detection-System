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

st.set_page_config(page_title="Intrusion Detection System", layout="wide")

# ----------------------------
# Cache Clearing Function
# ----------------------------
def clear_cache():
    st.cache_resource.clear()
    st.cache_data.clear()

with st.sidebar:
    st.button("Clear Cache", on_click=clear_cache)
    st.markdown("---")
    st.write("If you encounter errors after uploading a new file, try clearing the cache.")

# ----------------------------
# Load resources
# ----------------------------
@st.cache_resource(show_spinner="Loading essential model files...")
def load_resources():
    model = tf.keras.models.load_model('cnn_bilstm_model.h5')
    with open('scaler.pkl', 'rb') as f:
        scaler = pickle.load(f)
    with open('label_encoder.pkl', 'rb') as f:
        label_encoder = pickle.load(f)
    with open('selected_features.txt', 'r') as f:
        selected_features = [line.strip() for line in f.readlines()]
    return model, scaler, label_encoder, selected_features

model, scaler, label_encoder, selected_features = load_resources()

# ----------------------------
# Chunked CSV Loader
# ----------------------------
def load_csv_in_chunks(file, features, chunk_size=50000):
    dfs = []
    for chunk in pd.read_csv(file, usecols=features, chunksize=chunk_size):
        chunk.replace([np.inf, -np.inf], np.nan, inplace=True)
        chunk.dropna(inplace=True)
        dfs.append(chunk)
    return pd.concat(dfs, ignore_index=True)

# ----------------------------
# Preprocess & Cache
# ----------------------------
@st.cache_data
def preprocess_data(df, scaler, features):
    df_sel = df[features].copy()
    df_sel.replace([np.inf, -np.inf], np.nan, inplace=True)
    df_sel.dropna(inplace=True)
    X_scaled = scaler.transform(df_sel)
    return X_scaled, df_sel

# ----------------------------
# Helpers
# ----------------------------
def predict_2d_to_3d(x_2d):
    x_arr = np.array(x_2d, dtype=np.float32)
    n_samples, n_features = x_arr.shape
    x_3d = x_arr.reshape(n_samples, n_features, 1)
    return model.predict(x_3d, verbose=0)

@st.cache_resource
def get_shap_explainer(data):
    # Use at most 100 rows for background
    background = shap.sample(data, 100, random_state=42)
    return shap.KernelExplainer(predict_2d_to_3d, background)

@st.cache_resource
def get_lime_explainer(data, feature_names, class_names):
    return LimeTabularExplainer(
        data,
        feature_names=feature_names,
        class_names=class_names,
        mode='classification'
    )

def fig_to_bytes(fig):
    buf = BytesIO()
    fig.savefig(buf, format='png', bbox_inches='tight')
    buf.seek(0)
    return buf

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

def plot_lime_local(lime_exp, title="LIME Local Explanation"):
    fig = lime_exp.as_pyplot_figure()
    fig.suptitle(title, y=1.02)
    plt.tight_layout()
    return fig

def plot_shap_global(shap_explainer, data, feature_names, n_samples=200):
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
    ax.set_title("Global Feature Importance (Mean |SHAP|)")
    ax.set_xlabel("Mean |SHAP value|")
    plt.tight_layout()
    return fig

# ----------------------------
# Main App UI
# ----------------------------
st.title("🛡️ Network Intrusion Detection System — with SHAP & LIME")
st.info("Upload a CSV file with network traffic data to get predictions and explanations.")
st.markdown("Required features: " + ", ".join(selected_features))

uploaded_file = st.file_uploader("Upload network traffic CSV", type=["csv"])
if uploaded_file is None:
    st.info("Please upload a CSV file to begin.")
    st.stop()

with st.spinner("Processing file and generating predictions..."):
    try:
        df = load_csv_in_chunks(uploaded_file, selected_features)
    except Exception as e:
        st.error(f"Failed to read CSV: {e}")
        st.stop()

    X_scaled, df_sel = preprocess_data(df, scaler, selected_features)
    if df_sel.empty:
        st.error("No valid data found after cleaning.")
        st.stop()

    try:
        X_reshaped = X_scaled.reshape(X_scaled.shape[0], X_scaled.shape[1], 1)
        preds_probs = model.predict(X_reshaped, verbose=0)
        preds = np.argmax(preds_probs, axis=1)
        preds_labels = label_encoder.inverse_transform(preds)
        results_df = df.copy()
        results_df['Predicted_Label'] = preds_labels
        results_df['Predicted_Probability'] = np.max(preds_probs, axis=1)
    except Exception as e:
        st.error(f"Prediction failed: {e}")
        st.stop()

# Cache explainers
shap_explainer = get_shap_explainer(X_scaled)
lime_explainer = get_lime_explainer(X_scaled, selected_features, label_encoder.classes_)

# ----------------------------
# Tabs
# ----------------------------
tab1, tab2 = st.tabs(["📊 Predictions", "🔎 Explainability & Export"])

with tab1:
    st.subheader("Data Preview")
    st.dataframe(results_df.head())

    st.subheader("Prediction Distribution")
    counts = results_df['Predicted_Label'].value_counts()
    st.bar_chart(counts)

with tab2:
    st.header("Model Explainability (SHAP + LIME)")

    unique_labels = results_df['Predicted_Label'].unique().tolist()
    selected_class = st.selectbox("Select predicted class:", unique_labels)
    class_rows = results_df[results_df['Predicted_Label'] == selected_class]

    if not class_rows.empty:
        selected_row_index = st.selectbox(
            "Select row index:",
            class_rows.index.tolist()
        )
        numpy_row_idx = np.where(df_sel.index.values == selected_row_index)[0][0]
        st.subheader("Selected Row Details")
        st.dataframe(results_df.loc[[selected_row_index]])

        col_shap, col_lime = st.columns(2)

        with col_shap:
            st.subheader("SHAP (local)")
            shap_vals = shap_explainer.shap_values(X_scaled[numpy_row_idx:numpy_row_idx+1], nsamples=100)
            pred_class_idx = int(preds[numpy_row_idx])
            shap_for_row = np.array(shap_vals[pred_class_idx])[0]
            fig_local = plot_shap_local(shap_for_row, selected_features, f"SHAP for Row {selected_row_index}")
            st.pyplot(fig_local)
            plt.close(fig_local)

        with col_lime:
            st.subheader("LIME (local)")
            exp = lime_explainer.explain_instance(
                X_scaled[numpy_row_idx],
                predict_2d_to_3d,
                num_features=min(10, len(selected_features))
            )
            fig_lime = plot_lime_local(exp, f"LIME for Row {selected_row_index}")
            st.pyplot(fig_lime)
            plt.close(fig_lime)

            st.write("Top contributions:")
            st.write(pd.DataFrame(exp.as_list(), columns=["Feature", "Contribution"]))

        # ----------------------------
        # Batch Export
        # ----------------------------
        st.header("📦 Batch Export Explanations")
        row_limit = st.number_input(
            f"Rows to explain (max 50):",
            min_value=1, max_value=min(50, X_scaled.shape[0]), value=10, step=1
        )

        if st.button("Generate Batch Explanations (ZIP)"):
            with st.spinner("Generating batch explanations..."):
                zip_buf = BytesIO()
                with zipfile.ZipFile(zip_buf, "w", zipfile.ZIP_DEFLATED) as zf:
                    # Global SHAP summary
                    fig_glob = plot_shap_global(shap_explainer, X_scaled, selected_features)
                    zf.writestr("shap_global_summary.png", fig_to_bytes(fig_glob).getvalue())
                    plt.close(fig_glob)

                    all_lime_rows = []
                    progress_bar = st.progress(0)

                    for i in range(row_limit):
                        row_idx = df_sel.index[i]

                        # SHAP
                        shap_vals_row = shap_explainer.shap_values(X_scaled[i:i+1], nsamples=100)
                        pred_class_idx = int(preds[i])
                        shap_row_vals = np.array(shap_vals_row[pred_class_idx])[0]
                        fig_s = plot_shap_local(shap_row_vals, selected_features, f"SHAP Row {row_idx}")
                        zf.writestr(f"shap_row_{row_idx}.png", fig_to_bytes(fig_s).getvalue())
                        plt.close(fig_s)

                        # LIME
                        exp = lime_explainer.explain_instance(X_scaled[i], predict_2d_to_3d, num_features=10)
                        fig_l = exp.as_pyplot_figure()
                        fig_l.suptitle(f"LIME Row {row_idx}", y=1.02)
                        plt.tight_layout()
                        zf.writestr(f"lime_row_{row_idx}.png", fig_to_bytes(fig_l).getvalue())
                        plt.close(fig_l)

                        for f, c in exp.as_list():
                            all_lime_rows.append({"Row_Index": row_idx, "Feature": f, "Contribution": c})

                        progress_bar.progress(int(((i+1)/row_limit)*100))

                    lime_df_all = pd.DataFrame(all_lime_rows)
                    zf.writestr("lime_all_contributions.csv", lime_df_all.to_csv(index=False).encode('utf-8'))

                ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
                zipfname = f"explanations_batch_{ts}.zip"
                st.download_button("⬇️ Download ZIP", data=zip_buf.getvalue(),
                                   file_name=zipfname, mime="application/zip")
                progress_bar.empty()
                st.success("Batch export completed!")

st.caption("Note: SHAP KernelExplainer is slow — use fewer rows for explanations.")
