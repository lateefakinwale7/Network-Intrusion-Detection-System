# app.py (improved)
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
    try:
        st.cache_resource.clear()
    except Exception:
        pass
    try:
        st.cache_data.clear()
    except Exception:
        pass

with st.sidebar:
    st.button("Clear Cache", on_click=clear_cache)
    st.markdown("---")
    st.write("If you encounter errors after uploading a new file, try clearing the cache.")

# ----------------------------
# Load resources
# ----------------------------
@st.cache_resource(show_spinner="Loading essential model files...")
def load_resources():
    # adjust paths if needed
    model = tf.keras.models.load_model("cnn_bilstm_model.h5")
    with open("scaler.pkl", "rb") as f:
        scaler = pickle.load(f)
    with open("label_encoder.pkl", "rb") as f:
        label_encoder = pickle.load(f)
    with open("selected_features.txt", "r") as f:
        selected_features = [line.strip() for line in f.readlines() if line.strip()]
    return model, scaler, label_encoder, selected_features

model, scaler, label_encoder, selected_features = load_resources()

# ----------------------------
# Chunked CSV Loader
# ----------------------------
def load_csv_in_chunks(file, features, chunk_size=50000):
    """Read CSV in chunks but only with required columns; returns concatenated DataFrame."""
    dfs = []
    for chunk in pd.read_csv(file, usecols=features, chunksize=chunk_size):
        # replace infinities and drop rows that cannot be used
        chunk.replace([np.inf, -np.inf], np.nan, inplace=True)
        dfs.append(chunk)
    if not dfs:
        return pd.DataFrame(columns=features)
    df = pd.concat(dfs, ignore_index=True)
    return df

# ----------------------------
# Preprocess & Cache
# ----------------------------
@st.cache_data
def preprocess_and_align(df, scaler, features):
    """
    Cleans the selected features, drops rows with NaNs/infs for these features,
    and returns X_scaled plus aligned df and df_sel (both reset indexed).
    """
    # Defensive copy
    df = df.copy()
    # Keep only required columns in df_sel
    df_sel = df[features].copy()
    df_sel.replace([np.inf, -np.inf], np.nan, inplace=True)
    valid_mask = ~df_sel.isna().any(axis=1)
    if valid_mask.sum() == 0:
        return None, None, None
    # filter original df to keep only rows that have valid features so indices align
    df_clean = df.loc[valid_mask].reset_index(drop=True)
    df_sel_clean = df_sel.loc[valid_mask].reset_index(drop=True)
    # Scale
    X_scaled = scaler.transform(df_sel_clean.values.astype(np.float32))
    return X_scaled, df_clean, df_sel_clean

# ----------------------------
# Helpers
# ----------------------------
def predict_2d_to_3d(x_2d):
    """Function wrapper used by explainers (takes 2D numpy array)."""
    x_arr = np.array(x_2d, dtype=np.float32)
    if x_arr.ndim == 1:
        x_arr = x_arr.reshape(1, -1)
    n_samples, n_features = x_arr.shape
    x_3d = x_arr.reshape(n_samples, n_features, 1)
    preds = model.predict(x_3d, verbose=0)
    return preds

@st.cache_resource
def get_shap_explainer(background_data):
    """
    Create a SHAP explainer.
    For deep TF models, shap.DeepExplainer or GradientExplainer may be much faster than KernelExplainer.
    KernelExplainer is used as a stable fallback; keep background size small (<=100).
    """
    # keep background small to reduce overhead
    n_bg = min(100, max(1, background_data.shape[0]))
    try:
        bg = shap.sample(background_data, n_bg, random_state=42)
    except Exception:
        bg = background_data[:n_bg]
    # For neural networks, consider using DeepExplainer if your TF version and SHAP version support it:
    # try:
    #     return shap.DeepExplainer(model, bg.reshape(bg.shape[0], bg.shape[1], 1))
    # except Exception:
    #     return shap.KernelExplainer(predict_2d_to_3d, bg)
    return shap.KernelExplainer(predict_2d_to_3d, bg)

@st.cache_resource
def get_lime_explainer(data, feature_names, class_names):
    return LimeTabularExplainer(
        training_data=np.array(data),
        feature_names=feature_names,
        class_names=list(class_names),
        mode="classification"
    )

def fig_to_bytes(fig):
    buf = BytesIO()
    fig.savefig(buf, format="png", bbox_inches="tight")
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
    # shap_values for multiclass is list; KernelExplainer.shap_values returns list
    shap_vals_global = shap_explainer.shap_values(data[global_idx], nsamples=50)
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
        df_raw = load_csv_in_chunks(uploaded_file, selected_features)
    except Exception as e:
        st.error(f"Failed to read CSV: {e}")
        st.stop()

    if df_raw.empty:
        st.error("Uploaded CSV did not contain any of the required columns or is empty.")
        st.stop()

    X_scaled, df, df_sel = preprocess_and_align(df_raw, scaler, selected_features)
    if X_scaled is None or df_sel is None or df is None or df_sel.shape[0] == 0:
        st.error("No valid rows found after cleaning the required features.")
        st.stop()

    # Predict
    try:
        X_reshaped = X_scaled.reshape(X_scaled.shape[0], X_scaled.shape[1], 1)
        preds_probs = model.predict(X_reshaped, verbose=0)
        preds = np.argmax(preds_probs, axis=1)
        preds_labels = label_encoder.inverse_transform(preds)
        results_df = df.copy().reset_index(drop=True)
        results_df["Predicted_Label"] = preds_labels
        results_df["Predicted_Probability"] = np.max(preds_probs, axis=1)
    except Exception as e:
        st.error(f"Prediction failed: {e}")
        st.stop()

# Build explainers (use aligned X_scaled)
try:
    shap_explainer = get_shap_explainer(X_scaled)
except Exception as e:
    st.warning(f"Could not create SHAP explainer: {e}")
    shap_explainer = None

try:
    lime_explainer = get_lime_explainer(X_scaled, selected_features, label_encoder.classes_)
except Exception as e:
    st.warning(f"Could not create LIME explainer: {e}")
    lime_explainer = None

# ----------------------------
# Tabs
# ----------------------------
tab1, tab2 = st.tabs(["📊 Predictions", "🔎 Explainability & Export"])

with tab1:
    st.subheader("Data Preview")
    st.dataframe(results_df.head())

    st.subheader("Prediction Distribution")
    counts = results_df["Predicted_Label"].value_counts()
    st.bar_chart(counts)

with tab2:
    st.header("Model Explainability (SHAP + LIME)")

    unique_labels = results_df["Predicted_Label"].unique().tolist()
    selected_class = st.selectbox("Select predicted class:", unique_labels)

    class_rows = results_df[results_df["Predicted_Label"] == selected_class]
    if class_rows.empty:
        st.warning("No rows found for the selected class.")
    else:
        # Let user pick a displayed index (these are aligned reset indices)
        selected_row_index = st.selectbox("Select row index:", class_rows.index.tolist())
        st.subheader("Selected Row Details")
        st.dataframe(results_df.loc[[selected_row_index]])

        col_shap, col_lime = st.columns(2)

        # numpy_row_idx now equals selected_row_index because we ensured alignment
        numpy_row_idx = int(selected_row_index)

        if shap_explainer is not None:
            with col_shap:
                st.subheader("SHAP (local)")
                try:
                    # shap_values will be list (for multiclass) or array (for single class)
                    shap_vals = shap_explainer.shap_values(X_scaled[numpy_row_idx:numpy_row_idx+1], nsamples=50)
                    pred_class_idx = int(preds[numpy_row_idx])
                    shap_for_row = np.array(shap_vals[pred_class_idx])[0]
                    fig_local = plot_shap_local(shap_for_row, selected_features, f"SHAP for Row {selected_row_index}")
                    st.pyplot(fig_local)
                    plt.close(fig_local)
                except Exception as e:
                    st.error(f"SHAP local explanation failed: {e}")

        if lime_explainer is not None:
            with col_lime:
                st.subheader("LIME (local)")
                try:
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
                except Exception as e:
                    st.error(f"LIME local explanation failed: {e}")

        # ----------------------------
        # Batch Export
        # ----------------------------
        st.header("📦 Batch Export Explanations")
        max_allowed = min(50, X_scaled.shape[0])
        row_limit = st.number_input("Rows to explain (max 50):", min_value=1, max_value=max_allowed, value=min(10, max_allowed), step=1)

        if st.button("Generate Batch Explanations (ZIP)"):
            if shap_explainer is None or lime_explainer is None:
                st.error("Both SHAP and LIME explainers must be available to generate batch explanations.")
            else:
                with st.spinner("Generating batch explanations..."):
                    zip_buf = BytesIO()
                    with zipfile.ZipFile(zip_buf, "w", zipfile.ZIP_DEFLATED) as zf:
                        # Global SHAP summary (try/except)
                        try:
                            fig_glob = plot_shap_global(shap_explainer, X_scaled, selected_features, n_samples=200)
                            zf.writestr("shap_global_summary.png", fig_to_bytes(fig_glob).getvalue())
                            plt.close(fig_glob)
                        except Exception as e:
                            zf.writestr("shap_global_summary_error.txt", str(e).encode("utf-8"))

                        all_lime_rows = []
                        progress_bar = st.progress(0)

                        for i in range(row_limit):
                            row_idx = i  # because df and X_scaled are reset and aligned

                            # SHAP per-row
                            try:
                                shap_vals_row = shap_explainer.shap_values(X_scaled[row_idx:row_idx+1], nsamples=50)
                                pred_class_idx = int(preds[row_idx])
                                shap_row_vals = np.array(shap_vals_row[pred_class_idx])[0]
                                fig_s = plot_shap_local(shap_row_vals, selected_features, f"SHAP Row {row_idx}")
                                zf.writestr(f"shap_row_{row_idx}.png", fig_to_bytes(fig_s).getvalue())
                                plt.close(fig_s)
                            except Exception as e:
                                zf.writestr(f"shap_row_{row_idx}_error.txt", str(e).encode("utf-8"))

                            # LIME per-row
                            try:
                                exp = lime_explainer.explain_instance(X_scaled[row_idx], predict_2d_to_3d, num_features=10)
                                fig_l = exp.as_pyplot_figure()
                                fig_l.suptitle(f"LIME Row {row_idx}", y=1.02)
                                plt.tight_layout()
                                zf.writestr(f"lime_row_{row_idx}.png", fig_to_bytes(fig_l).getvalue())
                                plt.close(fig_l)

                                for f, c in exp.as_list():
                                    all_lime_rows.append({"Row_Index": row_idx, "Feature": f, "Contribution": c})
                            except Exception as e:
                                zf.writestr(f"lime_row_{row_idx}_error.txt", str(e).encode("utf-8"))

                            progress_bar.progress(int(((i + 1) / row_limit) * 100))

                        if all_lime_rows:
                            lime_df_all = pd.DataFrame(all_lime_rows)
                            zf.writestr("lime_all_contributions.csv", lime_df_all.to_csv(index=False).encode("utf-8"))

                    ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
                    zipfname = f"explanations_batch_{ts}.zip"
                    st.download_button("⬇️ Download ZIP", data=zip_buf.getvalue(), file_name=zipfname, mime="application/zip")
                    progress_bar.empty()
                    st.success("Batch export completed!")

st.caption("Note: SHAP KernelExplainer can be slow for many rows — keep rows small or switch to Deep/Gradient explainer for TF models.")
