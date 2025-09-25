import streamlit as st
import pandas as pd
import numpy as np
import shap
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
import joblib

# ----------------------------
# Load Model + Scaler
# ----------------------------
@st.cache_resource
def load_model(path):
    return joblib.load(path)

@st.cache_resource
def load_scaler(path):
    return joblib.load(path)

# ----------------------------
# Preprocess (NO caching here)
# ----------------------------
def preprocess_data(df, scaler, features):
    df_sel = df[features].copy()
    df_sel.replace([np.inf, -np.inf], np.nan, inplace=True)
    df_sel.dropna(inplace=True)
    X_scaled = scaler.transform(df_sel)
    return X_scaled, df_sel

# ----------------------------
# Streamlit UI
# ----------------------------
st.title("🚨 Network Intrusion Detection System with Explainability")

uploaded_file = st.file_uploader("Upload your CSV file", type=["csv"])
if uploaded_file:
    df = pd.read_csv(uploaded_file)
    st.write("📊 Data Preview", df.head())

    # Load artifacts
    model = load_model("model.pkl")
    scaler = load_scaler("scaler.pkl")

    # Features
    selected_features = [
        # replace with your actual feature list
        "duration", "src_bytes", "dst_bytes"
    ]

    if all(f in df.columns for f in selected_features):
        X_scaled, df_sel = preprocess_data(df, scaler, selected_features)

        preds = model.predict(X_scaled)
        df_sel["Prediction"] = preds

        st.write("✅ Prediction Results", df_sel.head())

        # ----------------------------
        # Explainability with SHAP
        # ----------------------------
        st.subheader("🔎 Model Explainability (SHAP)")

        # Initialize SHAP explainer
        try:
            explainer = shap.TreeExplainer(model)
            shap_values = explainer.shap_values(X_scaled)
        except Exception:
            explainer = shap.KernelExplainer(model.predict, X_scaled[:50])
            shap_values = explainer.shap_values(X_scaled[:10])

        # Global feature importance
        st.markdown("**Global Feature Importance**")
        fig, ax = plt.subplots()
        shap.summary_plot(shap_values, X_scaled, feature_names=selected_features, show=False)
        st.pyplot(fig)

        # Per-sample explanation (colored bar chart)
        st.markdown("**Per-sample Explanation**")
        sample_idx = st.slider("Pick a sample to explain", 0, len(df_sel) - 1, 0)

        shap_sample = shap_values[sample_idx]
        colors = ["red" if val > 0 else "blue" for val in shap_sample]

        fig, ax = plt.subplots()
        y_pos = np.arange(len(selected_features))
        ax.barh(y_pos, shap_sample, align="center", color=colors)
        ax.set_yticks(y_pos)
        ax.set_yticklabels(selected_features)
        ax.invert_yaxis()
        ax.set_xlabel("SHAP value (impact on prediction)")
        ax.set_title(f"Sample {sample_idx} Feature Contributions")
        st.pyplot(fig)

        # Download results
        st.download_button(
            "⬇️ Download Predictions",
            df_sel.to_csv(index=False).encode("utf-8"),
            "predictions.csv",
            "text/csv",
        )
    else:
        st.error("❌ Missing required features in uploaded dataset.")
