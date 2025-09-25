import streamlit as st
import pandas as pd
import numpy as np
import shap
import joblib
import matplotlib.pyplot as plt
from lime.lime_tabular import LimeTabularExplainer

# ================================================================
# 🔑 Universal Hashable Converter
# ================================================================
def to_hashable(x):
    import numpy as np
    import pandas as pd

    if isinstance(x, (np.ndarray, pd.Series)):
        return x.tolist()
    elif isinstance(x, pd.DataFrame):
        return x.to_dict(orient="list")
    elif isinstance(x, (list, tuple, set)):
        return tuple(to_hashable(i) for i in x)
    elif isinstance(x, dict):
        return {k: to_hashable(v) for k, v in x.items()}
    else:
        return x


# ================================================================
# 🔧 Cached Functions
# ================================================================
@st.cache_resource(show_spinner="Loading trained model...")
def load_model(path: str):
    try:
        model = joblib.load(path)
        st.success("✅ Model loaded successfully")
        return model
    except FileNotFoundError:
        st.error(f"❌ Model file not found at: {path}")
        return None
    except Exception as e:
        st.error(f"⚠️ Error loading model: {e}")
        return None


@st.cache_resource(show_spinner="Preparing SHAP explainer...")
def get_shap_explainer(data, labels, predict_func):
    try:
        data = np.array(data, dtype=np.float32)
        labels = to_hashable(labels)

        unique_labels = set(labels)
        background_samples = []

        for label in unique_labels:
            try:
                idx = labels.index(label) if label in labels else None
                if idx is not None:
                    background_samples.append(data[idx])
            except Exception:
                pass

        background = np.array(background_samples, dtype=np.float32)

        if background.shape[0] < 2:  # fallback
            rng = np.random.default_rng(seed=42)
            bg_idx = rng.choice(data.shape[0], size=min(50, data.shape[0]), replace=False)
            background = data[bg_idx]

        explainer = shap.KernelExplainer(predict_func, background)
        st.success("✅ SHAP explainer initialized")
        return explainer
    except Exception as e:
        st.error(f"⚠️ Error initializing SHAP explainer: {e}")
        return None


@st.cache_resource(show_spinner="Preparing LIME explainer...")
def get_lime_explainer(data, feature_names, class_names):
    try:
        data = np.array(data, dtype=np.float32)
        feature_names = to_hashable(feature_names)
        class_names = [str(c) for c in to_hashable(class_names)]

        explainer = LimeTabularExplainer(
            data,
            feature_names=feature_names,
            class_names=class_names,
            mode="classification"
        )
        st.success("✅ LIME explainer initialized")
        return explainer
    except Exception as e:
        st.error(f"⚠️ Error initializing LIME explainer: {e}")
        return None


# ================================================================
# 🚀 Streamlit App
# ================================================================
st.title("🚨 Network Intrusion Detection System with Explainability")

# ---- Upload model
st.sidebar.header("Upload Model")
model_file = st.sidebar.file_uploader("Upload your trained model (.pkl)", type=["pkl"])

if model_file:
    with open("uploaded_model.pkl", "wb") as f:
        f.write(model_file.read())
    model = load_model("uploaded_model.pkl")
else:
    model = None
    st.warning("⚠️ Please upload a trained model to continue.")

# ---- Upload dataset
st.sidebar.header("Upload Dataset")
data_file = st.sidebar.file_uploader("Upload CSV dataset", type=["csv"])

if data_file:
    df = pd.read_csv(data_file)
    st.subheader("📊 Data Preview")
    st.write(df.head())
else:
    df = None
    st.info("ℹ️ Upload a CSV dataset to proceed.")


# ---- Run Prediction
if model and df is not None:
    st.subheader("🔎 Run Predictions")

    X = df.select_dtypes(include=[np.number])  # only numeric
    preds = model.predict(X)
    df["Prediction"] = preds
    st.write("✅ Predictions generated")
    st.dataframe(df.head())


# ---- Explainability
if model and df is not None:
    st.subheader("🧠 Explainability")

    X = df.select_dtypes(include=[np.number])
    preds = df["Prediction"].tolist()
    class_names = np.unique(preds)

    # Pick class
    st.markdown("### 🎯 Select Class & Row for Explanation")
    chosen_class = st.selectbox("Choose a predicted class:", class_names)

    filtered_idx = df[df["Prediction"] == chosen_class].index.tolist()

    if filtered_idx:
        row_idx = st.selectbox("Select row index from chosen class:", filtered_idx)
        sample = X.loc[[row_idx]]

        # --- Show raw feature values
        st.markdown(f"### 📝 Feature Values for Row {row_idx} (Class: {chosen_class})")
        st.dataframe(sample.T.rename(columns={row_idx: "Value"}))

        # SHAP
        if st.checkbox("Enable SHAP Explanation"):
            shap_explainer = get_shap_explainer(X, preds, model.predict_proba)
            if shap_explainer:
                shap_values = shap_explainer.shap_values(sample)

                st.markdown(f"### 🔍 SHAP Force Plot (Row {row_idx}, Class {chosen_class})")
                shap.initjs()
                st_shap = shap.force_plot(
                    shap_explainer.expected_value[0], shap_values[0], sample, matplotlib=False
                )
                st.components.v1.html(st_shap.html(), height=300)

                st.markdown("### 📊 SHAP Summary Plot (Global Importance)")
                fig, ax = plt.subplots()
                shap.summary_plot(shap_values, X, feature_names=X.columns, plot_type="bar", show=False)
                st.pyplot(fig)
                plt.close(fig)

        # LIME
        if st.checkbox("Enable LIME Explanation"):
            lime_explainer = get_lime_explainer(X, X.columns, class_names)
            if lime_explainer:
                exp = lime_explainer.explain_instance(
                    sample.values[0], model.predict_proba, num_features=10
                )

                st.markdown(f"### 🔍 LIME Explanation (Row {row_idx}, Class {chosen_class})")
                fig = exp.as_pyplot_figure()
                st.pyplot(fig)
                plt.close(fig)
    else:
        st.warning(f"No rows found for class `{chosen_class}`.")
