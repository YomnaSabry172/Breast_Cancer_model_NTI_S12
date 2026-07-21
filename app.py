import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.datasets import load_breast_cancer
from my_module.model import predict_diagnosis

st.set_page_config(page_title="Breast Cancer Diagnosis", page_icon="🩺", layout="wide")
st.markdown(
    """
    <style>
    .block-container {
        max-width: 900px;
        margin: 0 auto;
        padding-top: 2rem;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

plt.rcParams.update({
    "figure.facecolor": "#1C242C",
    "axes.facecolor": "#1C242C",
    "axes.edgecolor": "#93A4B0",
    "axes.labelcolor": "#E8EEF2",
    "text.color": "#E8EEF2",
    "xtick.color": "#93A4B0",
    "ytick.color": "#93A4B0",
    "grid.color": "#2A343D",
    "font.size": 9,
})

FIGSIZE = (3.6, 2.4)
DPI = 130


@st.cache_data
def load_data():
    data = load_breast_cancer()
    df = pd.DataFrame(data.data, columns=data.feature_names)
    df["diagnosis"] = data.target
    df["diagnosis_label"] = df["diagnosis"].map({0: "Malignant", 1: "Benign"})
    return df, list(data.feature_names)


df, feature_names = load_data()


st.title("🦀 Breast Cancer Diagnosis Predictor")
st.markdown(
    """
    This app uses a **Logistic Regression** model trained on real diagnostic
    measurements from breast tumor biopsies to predict whether a tumor is
    **malignant** (cancerous) or **benign** (non-cancerous).

    Doctors and lab technicians take measurements of a tumor's cell nuclei —
    things like size, shape, and texture — and this model learns the patterns
    that tend to separate malignant tumors from benign ones.

    Scroll down to explore the dataset, then try the predictor yourself.
    """
)
st.divider()


st.header("🔬 About this dataset")
st.markdown(
    """
    The dataset contains **30 measurements** taken from digitized images of
    tumor cell nuclei, for **569 patients**, each labeled malignant or benign
    based on an actual medical diagnosis.

    The 30 measurements fall into 3 groups — the **mean**, **standard error**,
    and **worst (largest)** value of 10 underlying properties, such as radius,
    texture, and smoothness of the cell nuclei.
    """
)

# Toggle button — remembers its state across reruns via session_state
if "show_data" not in st.session_state:
    st.session_state.show_data = False

if st.button("Show raw data" if not st.session_state.show_data else "Hide raw data"):
    st.session_state.show_data = not st.session_state.show_data

if st.session_state.show_data:
    st.dataframe(df.drop(columns=["diagnosis"]), use_container_width=True, height=250)

st.divider()


st.header("📊 Exploring the data")

st.subheader("1. How common is each diagnosis?")
fig, ax = plt.subplots(figsize=FIGSIZE, dpi=DPI)
sns.countplot(data=df, x="diagnosis_label", hue="diagnosis_label",
              palette="Set2", legend=False, ax=ax)
ax.set_xlabel("")
left_col, _ = st.columns([1, 2])
with left_col:
    st.pyplot(fig, use_container_width=False)
st.caption(
    "The dataset is moderately imbalanced, with more benign cases than "
    "malignant — worth keeping in mind when judging accuracy alone."
)

st.subheader("2. Explore a feature by diagnosis")
selected_feature = st.selectbox("Choose a feature to explore:", feature_names, index=0)

fig, ax = plt.subplots(figsize=FIGSIZE, dpi=DPI)
sns.histplot(data=df, x=selected_feature, hue="diagnosis_label", kde=True,
             palette="Set2", ax=ax)
left_col, _ = st.columns([1, 2])
with left_col:
    st.pyplot(fig, use_container_width=False)
st.caption(f"Explore how {selected_feature} differs between malignant and benign cases.")

st.subheader("3. A closer look at one strong feature")
fig, ax = plt.subplots(figsize=FIGSIZE, dpi=DPI)
sns.boxplot(data=df, x="diagnosis_label", y="mean radius", hue="diagnosis_label",
            palette="Set2", legend=False, ax=ax)
ax.set_xlabel("")
left_col, _ = st.columns([1, 2])
with left_col:
    st.pyplot(fig, use_container_width=False)
st.caption(
    "Malignant tumors show a clearly higher median mean radius than benign "
    "ones, suggesting this feature alone carries real predictive signal."
)

st.subheader("Step 4 — how do features relate to each other?")
corr_features = ["mean radius", "mean texture", "mean concavity",
                  "mean smoothness", "mean symmetry", "diagnosis"]
fig, ax = plt.subplots(figsize=(4.2, 3.2), dpi=DPI)
sns.heatmap(df[corr_features].corr(), annot=True, cmap="coolwarm",
            annot_kws={"size": 7}, ax=ax)
left_col, _ = st.columns([1, 2])
with left_col:
    st.pyplot(fig, use_container_width=False)
st.caption(
    "Mean radius and mean concavity both correlate strongly with diagnosis, "
    "while mean symmetry contributes comparatively less."
)

st.divider()

st.header("🧪Try it yourself")
st.write(
    "Adjust the measurements below — they default to the dataset's average "
    "values — then click Predict to see the model's diagnosis."
)

groups = {
    "Mean values": feature_names[0:10],
    "Standard error values": feature_names[10:20],
    "Worst values": feature_names[20:30],
}

user_input = {}
for group_name, group_features in groups.items():
    with st.expander(group_name, expanded=(group_name == "Mean values")):
        cols = st.columns(2)
        for i, feature in enumerate(group_features):
            default_val = float(df[feature].mean())
            user_input[feature] = cols[i % 2].number_input(
                feature, value=default_val, format="%.4f"
            )

if st.button("Predict", type="primary"):
    features = [user_input[f] for f in feature_names]
    result = predict_diagnosis(features)
    if result == "Malignant":
        st.error(f"Prediction: {result}")
    else:
        st.success(f"Prediction: {result}")