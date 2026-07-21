# Breast Cancer Diagnosis Predictor

![Python](https://img.shields.io/badge/Python-3.10+-blue?logo=python&logoColor=white)
![Streamlit](https://img.shields.io/badge/Streamlit-App-red?logo=streamlit&logoColor=white)
![scikit-learn](https://img.shields.io/badge/scikit--learn-Model-orange?logo=scikitlearn&logoColor=white)
![License](https://img.shields.io/badge/License-MIT-green)

An interactive Streamlit web app that predicts whether a breast tumor is **malignant** or **benign** from 30 diagnostic cell-nuclei measurements, using a Logistic Regression model trained on the classic Breast Cancer Wisconsin (Diagnostic) dataset.

Built as part of the NTI Machine Learning Internship, Session 12 — following on from a first pass on the Iris dataset, this project applies the same deployment workflow to a real, medically-grounded binary classification problem.

 [_**Live app link on Streamlit community**_](https://breastcancermodelnti12-yomnasabry.streamlit.app/)

![App demo](assets/demo.gif)

---

## What this project does

- Loads the Breast Cancer Wisconsin dataset directly from scikit-learn (569 patients, 30 features each)
- Trains a Logistic Regression classifier to distinguish malignant from benign tumors
- Explores the dataset through an interactive, step-by-step walkthrough — target balance, feature distributions, a boxplot comparison, and a correlation heatmap
- Lets a user enter their own 30 measurements and get a live prediction
- Demonstrates a realistic train-once / predict-many deployment pattern, rather than retraining a model on every app launch

---

## Project structure

```
Breast_Cancer_model_NTI_S12/
├── app.py                      # the Streamlit web app itself
├── my_module/
│   ├── train_model.py          # run once — trains the model and saves it to disk
│   ├── model.py                # loads the saved model and exposes predict_diagnosis()
│   ├── logreg_model.pkl        # the trained Logistic Regression model (generated)
│   ├── scaler.pkl              # the fitted StandardScaler (generated)
│   └── __init__.py
├── .streamlit/
│   └── config.toml             # custom dark theme for the app
├── assets/
│   └── demo.gif                # short screen recording of the app in use
├── requirements.txt             # Python dependencies
├── .gitignore
├── LICENSE
└── README.md
```

**About the `assets` folder:** this holds non-code files used purely for documentation — right now just `demo.gif`, a short screen recording showing the app being used (exploring the data, then entering values and getting a prediction). It has no effect on how the app runs; it exists only so this README can display a live preview instead of a wall of text. If you add more images later (like a screenshot or a diagram), they'd go here too.

---

## The pipeline — how the three Python files work together

This project deliberately splits training from prediction into two separate files, rather than retraining the model every time the app runs. Here's the flow:

```
train_model.py  →  produces  →  logreg_model.pkl + scaler.pkl
                                         ↓
                                    model.py  (loads them, defines predict_diagnosis())
                                         ↓
                                     app.py  (imports predict_diagnosis(), builds the UI)
```

`app.py` never talks to `train_model.py` directly — it only ever imports from `model.py`.

### `train_model.py` — trains the model, once

This file is run manually, from the terminal, whenever you want to (re)train the model. It is **not** imported by the app. Its steps:

1. **Load the data** — `load_breast_cancer()` from `sklearn.datasets`, split into `X` (30 features) and `y` (target: 0 = malignant, 1 = benign).
2. **Split into train/test** — an 80/20 split via `train_test_split(..., random_state=42)`, so the model is judged on data it never saw during training. `random_state=42` just makes the split reproducible across runs.
3. **Scale the features** — a `StandardScaler` is fit on the training data only, then used to transform both the training and test sets. Fitting only on training data avoids leaking information from the test set into the scaling step.
4. **Train** — a `LogisticRegression` model (with `max_iter=5000`, since Logistic Regression trains iteratively and the default iteration cap of 100 isn't always enough for this dataset to fully converge — this only gives the optimizer more steps to settle, it isn't a form of overfitting) is fit on the scaled training data.
5. **Check it actually works** — accuracy is printed on the held-out test set before anything is saved, as a sanity check. This model reaches roughly **97–98% test accuracy**.
6. **Save** — the fitted model and the fitted scaler are written to disk as `logreg_model.pkl` and `scaler.pkl`, using `joblib.dump()`, saved into the same folder as this script (via `os.path.dirname(os.path.abspath(__file__))`, so the paths work no matter what directory you run the script from).

Run it with:
```bash
python my_module/train_model.py
```

### `model.py` — loads the trained model, defines the prediction function

This file does no training at all. When imported, it:

1. Loads `logreg_model.pkl` and `scaler.pkl` from disk using `joblib.load()`.
2. Defines `predict_diagnosis(features)` — takes a list of 30 raw feature values (in the same order as `load_breast_cancer().feature_names`), scales them using the **already-fitted** scaler (`.transform()`, never `.fit_transform()` — the scaler must not be refit on new input), runs the scaled values through the model, and returns a human-readable string: `"Malignant"` or `"Benign"`.

This file is meant to be **imported**, not run directly — running it alone does nothing visible, since it only defines a function and loads two files.

### `app.py` — the Streamlit interface

Imports `predict_diagnosis` from `model.py` and builds a single scrolling page (not tabs), structured to be understandable by someone with no ML background:

1. **Introduction** — a plain-language explanation of what the app does and why.
2. **About this dataset** — what the 30 measurements represent, with a toggleable "Show raw data" button (state is remembered across reruns using `st.session_state`, since Streamlit reruns the entire script on every interaction).
3. **Exploring the data** — a numbered walkthrough:
   - Target distribution (how balanced are malignant vs benign cases)
   - An interactive feature explorer (`st.selectbox`) letting the user pick any of the 30 features and see its distribution by diagnosis
   - A boxplot comparing one strong feature (mean radius) across diagnoses
   - A correlation heatmap across a handful of the most relevant features
4. **Try it yourself** — 30 number inputs, grouped into the dataset's natural "mean / standard error / worst" categories inside collapsible `st.expander` sections, each defaulting to the dataset's mean value. A **Predict** button runs the model and displays the result in green (benign) or red (malignant).

The app also applies a custom dark theme (`.streamlit/config.toml`) and matches matplotlib's plot styling to it via `plt.rcParams`, so charts don't render as jarring white boxes against the dark background.

---

## Why joblib?

`joblib` is used to **save trained Python objects to disk and reload them later**, rather than retraining the model every time the app starts.

Without it, `model.py` would need to call `load_breast_cancer()`, split, scale, and fit a fresh `LogisticRegression` every single time someone opened the app — wasted, repeated work for a result that doesn't change between runs. Instead:

- `train_model.py` trains the model **once** and calls `joblib.dump(model, "logreg_model.pkl")` and `joblib.dump(scaler, "scaler.pkl")`.
- `model.py` calls `joblib.load(...)` to instantly restore the exact same fitted model and scaler, with no retraining.

Both the model **and** the scaler are saved — not just the model — because at prediction time, new user input must be scaled using the exact same mean/standard deviation values learned during training. Fitting a new scaler on a single input row would be meaningless (you can't compute a standard deviation from one number), and would silently produce incorrect predictions.

---

## How to clone and run this project locally

1. **Clone the repository:**
   ```bash
   git clone https://github.com/<your-username>/Breast_Cancer_model_NTI_S12.git
   cd Breast_Cancer_model_NTI_S12
   ```

2. **Create and activate a virtual environment:**
   ```bash
   python -m venv venv
   ```
   Windows:
   ```bash
   venv\Scripts\activate
   ```
   macOS/Linux:
   ```bash
   source venv/bin/activate
   ```

3. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

4. **Train the model** (generates the two `.pkl` files inside `my_module`):
   ```bash
   python my_module/train_model.py
   ```

5. **Run the app:**
   ```bash
   streamlit run app.py
   ```
   This opens the app in your browser at `http://localhost:8501`.

---

## How to use the app

1. Read the introduction and dataset overview at the top of the page.
2. Click **Show raw data** to preview the underlying dataset in a table.
3. Work through the **Exploring the data** section — try selecting different features in the dropdown to see how their distributions differ between malignant and benign cases.
4. Scroll to **Try it yourself**, expand the **Mean values / Standard error values / Worst values** sections, adjust any measurements you want to test, and click **Predict**.
5. The result appears highlighted — green for benign, red for malignant.

---

## Tech stack

- **Python 3**
- **scikit-learn** — dataset, preprocessing, and Logistic Regression model
- **pandas** — data handling
- **seaborn / matplotlib** — visualizations
- **joblib** — model and scaler persistence
- **Streamlit** — the web app framework

---

## Dataset source

Breast Cancer Wisconsin (Diagnostic) dataset, available directly through `sklearn.datasets.load_breast_cancer()`. It contains 569 samples with 30 real-valued features computed from digitized images of fine needle aspirate (FNA) biopsies of breast masses, each labeled malignant or benign.

---

## Disclaimer

This project is an educational exercise built for the NTI Machine Learning Internship. It is **not a medical diagnostic tool** and should never be used as a substitute for professional medical evaluation.
