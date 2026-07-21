#model.py doesn't do anything on its own when run directly — it only defines a function (predict_diagnosis) and loads two files at import time.

#model.py is meant to be imported, not run — specifically, by app.py, the same way your Iris model.py was imported by app.py in the last task. The only thing model.py does when imported is load the already-trained model and scaler from .pkl files, and define a function (predict_diagnosis) that takes 30 raw feature values, scales them, and returns a diagnosis string.


import joblib
import os

# Locate this file's folder so paths work no matter where the script is run from
current_dir = os.path.dirname(os.path.abspath(__file__))

# Load the already-trained model and scaler (created by train_model.py)
model = joblib.load(os.path.join(current_dir, "logreg_model.pkl"))
scaler = joblib.load(os.path.join(current_dir, "scaler.pkl"))


def predict_diagnosis(features):
    """
    Predict breast cancer diagnosis from 30 raw feature values.

    Parameters:
    - features: list or array of 30 numeric values, in the same order
      as sklearn's load_breast_cancer().feature_names

    Returns:
    - diagnosis: str, either "Malignant" or "Benign"
    """
    # Model expects a 2D array (one row), same shape as during training
    features_array = [features]

    # Apply the SAME scaler fitted during training — transform only, never re-fit
    features_scaled = scaler.transform(features_array)

    # Predict: 0 = malignant, 1 = benign (per sklearn's own label convention)
    prediction = model.predict(features_scaled)[0]

    return "Benign" if prediction == 1 else "Malignant"