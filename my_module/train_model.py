# it does the loading, splitting, scaling, and fitting, then saves the model and scaler out to .pkl files
#model.py never trains anything itself — it just loads those saved .pkl files and hands app.py a ready-to-use predict_diagnosis() function. app.py never talks to train_model.py directly at all; it only ever imports from model.py.
#piplne: train_model.py -> model.py -> app.py
#pipline of the file : Imports — load_breast_cancer from sklearn.datasets, train_test_split from sklearn.model_selection, StandardScaler from sklearn.preprocessing, LogisticRegression from sklearn.linear_model, and joblib.  Load the data — data = load_breast_cancer(), then X = data.data, y = data.target. Split — X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42). The random_state just makes the split reproducible — you'll get the same split every time you rerun this file. Scale — fit StandardScaler on X_train only, then transform both X_train and X_test with that same fitted scaler. Train — fit LogisticRegression on the scaled X_train and y_train. Check it actually works — print the accuracy on X_test before saving anything, so you know the model is reasonable first. Save — joblib.dump() the fitted model and the fitted scaler to two .pkl files inside my_module, so model.py can find them right next door.
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
import joblib
import os

# 1. Load the data
data = load_breast_cancer()
X = data.data
y = data.target

# 2. Split into train/test
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# 3. Scale — fit on train only, transform both
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# 4. Train
model = LogisticRegression(max_iter=5000)
model.fit(X_train_scaled, y_train)

# 5. Check it actually works before saving anything
y_pred = model.predict(X_test_scaled)
print(f"Test accuracy: {accuracy_score(y_test, y_pred):.4f}")

# 6. Save model + scaler next to this file
current_dir = os.path.dirname(os.path.abspath(__file__))
joblib.dump(model, os.path.join(current_dir, "logreg_model.pkl"))
joblib.dump(scaler, os.path.join(current_dir, "scaler.pkl"))

print("Saved logreg_model.pkl and scaler.pkl")

# some notes worth knowing 
#max_iter=5000 — Logistic Regression trains iteratively (like SGD), and the default iteration limit (100) sometimes isn't enough to fully converge on this dataset. Raising it avoids a "did not converge" warning. This isn't cheating or overfitting — it just gives the optimizer enough steps to settle.
#os.path.dirname(os.path.abspath(__file__)) — this makes sure the two .pkl files always save inside my_module, no matter what folder i're standing in when you run the script from the terminal.
