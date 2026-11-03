"""
Main.py - Employee Attrition Prediction pipeline
Loads the IBM HR dataset, preprocesses it, trains an ANN (Keras) plus
Logistic Regression and Random Forest baselines, prints the evaluation
metrics, and saves the ANN and preprocessor for use by APP.py.

Run:  python Main.py
"""
import warnings
import numpy as np
import pandas as pd
import joblib
import tensorflow as tf
from tensorflow import keras
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (accuracy_score, precision_score, recall_score,
                             f1_score, roc_auc_score, confusion_matrix)

warnings.filterwarnings("ignore")

DATA_FILE = "WA_Fn-UseC_-HR-Employee-Attrition.csv"
MODEL_FILE = "attrition_ann.keras"
PREPROCESSOR_FILE = "attrition_preprocessor.joblib"
SEED = 42

np.random.seed(SEED)
tf.random.set_seed(SEED)


def load_and_clean(path):
    """Load the CSV, drop constant/identifier columns, split X and y."""
    df = pd.read_csv(path)
    print("Data shape:", df.shape)
    print("Missing values:", df.isnull().sum().sum(),
          "| Duplicates:", df.duplicated().sum())
    df = df.drop(columns=["EmployeeCount", "Over18", "StandardHours", "EmployeeNumber"])
    y = df["Attrition"].map({"Yes": 1, "No": 0})
    X = df.drop(columns=["Attrition"])
    return X, y


def build_preprocessor(X):
    """Standardise numeric columns and one-hot encode categorical columns."""
    num_cols = X.select_dtypes(include="number").columns.tolist()
    cat_cols = X.select_dtypes(exclude="number").columns.tolist()
    return ColumnTransformer([
        ("num", StandardScaler(), num_cols),
        ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), cat_cols),
    ])


def build_ann(n_inputs):
    """Two hidden layers (64, 32) with dropout and a sigmoid output."""
    model = keras.Sequential([
        keras.layers.Input(shape=(n_inputs,)),
        keras.layers.Dense(64, activation="relu"),
        keras.layers.Dropout(0.3),
        keras.layers.Dense(32, activation="relu"),
        keras.layers.Dropout(0.2),
        keras.layers.Dense(1, activation="sigmoid"),
    ])
    model.compile(optimizer=keras.optimizers.Adam(0.001),
                  loss="binary_crossentropy",
                  metrics=["accuracy", keras.metrics.AUC(name="auc")])
    return model


def evaluate(name, y_true, prob, thr=0.5):
    pred = (prob >= thr).astype(int)
    return {
        "Model": name,
        "Accuracy": accuracy_score(y_true, pred),
        "Precision": precision_score(y_true, pred, zero_division=0),
        "Recall": recall_score(y_true, pred),
        "F1-score": f1_score(y_true, pred),
        "ROC-AUC": roc_auc_score(y_true, prob),
    }


def main():
    X, y = load_and_clean(DATA_FILE)

    # Stratified 80:20 split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=SEED)
    print("Train:", X_train.shape, "Test:", X_test.shape)

    # Preprocessing (fit on training data only to avoid leakage)
    preprocessor = build_preprocessor(X)
    X_train_p = preprocessor.fit_transform(X_train).astype("float32")
    X_test_p = preprocessor.transform(X_test).astype("float32")
    print("Input features after encoding:", X_train_p.shape[1])

    # Class weights to handle the imbalance
    n0, n1 = (y_train == 0).sum(), (y_train == 1).sum()
    class_weight = {0: len(y_train) / (2 * n0), 1: len(y_train) / (2 * n1)}

    # Train the ANN
    model = build_ann(X_train_p.shape[1])
    model.summary()
    early_stop = keras.callbacks.EarlyStopping(
        monitor="val_loss", patience=15, restore_best_weights=True)
    history = model.fit(X_train_p, y_train.values, validation_split=0.15,
                        epochs=200, batch_size=32, class_weight=class_weight,
                        callbacks=[early_stop], verbose=0)
    print("ANN stopped after", len(history.history["loss"]), "epochs")

    # Baseline models
    lr = LogisticRegression(max_iter=1000, class_weight="balanced",
                            random_state=SEED).fit(X_train_p, y_train)
    rf = RandomForestClassifier(n_estimators=300, min_samples_leaf=2,
                                class_weight="balanced", random_state=SEED,
                                n_jobs=-1).fit(X_train_p, y_train)

    # Evaluation on the held-out test set
    probs = {
        "ANN (Keras)": model.predict(X_test_p, verbose=0).ravel(),
        "Logistic Regression": lr.predict_proba(X_test_p)[:, 1],
        "Random Forest": rf.predict_proba(X_test_p)[:, 1],
    }
    results = pd.DataFrame([evaluate(n, y_test, p) for n, p in probs.items()])
    print("\nTest-set performance (threshold 0.5)")
    print(results.set_index("Model").round(4).to_string())

    print("\nANN confusion matrix [[TN FP] [FN TP]]")
    print(confusion_matrix(y_test, (probs["ANN (Keras)"] >= 0.5).astype(int)))

    # Overfitting check (train vs test AUC)
    train_auc = roc_auc_score(y_train, model.predict(X_train_p, verbose=0).ravel())
    test_auc = roc_auc_score(y_test, probs["ANN (Keras)"])
    print(f"\nANN ROC-AUC  train: {train_auc:.3f} | test: {test_auc:.3f}")

    # Save artefacts for APP.py
    model.save(MODEL_FILE)
    joblib.dump(preprocessor, PREPROCESSOR_FILE)
    print(f"\nSaved {MODEL_FILE} and {PREPROCESSOR_FILE}")


if __name__ == "__main__":
    main()
