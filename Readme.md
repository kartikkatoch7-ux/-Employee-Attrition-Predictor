# 📉 Employee Attrition Predictor

Predict whether an employee will leave the company or stay. The prediction uses the employee's demographics, job role, compensation, overtime, satisfaction ratings and career history. The project compares **Logistic Regression**, a **Random Forest** and a **Neural Network (TensorFlow/Keras)**, and studies overfitting, decision thresholds and feature importance.

> Academic project - MBA (Data Science and Artificial Intelligence), Chandigarh University.
> Author: Kartik Katoch (25MDA10082)

---

## Dataset

[IBM HR Analytics Employee Attrition & Performance](https://www.kaggle.com/datasets/pavansubhasht/ibm-hr-analytics-attrition-dataset) (`WA_Fn-UseC_-HR-Employee-Attrition.csv`)

| Item | Value |
|---|---|
| Employees | 1,470 |
| Columns | 35 (including target) |
| Leavers | 237 (16.1%) |
| Model inputs | 51 (23 numeric + 28 one-hot columns) |
| Missing values / duplicates | None |

This is a **fictional** dataset created by IBM data scientists, so the findings illustrate methodology, not any real organisation.

## Project structure

```
Employee-Attrition-Prediction/
├── APP.py                                  # Streamlit web app
├── Main.py                                 # Training + evaluation pipeline
├── Employee_Attrition_ANN.ipynb            # Notebook: EDA, model, evaluation
├── Project_Report.pdf                      # Full project report
├── requirements.txt                        # Python packages
├── README.md
└── WA_Fn-UseC_-HR-Employee-Attrition.csv   # Dataset
```

## Method

1. Drop constant/identifier columns (`EmployeeCount`, `StandardHours`, `Over18`, `EmployeeNumber`).
2. Stratified 80:20 train-test split (`random_state=42`).
3. Standardise numeric features and one-hot encode categorical features (fitted on training data only).
4. Train the ANN: `Dense(64, ReLU) → Dropout(0.3) → Dense(32, ReLU) → Dropout(0.2) → Dense(1, Sigmoid)`, Adam (lr 0.001), binary cross-entropy, class weights and early stopping.
5. Compare with Logistic Regression and Random Forest on the same test set.
6. Check overfitting, decision-threshold effects and feature importance.

## Results (report, threshold 0.5, test set of 294 employees)

| Model | Accuracy | Precision | Recall | F1 | ROC-AUC |
|---|---|---|---|---|---|
| Logistic Regression | 86.1% | 0.615 | 0.340 | 0.438 | 0.812 |
| Random Forest | 84.4% | 0.545 | 0.128 | 0.207 | 0.794 |
| **ANN (Keras)** | 82.7% | 0.467 | **0.596** | **0.523** | 0.763 |

**Key findings**

- The ANN found the most leavers (best recall and F1), but Logistic Regression had the best accuracy and ROC-AUC.
- The ANN overfitted: ROC-AUC was 0.966 on training data vs 0.763 on test data.
- With only 47 leavers in the test set, the model differences are within sampling error.
- **OverTime** is the strongest driver of attrition (30.5% of overtime workers left vs 10.4% of others), followed by marital status, job satisfaction and number of companies worked.

> Exact numbers can vary slightly when you re-run `Main.py` (random initialisation, class weights, early stopping).

## How to run

```bash
# 1. Clone the repository
git clone https://github.com/YOUR-USERNAME/Employee-Attrition-Prediction.git
cd Employee-Attrition-Prediction

# 2. Install packages
pip install -r requirements.txt

# 3. Train the models and save attrition_ann.keras + attrition_preprocessor.joblib
python Main.py

# 4. Launch the web app
streamlit run APP.py
```

## Limitations and future work

- Small dataset; results come from a single train-test split.
- Fictional data; patterns may not carry over to a real company.
- Future work: cross-validation, hyperparameter tuning, SMOTE, SHAP explanations, validation on real HR data.

## Tech stack

Python · Pandas · NumPy · Scikit-learn · TensorFlow/Keras · Matplotlib · Seaborn · Streamlit
