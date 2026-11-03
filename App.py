"""
APP.py - Employee Attrition Predictor (Streamlit web app)
 
Run Main.py first to create attrition_ann.keras and attrition_preprocessor.joblib, then:
    streamlit run APP.py
"""
import os
import joblib
import pandas as pd
import streamlit as st
from tensorflow import keras
 
DATA_FILE = "WA_Fn-UseC_-HR-Employee-Attrition.csv"
MODEL_FILE = "attrition_ann.keras"
PREPROCESSOR_FILE = "attrition_preprocessor.joblib"
DROP_COLS = ["EmployeeCount", "Over18", "StandardHours", "EmployeeNumber", "Attrition"]
 
st.set_page_config(page_title="Employee Attrition Predictor", page_icon="📉", layout="wide")
 
 
@st.cache_resource
def load_artifacts():
    model = keras.models.load_model(MODEL_FILE)
    preprocessor = joblib.load(PREPROCESSOR_FILE)
    return model, preprocessor
 
 
@st.cache_data
def load_reference():
    df = pd.read_csv(DATA_FILE).drop(columns=DROP_COLS)
    # Typical value for every column: median for numbers, most common for text
    defaults = {c: (df[c].median() if pd.api.types.is_numeric_dtype(df[c]) else df[c].mode()[0])
                for c in df.columns}
    return df, defaults
 
 
st.title("📉 Employee Attrition Predictor")
st.write("Estimate the probability that an employee will leave, using an Artificial "
         "Neural Network trained on the IBM HR Analytics dataset.")
 
if not (os.path.exists(MODEL_FILE) and os.path.exists(PREPROCESSOR_FILE)):
    st.error("Model files not found. Run `python Main.py` first to train and save the model.")
    st.stop()
 
model, preprocessor = load_artifacts()
ref, defaults = load_reference()
 
st.sidebar.header("Settings")
threshold = st.sidebar.slider("Decision threshold", 0.10, 0.90, 0.50, 0.05,
                              help="Lower values flag more employees (higher recall, lower precision).")
st.sidebar.caption("Fields not shown below use the dataset's typical (median/most common) value.")
 
st.subheader("Employee details")
c1, c2, c3 = st.columns(3)
 
with c1:
    age = st.slider("Age", 18, 60, int(defaults["Age"]))
    marital = st.selectbox("Marital status", sorted(ref["MaritalStatus"].unique()),
                           index=sorted(ref["MaritalStatus"].unique()).index(defaults["MaritalStatus"]))
    gender = st.selectbox("Gender", sorted(ref["Gender"].unique()))
    distance = st.slider("Distance from home (km)", 1, 29, int(defaults["DistanceFromHome"]))
    education = st.slider("Education level (1-5)", 1, 5, int(defaults["Education"]))
    edu_field = st.selectbox("Education field", sorted(ref["EducationField"].unique()))
 
with c2:
    department = st.selectbox("Department", sorted(ref["Department"].unique()))
    job_role = st.selectbox("Job role", sorted(ref["JobRole"].unique()))
    job_level = st.slider("Job level (1-5)", 1, 5, int(defaults["JobLevel"]))
    overtime = st.selectbox("Works overtime?", ["No", "Yes"])
    travel = st.selectbox("Business travel", sorted(ref["BusinessTravel"].unique()),
                          index=sorted(ref["BusinessTravel"].unique()).index(defaults["BusinessTravel"]))
    stock = st.slider("Stock option level (0-3)", 0, 3, int(defaults["StockOptionLevel"]))
 
with c3:
    income = st.number_input("Monthly income", 1000, 20000, int(defaults["MonthlyIncome"]), step=100)
    total_years = st.slider("Total working years", 0, 40, int(defaults["TotalWorkingYears"]))
    years_company = st.slider("Years at company", 0, 40, int(defaults["YearsAtCompany"]))
    years_role = st.slider("Years in current role", 0, 18, int(defaults["YearsInCurrentRole"]))
    num_companies = st.slider("Companies worked for", 0, 9, int(defaults["NumCompaniesWorked"]))
    years_promo = st.slider("Years since last promotion", 0, 15, int(defaults["YearsSinceLastPromotion"]))
 
st.subheader("Satisfaction ratings (1 = low, 4 = high)")
s1, s2, s3, s4 = st.columns(4)
job_sat = s1.slider("Job satisfaction", 1, 4, int(defaults["JobSatisfaction"]))
env_sat = s2.slider("Environment satisfaction", 1, 4, int(defaults["EnvironmentSatisfaction"]))
wlb = s3.slider("Work-life balance", 1, 4, int(defaults["WorkLifeBalance"]))
job_inv = s4.slider("Job involvement", 1, 4, int(defaults["JobInvolvement"]))
 
if st.button("Predict attrition risk", type="primary"):
    row = dict(defaults)
    row.update({
        "Age": age, "MaritalStatus": marital, "Gender": gender,
        "DistanceFromHome": distance, "Education": education, "EducationField": edu_field,
        "Department": department, "JobRole": job_role, "JobLevel": job_level,
        "OverTime": overtime, "BusinessTravel": travel, "StockOptionLevel": stock,
        "MonthlyIncome": income, "TotalWorkingYears": total_years,
        "YearsAtCompany": years_company, "YearsInCurrentRole": years_role,
        "NumCompaniesWorked": num_companies, "YearsSinceLastPromotion": years_promo,
        "JobSatisfaction": job_sat, "EnvironmentSatisfaction": env_sat,
        "WorkLifeBalance": wlb, "JobInvolvement": job_inv,
    })
    X_new = pd.DataFrame([row])[ref.columns]
    X_p = preprocessor.transform(X_new).astype("float32")
    prob = float(model.predict(X_p, verbose=0)[0][0])
 
    st.subheader("Result")
    st.metric("Probability of leaving", f"{prob:.1%}")
    st.progress(min(max(prob, 0.0), 1.0))
    if prob >= threshold:
        st.error(f"⚠️ High risk: likely to leave (probability ≥ {threshold:.0%} threshold).")
    else:
        st.success(f"✅ Lower risk: likely to stay (probability < {threshold:.0%} threshold).")
    st.caption("This is a probability estimate from patterns in a fictional dataset. "
               "Use it to prioritise conversations, not to make individual decisions.")
