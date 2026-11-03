# Employee Attrition Prediction using an Artificial Neural Network

Predicts whether an employee will leave a company using a Keras ANN on the
IBM HR Analytics dataset (1,470 employees), compared against Logistic
Regression and Random Forest.

## Results (test set, threshold 0.5)
| Model | Accuracy | Recall | F1 | ROC-AUC |
|---|---|---|---|---|
| Logistic Regression | 86.1% | 0.340 | 0.438 | 0.812 |
| Random Forest | 84.4% | 0.128 | 0.207 | 0.794 |
| ANN (Keras) | 82.7% | 0.596 | 0.523 | 0.763 |

Key finding: OverTime is the strongest driver of attrition. The ANN overfits
the small dataset (train AUC 0.966 vs test 0.763).

## Project structure
- `attrition_ann.ipynb` – full workflow: EDA, preprocessing, model, evaluation
- `attrition_ann.keras` – trained model
- `WA_Fn-UseC_-HR-Employee-Attrition.csv` – dataset
- `KK_TIK_PRO.pdf` – full project report

## How to run
1. `pip install -r requirements.txt`
2. `jupyter notebook attrition_ann.ipynb`

## Tech stack
Python, Pandas, NumPy, Scikit-learn, TensorFlow/Keras, Matplotlib

## Author
Kartik Katoch – MBA (Data Science & AI), Chandigarh University
