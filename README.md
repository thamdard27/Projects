
# Diabetes Prediction Using Machine Learning

This project applies multiple machine learning techniques to predict diabetes using a medical dataset of **100,000 patient records**. The goal is to compare several classification models and evaluate their performance with a focus on **recall** and **specificity**, which are essential metrics for medical diagnosis and early disease detection.

---

## Project Files

* **diabetes_prediction.ipynb** – Full code for preprocessing, SMOTE balancing, modeling, hyperparameter tuning, and evaluation
* **Diabetes_Prediction_Report.pdf** – Comprehensive project report summarizing methodology and results
* **README.md** – Project documentation

---

## Dataset

* **Source:** Diabetes Prediction Dataset (Kaggle) "https://www.kaggle.com/datasets/iammustafatz/diabetes-prediction-dataset"
* **Records:** 100,000
* **Features:** age, gender, hypertension, heart disease, BMI, smoking history, HbA1c, blood glucose
* **Target Variable:** diabetes (0 = No, 1 = Yes)

---

## Methods

* Data cleaning and outlier removal
* One-hot encoding for categorical variables
* Feature scaling using StandardScaler
* Handling class imbalance using **SMOTE**
* Model training and hyperparameter tuning (GridSearchCV)
* Evaluation using:

  * Recall
  * Specificity
  * Accuracy
  * Precision
  * F1-score
  * ROC-AUC

---

## Models Evaluated

* Logistic Regression
* K-Nearest Neighbors (KNN)
* Naive Bayes
* Decision Tree
* Random Forest
* Gradient Boosting

---

## Key Findings

* Before SMOTE, all models showed **high accuracy but very low recall**, meaning most diabetic cases were missed.
* After balancing the dataset:

  * **Naive Bayes achieved the highest recall (0.89)**, making it the most sensitive model
  * **KNN provided the best balance** between recall and specificity
  * Ensemble methods performed strongly overall but were less effective at detecting minority-class cases

These results emphasize the importance of balancing techniques when working with medical datasets.

---

## Project Structure

```
📂 Diabetes Prediction Project
│── diabetes_prediction.ipynb.ipynb
│── Diabetes_Prediction_Report.pdf
│── README.md
```

---
## Skills Used
 * Python
 * Pandas, NumPy
 * Scikit-learn
 * Imbalanced-Learn (SMOTE)
 * Data Cleaning & Preprocessing
 * Feature Engineering
 * Hyperparameter Tuning (GridSearchCV)
 * Classification Models (KNN, Naive Bayes, Logistic Regression, Decision Tree, Random Forest, Gradient Boosting)
 * Model Evaluation (Recall, Specificity, F1-score, ROC-AUC)
 * Matplotlib, Seaborn
 * Jupyter Notebook

## Future Improvements

* Test deep learning models (MLP, TabNet)
* Incorporate additional medical or lifestyle features
* Compare SMOTE with other balancing methods (ADASYN, class weights)
* Convert the workflow into a full ML pipeline for deployment

