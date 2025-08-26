# Clinical Obesity Prediction Dashboard

## Project Overview
This project builds a machine learning pipeline to classify clinical obesity using health survey data. It extracts, cleans, and merges data from government health PDFs, engineers features based on medical criteria, trains several ML models, and provides an interactive dashboard for predictions and model comparison.

---

## 1. Data Extraction & Cleaning
- **Source:** Data comes from large health survey PDFs (NFHS-5).
- **Extraction:** We use `pdfplumber` to extract tables from PDFs and save them as CSV files.
- **Cleaning:** The CSVs are cleaned to remove metadata, standardize column names, and extract key fields (height, weight, age, sex, region, etc.).
- **Merging:** All datasets are merged using unique IDs to create a single, analysis-ready table.

---

## 2. Feature Engineering
- **BMI Calculation:** BMI = weight (kg) / (height (m))^2
- **Flags:**
  - `adl_limitation_flag`: 1 if the person has difficulty with daily activities (walking, dressing, etc.)
  - `organ_dysfunction_flag`: 1 if the person has organ issues (breathlessness, anaemia, etc.)
- **Labeling (Lancet Criteria):**
  - **0 = Normal:** BMI < 25
  - **1 = Preclinical Obesity:** 25 ≤ BMI < 30, no dysfunction
  - **2 = Clinical Obesity:** BMI ≥ 30 and any dysfunction flag is 1

---

## 3. Model Choices & Why We Use Them
We train and compare several popular ML models:

### Logistic Regression
- **What:** A simple, interpretable model for classification.
- **Why:** Good baseline for binary/multiclass problems.

### K-Nearest Neighbors (KNN)
- **What:** Classifies a sample based on the majority label of its 'k' closest data points.
- **Why:**
  - Very intuitive and non-parametric (no training phase, just stores data).
  - Works well on small datasets and when classes are well-separated in feature space.
  - **In this project, KNN performed best** because:
    - Our dataset is small and well-labeled.
    - The classes (Normal, Preclinical, Clinical) are clearly separated by BMI and flags.
    - KNN can easily find similar cases and assign the correct label.

### Random Forest
- **What:** An ensemble of decision trees.
- **Why:** Handles non-linearities and feature interactions, robust to overfitting.

### XGBoost
- **What:** A powerful gradient boosting algorithm.
- **Why:** Often wins ML competitions, handles complex patterns, but can overfit on small data.

---

## 4. Why KNN Stands Out Here
- **Small Dataset:** KNN is ideal for small datasets because it doesn't need to "learn" parameters, just compares new samples to existing ones.
- **Clear Feature Separation:** Since BMI and flags clearly separate the classes, KNN can easily find the right neighbors.
- **No Overfitting:** On small, clean data, KNN is less likely to overfit compared to complex models like XGBoost.
- **Result:** KNN achieved the highest accuracy and F1-score in our model comparison.

---

## 5. Using the Streamlit Dashboard
- **Manual Entry:** Enter height, weight, flags, age, sex, and region to get a prediction and see your BMI.
- **File Upload:** Upload a CSV with multiple records for batch predictions.
- **Visualizations:**
  - See your BMI and predicted label.
  - Compare all models (KNN, Logistic Regression, Random Forest, XGBoost) with interactive charts.
- **Download Results:** Download predictions as CSV for further analysis or Tableau visualization.

---

## 6. Interpreting Results
- **BMI:** Shows your calculated Body Mass Index.
- **Clinical Obesity Label:** Rule-based label using medical criteria.
- **Model Prediction:** What the ML model predicts for your input.
- **Model Comparison:**
  - Table and bar charts show how each model performed.
  - The best model for each metric is highlighted.
  - Use this to understand which model is most reliable for your data.

---

## 7. Next Steps
- Try with your own data or larger datasets for more robust results.
- Use the exported CSV for advanced visualizations in Tableau or Power BI.
- Experiment with more features or different ML models as you learn more!

---

**This project is beginner-friendly and designed to help you learn how real-world ML pipelines are built, evaluated, and deployed. If you have questions, just ask!** 