# 🤖 Employee Productivity Prediction using Machine Learning

---

## 🌟 Introduction
This project demonstrates a full machine learning workflow to predict garment workers' productivity using daily operational data. Built with Python in JupyterLab, the solution includes data preprocessing, exploratory analysis, multiple model training, evaluation, and deployment. By leveraging machine learning, organizations can make data-driven decisions to optimize workforce productivity.

---

## 📂 Dataset Overview

The dataset `garments_worker_productivity.csv` contains:

- **Date, Quarter, Department, Day**
- **Team , Targeted Productivity**
- **Operational Features**: SMV, WIP, Over Time, Incentive  
- **Behavioral Features**: Idle Time, Idle Men, Style Changes  
- **Workforce Metrics**: Number of Workers
- **Target:** actual_productivity

---

## 🔍 Project Pipeline

### 1. 📊 Data Preprocessing
- Converted `date` to `month` and dropped the original `date` column
- Dropped columns with >20% missing values (e.g., `wip`)
- Imputed remaining missing values using:
  - **Mode** for categorical features
  - **Mean** for numerical features
- Applied `LabelEncoder` to categorical columns: `quarter`, `department`, `day`

### 2. 🧪 Exploratory Analysis
- Generated correlation matrix and heatmap (via Seaborn)
- Analyzed descriptive statistics of both numerical and categorical features

### 3. 🧠 Model Training
Three regression models were trained:
- **Linear Regression**
- **Random Forest Regressor**
- **XGBoost Regressor**

### 4. 🏆 Model Evaluation
Models were evaluated using:
- Mean Absolute Error (MAE)
- Mean Squared Error (MSE)
- R² Score

### 5. 💾 Model Saving
- Best model selected: **Random Forest**
- Saved as `best_model.pkl` using `pickle`

---

## 📊 Model Performance Summary
_________________________________________________
| Model            | MAE    | MSE    | R² Score |
|------------------|--------|--------|----------|
| Linear Regression| 0.1075 | 0.0216 | 0.1862   |
| Random Forest    | 0.0669 | 0.0117 | 0.5583   |
| XGBoost          | 0.0727 | 0.0151 | 0.4331   |


- **Best Model:** 👉 Random Forest (highest R², lowest errors)

---

## 🧩 Feature Importance
- Visualized feature importances for Random Forest and XGBoost
- Key features: team, targeted_productivity, smv, over_time, no_of_workers, etc.

---

## 🤖 Web App Integration
- Built a Flask web application for user-friendly predictions
- HTML templates for Home, Predict, Submit, and About pages
- Model loaded and used for real-time predictions via web form

---

## 🎯 Results
- The system predicts employee productivity with good accuracy
- Users can input data and get instant predictions via the web app
- Youtube Link : [https://youtu.be/DzDXuIav9bc](https://youtu.be/DzDXuIav9bc)



---

## 🛠️ How to Run

### 1. Clone the repository
```bash
git clone https://github.com/chandradeepkumarraj/epp_ml_model.git
cd emp_ml_model
```

### 2. Install dependencies
```bash
pip install pandas numpy matplotlib seaborn scikit-learn xgboost
```

### 3. Run the Flask app
```bash
python app.py
```
Visit [http://localhost:5000](http://localhost:5000) in your browser.

---

## 🧠 Author
**Author:** Chandradeep Kumar Raj  
**Email:** chandradeepkumarraj@gmail.com  
**GitHub:** [@chandradeepkumarraj]

