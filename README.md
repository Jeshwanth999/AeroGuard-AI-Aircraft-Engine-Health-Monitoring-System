 AeroGuard AI – Aircraft Engine Health Monitoring System

An AI-powered predictive maintenance application that estimates the Remaining Useful Life (RUL) of aircraft engines using NASA C-MAPSS FD001 data and Machine Learning.

📌 Project Overview

AeroGuard AI uses a Random Forest Regression model to predict the remaining operating cycles of an aircraft engine before its dataset-defined failure point.

The application provides a web dashboard where users can enter engine operating conditions and sensor readings to receive an estimated RUL prediction.

🎯 Project Objectives

Predict aircraft engine Remaining Useful Life (RUL).

Analyze engine operating conditions and sensor readings.

Apply Machine Learning for predictive maintenance.

Evaluate and optimize model performance.

Display predictions through an interactive web dashboard.

🛠️ Technologies Used

Category

Technologies

Programming

Python

Machine Learning

Scikit-learn

Data Processing

Pandas, NumPy

Visualization

Matplotlib

Backend

FastAPI

Frontend

HTML, CSS, JavaScript

Database

SQLite

Dataset

NASA C-MAPSS FD001

🧠 Machine Learning Workflow

Load NASA C-MAPSS FD001 dataset.

Perform data preprocessing.

Calculate Remaining Useful Life (RUL).

Apply RUL clipping at 125 cycles.

Train Random Forest Regression model.

Evaluate model using MAE and RMSE.

Analyze feature importance.

Perform hyperparameter tuning.

Compare original and tuned models.

Integrate the tuned model with FastAPI.

📊 Model Performance

Evaluation Metric

Original Model

Tuned Model

MAE

13.37

13.34

RMSE

17.84

17.77

The tuned model achieved a slight improvement over the original model on the NASA FD001 test dataset.

🔍 Feature Importance

Top features identified by the original Random Forest model:

Sensor 11 – 49.80%

Cycle – 25.93%

Sensor 9 – 5.51%

Sensor 4 – 4.70%

These scores indicate how much the model relied on each feature when making predictions.

🌐 Application Features

Interactive engine health monitoring dashboard.

NASA engine sensor input form.

Sample sensor data loading.

Real-time RUL prediction through FastAPI.

SQLite-based prediction history.

Original and tuned Random Forest models.

Model evaluation and feature importance analysis.

🏗️ Project Architecture

AeroGuard_AI/
│
├── app/
│   ├── main.py
│   ├── ml_engine.py
│   └── static/
│       ├── index.html
│       ├── style.css
│       └── app.js
│
├── data/
│   └── nasa/
│       ├── train_FD001.txt
│       ├── test_FD001.txt
│       └── RUL_FD001.txt
│
├── models/
│   ├── nasa_fd001_rul.joblib
│   └── nasa_fd001_tuned.joblib
│
├── train_nasa_model.py
├── evaluate_nasa_model.py
├── feature_importance.py
├── tune_nasa_model.py
└── compare_models.py

⚙️ Installation and Execution

1. Clone the repository

git clone YOUR_GITHUB_REPOSITORY_URL
cd AeroGuard_AI

2. Create virtual environment

python -m venv venv

3. Activate environment (Windows)

.\venv\Scripts\Activate.ps1

4. Install dependencies

pip install fastapi uvicorn pandas numpy scikit-learn matplotlib joblib

5. Start the application

uvicorn app.main:app --reload

6. Open the dashboard

http://127.0.0.1:8000

API documentation:

http://127.0.0.1:8000/docs

📈 Future Enhancements

Compare multiple ML algorithms.

Add engine health status classification.

Develop advanced sensor trend visualizations.

Implement historical engine performance analytics.

Deploy the application to a cloud platform.

⚠️ Disclaimer

This is an educational Machine Learning project using the NASA C-MAPSS benchmark dataset. Predictions are experimental and must not be used for actual aircraft operation, safety, or maintenance decisions.

👨‍💻 Author

Your Name

Machine Learning | Python | Predictive Maintenance | Data Analytics
