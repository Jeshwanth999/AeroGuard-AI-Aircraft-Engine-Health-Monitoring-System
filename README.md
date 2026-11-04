# ✈️ AeroGuard AI – Aircraft Engine Health Monitoring System

An end-to-end Machine Learning project that predicts the **Remaining Useful Life (RUL)** of aircraft engines using NASA C-MAPSS sensor data.

## 🚀 Project Overview

AeroGuard AI is a predictive maintenance application designed to estimate how many operational cycles an aircraft engine can continue running before it reaches its end of useful life.

The system uses historical engine sensor measurements and Machine Learning regression models to predict Remaining Useful Life (RUL).

It includes a FastAPI backend and an interactive web dashboard for submitting engine sensor readings and viewing predictions.

## 🎯 Key Features

- NASA C-MAPSS FD001 dataset integration
- Aircraft engine Remaining Useful Life prediction
- Random Forest Regression model
- Hyperparameter tuning
- Model performance comparison
- Feature importance analysis
- Interactive web dashboard
- FastAPI prediction API
- Engine prediction history and metrics

## 🧠 Machine Learning Workflow

1. Load NASA C-MAPSS FD001 dataset.
2. Perform data preprocessing and feature preparation.
3. Generate RUL target values from engine cycles.
4. Train a Random Forest Regression model.
5. Evaluate model performance.
6. Perform hyperparameter tuning.
7. Compare original and tuned models.
8. Integrate the trained model with FastAPI.
9. Display predictions through the web dashboard.

## 📊 Model Performance

The following results were obtained on the NASA FD001 test dataset.

| Metric | Original Model | Tuned Model |
|---|---:|---:|
| MAE | 13.37 | 13.34 |
| RMSE | 17.84 | 17.77 |

**Observation:** Hyperparameter tuning produced a small improvement in both MAE and RMSE.

- MAE: Mean Absolute Error
- RMSE: Root Mean Squared Error

## 🔍 Feature Importance

Top important features identified by the Random Forest model:

| Feature | Importance |
|---|---:|
| sensor_11 | 0.497996 |
| cycle | 0.259288 |
| sensor_9 | 0.055090 |
| sensor_4 | 0.047006 |
| sensor_12 | 0.021992 |
| sensor_7 | 0.017822 |
| sensor_14 | 0.016166 |
| sensor_15 | 0.015662 |
| sensor_21 | 0.010687 |
| sensor_2 | 0.010080 |

## 🛠️ Technologies Used

**Programming:** Python

**Machine Learning:** Scikit-learn, Random Forest Regression

**Data Processing:** Pandas, NumPy

**Visualization:** Matplotlib

**Backend:** FastAPI, Uvicorn

**Frontend:** HTML, CSS, JavaScript

**Model Storage:** Joblib

**Dataset:** NASA C-MAPSS FD001

## 📁 Project Structure

```text
AeroGuard_AI/
│
├── app/
│   └── main.py
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
├── static/
│   ├── index.html
│   ├── style.css
│   └── app.js
│
├── train_nasa_model.py
├── evaluate_nasa_model.py
├── tune_nasa_model.py
├── compare_models.py
├── feature_importance.py
├── requirements.txt
└── README.md
```

*Note: Update this structure if your actual repository folders or filenames are different.*

## ⚙️ Installation and Setup

### 1. Clone the Repository

```bash
git clone YOUR_GITHUB_REPOSITORY_URL
```

### 2. Navigate to Project Folder

```bash
cd AeroGuard_AI
```

### 3. Create Virtual Environment

```bash
python -m venv venv
```

### 4. Activate Virtual Environment

**Windows PowerShell:**

```powershell
.\venv\Scripts\Activate.ps1
```

### 5. Install Dependencies

```bash
pip install -r requirements.txt
```

### 6. Run the Application

```bash
uvicorn app.main:app --reload
```

### 7. Open the Dashboard

```text
http://127.0.0.1:8000
```

### 8. Explore API Documentation

```text
http://127.0.0.1:8000/docs
```

## 📈 Future Enhancements

- Compare additional Machine Learning algorithms.
- Improve model performance through advanced tuning.
- Add real-time sensor data integration.
- Develop advanced predictive maintenance analytics.
- Deploy the application to a cloud platform.

## ⚠️ Disclaimer

This project is developed for educational and demonstration purposes using the NASA C-MAPSS dataset. It is not certified or validated for real-world aircraft maintenance or aviation safety decisions.

## 👨‍💻 Author

G.Jeshwanth

Machine Learning | Python | Data Analytics

---

⭐ If you find this project interesting, consider giving this repository a star!
