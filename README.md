# weather-hadoop-project
Weather Data Analysis and Prediction Using Hadoop

This project demonstrates weather data analysis using Python, feature engineering, and Random Forest model tuning. It is part of the **Data Analytics and Big Data Fundamentals** course.

---

## 📁 Project Structure

weather-hadoop-project/
├── data/ # Cleaned and merged weather datasets but also includes original datatsets
│ └── weather_data_for_ml.csv
│ └── weather_data.csv
│ └── humidity.csv
│ └── pressure.csv
│ └── temperature.csv
│ └── wind_speed.csv
├── notebooks/ # All step-by-step scripts
│ ├── step1_merge_datasets.py
│ ├── step2_eda_analysis.py
│ ├── step3_feature_engineering_and_model.py
│ ├── step4_gridsearch_tuning.py
│ ├── step5_final_model_training.py
│ └── step6_save_model_joblib.py
└── README.md


---

## ✅ Objectives

- Analyze weather datasets using Hadoop and Python
- Perform EDA and feature engineering
- Train a temperature prediction model using Random Forest
- Tune model hyperparameters using GridSearchCV
- Save and reload models with `joblib`

---

## ⚙️ Technologies Used

- Python (Pandas, Scikit-learn, Seaborn)
- Jupyter Notebook
- Cloudera QuickStart VM (Hadoop and Hive)
- GridSearchCV for tuning
- Joblib for model persistence

---

## Model Summary

- Model Type: Random Forest Regressor
- Tuned Parameters: `n_estimators=50`, `max_depth=None`, `min_samples_split=2`
- RMSE (Test): **3.69**
- R² Score (Test): **0.8759**

---

## How to Use

1. Run scripts in order from the `notebooks/` folder.
2. To regenerate the model:
   - Run: `step5_final_model_training.py`
   - Save using: `step6_save_model_joblib.py`
3. To make predictions:
   ```python
   import joblib
   import pandas as pd

   # Load the saved model
   model = joblib.load("model/final_rf_model.pkl")

   # Example input: match the order of features used in training
   # (e.g., year, month, day, hour, weekday, city_encoded columns)
   sample_input = pd.DataFrame([{
       'humidity': 85,
       'wind_speed': 2.5,
       'pressure': 1012,
       'year': 2016,
       'month': 6,
       'day': 15,
       'hour': 14,
       'weekday': 2,
       # Include city one-hot encoded fields like:
       'city_Bangalore': 0,
       'city_Delhi': 1,
       'city_Mumbai': 0,
       # ...
   }])

   # Predict temperature
   prediction = model.predict(sample_input)
   print("Predicted temperature:", prediction[0])

## Notes
The model file (.pkl) is not uploaded due to GitHub's size restrictions.
To deploy or test predictions, run the training scripts to recreate the model.

## Author
Neeharika Solasa
B.Tech CSE, VIT-AP
