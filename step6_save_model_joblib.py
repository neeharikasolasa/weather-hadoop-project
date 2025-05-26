import joblib
import os

model_path = r"E:\weather-hadoop-project\model\final_rf_model.pkl"
os.makedirs(os.path.dirname(model_path), exist_ok=True)
joblib.dump(final_model, model_path)
print(f"✅ Model saved to: {model_path}")

# Optional load and test
loaded_model = joblib.load(model_path)
sample_prediction = loaded_model.predict([X_test.iloc[0]])
print(f"🔍 Sample Prediction: {sample_prediction[0]:.2f} (Actual: {y_test.iloc[0]:.2f})")
