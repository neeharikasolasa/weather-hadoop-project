final_model = RandomForestRegressor(
    n_estimators=50,
    max_depth=None,
    min_samples_split=2,
    random_state=42
)
final_model.fit(X_train, y_train)
y_pred = final_model.predict(X_test)

rmse = sqrt(mean_squared_error(y_test, y_pred))
r2 = r2_score(y_test, y_pred)

print("✅ Final Model Training Complete")
print(f"🔹 RMSE: {rmse:.2f}")
print(f"🔹 R² Score: {r2:.4f}")
