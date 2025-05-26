from sklearn.model_selection import GridSearchCV

param_grid = {
    'n_estimators': [10, 50],
    'max_depth': [5, 10, None],
    'min_samples_split': [2, 5]
}

grid_search = GridSearchCV(
    estimator=RandomForestRegressor(random_state=42),
    param_grid=param_grid,
    cv=3,
    n_jobs=-1,
    scoring='neg_mean_squared_error'
)
grid_search.fit(X_train, y_train)

best_model = grid_search.best_estimator_
y_pred = best_model.predict(X_test)
rmse = sqrt(mean_squared_error(y_test, y_pred))
r2 = r2_score(y_test, y_pred)

print("✅ GridSearchCV Complete")
print(f"🔍 Best Parameters: {grid_search.best_params_}")
print(f"🔹 RMSE: {rmse:.2f}")
print(f"🔹 R² Score: {r2:.4f}")
