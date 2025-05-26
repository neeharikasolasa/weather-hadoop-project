import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
from math import sqrt

df = pd.read_csv(
    r"E:\weather-hadoop-project\data\weather_data_for_ml.csv",
    skiprows=1,
    names=['datetime', 'city', 'temperature', 'humidity', 'wind_speed', 'pressure']
)

for col in ['temperature', 'humidity', 'wind_speed', 'pressure']:
    df[col] = pd.to_numeric(df[col], errors='coerce')

df['datetime'] = pd.to_datetime(df['datetime'], errors='coerce')
df['year'] = df['datetime'].dt.year
df['month'] = df['datetime'].dt.month
df['day'] = df['datetime'].dt.day
df['hour'] = df['datetime'].dt.hour
df['weekday'] = df['datetime'].dt.weekday

df = pd.get_dummies(df, columns=['city'], drop_first=True)
df.dropna(inplace=True)
df = df.sample(frac=0.05, random_state=42)

X = df.drop(columns=['temperature', 'datetime'])
y = df['temperature']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = RandomForestRegressor(n_estimators=10, random_state=42)
model.fit(X_train, y_train)

y_pred = model.predict(X_test)
rmse = sqrt(mean_squared_error(y_test, y_pred))
r2 = r2_score(y_test, y_pred)

print("✅ Model Training Complete")
print(f"🔹 RMSE: {rmse:.2f}")
print(f"🔹 R² Score: {r2:.4f}")
