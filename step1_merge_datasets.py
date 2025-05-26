import pandas as pd

# Load weather files
temperature = pd.read_csv(r"E:\weather-hadoop-project\data\temperature.csv")
humidity = pd.read_csv(r"E:\weather-hadoop-project\data\humidity.csv")
wind_speed = pd.read_csv(r"E:\weather-hadoop-project\data\wind_speed.csv")
pressure = pd.read_csv(r"E:\weather-hadoop-project\data\pressure.csv")

# Merge on datetime and city
df = temperature.merge(humidity, on=["datetime", "city"])
df = df.merge(wind_speed, on=["datetime", "city"])
df = df.merge(pressure, on=["datetime", "city"])

# Save merged dataset
df.to_csv(r"E:\weather-hadoop-project\data\weather_data_for_ml.csv", index=False)
print("✅ Merged dataset saved.")
