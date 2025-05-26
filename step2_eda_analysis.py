import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Load merged dataset
df = pd.read_csv(
    r"E:\weather-hadoop-project\data\weather_data_for_ml.csv",
    skiprows=1,
    names=['datetime', 'city', 'temperature', 'humidity', 'wind_speed', 'pressure']
)

df['datetime'] = pd.to_datetime(df['datetime'], errors='coerce')
df['year'] = df['datetime'].dt.year
df['month'] = df['datetime'].dt.month
df['day'] = df['datetime'].dt.day
df['hour'] = df['datetime'].dt.hour
df['weekday'] = df['datetime'].dt.weekday

eda_df = df.sample(frac=0.01, random_state=42)

print("\n🔍 Missing values:")
print(eda_df.isnull().sum())
print("\n📊 Summary statistics:")
print(eda_df.describe())

plt.figure(figsize=(12, 5))
sns.lineplot(data=eda_df.sort_values('datetime'), x='datetime', y='temperature', hue='city')
plt.title("Temperature Trends Over Time by City")
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()

plt.figure(figsize=(8, 5))
sns.barplot(data=eda_df, x='city', y='humidity', estimator='mean')
plt.title("Average Humidity by City")
plt.tight_layout()
plt.show()

for col in ['temperature', 'humidity', 'wind_speed', 'pressure']:
    eda_df[col] = pd.to_numeric(eda_df[col], errors='coerce')

plt.figure(figsize=(6, 4))
sns.heatmap(eda_df[['temperature', 'humidity', 'wind_speed', 'pressure']].corr(), annot=True, cmap='coolwarm')
plt.title("Correlation Matrix")
plt.tight_layout()
plt.show()
