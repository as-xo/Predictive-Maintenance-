import numpy as np
import pandas as pd
from datetime import datetime, timedelta
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix



np.random.seed(42)

data_points = 1000

start_date = datetime.now()

date_range = [start_date + timedelta(hours=i) for i in range(data_points)]

# Simulating sensor data
data = {
    "timestamp": date_range,
    "vibration_level": np.random.normal(loc=5, scale=1, size=data_points),  # mean 5, std 1
    "temperature": np.random.normal(loc=75, scale=5, size=data_points),  # mean 75, std 5
    "motor_speed": np.random.normal(loc=1000, scale=100, size=data_points),  # mean 1000, std 100
    "maintenance_flag": [0] * (data_points - 50) + [1] * 50  # Random failures towards the end
}

for i in range(data_points - 50, data_points):
    data["vibration_level"][i] += i * 0.1  

# Create DataFrame
df = pd.DataFrame(data)
print(df.head())


sns.set_theme(style="whitegrid")
plt.figure(figsize=(14, 8))

# Subplots for each feature
plt.subplot(3, 1, 1)
plt.plot(df['timestamp'], df['vibration_level'], color='blue')
plt.title('Vibration Level Over Time')
plt.xlabel('Time')
plt.ylabel('Vibration Level')

plt.subplot(3, 1, 2)
plt.plot(df['timestamp'], df['temperature'], color='green')
plt.title('Temperature Over Time')
plt.xlabel('Time')
plt.ylabel('Temperature (°F)')

plt.subplot(3, 1, 3)
plt.plot(df['timestamp'], df['motor_speed'], color='purple')
plt.title('Motor Speed Over Time')
plt.xlabel('Time')
plt.ylabel('Motor Speed (RPM)')

plt.tight_layout()
plt.show()


# Feature selection (excluding timestamp and the label)
features = ['vibration_level', 'temperature', 'motor_speed']
X = df[features]
y = df['maintenance_flag']

# Splitting data (80% train, 20% test)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


# Standardizing features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Initialize and train the model
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train_scaled, y_train)

# Predict on the test data
y_pred = model.predict(X_test_scaled)

# Evaluate the model
print("Classification Report:")
print(classification_report(y_test, y_pred))

# Confusion matrix to visualize performance
conf_matrix = confusion_matrix(y_test, y_pred)
print("Confusion Matrix:")
print(conf_matrix)