import numpy as np
import pandas as pd
from datetime import datetime, timedelta

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