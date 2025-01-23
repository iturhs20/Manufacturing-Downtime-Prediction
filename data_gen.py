import pandas as pd
import numpy as np

# Set random seed for reproducibility
np.random.seed(42)

# Create synthetic data
machine_ids = [f"Machine_{i+1}" for i in range(50)]
temperatures = np.random.randint(60, 120, size=50)
run_times = np.random.randint(30, 180, size=50)
downtime_flags = np.random.choice(['Yes', 'No'], size=50, p=[0.3, 0.7])

# Create DataFrame
data = {
    'Machine_ID': machine_ids,
    'Temperature': temperatures,
    'Run_Time': run_times,
    'Downtime_Flag': downtime_flags
}

df = pd.DataFrame(data)

# Save the DataFrame as CSV
df.to_csv('manufacturing_data.csv', index=False)

print("Dataset saved as 'manufacturing_data.csv'")
