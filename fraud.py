import pandas as pd
import numpy as np
from scipy.stats import norm

# File configurations
input_path = "Datasets/AEP_hourly.csv"
output_path = "Datasets/AEP_hourly_fraud.csv"

# 1. Load the dataset
df = pd.read_csv(input_path)

# 2. Rename columns: Datetime -> datetime, AEP_MW -> electricity_consumption
df = df.rename(columns={
    'Datetime': 'datetime',
    'AEP_MW': 'electricity_consumption'
})

# 3. Extract the energy values
energy_orig = df['electricity_consumption'].values
mu = np.mean(energy_orig)
std = np.std(energy_orig)

# 4. Define Shaving Bounds
# Using 5th and 95th percentiles ensures the range is realistic for the AEP grid
e_min = np.percentile(energy_orig, 5)
e_max = np.percentile(energy_orig, 95)

# 5. Transform Gaussian n(E) -> Flat n(E)
# The CDF (Cumulative Distribution Function) maps the bell curve to a uniform [0, 1]
uniform_mapped = norm.cdf(energy_orig, loc=mu, scale=std)

# Scale the uniform [0, 1] values to our desired [e_min, e_max] range
energy_fraud = e_min + (e_max - e_min) * uniform_mapped

# 6. Update the column and save to CSV
df['electricity_consumption'] = energy_fraud
df.to_csv(output_path, index=False)

print(f"File saved: {output_path}")
print(f"Final columns: {list(df.columns)}")