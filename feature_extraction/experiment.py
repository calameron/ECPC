# %% [markdown]
# # **Energy consumption - EDA Project**
# 
# ---
# Objective:Analyze the data (energy consumption) and extract features for labeling, and creating a feature-vector. This vector will be used for further labeling.
#
# Input: DateTime, energy_consumption. 
#
# The date-time also reflects seasonality and time of day. 
# it is widely believed that there is correlation between those parameters and electric energy consumption,
#  therefore energy consumption may be future forecasted through time-series regression machine learning. 
# The seasonality/day-of-week {Monday-Friday,Saturday,Sunday}/time of day is named professionally Time-Of-Use, and affects electricity Tariff time. 
# This is called professionally electricity load forecasting.
# The data is hourly/quarter-hourlySo it's electricity-consumption=f(day and time,  var2. Var2 is the peak demand segmentation. it is anually split into: on-going continuous (A), 100 peak-demand anuall hours (B), and the rest of time (C). Project is intending to perform time-series load-forecasting. this paragraph however focus on preliminary analytics


# Input data files are available in the "../input/" directory.

# %% importing libraries
import os
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt # plotting
from sklearn.preprocessing import MinMaxScaler
from sklearn.decomposition import PCA
from sklearn.mixture import GaussianMixture
# settings:
# visualization settings
plt.style.use('default')  # use default style for plots
# display settings
pd.options.display.float_format = '{:.2e}'.format   # display floats in scientific notation with 3 decimal places
# %% IO variables
# input path
input_dir = "./input_data/"
input_filename = "AEP_hourly.csv"
input_path = input_dir + input_filename
# create output directory
output_dir = "./graphs/" + input_filename.split(".")[0] + "/"
if not os.path.exists(output_dir):
    os.makedirs(output_dir)


# %% [markdown]
# ## **1. Data loading and preliminary analysis**
# load data and sort by date-time (ascending). 
#
# Be aware that the data is shortened currently to 2004, for the sake of faster processing and visualization.
#
# %% read data
df_io = pd.read_csv(input_path, parse_dates=['datetime'], index_col='datetime')
df_training = df_io.sort_values(by="datetime")
# Rename the first column to the standard name
df_training.columns.values[0] = "electricity_consumption"
# get subset of data for faster processing and visualization
df_training_partial = df_training.loc['2004':'2010']
# print data loading status
print("Data loaded successfully!")
# %% [markdown]
# **Statistical Overview of data**
# %% display first few rows of data
head = df_training.head()
stats = df_training.describe().T
print("First few rows of data:")
print(head)
print("Statistical Overview:")
print(stats)

# %% [markdown]
# ## **Distribution of electricity load consumption:**

# %%
# Force inline rendering to prevent GUI thread crashes
import matplotlib.pyplot as plt
import os
import pandas as pd

# Force numeric type; turns any hidden strings/errors into NaN
numeric_data = pd.to_numeric(df_training['electricity_consumption'], errors='coerce')
clean_data = numeric_data.dropna()

bin_count = 100
plt.figure(figsize=(12, 6))

# Generate histogram
n, bins, _ = plt.hist(clean_data, bins=bin_count, histtype='stepfilled', alpha=0.9)

plt.xlabel('Electricity Consumption')
plt.ylabel('Frequency')
plt.title('Distribution of Electricity Load Consumption')

# Save the plot
save_path = os.path.join(output_dir, "distribution.png")
plt.savefig(save_path)
print(f"Graph saved to: {save_path}")

# Show the plot
plt.show()

# Free the memory immediately after displaying
plt.close()

# %% [markdown]
# ## **Distribution of electricity load consumption:**
# %%
bin_count = 100
plt.figure(figsize=(12, 6))
n, bins, _ = plt.hist(df_training['electricity_consumption'], bins=bin_count, histtype='stepfilled', alpha=0.9)
plt.xlabel('Electricity Consumption')
plt.ylabel('Frequency')
plt.title('Distribution of Electricity Load Consumption')


plt.show()
# save the plot
save_path = os.path.join(output_dir, "distribution.png")
plt.savefig(save_path)
print(f"Graph saved to: {save_path}")

# %% Calculate GMM of raw data, and extract features: hills, peaks, covariances.
def makeGMM(data, n_clusters=3):
    """Cluster the electricity consumption data into separate gaussians."""   
    # Reshape to 2D for sklearn (n_samples, n_features)
    X = consumption_data.reshape(-1, 1)
    
    # Fit Gaussian Mixture Model
    gmm = GaussianMixture(n_components=n_clusters, random_state=42).fit(X)
    labels = gmm.predict(X)
    
    # Add cluster labels to data
    data_with_clusters = data.copy()
    data_with_clusters['cluster'] = labels
    
    # Plot clustering result
    plt.figure(figsize=(12, 6))
    plt.scatter(range(len(consumption_data)), consumption_data, c=labels, cmap='viridis', s=30, alpha=0.6)
    plt.title(f"GMM Clustering (n={n_clusters})")
    plt.xlabel("Sample Index")
    plt.ylabel("Electricity Consumption (kWatt)")
    plt.colorbar(label="Cluster")
    plt.show()
    
    # Extract GMM parameters
    hills, peaks, covars = find_values_gmm(data_with_clusters, gmm, consumption_data)
    return (hills, peaks, covars, gmm)

def find_values_gmm(data, gmm, consumption_data):
    """Extract GMM statistics: cluster sizes, peaks, and covariances."""
    hills = []
    peaks = []
    covars = []
    
    # Get cluster information
    for i in range(gmm.n_components):
        # Get points in this cluster
        cluster_mask = data['cluster'] == i
        cluster_points = consumption_data[cluster_mask]
        
        if len(cluster_points) > 0:
            # Hill size: number of points in this cluster
            hills.append(len(cluster_points))
            # Peak: maximum consumption in this cluster
            peaks.append(np.max(cluster_points))
            # Covariance: variance of this cluster
            covars.append(gmm.covariances_[i][0, 0])
        else:
            hills.append(0)
            peaks.append(0)
            covars.append(0)
    
    return (hills, peaks, covars)

# Execution
print("\n=== Starting Gaussian Mixture Model Analysis ===\n")
hills, peaks, covars, gmm = makeGMM(df_training, n_clusters=3)
print("\nResults:")
print("Hill Sizes (cluster counts):", hills)
print("Peak Values (max per cluster):", peaks)
print("Covariances (cluster variances):", covars)
print("\nGMM Information:")
print("Number of components:", gmm.n_components)
print("Weights:", gmm.weights_)
print("Means:", gmm.means_)