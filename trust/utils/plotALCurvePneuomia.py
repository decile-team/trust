import matplotlib.pyplot as plt
import pandas as pd

# File paths
us_filepath = "/home/wassal/trust-wassal/tutorials/results/pneumoniamnist/classimb/rounds10/coreset/40/exp2/pneumoniamnist_classimb_AL_2_coreset_budget:40_rounds:10_runsexp2.csv"
us_soft_filepath = "/home/wassal/trust-wassal/tutorials/results/pneumoniamnist/classimb/rounds10/coreset_soft/40/exp2/pneumoniamnist_classimb_AL_WITHSOFT_2_coreset_soft_budget:40_rounds:10_runsexp2.csv"

# Read data without headers and manually assign column names
column_names = ["Class0", "Class1", "Avg"]
# Assuming the data in each file is stored as comma-separated values
us_data = pd.read_csv(us_filepath ,header=None, names=column_names)
us_soft_data = pd.read_csv(us_soft_filepath, header=None, names=column_names)

# Extract the data for Class 0, Class 1, and Average for each approach
us_class0 = us_data['Class0'].values
us_class1 = us_data['Class1'].values
us_avg = us_data['Avg'].values

us_soft_class0 = us_soft_data['Class0'].values
us_soft_class1 = us_soft_data['Class1'].values
us_soft_avg = us_soft_data['Avg'].values

# Create a plot
plt.figure(figsize=(10, 6))

# Plot the data for Uncertainty Sampling
plt.plot(us_class0, label='Class 0', marker='o')
plt.plot(us_class1, label='Class 1', marker='o')
plt.plot(us_avg, label='US Avg', marker='o', linestyle='--')

# Plot the data for Uncertainty Sampling Soft Approach
plt.plot(us_soft_class0, label='Soft Class 0', marker='s')
plt.plot(us_soft_class1, label='Soft Class 1', marker='s')
plt.plot(us_soft_avg, label='US Soft Avg', marker='s', linestyle='--')

plt.title('Uncertainty Sampling vs. Uncertainty Sampling Soft Approach')
plt.xlabel('AL Rounds')
plt.ylabel('Value')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig('ALcurve.png')
