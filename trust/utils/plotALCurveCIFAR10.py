import matplotlib.pyplot as plt
import pandas as pd

# File paths
us_filepath = "/home/wassal/trust-wassal/tutorials/results/cifar10/classimb/rounds5/coreset/100/exp1/cifar10_classimb_AL_10_coreset_budget:100_rounds:5_runsexp1.csv"
us_soft_filepath = "/home/wassal/trust-wassal/tutorials/results/cifar10/classimb/rounds5/coreset_soft/100/exp1/cifar10_classimb_AL_WITHSOFT_10_coreset_soft_budget:100_rounds:5_runsexp1.csv"

# Read data without headers and manually assign column names based on CIFAR10
column_names=["Class0", "Class1", "Class2", "Class3", "Class4", "Class5", "Class6", "Class7", "Class8", "Class9", "Avg"]

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
plt.plot(us_class0, label='US Class 0', marker='o')
plt.plot(us_class1, label='US Class 1', marker='o')
plt.plot(us_avg, label='US Avg', marker='o', linestyle='--')

# Plot the data for Uncertainty Sampling Soft Approach
plt.plot(us_soft_class0, label='US Soft Class 0', marker='s')
plt.plot(us_soft_class1, label='US Soft Class 1', marker='s')
plt.plot(us_soft_avg, label='US Soft Avg', marker='s', linestyle='--')

plt.title('Uncertainty Sampling vs. Uncertainty Sampling Soft Approach')
plt.xlabel('AL Rounds')
plt.ylabel('Value')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig('ALcurve.png')
