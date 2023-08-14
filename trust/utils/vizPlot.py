import pandas as pd
import matplotlib.pyplot as plt
import os
# Load data from CSV
base_dir = "/home/wassal/trust-wassal/tutorials/results/cifar10/classimb"
output_path = os.path.join(base_dir, "output_statistics.csv")


# Create a scatter plot for mean gain
plt.figure(figsize=(10, 6))

for index, row in df.iterrows():
    label = row['Strategy']
    budget = row['Budget']
    mean_gain = row['Mean Gain']
    std_dev = row['Standard Deviation']

    plt.scatter(budget, mean_gain, label=f"{label} (SD: {std_dev:.2f})", s=50)

# Adding labels and title
plt.xlabel("Budget")
plt.ylabel("Mean Gain")
plt.title("Scatter plot of Mean Gain vs. Budget")
plt.legend(loc='upper left', bbox_to_anchor=(1,1), ncol=1)
plt.tight_layout()
plt.grid(True)
plt.savefig(os.path.join(base_dir, "output_statistics.png"), dpi=300, bbox_inches='tight', pad_inches=0.1)
# Show the plot
plt.show()
