import os
import pandas as pd
import csv
import matplotlib.pyplot as plt
#for cifar
#base_dir = "/home/wassal/trust-wassal/tutorials/results/cifar10/classimb"
#budgets=['50', '100', '150', '200', '250', '300', '350', '400', '450', '500']
filename = "output_statistics_cifr10_classimb_withAL"

#for pneumonia
base_dir = "/home/wassal/trust-wassal/tutorials/results/pneumoniamnist/classimb"
budgets=['5', '10', '15', '20', '25']
filename = "output_statistics_pneu_classimb_withAL"

strategies = ["WASSAL", "WASSAL_P", "fl1mi", "fl2mi", "gcmi", "logdetmi", "random","badge","us","glister","coreset","glister","gradmatch-tss","leastconf","logdetcmi","flcmi","margin"]


experiments=['exp1']
rounds=5
# Prepare the CSV file for saving stats
output_path = os.path.join(base_dir, filename+"rounds_"+str(rounds)+".csv")


def compute_stats(gains):
    mean_gain = sum(gains) / len(gains)
    variance = sum([(gain - mean_gain) ** 2 for gain in gains]) / len(gains)
    sd_gain = variance ** 0.5
    return mean_gain, variance, sd_gain



with open(output_path, "w", newline='') as csvfile:
    writer = csv.writer(csvfile)
    writer.writerow(["Strategy", "Budget", "Mean Gain", "Variance", "Standard Deviation"])

    # Store data in dictionaries
    data = {}
    
    #for each budget
    for budget in budgets:
         
        #for each strategy
        for strategy in strategies:
           # Reset these lists for each strategy
            means = []
            sds = []
            budgets_list = []  # renamed to avoid conflict with 'budgets'
            
            #continue if a path does not exist with strategy and budget
            if not os.path.exists(os.path.join(base_dir, strategy,str(budget))):
                continue

            #calculate mean and sd of strategy across all runs
            gains = []
            for experiment in experiments:
                path = os.path.join(base_dir, strategy,str(budget),experiment)
                if not os.path.exists(path):
                    continue
                for csv_file in os.listdir(path):
                    if csv_file.endswith('.csv'):
                        csv_path = os.path.join(path, csv_file)
                    else:
                        continue
                    df = pd.read_csv(csv_path,header=None)
                    y1 = df.iloc[0, 1]
                    y2 = df.iloc[1, 1]
                    gain = y2 - y1
                    gains.append(gain)
                if not gains:
                    continue
            
             # Compute stats after processing all experiments for the current strategy and budget
            mean_gain, variance, sd_gain = compute_stats(gains)
            writer.writerow([strategy, budget, mean_gain, variance, sd_gain])
            print(f"Strategy: {strategy}, Budget: {budget}")
            print("Mean Gain:", mean_gain)
            print("Variance:", variance)
            print("Standard Deviation of Gain:", sd_gain)
            print("----------------------------------------------------")

            # Save data to the dictionary
            if gains:
                
                if strategy not in data:
                    data[strategy] = {'means': [], 'sds': [], 'budgets': []}
                data[strategy]['means'].append(mean_gain)
                data[strategy]['sds'].append(sd_gain)
                data[strategy]['budgets'].append(budget)
    print(f"Statistics saved to {output_path}")

    # Plot data
# Plot data
plt.figure(figsize=(10, 6))

# Loop through the data dictionary to extract the values
for strategy, values in data.items():
    plt.plot(values['budgets'], values['means'], label=strategy)
    #plt.errorbar(values['budgets'], values['means'],values['sds'], label=strategy)

plt.xlabel('Budget')
plt.ylabel('Mean Gain for rare class')
plt.title('Mean Gain by Strategy for '+str(rounds)+'AL rounds')
plt.legend()
plt.grid(True, which='both', linestyle='--', linewidth=0.5)
plt.tight_layout()
plt.savefig(os.path.join(base_dir, filename+"rounds_"+str(rounds)+".png"), dpi=300, bbox_inches='tight', pad_inches=0.1)
plt.show()

