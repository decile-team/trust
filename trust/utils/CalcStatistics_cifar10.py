import os
import pandas as pd
import csv
import matplotlib.pyplot as plt
#for cifar
#base_dir = "/home/wassal/trust-wassal/tutorials/results/cifar10/classimb"
#budgets=['50', '100', '150', '200']

#budgets=['5']
#filename = "output_statistics_cifar_classimb_withAL_"
rounds=5

#for pneumonia
base_dir = "/home/wassal/trust-wassal/tutorials/results/cifar10/classimb/rounds"+str(rounds)
#budgets=['5', '10', '15', '20', '25']
budgets=[100, 200]
filename = "output_statistics_cifar10_vanilla"

#strategies = ["WASSAL", "WASSAL_P", "fl1mi", "fl2mi", "gcmi", "logdetmi", "random","badge","us","glister","coreset","glister","gradmatch-tss","leastconf","logdetcmi","flcmi","margin"]
#strategy_group="allstrategies"
#strategies = ["WASSAL", "WASSAL_P","random","badge","us","glister","coreset","glister","gradmatch-tss","leastconf","margin"]
#strategy_group="AL"
#strategies = ["WASSAL_P","random","logdetcmi","flcmi"]
#strategy_group="withprivate"
#strategies = ["WASSAL",  "fl1mi", "fl2mi", "gcmi", "logdetmi","fl1mi_soft", "fl2mi_soft", "gcmi_soft", "logdetmi_soft", "random","WASSAL_P","logdetcmi","flcmi","logdetcmi_soft","flcmi_soft"]
#strategy_group="WASSAL_SOFT"
#strategies = ["random","badge","us","glister","coreset","glister","gradmatch-tss","leastconf","margin","badge_soft","us_soft","glister_soft","coreset_soft","glister_soft","gradmatch-tss_soft","leastconf_soft","margin_soft"]
strategies = ['us','us_soft','coreset','coreset_soft','leastconf','leastconf_soft','margin','margin_soft','random']
strategy_group="AL_WITH_SOFT"

#experiments=['exp1','exp2','exp3','exp4','exp5']
experiments=['exp1']

# Prepare the CSV file for saving stats
output_path = os.path.join(base_dir, filename+"_group_"+strategy_group+"_rounds_"+str(rounds))


def compute_stats(gains):
    mean_gain = sum(gains) / len(gains)
    variance = sum([(gain - mean_gain) ** 2 for gain in gains]) / len(gains)
    sd_gain = variance ** 0.5
    return mean_gain, variance, sd_gain

#mean gain of targeted class

with open(output_path+".csv", "w", newline='') as csvfile:
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
                    #cumlative gains from each row
                    #for i in range(0,5):
                    #gains=
                    y1 = df.iloc[0, 1]
                    y2 = df.iloc[rounds-1, 1]
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

# Define 10 distinct colors
# colors = [
#     '#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd',
#     '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf',
#     '#1fa2b4', '#b27f0e', '#a2a02c', '#662728', '#9467ad',
#     '#2c564b', '#e377a2', '#a27f7f', '#2c2d22', '#b7beaf'
# ]
#https://chat.openai.com/c/6b80df15-56ba-4584-aaee-38ecfbbdfc1d

colors = [
    '#FF0000',  # Red
    '#00FF00',  # Green
    '#0000FF',  # Blue
    '#FFFF00',  # Yellow
    '#00FFFF',  # Aqua
    '#FF00FF',  # Magenta
    '#FF4500',  # OrangeRed
    '#8A2BE2',  # BlueViolet
    '#A52A2A',  # Brown
    '#DEB887',  # BurlyWood
    '#5F9EA0',  # CadetBlue
    '#7FFF00',  # Chartreuse
    '#D2691E',  # Chocolate
    '#FF7F50',  # Coral
    '#6495ED',  # CornflowerBlue
    '#DC143C',  # Crimson
    '#00CED1',  # DarkTurquoise
    '#9400D3',  # DarkViolet
    '#FF1493',  # DeepPink
    '#00BFFF',  # DeepSkyBlue
]


    # Plot data
# Plot data
plt.figure(figsize=(10, 6))
color_index = 0

# Loop through the data dictionary to extract the values
for strategy, values in data.items():
    plt.plot(values['budgets'], values['means'], label=strategy,color=colors[color_index])
    # Add the strategy name to the end of the line using plt.text()
    x_pos = values['budgets'][-1]  # x-coordinate of the last point on the line
    y_pos = values['means'][-1]    # y-coordinate of the last point on the line
    plt.text(x_pos, y_pos, strategy, fontsize=12, color=colors[color_index])
    color_index += 1
    #plt.errorbar(values['budgets'], values['means'],values['sds'], label=strategy)

plt.xlabel('Budget')
plt.ylabel('Mean Gain for class 1')
plt.title('Mean Gain for class 1 for '+str(rounds)+'AL rounds')
plt.legend()
plt.grid(True, which='both', linestyle='--', linewidth=0.5)
plt.tight_layout()
plt.savefig(output_path+".png", dpi=300, bbox_inches='tight', pad_inches=0.1)
plt.show()

#mean gain of all classes
with open(output_path+"_allclasses.csv", "w", newline='') as csvfile:
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
                    #cumlative gains from each row
                    #for i in range(0,5):
                    #gains=
                    y1 = df.iloc[0, 2]
                    y2 = df.iloc[rounds-1, 2]
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
plt.figure(figsize=(10, 6))
color_index = 0

# Loop through the data dictionary to extract the values
for strategy, values in data.items():
    plt.plot(values['budgets'], values['means'], label=strategy,color=colors[color_index])
    # Add the strategy name to the end of the line using plt.text()
    x_pos = values['budgets'][-1]  # x-coordinate of the last point on the line
    y_pos = values['means'][-1]    # y-coordinate of the last point on the line
    plt.text(x_pos, y_pos, strategy, fontsize=12, color=colors[color_index])
    color_index += 1
    #plt.errorbar(values['budgets'], values['means'],values['sds'], label=strategy)

plt.xlabel('Budget')
plt.ylabel('Mean Gain for all classes')
plt.title('Mean Gain for all classes for '+str(rounds)+'AL rounds')
plt.legend()
plt.grid(True, which='both', linestyle='--', linewidth=0.5)
plt.tight_layout()
plt.savefig(output_path+"_allclasses.png", dpi=300, bbox_inches='tight', pad_inches=0.1)

#mean gain of majority class
with open(output_path+"_majorityclass.csv", "w", newline='') as csvfile:
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
                    #cumlative gains from each row
                    #for i in range(0,5):
                    #gains=
                    y1 = df.iloc[0, 0]
                    y2 = df.iloc[rounds-1, 0]
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
plt.figure(figsize=(10, 6))
color_index = 0

# Loop through the data dictionary to extract the values
for strategy, values in data.items():
    plt.plot(values['budgets'], values['means'], label=strategy,color=colors[color_index])
    # Add the strategy name to the end of the line using plt.text()
    x_pos = values['budgets'][-1]  # x-coordinate of the last point on the line
    y_pos = values['means'][-1]    # y-coordinate of the last point on the line
    plt.text(x_pos, y_pos, strategy, fontsize=12, color=colors[color_index])
    color_index += 1
    #plt.errorbar(values['budgets'], values['means'],values['sds'], label=strategy)

plt.xlabel('Budget')
plt.ylabel('Mean Gain for class 0')
plt.title('Mean Gain for class 0 for '+str(rounds)+'AL rounds')
plt.legend()
plt.grid(True, which='both', linestyle='--', linewidth=0.5)
plt.tight_layout()
plt.savefig(output_path+"_majorityclass.png", dpi=300, bbox_inches='tight', pad_inches=0.1)
