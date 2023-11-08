#!/bin/bash

# Define the skip_strategies and skip_budgets
SKIP_STRATEGIES="AL random WASSAL"
SKIP_METHODS="WASSAL_WITHSOFT glister_withsoft gradmatch-tss_withsoft coreset_withsoft"
SKIP_BUDGETS="25 50 75 100 200"
DEVICE_ID="0"
# Call the Python script with the defined arguments
python3 -u tutorials/All_Wassal/wassal_svhn_multiclass_vanilla.py "$SKIP_STRATEGIES" "$SKIP_METHODS" "$SKIP_BUDGETS" "$DEVICE_ID" 2>&1 | tee  tutorials/results/svhn_10rounds_wassal1_small.log
python3 informme.py