#!/bin/bash

# Define the skip_strategies and skip_budgets
SKIP_STRATEGIES="AL random"
SKIP_BUDGETS="25 50 75 100 200"
DEVICE_ID="1"
# Call the Python script with the defined arguments
python3 tutorials/All_Wassal/wassal_cifar10_multiclass_vanilla.py "$SKIP_STRATEGIES" "$SKIP_BUDGETS" "$DEVICE_ID" 2>&1 | tee tutorials/results/cifar10wassal_small.log
python3 informme.py