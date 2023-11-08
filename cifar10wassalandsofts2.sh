#!/bin/bash

# Define the skip_strategies and skip_budgets
SKIP_STRATEGIES="AL random WASSAL"
SKIP_METHODS="leastconf_withsoft margin_withsoft badge_withsoft us_withsoft"
SKIP_BUDGETS="25 50 75 100 200"
DEVICE_ID="1"
# Call the Python script with the defined arguments
python3 -u tutorials/All_Wassal/wassal_cifar10_multiclass_vanilla.py "$SKIP_STRATEGIES" "$SKIP_METHODS" "$SKIP_BUDGETS" "$DEVICE_ID" 2>&1 | tee tutorials/results/cifar10_10rounds_wassal2_small.log
python3 informme.py