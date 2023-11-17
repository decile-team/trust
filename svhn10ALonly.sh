#!/bin/bash

# Define the skip_strategies and skip_budgets
SKIP_STRATEGIES="AL_WITHSOFT WASSAL WASSAL_WITHSOFT"
SKIP_METHODS=""
SKIP_BUDGETS="25 50 75 100 200"
DEVICE_ID="0"
# Call the Python script with the defined arguments
python3 tutorials/All_Wassal/wassal_svhn_multiclass_vanilla.py "$SKIP_STRATEGIES" "$SKIP_METHODS" "$SKIP_BUDGETS" "$DEVICE_ID" 2>&1 | tee tutorials/results/svhn_10rounds_al_small.log
python3 informme.py