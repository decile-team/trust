python3 tutorials/All_Wassal/wassal_cifar10_multiclass_vanilla.py 2>&1 | tee tutorials/results/cifar10.log
#!/bin/bash

# Define the skip_strategies and skip_budgets
SKIP_STRATEGIES="strategy1 strategy2"
SKIP_BUDGETS="100 200"

python3 informme.py