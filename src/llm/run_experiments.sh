#!/bin/bash

# Parameter values to iterate over
K_VALUES=(10)  # Number of examples k
NOISE_LEVELS=(0.1 0.2 0.3 0.4 0.5 0.6 1)  # Noise levels
# NUMBER_SUBSETS=(20 40 100 263)  # Size of the dataset subset to use

# Create log directory if it doesn't exist
mkdir -p run_logs

N=523  # Initial log file number

# Iterate over all parameter combinations
for noise_level in "${NOISE_LEVELS[@]}"; do
# for subset in "${NUMBER_SUBSETS[@]}"; do
  for k in "${K_VALUES[@]}"; do

    # Define log filename dynamically
    LOG_FILE="run_logs/inference${N}.log"

    # Run the Python script with parameters
    echo "Running inference with k=$k, noise_level=$noise_level"
    CUDA_VISIBLE_DEVICES=1,3 python inference_sh.py --k "$k" --noise_level "$noise_level" &> "$LOG_FILE"
    
    # echo "Running inference with k=$k, subset=$subset"
    # CUDA_VISIBLE_DEVICES=1,2 python inference2.py --k "$k" --subset "$subset" &> "$LOG_FILE"

    # Increment log file number
    ((N++))

  done
done

echo "All experiments completed!"
