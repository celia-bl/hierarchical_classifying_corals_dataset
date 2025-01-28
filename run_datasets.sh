#!/bin/bash
# SBATCH FOR CLUSTER

#SBATCH --job-name=test-RIO-MLC-TasCPC          # Job name
#SBATCH --output=output_%j.log                  # Output file (with job ID)
#SBATCH --error=error_%j.log                    # Error file (with job ID)
#SBATCH --time=82:00:00                         # Maximum run time (hh:mm:ss)
#SBATCH --partition=dfq                         # Partition to use (e.g., 'compute', 'gpu', etc.)
#SBATCH --nodes=1                               # Number of nodes
#SBATCH --ntasks=1                              # Number of tasks

echo "Running on: $SLURM_NODELIST"

# Bash script to run Python commands in the specified order

# Define the parameter files
params=("params-RIO.yaml" "params-MLC.yaml" "params-TasCPC.yaml")

# Loop to execute commands for each parameter file
for param_file in "${params[@]}"; do
    echo "Running main.py extract for $param_file"
    python3 main.py extract "./params/$param_file"

    # Check for errors after running main.py extract
    if [ $? -ne 0 ]; then
        echo "Error occurred while running main.py extract with $param_file. Exiting the script."
        exit 1
    fi

    echo "Running main.py train for $param_file"
    python3 main.py train "./params/$param_file"

    # Check for errors after running train.py
    if [ $? -ne 0 ]; then
        echo "Error occurred while running train.py with $param_file. Exiting the script."
        exit 1
    fi
done

# Final message if all commands succeed
echo "All commands executed successfully."
