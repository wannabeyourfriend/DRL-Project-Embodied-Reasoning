#!/bin/bash

# --- Configuration ---
# Path to your modified Python script
PYTHON_SCRIPT_PATH="/cluster/home1/wzx/EgoReasoner/data/data_preprocess_imitation.py" # Assumes script is in the current directory

# Directory to scan for task folder names
# Subdirectories within this path will be used as TASK_TYPES
TASKS_DISCOVERY_DIR="/nfs/home1/wzx/EgoReasoner/data/egoreasoner/data/images"
# --- Configuration End ---

# Initialize TASK_TYPES array
TASK_TYPES=()

# Check if python3 is available
if ! command -v python3 &> /dev/null; then
    echo "ERROR: python3 could not be found. Please ensure it's installed and in your PATH."
    exit 1
fi

# Check if the Python script exists
if [ ! -f "$PYTHON_SCRIPT_PATH" ]; then
    echo "ERROR: Python script '$PYTHON_SCRIPT_PATH' not found!"
    echo "Please check the PYTHON_SCRIPT_PATH variable in this bash script."
    exit 1
fi

# Automatically discover task types (folder names)
echo "Discovering task types from: $TASKS_DISCOVERY_DIR"
if [ -d "$TASKS_DISCOVERY_DIR" ]; then
    # Use find to get directory names and mapfile (or readarray) to populate the array
    # -mindepth 1 and -maxdepth 1 ensure we only get immediate subdirectories
    # -type d specifies that we are looking for directories
    # -exec basename {} \; extracts just the directory name
    mapfile -t TASK_TYPES < <(find "$TASKS_DISCOVERY_DIR" -mindepth 1 -maxdepth 1 -type d -exec basename {} \; | sort)
    
    if [ ${#TASK_TYPES[@]} -eq 0 ]; then
        echo "Warning: No subdirectories found in '$TASKS_DISCOVERY_DIR'. No tasks to process."
    else
        echo "Found the following task types:"
        for task_name in "${TASK_TYPES[@]}"; do
            echo "  - $task_name"
        done
    fi
else
    echo "ERROR: Tasks discovery directory '$TASKS_DISCOVERY_DIR' not found."
    echo "Cannot automatically populate TASK_TYPES. Please check the TASKS_DISCOVERY_DIR path."
    exit 1 # Exit if the discovery directory doesn't exist
fi

# Exit if no tasks were found
if [ ${#TASK_TYPES[@]} -eq 0 ]; then
    echo "Exiting as no task types were found to process."
    exit 0
fi

echo ""
echo "Starting batch processing of datasets..."
echo "========================================"

# Loop through each discovered task type
for task_name in "${TASK_TYPES[@]}"; do
    echo ""
    echo "------------------------------------------------------------"
    echo "Processing task type: $task_name"
    echo "------------------------------------------------------------"

    # Call the Python script with the current task name as an argument
    python3 "$PYTHON_SCRIPT_PATH" "$task_name"
    exit_code=$? # Capture the exit code of the Python script

    if [ $exit_code -ne 0 ]; then
        echo ""
        echo "############################################################"
        echo "ERROR: Python script failed for task type: $task_name (Exit Code: $exit_code)"
        echo "############################################################"
        # Decide if you want to stop on error or continue with other tasks
        # To stop on first error, uncomment the next line:
        # exit 1
    else
        echo ""
        echo "Successfully processed task type: $task_name"
    fi
done

echo ""
echo "========================================"
echo "All specified tasks have been processed."
echo "========================================"

exit 0