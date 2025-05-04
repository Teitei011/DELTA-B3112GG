#!/bin/bash

# --- Configuration ---
STRESS_TIME=600          # Stress test duration in seconds
INTERVAL=5               # Temperature sampling interval in seconds

# --- Argument Handling ---
# Check if an argument was provided
if [ -z "$1" ]; then
  echo "Usage: $0 <TEST_NUMBER>"
  echo "Error: Please provide a number to identify this test run."
  exit 1
fi

# Check if the provided argument is a number (integer)
if ! [[ "$1" =~ ^[0-9]+$ ]]; then
  echo "Usage: $0 <TEST_NUMBER>"
  echo "Error: The provided argument '$1' is not a valid integer."
  exit 1
fi

TEST_NUMBER="$1"
#CSV_FILE="cooling_Test_${TEST_NUMBER}.csv"
CSV_FILE="cooling_stand.csv"

# --- Script Logic ---
echo "Starting cooling test number: $TEST_NUMBER"
echo "Output will be saved to: $CSV_FILE"
echo "Stress test duration: $STRESS_TIME seconds"
echo "Sampling interval: $INTERVAL seconds"
echo "---"


# Clear previous CSV file (if it exists)
> "$CSV_FILE"

# Log headers to CSV
echo "Timestamp,CPU_Temp,CPU_Temp2,GPU_Temp" > "$CSV_FILE"

# Run stress test in the background
echo "Starting stress test..."
stress-ng --cpu $(nproc) --timeout $STRESS_TIME &
STRESS_PID=$! # Store the PID of the stress-ng process

# Wait briefly for stress-ng to potentially start impacting temps
sleep 2

# Monitor temperatures
echo "Monitoring temperatures..."
while true; do
    # Check if stress test process is still running using its PID
    # This is more reliable than checking $! inside the loop,
    # as $! changes with every background command.
    if ! ps -p $STRESS_PID > /dev/null; then
        echo "Stress test process (PID: $STRESS_PID) finished."
        break # Exit the loop if stress-ng is done
    fi

    # Get current timestamp
    TIMESTAMP=$(date +"%Y-%m-%d %T")

    # Extract CPU temperatures (adjust selectors based on your hardware's 'sensors' output)
    # Using more robust awk for parsing to handle variations
    CPU_TEMP=$(sensors | awk '/Package id 0:/ {print $4}' | sed 's/+//;s/°C//')
    # Attempt to get Core 0 temp as CPU_TEMP2, adjust if needed
    CPU_TEMP2=$(sensors | awk '/Core 0:/ {print $3}' | sed 's/+//;s/°C//')

    # Extract GPU temperature (adjust grep pattern and awk field for your GPU if needed)
    # Example for NVIDIA using nvidia-smi (often more reliable than sensors for GPU)
    # If you don't have nvidia-smi or use AMD, adjust or comment out
    # GPU_TEMP=$(sensors | grep -E 'junction:|edge:' | grep -oE '[0-9]+\.[0-9]+' | head -n 1) # Example for some AMD GPUs via sensors
    GPU_TEMP=$(nvidia-smi --query-gpu=temperature.gpu --format=csv,noheader,nounits 2>/dev/null || echo "") # Example for NVIDIA

    # Handle cases where a temp might not be found
    CPU_TEMP=${CPU_TEMP:-"N/A"}
    CPU_TEMP2=${CPU_TEMP2:-"N/A"}
    GPU_TEMP=${GPU_TEMP:-"N/A"}


    # Append to CSV
    echo "$TIMESTAMP,$CPU_TEMP,$CPU_TEMP2,$GPU_TEMP" >> "$CSV_FILE"
    # Optional: print to console as well
    # echo "$TIMESTAMP | CPU1: $CPU_TEMP°C | CPU2: $CPU_TEMP2°C | GPU: $GPU_TEMP°C"


    # Wait for next interval
    sleep $INTERVAL

done # End of while loop

echo "---"
echo "Test completed. Results saved to $CSV_FILE"

exit 0
