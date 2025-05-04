import pandas as pd
import matplotlib.pyplot as plt
import glob
import os
import re # For extracting speed
import sys # To check for baseline files
import numpy as np # For calculations like mean and polyfit
from scipy.stats import pearsonr # For correlation coefficient

# --- Configuration ---
DATA_DIR = '.' # Directory containing the CSV files (current directory)
BASELINE_FILES = {
    # Define the order you want baselines to appear in plots
    'Ground': 'ground.csv',
    'Normal Stand': 'normal_stand.csv',
    'Cooling Stand': 'cooling_stand.csv', # Cooling stand with fans OFF
}
TEST_FILE_PATTERN = 'cooling_[Tt]est_*.csv' # Match both 'Test' and 'test'
OUTPUT_DIR = './cooling_plots' # Optional: Directory to save plots
STEADY_STATE_START_TIME = 550 # seconds - **ADJUST** based on test duration/stabilization
BASELINE_FOR_DELTA_PLOT = 'Normal Stand' # Choose which baseline to compare against ('Ground', 'Normal Stand')

# --- Helper Functions ---
def extract_speed(filename):
    """Extracts numeric speed from filenames like 'cooling_Test_1200.csv' or 'cooling_test_500.csv'."""
    match = re.search(r'_(\d+)\.csv$', filename, re.IGNORECASE)
    if match:
        try:
            return int(match.group(1))
        except ValueError:
            return None
    return None

def process_csv(filepath, label):
    """Loads, cleans, and processes a single CSV file."""
    try:
        df = pd.read_csv(filepath, parse_dates=['Timestamp'])
        if df.empty:
            print(f"Warning: File is empty: {filepath}")
            return None, None, None

        required_cols = ['Timestamp', 'CPU_Temp', 'CPU_Temp2', 'GPU_Temp']
        if not all(col in df.columns for col in required_cols):
            print(f"Warning: Missing required columns in {filepath}. Required: {required_cols}. Found: {df.columns.tolist()}. Skipping.")
            return None, None, None

        for col in ['CPU_Temp', 'CPU_Temp2', 'GPU_Temp']:
             df[col] = pd.to_numeric(df[col], errors='coerce')

        df.dropna(subset=['CPU_Temp', 'CPU_Temp2', 'GPU_Temp'], inplace=True)

        if df.empty:
            print(f"Warning: No valid numeric temperature data in {filepath} after cleaning.")
            return None, None, None

        df['CPU_Avg_Temp'] = df[['CPU_Temp', 'CPU_Temp2']].mean(axis=1)
        df.sort_values('Timestamp', inplace=True)
        start_time = df['Timestamp'].iloc[0]
        df['Elapsed_Time'] = (df['Timestamp'] - start_time).dt.total_seconds()
        df['Label'] = label

        # Extract speed if applicable
        speed = extract_speed(os.path.basename(filepath))

        return df, label, speed
    except FileNotFoundError:
        print(f"Error: File not found: {filepath}")
        return None, None, None
    except Exception as e:
        print(f"Error processing file {filepath}: {e}")
        return None, None, None

# --- Main Script Logic ---
all_data = {} # Store processed dataframes {label: df}
speeds_data = {} # Store data for RPM tests {speed: df}
found_baseline_files = {}

# 1. Process Baseline Files
print("Processing baseline files...")
missing_baseline = False
for label, filename in BASELINE_FILES.items():
    filepath = os.path.join(DATA_DIR, filename)
    if not os.path.exists(filepath):
        print(f"Error: Baseline file not found: {filepath}. Please ensure it exists.")
        missing_baseline = True
    else:
        df, processed_label, _ = process_csv(filepath, label) # Speed is None for baselines
        if df is not None:
            found_baseline_files[processed_label] = df
            print(f"  Processed: {filename} as '{processed_label}'")

if missing_baseline:
    print("Exiting due to missing baseline file(s).")
    sys.exit(1)

# Add found baseline data to the main dictionary
all_data.update(found_baseline_files)

# 2. Process Test Files
print("\nProcessing test files...")
test_files = glob.glob(os.path.join(DATA_DIR, TEST_FILE_PATTERN))
# Filter out baseline filenames if they accidentally match the pattern
test_files = [f for f in test_files if os.path.basename(f) not in BASELINE_FILES.values()]

if not test_files:
    print(f"Warning: No test files found matching pattern '{TEST_FILE_PATTERN}' in {DATA_DIR}")

# Use a dictionary to store test data keyed by speed
test_data_by_speed = {}

for filepath in test_files:
    filename = os.path.basename(filepath)
    # We still use extract_speed to find the RPM value
    speed_val = extract_speed(filename)
    if speed_val is not None:
        label = f"{speed_val} RPM"
        df, processed_label, speed_out = process_csv(filepath, label)
        if df is not None and speed_out is not None:
            # Use speed_val (the extracted integer) as the key
            if speed_val in test_data_by_speed:
                 print(f"  Warning: Multiple files found for speed {speed_val} RPM. Using data from: {filename}")
            test_data_by_speed[speed_val] = (df, processed_label) # Store df and its label
            print(f"  Processed: {filename} (Speed: {speed_val} RPM)")
    else:
         # Check if it's the cpu_gpu_test.csv file or other non-speed files
        if 'cpu_gpu_test.csv' in filename or 'cooling_pad_tests.csv' in filename :
             print(f"  Skipping non-speed test file: {filename}")
        else:
             print(f"  Warning: Could not extract speed from filename: {filename}. Skipping.")


# Add the processed test data (unique by speed) to all_data and speeds_data
# Sort by speed for consistent plotting order
sorted_speeds = sorted(test_data_by_speed.keys())
for speed in sorted_speeds:
    df, label = test_data_by_speed[speed]
    all_data[label] = df
    speeds_data[speed] = df # Store only RPM data separately

# Check if we have any RPM data to plot against speed
if not speeds_data:
    print("\nWarning: No valid RPM test data found. Speed-related plots will be skipped.")

# Check if we have any data beyond baseline to plot
if len(all_data) <= len(found_baseline_files):
    print("\nError: No valid test data successfully processed. Only baseline data found. Exiting.")
    sys.exit(1)

print(f"\nProcessed {len(all_data)} total datasets for plotting.")
print("Dataset labels:", list(all_data.keys()))


# --- Hypothesis Statement ---
print("\n--- Hypothesis ---")
print("Null Hypothesis (H₀): Increasing fan speed does not decrease steady-state CPU/GPU temperature (correlation is ≥ 0).")
print("Alternative Hypothesis (H₁): Increasing fan speed decreases steady-state CPU/GPU temperature (correlation is < 0).")
print("--------------------")


# Optional: Create output directory
if OUTPUT_DIR:
    try:
        os.makedirs(OUTPUT_DIR, exist_ok=True)
        print(f"\nPlots will be saved to: {OUTPUT_DIR}")
    except OSError as e:
        print(f"Error creating output directory {OUTPUT_DIR}: {e}. Plots will not be saved.")
        OUTPUT_DIR = None # Disable saving


# --- Calculate Steady State Means and Prepare Data ---
print(f"\nCalculating steady-state means (data after {STEADY_STATE_START_TIME} seconds)...")
steady_state_means = {}
valid_labels_for_steady_state = []

# Define plot order: Baselines first, then RPMs sorted numerically
plot_order = list(BASELINE_FILES.keys()) + [f"{speed} RPM" for speed in sorted_speeds]

for label in plot_order:
    if label in all_data:
        df = all_data[label]
        steady_df = df[df['Elapsed_Time'] >= STEADY_STATE_START_TIME].copy()
        if not steady_df.empty:
            mean_cpu = steady_df['CPU_Avg_Temp'].mean()
            mean_gpu = steady_df['GPU_Temp'].mean()
            steady_state_means[label] = {'CPU': mean_cpu, 'GPU': mean_gpu}
            valid_labels_for_steady_state.append(label)
            print(f"  {label}: Mean CPU = {mean_cpu:.2f}°C, Mean GPU = {mean_gpu:.2f}°C")
        else:
            print(f"  Warning: No steady-state data >= {STEADY_STATE_START_TIME}s found for '{label}'. Skipping for means/boxplots.")

# Filter plot_order to only include labels with valid steady state data
plot_order_steady = [label for label in plot_order if label in valid_labels_for_steady_state]


# --- Plotting ---
print("\nGenerating plots...")

# Function to generate a color map for consistency
def get_color_map(labels):
    color_cycle = plt.rcParams['axes.prop_cycle'].by_key()['color'] # Use standard matplotlib cycle
    # Add more colors if needed, e.g., from a colormap
    if len(labels) > len(color_cycle):
        cmap = plt.get_cmap('tab20') # Or 'viridis', 'plasma' etc.
        extra_colors = [cmap(i/len(labels)) for i in range(len(labels))]
        color_cycle = extra_colors # Override if needed
    colors = {label: color_cycle[i % len(color_cycle)] for i, label in enumerate(labels)}
    return colors

# Use the order that includes ALL successfully processed files for time series plots
color_map = get_color_map(plot_order)
# Use the order that includes only those with STEADY STATE data for summary plots
color_map_steady = get_color_map(plot_order_steady)


# Plot 1: CPU Temperature vs. Elapsed Time (Existing)
plt.figure(figsize=(14, 8))
for label in plot_order: # Use original plot_order here
    if label in all_data:
        df = all_data[label]
        plt.plot(df['Elapsed_Time'], df['CPU_Avg_Temp'], label=label, color=color_map.get(label, 'gray')) # Use .get for safety
plt.xlabel("Elapsed Time (seconds)")
plt.ylabel("Average CPU Temperature (°C)")
plt.title("CPU Temperature vs. Time Under Different Cooling Conditions")
plt.legend(loc='center left', bbox_to_anchor=(1, 0.5))
plt.grid(True, linestyle='--', alpha=0.6)
plt.tight_layout(rect=[0, 0, 0.85, 1])
if OUTPUT_DIR:
    plt.savefig(os.path.join(OUTPUT_DIR, 'cpu_temp_vs_time.png'))
plt.show()

# Plot 2: GPU Temperature vs. Elapsed Time (Existing)
plt.figure(figsize=(14, 8))
for label in plot_order: # Use original plot_order here
     if label in all_data:
        df = all_data[label]
        plt.plot(df['Elapsed_Time'], df['GPU_Temp'], label=label, color=color_map.get(label, 'gray'))
plt.xlabel("Elapsed Time (seconds)")
plt.ylabel("GPU Temperature (°C)")
plt.title("GPU Temperature vs. Time Under Different Cooling Conditions")
plt.legend(loc='center left', bbox_to_anchor=(1, 0.5))
plt.grid(True, linestyle='--', alpha=0.6)
plt.tight_layout(rect=[0, 0, 0.85, 1])
if OUTPUT_DIR:
    plt.savefig(os.path.join(OUTPUT_DIR, 'gpu_temp_vs_time.png'))
plt.show()

# Plot 3: Box Plot Comparison (Existing, but uses filtered plot_order_steady)
if len(plot_order_steady) > 1 :
    cpu_data_for_boxplot = []
    gpu_data_for_boxplot = []
    labels_for_boxplot = []

    for label in plot_order_steady: # Use filtered list
        df = all_data[label]
        steady_df = df[df['Elapsed_Time'] >= STEADY_STATE_START_TIME]
        if not steady_df.empty: # Should be true based on how plot_order_steady was created
            cpu_data_for_boxplot.append(steady_df['CPU_Avg_Temp'].dropna())
            gpu_data_for_boxplot.append(steady_df['GPU_Temp'].dropna())
            labels_for_boxplot.append(label)

    if cpu_data_for_boxplot: # Check if list is not empty
        fig, axes = plt.subplots(1, 2, figsize=(18, 7), sharey=False)

        bp1 = axes[0].boxplot(cpu_data_for_boxplot, labels=labels_for_boxplot, patch_artist=True, showfliers=False)
        axes[0].set_title(f'CPU Temperature Distribution (After {STEADY_STATE_START_TIME}s)')
        axes[0].set_ylabel('Average CPU Temperature (°C)')
        axes[0].tick_params(axis='x', rotation=45, labelsize=9)
        axes[0].grid(True, axis='y', linestyle='--', alpha=0.6)
        for patch, label in zip(bp1['boxes'], labels_for_boxplot):
             patch.set_facecolor(color_map_steady.get(label, 'gray'))
             patch.set_alpha(0.7)

        bp2 = axes[1].boxplot(gpu_data_for_boxplot, labels=labels_for_boxplot, patch_artist=True, showfliers=False)
        axes[1].set_title(f'GPU Temperature Distribution (After {STEADY_STATE_START_TIME}s)')
        axes[1].set_ylabel('GPU Temperature (°C)')
        axes[1].tick_params(axis='x', rotation=45, labelsize=9)
        axes[1].grid(True, axis='y', linestyle='--', alpha=0.6)
        for patch, label in zip(bp2['boxes'], labels_for_boxplot):
            patch.set_facecolor(color_map_steady.get(label, 'gray'))
            patch.set_alpha(0.7)

        plt.suptitle('Steady-State Temperature Distribution Comparison', fontsize=16)
        plt.tight_layout(rect=[0, 0.05, 1, 0.95]) # Adjust rect for rotated labels and title
        if OUTPUT_DIR:
            plt.savefig(os.path.join(OUTPUT_DIR, 'temp_distribution_boxplot.png'))
        plt.show()
    else:
        print("\nSkipping box plots: No datasets had sufficient steady-state data.")
elif len(plot_order_steady) <=1 :
      print("\nSkipping box plots as fewer than two datasets had valid steady-state data.")


# --- NEW PLOTS ---

# Plot 4: Average Steady-State Temperature Bar Chart
if steady_state_means and len(plot_order_steady) > 0:
    cpu_means = [steady_state_means[label]['CPU'] for label in plot_order_steady]
    gpu_means = [steady_state_means[label]['GPU'] for label in plot_order_steady]
    x = np.arange(len(plot_order_steady)) # label locations
    width = 0.35 # width of the bars

    fig, ax = plt.subplots(figsize=(14, 7))
    rects1 = ax.bar(x - width/2, cpu_means, width, label='CPU Avg Temp', color=[color_map_steady.get(l,'grey') for l in plot_order_steady], alpha=0.8)
    rects2 = ax.bar(x + width/2, gpu_means, width, label='GPU Temp', color=[color_map_steady.get(l,'grey') for l in plot_order_steady], alpha=0.6) # Slightly different alpha

    ax.set_ylabel('Average Temperature (°C)')
    ax.set_title(f'Average Steady-State Temperature (After {STEADY_STATE_START_TIME}s)')
    ax.set_xticks(x)
    ax.set_xticklabels(plot_order_steady, rotation=45, ha='right')
    ax.legend()
    ax.grid(True, axis='y', linestyle='--', alpha=0.6)

    # Add labels on bars
    ax.bar_label(rects1, padding=3, fmt='%.1f')
    ax.bar_label(rects2, padding=3, fmt='%.1f')

    fig.tight_layout()
    if OUTPUT_DIR:
        plt.savefig(os.path.join(OUTPUT_DIR, 'steady_state_average_temps_bar.png'))
    plt.show()
else:
    print("\nSkipping Average Steady-State Temperature Bar Chart: No steady-state data available.")


# Plot 5: Temperature vs. Fan Speed Scatter Plot (and Correlation)
rpm_labels = [label for label in plot_order_steady if "RPM" in label] # Get only RPM labels with steady state data
if rpm_labels and len(rpm_labels) >= 2: # Need at least 2 points for correlation/line
    rpms = [int(re.search(r'(\d+)', label).group(1)) for label in rpm_labels]
    cpu_temps_at_rpm = [steady_state_means[label]['CPU'] for label in rpm_labels]
    gpu_temps_at_rpm = [steady_state_means[label]['GPU'] for label in rpm_labels]

    fig, axes = plt.subplots(1, 2, figsize=(15, 6), sharex=True)

    # CPU vs RPM
    axes[0].scatter(rpms, cpu_temps_at_rpm, color='red', label='CPU Data Points')
    # Fit and plot trendline for CPU
    try:
        coeffs_cpu = np.polyfit(rpms, cpu_temps_at_rpm, 1) # Linear fit
        poly_cpu = np.poly1d(coeffs_cpu)
        rpm_line = np.linspace(min(rpms), max(rpms), 100)
        axes[0].plot(rpm_line, poly_cpu(rpm_line), linestyle="--", color='darkred', label=f'CPU Trend (y={coeffs_cpu[0]:.3f}x + {coeffs_cpu[1]:.1f})')
        # Calculate Correlation
        corr_cpu, p_cpu = pearsonr(rpms, cpu_temps_at_rpm)
        axes[0].set_title(f'CPU Steady State Temp vs. Fan Speed\nCorrelation: {corr_cpu:.3f}') #(p={p_cpu:.3f})') # Add p-value if desired
    except np.linalg.LinAlgError:
         axes[0].set_title(f'CPU Steady State Temp vs. Fan Speed\n(Trendline fit failed)')
         corr_cpu = np.nan
         print("Warning: Could not fit trendline for CPU vs RPM.")
    except ValueError as ve:
         axes[0].set_title(f'CPU Steady State Temp vs. Fan Speed\n(Correlation error: {ve})')
         corr_cpu = np.nan
         print(f"Warning: Could not calculate correlation for CPU vs RPM: {ve}")


    axes[0].set_ylabel('Average CPU Temperature (°C)')
    axes[0].set_xlabel('Fan Speed (RPM)')
    axes[0].legend()
    axes[0].grid(True, linestyle='--', alpha=0.6)

    # GPU vs RPM
    axes[1].scatter(rpms, gpu_temps_at_rpm, color='blue', label='GPU Data Points')
    # Fit and plot trendline for GPU
    try:
        coeffs_gpu = np.polyfit(rpms, gpu_temps_at_rpm, 1) # Linear fit
        poly_gpu = np.poly1d(coeffs_gpu)
        axes[1].plot(rpm_line, poly_gpu(rpm_line), linestyle="--", color='darkblue', label=f'GPU Trend (y={coeffs_gpu[0]:.3f}x + {coeffs_gpu[1]:.1f})')
         # Calculate Correlation
        corr_gpu, p_gpu = pearsonr(rpms, gpu_temps_at_rpm)
        axes[1].set_title(f'GPU Steady State Temp vs. Fan Speed\nCorrelation: {corr_gpu:.3f}')# (p={p_gpu:.3f})')
    except np.linalg.LinAlgError:
        axes[1].set_title(f'GPU Steady State Temp vs. Fan Speed\n(Trendline fit failed)')
        corr_gpu = np.nan
        print("Warning: Could not fit trendline for GPU vs RPM.")
    except ValueError as ve:
         axes[1].set_title(f'GPU Steady State Temp vs. Fan Speed\n(Correlation error: {ve})')
         corr_gpu = np.nan
         print(f"Warning: Could not calculate correlation for GPU vs RPM: {ve}")

    axes[1].set_ylabel('GPU Temperature (°C)')
    axes[1].set_xlabel('Fan Speed (RPM)')
    axes[1].legend()
    axes[1].grid(True, linestyle='--', alpha=0.6)

    plt.suptitle('Steady-State Temperature Dependence on Fan Speed', fontsize=16)
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    if OUTPUT_DIR:
        plt.savefig(os.path.join(OUTPUT_DIR, 'temp_vs_fan_speed_scatter.png'))
    plt.show()

    # Print correlation summary related to hypothesis
    print("\n--- Correlation Analysis (RPM vs Steady State Temp) ---")
    print(f"CPU: Pearson Correlation Coefficient = {corr_cpu:.3f}")
    print(f"GPU: Pearson Correlation Coefficient = {corr_gpu:.3f}")
    if not np.isnan(corr_cpu) and corr_cpu < -0.5: print("  Strong negative correlation found for CPU, supporting the alternative hypothesis.")
    elif not np.isnan(corr_cpu) and corr_cpu < -0.1: print("  Weak/Moderate negative correlation found for CPU, leaning towards the alternative hypothesis.")
    else: print("  No significant negative correlation found for CPU (or correlation is positive/near zero), fails to reject the null hypothesis.")
    if not np.isnan(corr_gpu) and corr_gpu < -0.5: print("  Strong negative correlation found for GPU, supporting the alternative hypothesis.")
    elif not np.isnan(corr_gpu) and corr_gpu < -0.1: print("  Weak/Moderate negative correlation found for GPU, leaning towards the alternative hypothesis.")
    else: print("  No significant negative correlation found for GPU (or correlation is positive/near zero), fails to reject the null hypothesis.")
    print("-------------------------------------------------------")

else:
    print("\nSkipping Temperature vs. Fan Speed plot: Not enough RPM data points with steady-state results (need >= 2).")

print("\nProcessing complete.")
D
