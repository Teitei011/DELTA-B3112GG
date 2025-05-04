import pandas as pd
import matplotlib.pyplot as plt
import glob
import os
import re # For extracting speed
import sys # To check for baseline files
import numpy as np # For calculations like mean and polyfit
from scipy.stats import pearsonr # For correlation coefficient
import logging # Import the logging module

# --- Configuration ---
# MODIFIED: Point DATA_DIR to the 'csv' subdirectory
DATA_DIR = './csv' # Directory containing the CSV files
BASELINE_FILES = {
    # Define the order you want baselines to appear in plots
    'Ground': 'ground.csv',
    'Normal Stand': 'normal_stand.csv',
    'Cooling Stand': 'cooling_stand.csv', # Cooling stand with fans OFF
}
TEST_FILE_PATTERN = 'cooling_[Tt]est_*.csv' # Match both 'Test' and 'test'
BASE_OUTPUT_DIR = './cooling_plots_multilang' # Directory to save plots and logs
STEADY_STATE_START_TIME = 420 # seconds - ADJUST based on test duration (e.g., 420 for 10min test)
BASELINE_FOR_DELTA_PLOT = 'Normal Stand' # Choose which baseline to compare against (Not currently used, but kept for potential future use)
LANGUAGES = ['en', 'pt'] # Add more language codes if needed

# --- Translation Dictionary ---
translations = {
    'en': {
        'processing_baselines': "Processing baseline files...",
        'processed_file_as': "  Processed: {} as '{}'",
        'error_baseline_not_found': "Error: Baseline file not found: {}. Please ensure it exists.",
        'exiting_missing_baseline': "Exiting due to missing baseline file(s).",
        'processing_tests': "\nProcessing test files...",
        # Updated warning message to reflect the change implicitly via f-string
        'warning_no_test_files': f"Warning: No test files found matching pattern '{TEST_FILE_PATTERN}' in {DATA_DIR}",
        'warning_multiple_files_speed': "  Warning: Multiple files found for speed {} RPM. Using data from: {}",
        'processed_test_file': "  Processed: {} (Speed: {} RPM)",
        'skipping_non_speed': "  Skipping non-speed test file: {}",
        'warning_extract_speed': "  Warning: Could not extract speed from filename: {}. Skipping.",
        'warning_no_rpm_data': "\nWarning: No valid RPM test data found. Speed-related plots will be skipped.",
        'error_no_valid_test_data': "\nError: No valid test data successfully processed. Only baseline data found. Exiting.",
        'processed_datasets_total': "\nProcessed {} total datasets for plotting.",
        'dataset_labels': "Dataset labels: {}",
        'hypothesis_title': "\n--- Hypothesis ---",
        'hypothesis_null': "Null Hypothesis (H₀): Increasing fan speed does not decrease steady-state CPU/GPU temperature (correlation is ≥ 0).",
        'hypothesis_alt': "Alternative Hypothesis (H₁): Increasing fan speed decreases steady-state CPU/GPU temperature (correlation is < 0).",
        'hypothesis_separator': "--------------------",
        'creating_output_dir': "\nOutput files (plots and logs) will be saved to subdirectories within: {}",
        'error_creating_output_dir': "Error creating base output directory {}: {}. Output may not be saved.",
        'calculating_steady_state': f"\nCalculating steady-state means (data after {STEADY_STATE_START_TIME} seconds)...",
        'steady_state_means_label': "  {}: Mean CPU = {:.2f}°C, Mean GPU = {:.2f}°C",
        'warning_no_steady_state_data': "  Warning: No steady-state data >= {}s found for '{}'. Skipping for means/boxplots.",
        'generating_plots': "\nGenerating plots...", # Kept for potential future use, replaced by lang specific one
        'plot_cpu_time_title': "CPU Temperature vs. Time Under Different Cooling Conditions",
        'plot_gpu_time_title': "GPU Temperature vs. Time Under Different Cooling Conditions",
        'plot_elapsed_time_label': "Elapsed Time (seconds)",
        'plot_cpu_temp_label': "Average CPU Temperature (°C)",
        'plot_gpu_temp_label': "GPU Temperature (°C)",
        'plot_boxplot_cpu_title': f'CPU Temperature Distribution (After {STEADY_STATE_START_TIME}s)',
        'plot_boxplot_gpu_title': f'GPU Temperature Distribution (After {STEADY_STATE_START_TIME}s)',
        'plot_boxplot_main_title': 'Steady-State Temperature Distribution Comparison',
        'plot_boxplot_cpu_ylabel': 'Average CPU Temperature (°C)',
        'plot_boxplot_gpu_ylabel': 'GPU Temperature (°C)',
        'skipping_boxplots_insufficient_data': "\nSkipping box plots: Not enough datasets had sufficient steady-state data.",
        'skipping_boxplots_fewer_than_two': "\nSkipping box plots as fewer than two datasets had valid steady-state data.",
        'plot_bar_title': f'Average Steady-State Temperature (After {STEADY_STATE_START_TIME}s)',
        'plot_bar_ylabel': 'Average Temperature (°C)',
        'plot_bar_legend_cpu': 'CPU Avg Temp',
        'plot_bar_legend_gpu': 'GPU Temp',
        'skipping_bar_chart_no_data': "\nSkipping Average Steady-State Temperature Bar Chart: No steady-state data available.",
        'plot_scatter_main_title': 'Steady-State Temperature Dependence on Fan Speed',
        'plot_scatter_cpu_title': 'CPU Steady State Temp vs. Fan Speed\n(H₀: slope ≥ 0 vs H₁: slope < 0)',
        'plot_scatter_gpu_title': 'GPU Steady State Temp vs. Fan Speed\n(H₀: slope ≥ 0 vs H₁: slope < 0)',
        'plot_scatter_cpu_trend_label': 'CPU Trend (y={:.3f}x + {:.1f})',
        'plot_scatter_gpu_trend_label': 'GPU Trend (y={:.3f}x + {:.1f})',
        'plot_scatter_cpu_fail_label': 'CPU Trend (Fit Failed)',
        'plot_scatter_gpu_fail_label': 'GPU Trend (Fit Failed)',
        'plot_scatter_cpu_data_label': 'CPU Data Points',
        'plot_scatter_gpu_data_label': 'GPU Data Points',
        'plot_scatter_fan_speed_label': 'Fan Speed (RPM)',
        'plot_scatter_cpu_ylabel': 'Average CPU Temperature (°C)',
        'plot_scatter_gpu_ylabel': 'GPU Temperature (°C)',
        'plot_scatter_cpu_title_corr': 'CPU Steady State Temp vs. Fan Speed\nCorrelation: {:.3f} (H₀: ≥ 0)',
        'plot_scatter_gpu_title_corr': 'GPU Steady State Temp vs. Fan Speed\nCorrelation: {:.3f} (H₀: ≥ 0)',
        'plot_scatter_cpu_title_fail': 'CPU Steady State Temp vs. Fan Speed\n(Trendline fit failed)',
        'plot_scatter_gpu_title_fail': 'GPU Steady State Temp vs. Fan Speed\n(Trendline fit failed)',
        'plot_scatter_cpu_title_corr_error': 'CPU Steady State Temp vs. Fan Speed\n(Correlation error: {})', # Currently not used directly in title
        'plot_scatter_gpu_title_corr_error': 'GPU Steady State Temp vs. Fan Speed\n(Correlation error: {})', # Currently not used directly in title
        'warning_trendline_cpu': "Warning: Could not fit trendline for CPU vs RPM.",
        'warning_trendline_gpu': "Warning: Could not fit trendline for GPU vs RPM.",
        'warning_correlation_cpu': "Warning: Could not calculate correlation for CPU vs RPM: {}",
        'warning_correlation_gpu': "Warning: Could not calculate correlation for GPU vs RPM: {}",
        'correlation_analysis_title': "\n--- Correlation Analysis (RPM vs Steady State Temp) ---",
        'correlation_cpu_label': "CPU: Pearson Correlation Coefficient = {:.3f}",
        'correlation_gpu_label': "GPU: Pearson Correlation Coefficient = {:.3f}",
        'correlation_cpu_strong_neg': "  Strong negative correlation found for CPU, supporting the alternative hypothesis (H₁).",
        'correlation_cpu_weak_neg': "  Weak/Moderate negative correlation found for CPU, leaning towards the alternative hypothesis (H₁).",
        'correlation_cpu_no_neg': "  No significant negative correlation found for CPU (or correlation is positive/near zero), fails to reject the null hypothesis (H₀).",
        'correlation_gpu_strong_neg': "  Strong negative correlation found for GPU, supporting the alternative hypothesis (H₁).",
        'correlation_gpu_weak_neg': "  Weak/Moderate negative correlation found for GPU, leaning towards the alternative hypothesis (H₁).",
        'correlation_gpu_no_neg': "  No significant negative correlation found for GPU (or correlation is positive/near zero), fails to reject the null hypothesis (H₀).",
        'correlation_analysis_separator': "-------------------------------------------------------",
        'skipping_scatter_plot': "\nSkipping Temperature vs. Fan Speed plot: Not enough RPM data points with steady-state results (need >= 2).",
        'processing_complete': "\nProcessing complete.",
        'file_is_empty': "Warning: File is empty: {}",
        'missing_columns': "Warning: Missing required columns in {}. Required: {}. Found: {}. Skipping.",
        'no_numeric_data': "Warning: No valid numeric temperature data in {} after cleaning.",
        'error_processing_file': "Error processing file {}: {}",
        'legend_location': 'center left',
        'legend_anchor': (1, 0.5),
        'tight_layout_rect_legend': [0, 0, 0.85, 1], # For plots with legend outside
        'tight_layout_rect_normal': [0, 0, 1, 0.95], # Standard rect
        'tight_layout_rect_rotated': [0, 0.05, 1, 0.95], # For rotated x-labels
        'setting_up_logging': "Setting up logging...",
        'log_file_created': "Log file created: {}",
        'generating_plots_for_lang': "\n--- Generating plots and logging for language: {} ---",
        'removing_log_handler': "Removing log handler for language: {}",
    },
    'pt': {
        'processing_baselines': "Processando arquivos de base...",
        'processed_file_as': "  Processado: {} como '{}'",
        'error_baseline_not_found': "Erro: Arquivo de base não encontrado: {}. Verifique se existe.",
        'exiting_missing_baseline': "Saindo devido a arquivo(s) de base ausente(s).",
        'processing_tests': "\nProcessando arquivos de teste...",
        # Updated warning message to reflect the change implicitly via f-string
        'warning_no_test_files': f"Aviso: Nenhum arquivo de teste encontrado com o padrão '{TEST_FILE_PATTERN}' em {DATA_DIR}",
        'warning_multiple_files_speed': "  Aviso: Múltiplos arquivos encontrados para velocidade {} RPM. Usando dados de: {}",
        'processed_test_file': "  Processado: {} (Velocidade: {} RPM)",
        'skipping_non_speed': "  Ignorando arquivo de teste sem velocidade: {}",
        'warning_extract_speed': "  Aviso: Não foi possível extrair velocidade do nome do arquivo: {}. Ignorando.",
        'warning_no_rpm_data': "\nAviso: Nenhum dado de teste RPM válido encontrado. Gráficos relacionados à velocidade serão ignorados.",
        'error_no_valid_test_data': "\nErro: Nenhum dado de teste válido processado com sucesso. Apenas dados de base encontrados. Saindo.",
        'processed_datasets_total': "\n{} conjuntos de dados totais processados para plotagem.",
        'dataset_labels': "Rótulos dos conjuntos de dados: {}",
        'hypothesis_title': "\n--- Hipótese ---",
        'hypothesis_null': "Hipótese Nula (H₀): Aumentar a velocidade da ventoinha não diminui a temperatura estacionária da CPU/GPU (correlação ≥ 0).",
        'hypothesis_alt': "Hipótese Alternativa (H₁): Aumentar a velocidade da ventoinha diminui a temperatura estacionária da CPU/GPU (correlação < 0).",
        'hypothesis_separator': "------------------",
        'creating_output_dir': "\nArquivos de saída (gráficos e logs) serão salvos em subdiretórios dentro de: {}",
        'error_creating_output_dir': "Erro ao criar diretório de saída base {}: {}. Saída pode não ser salva.",
        'calculating_steady_state': f"\nCalculando médias de estado estacionário (dados após {STEADY_STATE_START_TIME} segundos)...",
        'steady_state_means_label': "  {}: Média CPU = {:.2f}°C, Média GPU = {:.2f}°C",
        'warning_no_steady_state_data': "  Aviso: Nenhum dado de estado estacionário >= {}s encontrado para '{}'. Ignorando para médias/boxplots.",
        'generating_plots': "\nGerando gráficos...", # Kept for potential future use
        'plot_cpu_time_title': "Temperatura CPU vs. Tempo Sob Diferentes Condições de Resfriamento",
        'plot_gpu_time_title': "Temperatura GPU vs. Tempo Sob Diferentes Condições de Resfriamento",
        'plot_elapsed_time_label': "Tempo Decorrido (segundos)",
        'plot_cpu_temp_label': "Temperatura Média CPU (°C)",
        'plot_gpu_temp_label': "Temperatura GPU (°C)",
        'plot_boxplot_cpu_title': f'Distribuição Temperatura CPU (Após {STEADY_STATE_START_TIME}s)',
        'plot_boxplot_gpu_title': f'Distribuição Temperatura GPU (Após {STEADY_STATE_START_TIME}s)',
        'plot_boxplot_main_title': 'Comparação da Distribuição de Temperatura em Estado Estacionário',
        'plot_boxplot_cpu_ylabel': 'Temperatura Média CPU (°C)',
        'plot_boxplot_gpu_ylabel': 'Temperatura GPU (°C)',
        'skipping_boxplots_insufficient_data': "\nIgnorando box plots: Dados de estado estacionário insuficientes.",
        'skipping_boxplots_fewer_than_two': "\nIgnorando box plots: Menos de dois conjuntos de dados com dados de estado estacionário válidos.",
        'plot_bar_title': f'Temperatura Média em Estado Estacionário (Após {STEADY_STATE_START_TIME}s)',
        'plot_bar_ylabel': 'Temperatura Média (°C)',
        'plot_bar_legend_cpu': 'Temp Média CPU',
        'plot_bar_legend_gpu': 'Temp GPU',
        'skipping_bar_chart_no_data': "\nIgnorando Gráfico de Barras de Temperatura Média Estacionária: Nenhum dado de estado estacionário disponível.",
        'plot_scatter_main_title': 'Dependência da Temperatura Estacionária na Velocidade da Ventoinha',
        'plot_scatter_cpu_title': 'Temp CPU Estacionária vs. Vel. Ventoinha\n(H₀: inclinação ≥ 0 vs H₁: inclinação < 0)',
        'plot_scatter_gpu_title': 'Temp GPU Estacionária vs. Vel. Ventoinha\n(H₀: inclinação ≥ 0 vs H₁: inclinação < 0)',
        'plot_scatter_cpu_trend_label': 'Tendência CPU (y={:.3f}x + {:.1f})',
        'plot_scatter_gpu_trend_label': 'Tendência GPU (y={:.3f}x + {:.1f})',
        'plot_scatter_cpu_fail_label': 'Tendência CPU (Ajuste falhou)',
        'plot_scatter_gpu_fail_label': 'Tendência GPU (Ajuste falhou)',
        'plot_scatter_cpu_data_label': 'Pontos Dados CPU',
        'plot_scatter_gpu_data_label': 'Pontos Dados GPU',
        'plot_scatter_fan_speed_label': 'Velocidade Ventoinha (RPM)',
        'plot_scatter_cpu_ylabel': 'Temperatura Média CPU (°C)',
        'plot_scatter_gpu_ylabel': 'Temperatura GPU (°C)',
        'plot_scatter_cpu_title_corr': 'Temp CPU Estacionária vs. Vel. Ventoinha\nCorrelação: {:.3f} (H₀: ≥ 0)',
        'plot_scatter_gpu_title_corr': 'Temp GPU Estacionária vs. Vel. Ventoinha\nCorrelação: {:.3f} (H₀: ≥ 0)',
        'plot_scatter_cpu_title_fail': 'Temp CPU Estacionária vs. Vel. Ventoinha\n(Ajuste de tendência falhou)',
        'plot_scatter_gpu_title_fail': 'Temp GPU Estacionária vs. Vel. Ventoinha\n(Ajuste de tendência falhou)',
        'plot_scatter_cpu_title_corr_error': 'Temp CPU Estacionária vs. Vel. Ventoinha\n(Erro correlação: {})', # Currently not used
        'plot_scatter_gpu_title_corr_error': 'Temp GPU Estacionária vs. Vel. Ventoinha\n(Erro correlação: {})', # Currently not used
        'warning_trendline_cpu': "Aviso: Não foi possível ajustar linha de tendência para CPU vs RPM.",
        'warning_trendline_gpu': "Aviso: Não foi possível ajustar linha de tendência para GPU vs RPM.",
        'warning_correlation_cpu': "Aviso: Não foi possível calcular correlação para CPU vs RPM: {}",
        'warning_correlation_gpu': "Aviso: Não foi possível calcular correlação para GPU vs RPM: {}",
        'correlation_analysis_title': "\n--- Análise de Correlação (RPM vs Temp Estacionária) ---",
        'correlation_cpu_label': "CPU: Coeficiente de Correlação Pearson = {:.3f}",
        'correlation_gpu_label': "GPU: Coeficiente de Correlação Pearson = {:.3f}",
        'correlation_cpu_strong_neg': "  Forte correlação negativa encontrada para CPU, apoiando a hipótese alternativa (H₁).",
        'correlation_cpu_weak_neg': "  Correlação negativa fraca/moderada encontrada para CPU, inclinando para a hipótese alternativa (H₁).",
        'correlation_cpu_no_neg': "  Nenhuma correlação negativa significativa encontrada para CPU (ou correlação é positiva/próxima de zero), falha em rejeitar a hipótese nula (H₀).",
        'correlation_gpu_strong_neg': "  Forte correlação negativa encontrada para GPU, apoiando a hipótese alternativa (H₁).",
        'correlation_gpu_weak_neg': "  Correlação negativa fraca/moderada encontrada para GPU, inclinando para a hipótese alternativa (H₁).",
        'correlation_gpu_no_neg': "  Nenhuma correlação negativa significativa encontrada para GPU (ou correlação é positiva/próxima de zero), falha em rejeitar a hipótese nula (H₀).",
        'correlation_analysis_separator': "---------------------------------------------------------",
        'skipping_scatter_plot': "\nIgnorando gráfico Temperatura vs. Velocidade da Ventoinha: Pontos de dados RPM insuficientes com resultados estacionários (necessário >= 2).",
        'processing_complete': "\nProcessamento completo.",
        'file_is_empty': "Aviso: O arquivo está vazio: {}",
        'missing_columns': "Aviso: Colunas obrigatórias ausentes em {}. Necessário: {}. Encontrado: {}. Ignorando.",
        'no_numeric_data': "Aviso: Nenhum dado numérico de temperatura válido em {} após limpeza.",
        'error_processing_file': "Erro ao processar arquivo {}: {}",
        'legend_location': 'center left',
        'legend_anchor': (1, 0.5),
        'tight_layout_rect_legend': [0, 0, 0.85, 1],
        'tight_layout_rect_normal': [0, 0, 1, 0.95],
        'tight_layout_rect_rotated': [0, 0.05, 1, 0.95],
        'setting_up_logging': "Configurando logging...",
        'log_file_created': "Arquivo de log criado: {}",
        'generating_plots_for_lang': "\n--- Gerando gráficos e log para o idioma: {} ---",
        'removing_log_handler': "Removendo manipulador de log para o idioma: {}",
    }
    # Add more languages here...
}

# --- Helper Functions ---

def get_translation(key, lang):
    """Fetches translation for a key in the specified language."""
    # Fallback to English if lang or key is missing
    return translations.get(lang, translations['en']).get(key, f"<{key}_{lang}_missing>")

# Define other helper functions before they are called
def extract_speed(filename):
    """Extracts numeric speed from filenames like 'cooling_Test_1200.csv' or 'cooling_test_500.csv'."""
    match = re.search(r'_(\d+).csv$', filename, re.IGNORECASE)
    if match:
        try:
            return int(match.group(1))
        except ValueError:
            return None
    return None

def process_csv(filepath, label, lang='en'):
    """Loads, cleans, and processes a single CSV file, logging messages."""
    try:
        # Check file size first
        if os.path.getsize(filepath) == 0:
             logger.warning(get_translation('file_is_empty', lang).format(filepath))
             return None, None, None

        df = pd.read_csv(filepath, parse_dates=['Timestamp'])
        # Note: df.empty check after loading isn't strictly needed if size is checked,
        # but kept for safety against header-only files etc.
        if df.empty:
            logger.warning(get_translation('file_is_empty', lang).format(filepath))
            return None, None, None

        required_cols = ['Timestamp', 'CPU_Temp', 'CPU_Temp2', 'GPU_Temp']
        if not all(col in df.columns for col in required_cols):
            logger.warning(get_translation('missing_columns', lang).format(filepath, required_cols, df.columns.tolist()))
            return None, None, None

        for col in ['CPU_Temp', 'CPU_Temp2', 'GPU_Temp']:
            df[col] = pd.to_numeric(df[col], errors='coerce') # Convert non-numeric to NaN

        # Drop rows where ANY of the temp columns are NaN AFTER conversion
        initial_rows = len(df)
        df.dropna(subset=['CPU_Temp', 'CPU_Temp2', 'GPU_Temp'], inplace=True)
        dropped_rows = initial_rows - len(df)
        # if dropped_rows > 0:
        #     logger.debug(f"Dropped {dropped_rows} rows with NaN temperatures from {filepath}") # Optional debug

        if df.empty:
            logger.warning(get_translation('no_numeric_data', lang).format(filepath))
            return None, None, None

        df['CPU_Avg_Temp'] = df[['CPU_Temp', 'CPU_Temp2']].mean(axis=1)
        df.sort_values('Timestamp', inplace=True)
        start_time = df['Timestamp'].iloc[0]
        df['Elapsed_Time'] = (df['Timestamp'] - start_time).dt.total_seconds()
        df['Label'] = label # Use the provided label

        # Extract speed if applicable
        speed = extract_speed(os.path.basename(filepath))

        return df, label, speed # Return the original label passed in
    except FileNotFoundError:
        # No translation needed for critical error, keep it clear
        logger.error(f"Error: File not found: {filepath}")
        return None, None, None
    except pd.errors.EmptyDataError:
         logger.warning(get_translation('file_is_empty', lang).format(filepath))
         return None, None, None
    except Exception as e:
        logger.error(get_translation('error_processing_file', lang).format(filepath, e))
        return None, None, None

def get_color_map(labels):
    """Generates a consistent color map for labels."""
    color_cycle = plt.rcParams['axes.prop_cycle'].by_key()['color']
    # Ensure enough unique colors, fall back to a colormap if needed
    num_labels = len(labels)
    if num_labels > len(color_cycle):
        try:
            # Viridis is perceptually uniform and often a good default
            cmap = plt.get_cmap('viridis', num_labels)
            # Or use tab20 if you expect up to 20 distinct categories
            # cmap = plt.get_cmap('tab20', num_labels)
            colors_list = [cmap(i) for i in range(num_labels)]
        except ValueError: # Fallback if cmap doesn't support N argument
             cmap = plt.get_cmap('viridis')
             colors_list = [cmap(i/num_labels) for i in range(num_labels)]

    else:
        colors_list = color_cycle

    colors = {label: colors_list[i % len(colors_list)] for i, label in enumerate(labels)}
    return colors


# --- Logging Setup ---
# We'll set up the main logger and add handlers dynamically
logger = logging.getLogger('CoolingAnalysis')
logger.setLevel(logging.INFO) # Set the lowest level to capture

# Prevent duplicate handlers if the script is run multiple times in the same session (e.g., Jupyter)
if logger.hasHandlers():
    logger.handlers.clear()

# Console Handler (always active)
console_handler = logging.StreamHandler(sys.stdout)
console_formatter = logging.Formatter('%(message)s') # Simple format for console
console_handler.setFormatter(console_formatter)
logger.addHandler(console_handler)

# --- Create Base Output Directory ---
# Do this early so log files can be placed correctly
# NOW it's safe to call get_translation here
logger.info(get_translation('setting_up_logging', 'en'))
output_dir_created = False
if BASE_OUTPUT_DIR:
    try:
        os.makedirs(BASE_OUTPUT_DIR, exist_ok=True)
        logger.info(get_translation('creating_output_dir', 'en').format(BASE_OUTPUT_DIR))
        output_dir_created = True
    except OSError as e:
        logger.error(get_translation('error_creating_output_dir', 'en').format(BASE_OUTPUT_DIR, e))
        # Continue without saving if directory creation fails
else:
    logger.warning("BASE_OUTPUT_DIR is not set. Logs and plots will not be saved to files.")

# Dictionary to hold file handlers for each language
file_handlers = {}

# Setup file handlers for all languages defined
if output_dir_created:
    for lang in LANGUAGES:
        log_file_path = os.path.join(BASE_OUTPUT_DIR, f'results_{lang}.txt')
        try:
            # Use 'w' mode to overwrite the log file each time the script runs
            file_handler = logging.FileHandler(log_file_path, mode='w', encoding='utf-8')
            file_formatter = logging.Formatter('%(asctime)s - %(message)s', datefmt='%Y-%m-%d %H:%M:%S')
            file_handler.setFormatter(file_formatter)
            file_handlers[lang] = file_handler
            # Don't add the handler to the logger yet, we do it selectively
        except Exception as e:
            logger.error(f"Error creating log file handler for {lang} at {log_file_path}: {e}")
            file_handlers[lang] = None # Mark as failed


# --- Main Script Logic ---

# Add English file handler for initial processing steps
current_lang_handlers = {}
if 'en' in file_handlers and file_handlers['en']:
    logger.addHandler(file_handlers['en'])
    current_lang_handlers['en'] = file_handlers['en']
    # Log file creation AFTER adding the handler
    logger.info(get_translation('log_file_created', 'en').format(file_handlers['en'].baseFilename))


all_data = {} # Store processed dataframes {label: df}
speeds_data = {} # Store data for RPM tests {speed: df}
found_baseline_files = {}

# Use English for initial file processing messages
lang = 'en'

# 1. Process Baseline Files
logger.info(get_translation('processing_baselines', lang))
missing_baseline = False
for label, filename in BASELINE_FILES.items():
    # --- Filepath construction now uses DATA_DIR which points to './csv' ---
    filepath = os.path.join(DATA_DIR, filename)
    # --- End change reflection ---
    if not os.path.exists(filepath):
        logger.error(get_translation('error_baseline_not_found', lang).format(filepath))
        missing_baseline = True
    else:
        # Process with the English label initially
        df, processed_label, _ = process_csv(filepath, label, lang) # Uses logger inside
        if df is not None:
            found_baseline_files[processed_label] = df
            logger.info(get_translation('processed_file_as', lang).format(filename, processed_label))

if missing_baseline:
    logger.critical(get_translation('exiting_missing_baseline', lang))
    # Clean up handlers before exiting
    for handler in list(current_lang_handlers.values()): # Use list to avoid modifying during iteration
        logger.removeHandler(handler)
        handler.close()
    sys.exit(1)

all_data.update(found_baseline_files)

# 2. Process Test Files
logger.info(get_translation('processing_tests', lang))
# --- Glob pattern now uses DATA_DIR which points to './csv' ---
test_files = glob.glob(os.path.join(DATA_DIR, TEST_FILE_PATTERN))
# --- End change reflection ---
# Ensure we don't re-process files listed as baselines if the pattern matches
baseline_filenames = [os.path.basename(f) for f in BASELINE_FILES.values()]
test_files = [f for f in test_files if os.path.basename(f) not in baseline_filenames]


if not test_files:
    logger.warning(get_translation('warning_no_test_files', lang))
    # Don't exit yet, maybe user only wants baseline plots

test_data_by_speed = {}
processed_test_files_count = 0

for filepath in test_files:
    filename = os.path.basename(filepath)
    speed_val = extract_speed(filename)
    if speed_val is not None:
        label = f"{speed_val} RPM" # Consistent label format
        # process_csv uses the filepath directly, which is correct from glob
        df, processed_label, speed_out = process_csv(filepath, label, lang) # Uses logger inside
        if df is not None and speed_out is not None:
            if speed_val in test_data_by_speed:
                 logger.warning(get_translation('warning_multiple_files_speed', lang).format(speed_val, filename))
            # Store df and its English label, keyed by speed
            test_data_by_speed[speed_val] = (df, processed_label)
            logger.info(get_translation('processed_test_file', lang).format(filename, speed_val))
            processed_test_files_count += 1
        # else: processing errors already logged in process_csv
    else:
        # Check if it's known non-speed files or truly unexpected
        known_skips = ['cpu_gpu_test.csv', 'cooling_pad_tests.csv'] # Add others if needed
        if any(skip_name in filename for skip_name in known_skips):
             logger.info(get_translation('skipping_non_speed', lang).format(filename))
        else:
             logger.warning(get_translation('warning_extract_speed', lang).format(filename))


# Add the processed test data (unique by speed) to all_data
sorted_speeds = sorted(test_data_by_speed.keys())
for speed in sorted_speeds:
    df, label = test_data_by_speed[speed]
    all_data[label] = df
    speeds_data[speed] = df # Store only RPM data separately

if not speeds_data and processed_test_files_count > 0:
     logger.warning(get_translation('warning_no_rpm_data', lang))

if len(all_data) <= len(found_baseline_files) and test_files:
     logger.critical(get_translation('error_no_valid_test_data', lang))
     # Clean up handlers before exiting
     for handler in list(current_lang_handlers.values()):
        logger.removeHandler(handler)
        handler.close()
     sys.exit(1)
elif len(all_data) == 0:
     logger.critical("Error: No data at all (baseline or test) could be processed.")
     # Clean up handlers before exiting
     for handler in list(current_lang_handlers.values()):
        logger.removeHandler(handler)
        handler.close()
     sys.exit(1)


logger.info(get_translation('processed_datasets_total', lang).format(len(all_data)))
logger.info(get_translation('dataset_labels', lang).format(list(all_data.keys())))

# --- Hypothesis Statement (Printed once in English, logged to EN file) ---
logger.info(get_translation('hypothesis_title', 'en'))
logger.info(get_translation('hypothesis_null', 'en'))
logger.info(get_translation('hypothesis_alt', 'en'))
logger.info(get_translation('hypothesis_separator', 'en'))

# --- Calculate Steady State Means and Prepare Data (Done once) ---
logger.info(get_translation('calculating_steady_state', 'en'))
steady_state_means = {}
valid_labels_for_steady_state = []
# Define plot order based on desired baseline order then sorted speeds
plot_order = list(BASELINE_FILES.keys()) + [f"{speed} RPM" for speed in sorted_speeds]

for label in plot_order:
    if label in all_data:
        df = all_data[label]
        steady_df = df[df['Elapsed_Time'] >= STEADY_STATE_START_TIME].copy()
        if not steady_df.empty:
            mean_cpu = steady_df['CPU_Avg_Temp'].mean()
            mean_gpu = steady_df['GPU_Temp'].mean()
            # Check for NaN means which can happen if all values in steady state were NaN
            if pd.notna(mean_cpu) and pd.notna(mean_gpu):
                steady_state_means[label] = {'CPU': mean_cpu, 'GPU': mean_gpu}
                valid_labels_for_steady_state.append(label)
                logger.info(get_translation('steady_state_means_label', 'en').format(label, mean_cpu, mean_gpu))
            else:
                logger.warning(f"  Warning: Calculated steady-state means for '{label}' resulted in NaN. Skipping.")
        else:
            logger.warning(get_translation('warning_no_steady_state_data', 'en').format(STEADY_STATE_START_TIME, label))

# Filter plot_order to only include labels with valid steady state data for certain plots
plot_order_steady = [label for label in plot_order if label in valid_labels_for_steady_state]

# Consistent color maps based on the full potential plot order
color_map = get_color_map(plot_order)
color_map_steady = get_color_map(plot_order_steady) # Use potentially shorter list for steady plots

# --- Correlation Calculation (Done once before plotting loop) ---
rpm_labels_for_corr = [label for label in plot_order_steady if "RPM" in label]
corr_cpu, corr_gpu = np.nan, np.nan # Default to NaN
p_cpu, p_gpu = np.nan, np.nan

if len(rpm_labels_for_corr) >= 2:
    # Ensure rpms list corresponds exactly to the labels being used
    rpms = []
    cpu_temps_at_rpm = []
    gpu_temps_at_rpm = []
    for label in rpm_labels_for_corr:
         speed_match = re.search(r'(\d+)', label)
         if speed_match:
             rpms.append(int(speed_match.group(1)))
             cpu_temps_at_rpm.append(steady_state_means[label]['CPU'])
             gpu_temps_at_rpm.append(steady_state_means[label]['GPU'])
         else:
             logger.warning(f"Could not extract RPM from label '{label}' during correlation setup. Skipping this label.")

    # Recalculate length in case some labels were skipped
    if len(rpms) >= 2:
        try:
            corr_cpu, p_cpu = pearsonr(rpms, cpu_temps_at_rpm)
        except ValueError as ve:
            logger.warning(get_translation('warning_correlation_cpu', 'en').format(ve))
        try:
            corr_gpu, p_gpu = pearsonr(rpms, gpu_temps_at_rpm)
        except ValueError as ve:
            logger.warning(get_translation('warning_correlation_gpu', 'en').format(ve))
    else:
        # If not enough valid RPM data after filtering, reset correlations
        corr_cpu, corr_gpu = np.nan, np.nan
        rpm_labels_for_corr = [] # Clear the list as correlation is not possible
        logger.warning("Not enough valid RPM data points (>= 2) after label checking for correlation.")


# --- Plotting Loop ---
for lang in LANGUAGES:
    # Add the specific language handler if it exists and isn't already added
    added_handler = None
    if lang in file_handlers and file_handlers[lang] and lang not in current_lang_handlers:
        logger.addHandler(file_handlers[lang])
        current_lang_handlers[lang] = file_handlers[lang]
        added_handler = file_handlers[lang] # Track which handler was added in this iteration
        # Log file creation message to the newly added handler *and* console/existing handlers
        logger.info(get_translation('log_file_created', lang).format(file_handlers[lang].baseFilename))

    # Log the start of plotting for this language
    logger.info(get_translation('generating_plots_for_lang', lang).format(lang))

    current_output_dir = None
    if output_dir_created:
        current_output_dir = os.path.join(BASE_OUTPUT_DIR, lang)
        try:
            os.makedirs(current_output_dir, exist_ok=True)
        except OSError as e:
            logger.error(f"Error creating language directory {current_output_dir}: {e}")
            current_output_dir = None # Disable saving plots for this language

    # --- Plot 1: CPU Temperature vs. Elapsed Time ---
    try:
        plt.figure(figsize=(14, 8))
        for label in plot_order: # Plot all available data
            if label in all_data:
                df = all_data[label]
                plt.plot(df['Elapsed_Time'], df['CPU_Avg_Temp'], label=label, color=color_map.get(label, 'gray'))
        plt.xlabel(get_translation('plot_elapsed_time_label', lang))
        plt.ylabel(get_translation('plot_cpu_temp_label', lang))
        plt.title(get_translation('plot_cpu_time_title', lang))
        # Use language-specific legend settings if needed
        plt.legend(loc=get_translation('legend_location', lang), bbox_to_anchor=get_translation('legend_anchor', lang))
        plt.grid(True, linestyle='--', alpha=0.6)
        plt.tight_layout(rect=get_translation('tight_layout_rect_legend', lang))
        if current_output_dir:
            save_path = os.path.join(current_output_dir, 'cpu_temp_vs_time.png')
            plt.savefig(save_path)
            logger.info(f"Plot saved: {save_path}")
        plt.close() # Close plot to free memory
    except Exception as e:
        logger.error(f"Error generating CPU vs Time plot for {lang}: {e}")
        plt.close() # Ensure plot is closed even on error

    # --- Plot 2: GPU Temperature vs. Elapsed Time ---
    try:
        plt.figure(figsize=(14, 8))
        for label in plot_order: # Plot all available data
            if label in all_data:
                df = all_data[label]
                plt.plot(df['Elapsed_Time'], df['GPU_Temp'], label=label, color=color_map.get(label, 'gray'))
        plt.xlabel(get_translation('plot_elapsed_time_label', lang))
        plt.ylabel(get_translation('plot_gpu_temp_label', lang))
        plt.title(get_translation('plot_gpu_time_title', lang))
        plt.legend(loc=get_translation('legend_location', lang), bbox_to_anchor=get_translation('legend_anchor', lang))
        plt.grid(True, linestyle='--', alpha=0.6)
        plt.tight_layout(rect=get_translation('tight_layout_rect_legend', lang))
        if current_output_dir:
            save_path = os.path.join(current_output_dir, 'gpu_temp_vs_time.png')
            plt.savefig(save_path)
            logger.info(f"Plot saved: {save_path}")
        plt.close()
    except Exception as e:
        logger.error(f"Error generating GPU vs Time plot for {lang}: {e}")
        plt.close()

    # --- Plot 3: Box Plot Comparison ---
    if len(plot_order_steady) >= 2 : # Need at least two valid steady datasets for comparison
        try:
            cpu_data_for_boxplot = []
            gpu_data_for_boxplot = []
            labels_for_boxplot = [] # Use the steady state labels only

            # Use plot_order_steady which only contains labels with valid steady state means
            for label in plot_order_steady:
                df = all_data[label]
                steady_df = df[df['Elapsed_Time'] >= STEADY_STATE_START_TIME]
                cpu_data = steady_df['CPU_Avg_Temp'].dropna()
                gpu_data = steady_df['GPU_Temp'].dropna()
                # Ensure there's actually data after filtering/dropping NaN for this label
                if not cpu_data.empty and not gpu_data.empty:
                    cpu_data_for_boxplot.append(cpu_data)
                    gpu_data_for_boxplot.append(gpu_data)
                    labels_for_boxplot.append(label)
                else:
                     logger.warning(f"Skipping label '{label}' for boxplot ({lang}) due to empty data after dropna/steady state filtering.")

            # Check again if we *still* have enough data after potentially skipping labels
            if len(labels_for_boxplot) >= 2:
                fig, axes = plt.subplots(1, 2, figsize=(18, 7), sharey=False)

                # Use the potentially reduced list 'labels_for_boxplot'
                bp1 = axes[0].boxplot(cpu_data_for_boxplot, labels=labels_for_boxplot, patch_artist=True, showfliers=False)
                axes[0].set_title(get_translation('plot_boxplot_cpu_title', lang))
                axes[0].set_ylabel(get_translation('plot_boxplot_cpu_ylabel', lang))
                axes[0].tick_params(axis='x', rotation=45, labelsize=9)
                axes[0].grid(True, axis='y', linestyle='--', alpha=0.6)
                # Use color_map_steady but access using labels_for_boxplot
                for patch, label in zip(bp1['boxes'], labels_for_boxplot):
                    patch.set_facecolor(color_map_steady.get(label, 'gray'))
                    patch.set_alpha(0.7)

                bp2 = axes[1].boxplot(gpu_data_for_boxplot, labels=labels_for_boxplot, patch_artist=True, showfliers=False)
                axes[1].set_title(get_translation('plot_boxplot_gpu_title', lang))
                axes[1].set_ylabel(get_translation('plot_boxplot_gpu_ylabel', lang))
                axes[1].tick_params(axis='x', rotation=45, labelsize=9)
                axes[1].grid(True, axis='y', linestyle='--', alpha=0.6)
                for patch, label in zip(bp2['boxes'], labels_for_boxplot):
                    patch.set_facecolor(color_map_steady.get(label, 'gray')) # color_map_steady should have these labels
                    patch.set_alpha(0.7)

                plt.suptitle(get_translation('plot_boxplot_main_title', lang), fontsize=16)
                plt.tight_layout(rect=get_translation('tight_layout_rect_rotated', lang))
                if current_output_dir:
                    save_path = os.path.join(current_output_dir, 'temp_distribution_boxplot.png')
                    plt.savefig(save_path)
                    logger.info(f"Plot saved: {save_path}")
                plt.close()
            # Log skipping message based on final count of usable labels
            elif len(labels_for_boxplot) == 1:
                 logger.warning(get_translation('skipping_boxplots_fewer_than_two', lang))
            else: # len is 0
                 logger.warning(get_translation('skipping_boxplots_insufficient_data', lang))

        except Exception as e:
            logger.error(f"Error generating Boxplot for {lang}: {e}")
            plt.close() # Ensure plot is closed even on error

    elif len(plot_order_steady) == 1:
         # Logged only if initial check fails (only 1 dataset had steady data)
         logger.warning(get_translation('skipping_boxplots_fewer_than_two', lang))
    else: # len is 0
         # Logged only if initial check fails (0 datasets had steady data)
         logger.warning(get_translation('skipping_boxplots_insufficient_data', lang))


    # --- Plot 4: Average Steady-State Temperature Bar Chart ---
    if steady_state_means and len(plot_order_steady) > 0: # Check if there's data to plot
        try:
            # Use plot_order_steady for labels and order
            cpu_means = [steady_state_means[label]['CPU'] for label in plot_order_steady]
            gpu_means = [steady_state_means[label]['GPU'] for label in plot_order_steady]
            x = np.arange(len(plot_order_steady)) # label locations
            width = 0.35 # width of the bars

            fig, ax = plt.subplots(figsize=(14, 7))
            # Use color_map_steady, keyed by labels in plot_order_steady
            rects1 = ax.bar(x - width/2, cpu_means, width, label=get_translation('plot_bar_legend_cpu', lang), color=[color_map_steady.get(l,'grey') for l in plot_order_steady], alpha=0.8)
            rects2 = ax.bar(x + width/2, gpu_means, width, label=get_translation('plot_bar_legend_gpu', lang), color=[color_map_steady.get(l,'lightgrey') for l in plot_order_steady], alpha=0.6) # Different default/alpha

            ax.set_ylabel(get_translation('plot_bar_ylabel', lang))
            ax.set_title(get_translation('plot_bar_title', lang))
            ax.set_xticks(x)
            ax.set_xticklabels(plot_order_steady, rotation=45, ha='right') # Use plot_order_steady labels
            ax.legend()
            ax.grid(True, axis='y', linestyle='--', alpha=0.6)

            # Add labels if possible (check for numpy version compatibility if needed)
            try:
                ax.bar_label(rects1, padding=3, fmt='%.1f')
                ax.bar_label(rects2, padding=3, fmt='%.1f')
            except AttributeError:
                 # Don't log repeatedly inside loop, maybe log once before loop if needed
                 pass # Silently ignore if not available

            fig.tight_layout()
            if current_output_dir:
                save_path = os.path.join(current_output_dir, 'steady_state_average_temps_bar.png')
                plt.savefig(save_path)
                logger.info(f"Plot saved: {save_path}")
            plt.close()
        except Exception as e:
            logger.error(f"Error generating Bar Chart for {lang}: {e}")
            plt.close() # Ensure plot is closed even on error
    else:
        # Log skipping only once per language if no data
        if lang == LANGUAGES[0]: # Log only for the first language
             logger.warning(get_translation('skipping_bar_chart_no_data', lang))

    # --- Plot 5: Temperature vs. Fan Speed Scatter Plot (Hypothesis Visualization) ---
    # Check rpm_labels_for_corr which was potentially updated during correlation calculation
    if len(rpm_labels_for_corr) >= 2:
        try:
            # Use the potentially filtered 'rpms', 'cpu_temps_at_rpm', 'gpu_temps_at_rpm' from correlation step
            fig, axes = plt.subplots(1, 2, figsize=(16, 7), sharex=True) # Wider figure

            # --- CPU vs RPM ---
            axes[0].scatter(rpms, cpu_temps_at_rpm, color='red', label=get_translation('plot_scatter_cpu_data_label', lang))
            # Fit and plot trendline
            coeffs_cpu, poly_cpu = None, None
            try:
                if len(rpms) >= 2: # Re-check just in case
                    coeffs_cpu = np.polyfit(rpms, cpu_temps_at_rpm, 1) # Linear fit
                    poly_cpu = np.poly1d(coeffs_cpu)
                    # Ensure rpm_line spans the actual data range
                    rpm_min, rpm_max = min(rpms), max(rpms)
                    # Handle case with only two points at same RPM (unlikely but safe) or single point (shouldn't happen with check)
                    if rpm_min == rpm_max or len(rpms) < 2:
                       rpm_line = np.array([rpm_min]) # Plot just the point if needed
                       if len(rpms) == 1: # Cannot plot line for single point
                            pass # Just scatter point shown
                       elif len(rpms) >=2 : # Plot line segment if multiple points at same rpm_min
                           axes[0].plot(rpm_line, poly_cpu(rpm_line), linestyle="--", color='darkred',
                                        label=get_translation('plot_scatter_cpu_trend_label', lang).format(coeffs_cpu[0], coeffs_cpu[1]))
                    else:
                       rpm_line = np.linspace(rpm_min, rpm_max, 100)
                       axes[0].plot(rpm_line, poly_cpu(rpm_line), linestyle="--", color='darkred',
                                    label=get_translation('plot_scatter_cpu_trend_label', lang).format(coeffs_cpu[0], coeffs_cpu[1]))
                else:
                     # This case should ideally not be reached due to outer check len(rpm_labels_for_corr) >= 2
                     # but added for safety
                     raise ValueError("Not enough data points for trendline fit")

                # Title including correlation and hypothesis context
                if not np.isnan(corr_cpu):
                     axes[0].set_title(get_translation('plot_scatter_cpu_title_corr', lang).format(corr_cpu))
                else: # Fallback title if correlation failed
                     axes[0].set_title(get_translation('plot_scatter_cpu_title', lang)) # Generic hypothesis title
            except (np.linalg.LinAlgError, ValueError, TypeError) as fit_err:
                 # Log warning only once (preferably before loop)
                 # logger.warning(get_translation('warning_trendline_cpu', lang) + f" Error: {fit_err}")
                 axes[0].plot([], [], linestyle="--", color='darkred', label=get_translation('plot_scatter_cpu_fail_label', lang)) # Add dummy entry for legend
                 axes[0].set_title(get_translation('plot_scatter_cpu_title_fail', lang))


            axes[0].set_ylabel(get_translation('plot_scatter_cpu_ylabel', lang))
            axes[0].set_xlabel(get_translation('plot_scatter_fan_speed_label', lang))
            axes[0].legend()
            axes[0].grid(True, linestyle='--', alpha=0.6)

            # --- GPU vs RPM ---
            axes[1].scatter(rpms, gpu_temps_at_rpm, color='blue', label=get_translation('plot_scatter_gpu_data_label', lang))
            # Fit and plot trendline
            coeffs_gpu, poly_gpu = None, None
            try:
                if len(rpms) >= 2:
                    coeffs_gpu = np.polyfit(rpms, gpu_temps_at_rpm, 1)
                    poly_gpu = np.poly1d(coeffs_gpu)
                    rpm_min, rpm_max = min(rpms), max(rpms)
                    if rpm_min == rpm_max or len(rpms) < 2:
                        rpm_line = np.array([rpm_min])
                        if len(rpms) == 1:
                            pass
                        elif len(rpms) >=2 :
                            axes[1].plot(rpm_line, poly_gpu(rpm_line), linestyle="--", color='darkblue',
                                         label=get_translation('plot_scatter_gpu_trend_label', lang).format(coeffs_gpu[0], coeffs_gpu[1]))
                    else:
                        rpm_line = np.linspace(rpm_min, rpm_max, 100)
                        axes[1].plot(rpm_line, poly_gpu(rpm_line), linestyle="--", color='darkblue',
                                     label=get_translation('plot_scatter_gpu_trend_label', lang).format(coeffs_gpu[0], coeffs_gpu[1]))
                else:
                    raise ValueError("Not enough data points for trendline fit")

                if not np.isnan(corr_gpu):
                     axes[1].set_title(get_translation('plot_scatter_gpu_title_corr', lang).format(corr_gpu))
                else:
                     axes[1].set_title(get_translation('plot_scatter_gpu_title', lang))
            except (np.linalg.LinAlgError, ValueError, TypeError) as fit_err:
                # Log warning only once (preferably before loop)
                # logger.warning(get_translation('warning_trendline_gpu', lang) + f" Error: {fit_err}")
                axes[1].plot([], [], linestyle="--", color='darkblue', label=get_translation('plot_scatter_gpu_fail_label', lang))
                axes[1].set_title(get_translation('plot_scatter_gpu_title_fail', lang))


            axes[1].set_ylabel(get_translation('plot_scatter_gpu_ylabel', lang))
            axes[1].set_xlabel(get_translation('plot_scatter_fan_speed_label', lang))
            axes[1].legend()
            axes[1].grid(True, linestyle='--', alpha=0.6)

            plt.suptitle(get_translation('plot_scatter_main_title', lang), fontsize=16)
            plt.tight_layout(rect=get_translation('tight_layout_rect_normal', lang)) # Use normal rect here
            if current_output_dir:
                save_path = os.path.join(current_output_dir, 'temp_vs_fan_speed_scatter.png')
                plt.savefig(save_path)
                logger.info(f"Plot saved: {save_path}")
            plt.close()

        except Exception as e:
            logger.error(f"Error generating Scatter plot for {lang}: {e}")
            plt.close() # Ensure plot is closed even on error

    elif lang == LANGUAGES[0]: # Only log skip message once (based on first language)
        # Check if the reason was insufficient points from the start
        if len([lbl for lbl in plot_order_steady if "RPM" in lbl]) < 2:
            logger.warning(get_translation('skipping_scatter_plot', lang))


    # --- Remove Language Handler ---
    # Remove the handler specific to this language iteration IF it was added in this loop
    # Don't remove the 'en' handler yet, as it might be needed for the final summary
    if added_handler and lang != 'en':
        logger.info(get_translation('removing_log_handler', lang).format(lang))
        logger.removeHandler(added_handler)
        added_handler.close() # Close the file handle
        if lang in current_lang_handlers: # Remove from active dict
            del current_lang_handlers[lang]
    elif added_handler and lang == 'en':
        # If 'en' was added dynamically (shouldn't normally happen after initial setup),
        # keep it active for the summary. It will be closed at the very end.
        pass


# --- Final Correlation Summary (Logged once to EN file and console) ---
# Ensure the English handler is still active if it exists
if 'en' in file_handlers and file_handlers['en'] and 'en' not in current_lang_handlers:
    # This case might happen if 'en' was the only language and got removed early. Re-add it.
    logger.addHandler(file_handlers['en'])
    current_lang_handlers['en'] = file_handlers['en']

if len(rpm_labels_for_corr) >= 2: # Use the list potentially updated during correlation calc
    logger.info(get_translation('correlation_analysis_title', 'en'))
    # Use np.isnan to check before formatting
    corr_cpu_str = f"{corr_cpu:.3f}" if not np.isnan(corr_cpu) else "N/A"
    corr_gpu_str = f"{corr_gpu:.3f}" if not np.isnan(corr_gpu) else "N/A"
    logger.info(get_translation('correlation_cpu_label', 'en').replace("{:.3f}", "{}").format(corr_cpu_str))
    logger.info(get_translation('correlation_gpu_label', 'en').replace("{:.3f}", "{}").format(corr_gpu_str))

    # Hypothesis evaluation based on correlation
    if not np.isnan(corr_cpu):
        if corr_cpu < -0.5: logger.info(get_translation('correlation_cpu_strong_neg', 'en'))
        elif corr_cpu < -0.1: logger.info(get_translation('correlation_cpu_weak_neg', 'en'))
        else: logger.info(get_translation('correlation_cpu_no_neg', 'en'))
    else:
        logger.info("  CPU correlation could not be calculated.")

    if not np.isnan(corr_gpu):
        if corr_gpu < -0.5: logger.info(get_translation('correlation_gpu_strong_neg', 'en'))
        elif corr_gpu < -0.1: logger.info(get_translation('correlation_gpu_weak_neg', 'en'))
        else: logger.info(get_translation('correlation_gpu_no_neg', 'en'))
    else:
        logger.info("  GPU correlation could not be calculated.")

    logger.info(get_translation('correlation_analysis_separator', 'en'))
else: # If not enough data for correlation from the start
    logger.info("\n--- Correlation Analysis Skipped (requires >= 2 RPM data points with steady-state results) ---")


logger.info(get_translation('processing_complete', 'en'))

# --- Final Cleanup ---
# Remove and close any remaining file handlers
for lang_code, handler in list(current_lang_handlers.items()): # Use list copy
    logger.info(f"Closing final log handler for language: {lang_code}")
    logger.removeHandler(handler)
    handler.close()

# It's generally good practice to shut down logging, though often not strictly necessary
# logging.shutdown()