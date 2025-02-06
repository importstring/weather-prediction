# main.py
"""
WeatherAI Script

This script handles the ingestion, preprocessing, and modeling of weather and space event data.
It continuously monitors specified directories for new or modified CSV files, processes the data,
trains a machine learning model, and makes predictions based on the updated data.
It also includes functionalities for generating forecasts, evaluating model accuracy, and updating the model incrementally.

Author: Simon
Date: 2024-12-14
"""

import os
import re
import sys
import time
import json
import logging
import numbers
from datetime import datetime, timedelta
from logging.handlers import RotatingFileHandler

import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.exceptions import NotFittedError
from sklearn.utils.validation import check_is_fitted
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.multioutput import MultiOutputRegressor
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.linear_model import SGDRegressor, LinearRegression
from sklearn.metrics import mean_squared_error, r2_score, roc_auc_score, roc_curve, auc
from watchdog.events import FileSystemEventHandler

from utils import update_data, clear

# ============================
# Configuration and Setup
# ============================

# Define the base directory paths
BASE_DIR = "/Users/simon/Desktop/Areas/TKS/Focus1Rep2/WeatherAI"
KNOWLEDGE_PATH = os.path.join(BASE_DIR, 'Databases', 'QuickFix_Knowledge')  # Central location for logs and models

# Ensure the knowledge directory exists
try:
    os.makedirs(KNOWLEDGE_PATH, exist_ok=True)
except Exception as e:
    print(f"Failed to create knowledge directory: {e}")
    sys.exit(1)

# Configure logging
log_file = os.path.join(KNOWLEDGE_PATH, 'weather_ai.log')
try:
    handler = RotatingFileHandler(log_file, maxBytes=10*1024*1024, backupCount=5)  # 10MB per log file
    logging.basicConfig(
        level=logging.DEBUG, format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            handler,
            logging.StreamHandler(sys.stdout)
        ]
    )
    logging.info("Logging configured successfully.")
except Exception as e:
    print(f"Failed to configure logging: {e}")
    sys.exit(1)

# Define Paths
REAL_TIME_DATA_PATH = os.path.join(BASE_DIR, 'Databases', 'Data', 'Weather')
SPACE_DATA_PATH = os.path.join(BASE_DIR, 'Databases', 'Data', 'Space Data')
NEAR_EARTH_SPACE_DATA_PATH = os.path.join(BASE_DIR, 'Databases', 'Data', 'Near Earth Space Data')

# Verify that data directories exist
DATA_DIRECTORIES = [
    REAL_TIME_DATA_PATH,
    SPACE_DATA_PATH,
    NEAR_EARTH_SPACE_DATA_PATH
]

for directory in DATA_DIRECTORIES:
    if not os.path.exists(directory):
        logging.warning(f"Data directory does not exist: {directory}")
    else:
        logging.info(f"Data directory exists: {directory}")

# Initialize global variables
model = None
preprocessor = None

# Define target variables
POSSIBLE_TARGETS = {
    'visibility', 'dew_point_temp', 'feels_like', 'temp_min', 'temp_max', 
    'pressure_average', 'humidity', 'wind_speed', 'wind_deg', 'wind_gust', 
    'clouds_all', 'weather_id', 'weather_main', 'weather_description', 'weather_icon' 
}
TARGET_VARIABLES = set()
FEATURE_COLUMNS = set()
NUMERICAL_FEATURES = set()
CATEGORICAL_FEATURES = set()

# Define minimum target data points required
MIN_TARGET_DATA_POINTS = 100  # Adjust based on dataset size and requirements

# ============================
# File System Event Handler
# ============================

class RealTimeDataHandler(FileSystemEventHandler):
    """
    Handles file system events for real-time data ingestion.
    """

    def on_created(self, event):
        """
        Called when a file or directory is created.
        """
        if not event.is_directory and event.src_path.endswith('.csv'):
            logging.info(f'New file detected: {event.src_path}')
            process_new_data(event.src_path)

    def on_modified(self, event):
        """
        Called when a file or directory is modified.
        """
        if not event.is_directory and event.src_path.endswith('.csv'):
            logging.info(f'File modified: {event.src_path}')
            process_new_data(event.src_path)

# ============================
# Data Collection and Loading
# ============================

def collect_all_columns():
    """
    Collects all possible columns from specified data sources and categorizes them into 
    target variables, numerical features, and categorical features.
    This function performs the following steps:
    1. Collects columns from all data directories specified in `data_directories`.
    2. Identifies target variables by intersecting all collected columns with `POSSIBLE_TARGETS`.
    3. Defines feature columns as the set difference between all collected columns and target variables.
    4. Automatically detects categorical and numerical features based on sample data types.
    5. Excludes 'UTC_DATE' from numerical features.
    6. Logs the identified target variables, categorical features, and numerical features.
    Global Variables:
    - FEATURE_COLUMNS: Set of feature columns identified from the data sources.
    - TARGET_VARIABLES: Set of target variables identified from the data sources.
    - NUMERICAL_FEATURES: Set of numerical feature columns identified from the data sources.
    - CATEGORICAL_FEATURES: Set of categorical feature columns identified from the data sources.
    Logs:
    - Logs the process of collecting columns from data sources.
    - Logs warnings if no sample data is available for a column.
    - Logs the identified target variables, categorical features, and numerical features.
    """
    global FEATURE_COLUMNS, TARGET_VARIABLES, NUMERICAL_FEATURES, CATEGORICAL_FEATURES
    logging.info('Collecting all possible columns from data sources...')

    # Collect columns from all data directories
    data_directories = [
        REAL_TIME_DATA_PATH,
        SPACE_DATA_PATH,
        NEAR_EARTH_SPACE_DATA_PATH
    ]

    all_columns = set()
    column_samples = {}
    for directory in data_directories:
        data_frames = load_data_from_directory(directory) 
            
        for df in data_frames.values():
            
            all_columns.update(df.columns)
            for col in df.columns:
                if col not in column_samples:
                    # Get a sample value for the column
                    sample_series = df[col].dropna()
                    sample_value = sample_series.iloc[0] if not sample_series.empty else None
                    column_samples[col] = sample_value

    logging.info(f'All columns found: {all_columns}')

    # Define target variables and feature columns
    TARGET_VARIABLES = all_columns.intersection(POSSIBLE_TARGETS)
    FEATURE_COLUMNS = all_columns - TARGET_VARIABLES

    # Automatically detect categorical features based on data types
    CATEGORICAL_FEATURES = set()
    NUMERICAL_FEATURES = set()
    for col in FEATURE_COLUMNS:
        sample_value = column_samples.get(col, None)
        if sample_value is not None:
            if isinstance(sample_value, numbers.Number):
                NUMERICAL_FEATURES.add(col)
            else:
                CATEGORICAL_FEATURES.add(col)
        else:
            logging.warning(f'No sample data for column "{col}". Cannot determine data type.')

    # Exclude 'UTC_DATE' from numerical features
    NUMERICAL_FEATURES.discard('UTC_DATE')
    NUMERICAL_FEATURES.discard('UTC_DATE_+0000 UTC')

    # Log the identified features
    logging.info(f'Identified TARGET_VARIABLES: {TARGET_VARIABLES}')
    logging.info(f'Identified CATEGORICAL_FEATURES: {CATEGORICAL_FEATURES}')
    logging.info(f'Identified NUMERICAL_FEATURES: {NUMERICAL_FEATURES}')

def get_columns_from_directory(directory_path):
    """
    Retrieves column names from all CSV files within a directory.

    Parameters:
        directory_path (str): Path to the data directory.

    Returns:
        set: Set of column names found in the directory.
    """
    columns = set()
    data_frames = load_data_from_directory(directory_path)
    for df in data_frames:
        columns.update(df.columns)
    return columns

def get_unique_value_counts(column, directories):
    """
    Calculates the total number of unique values for a given column across multiple directories.

    Parameters:
        column (str): Column name.
        directories (list): List of directory paths.

    Returns:
        int: Total number of unique values.
    """
    unique_values = set()
    for directory in directories:
        data_frames = load_data_from_directory(directory)
        for df in data_frames:
            if column in df.columns:
                unique_values.update(df[column].dropna().unique())
    return len(unique_values)


def load_data_from_directory(directory_path, expected_pattern=None):
    """
    Loads CSV files from a directory with varying filename patterns and parses them accordingly.
    Parameters:
        directory_path (str): Path to the parent directory containing CSV files in subdirectories.
    Returns:
        dict: A dictionary with keys as (category, year) tuples and values as DataFrames.
    """
    data_frames = {}
    if not os.path.exists(directory_path):
        logging.warning(f'Directory "{directory_path}" does not exist.')
        return data_frames

    for root, _, files in os.walk(directory_path):
        parent_dir = os.path.basename(root)
        for file in files:
            if not file.endswith('.csv'):
                continue

            if parent_dir == 'Weather':
                pattern = r'^Toronto\.csv$'
            elif parent_dir == 'Space Data':
                pattern = r'^space_events_(?P<year>\d{4})\.csv$'
            elif parent_dir == 'Near Earth Space Data':
                pattern = r'^(?P<category>\w+)-(?P<year>\d{4})\.csv$'
            else:
                logging.warning(f'Unrecognized directory "{parent_dir}". Skipping file "{file}".')
                continue

            regex = re.compile(pattern)
            match = regex.match(file)
            if not match:
                logging.warning(f'File "{file}" does not match the pattern in directory "{parent_dir}".')
                continue

            category = match.group('category') if 'category' in match.groupdict() else None
            if 'year' in match.groupdict():
                year = int(match.group('year'))
            else:
                logging.warning(f'Year not found in file name: {file}. Attempting to parse file for year information.')
                try:
                    df = pd.read_csv(os.path.join(root, file))
                    if 'dt' in df.columns:
                        df['dt'] = pd.to_datetime(df['dt'], unit='s', errors='coerce')
                        df['year'] = df['dt'].dt.year
                        year = df['year'].iloc[0] if not df['year'].isnull().all() else None
                        if year is None:
                            logging.warning(f'Failed to extract year from file: {file}')
                            continue
                    elif 'UNIX_TIMESTAMP' in df.columns:
                        df['UNIX_TIMESTAMP'] = pd.to_datetime(df['UNIX_TIMESTAMP'], unit='s', errors='coerce')
                        df['year'] = df['UNIX_TIMESTAMP'].dt.year
                        year = df['year'].iloc[0] if not df['year'].isnull().all() else None
                        if year is None:
                            logging.warning(f'Failed to extract year from UNIX_TIMESTAMP in file: {file}')
                            continue
                    else:
                        logging.warning(f'dt or UNIX_TIMESTAMP column not found in file: {file}')
                        continue
                except Exception as e:
                    logging.warning(f'Failed to parse file "{file}" for year information: {e}')
                    continue

            file_path = os.path.join(root, file)
            try:
                df = pd.read_csv(os.path.join(root, file))
                df = apply_column_mappings(df)  # Apply column mappings here
                key = (category, year)
                if key in data_frames:
                    data_frames[key] = pd.concat([data_frames[key], df], ignore_index=True)
                else:
                    data_frames[key] = df
                logging.info(f'Loaded file "{file}" with shape {df.shape}.')
            except Exception as e:
                logging.error(f'Error processing file "{file}": {e}')
            except Exception as e:
                logging.warning(f'Failed to read file "{file_path}": {e}')

    logging.info(f'Total files loaded from "{directory_path}": {len(data_frames)}')
    return data_frames

def convert_OpenWeatherMap_to_standard(df):
    # Deprecated due to centralized UTC_DATE handling
    logging.info("convert_OpenWeatherMap_to_standard is deprecated. Use standardize_utc_date instead.")
    return standardize_utc_date(df, df_name='convert_OpenWeatherMap_to_standard')

def apply_column_mappings(df):
    """
    Standardizes column names using predefined mappings.

    Parameters:
        df (pd.DataFrame): DataFrame to apply column mappings to.

    Returns:
        pd.DataFrame: DataFrame with standardized column names.
    """
    COLUMN_MAPPING = {
        'dt': "UNIX_TIMESTAMP", # Ignore
        'dt_iso': "UTC_DATE",  #TODO: Handle the +0000 UTC   
        'timezone': "TIMEZONE", # Ignore
        'city_name': "CITY_NAME", # Ignore
        'visability': 'visibility', # Good
        'dew_point': 'dew_point_temp', # Good
        'feels_like': 'feels_like', # New and good
        'temp_min': 'temp_min', # New and good
        'temp_max': 'temp_max', # New and good
        'pressure': 'pressure_average', # Pressure Average and now good
        'sea_level': 'sea_level', # New and good 
        'grnd_level': "grnd_level", # New and good
        'humidity': "humidity", # New and uncertain
        'wind_speed': 'wind_speed', # New and uncertain
        'wind_deg': 'wind_deg', # New and uncertain
        'wind_gust': 'wind_gust',
        'rain_1h': 'rain_1h',
        'rain_3h': 'rain_3h',
        'snow_1h': 'snow_1h',
        'snow_3h': 'snow_3h',
        'clouds_all': 'clouds_all',
        'weather_id': 'weather_id',
        'weather_main': 'weather_main',
        'weather_description': 'weather_description',
        'weather_icon': 'weather_icon',
        
        'UTC_DATE': 'UTC_DATE', # Old file: 
        'LOCAL_DATE': 'LOCAL_DATE', # Old file: 
        'UTC_YEAR': 'UTC_YEAR', # Old file: 
        'UTC_MONTH': 'UTC_MONTH', # Old file: 
        'UTC_DAY': 'UTC_DAY', # Old file: 
        'LOCAL_YEAR': 'LOCAL_YEAR', # Old file: 
        'LOCAL_MONTH': 'LOCAL_MONTH', # Old file: 
        'LOCAL_DAY': 'LOCAL_DAY', # Old file: 
        'LOCAL_HOUR': 'LOCAL_HOUR', #Old file: 
        'date': 'UTC_DATE', 
        'startTime': 'UTC_DATE',
        'eventTime': 'UTC_DATE', 
        'beginTime': 'UTC_DATE',
        'peakTime': 'UTC_DATE',
        'endTime': 'UTC_DATE',
        'submissionTime': 'UTC_DATE',
        'time21_5': 'UTC_DATE',
        'messageIssueTime': 'UTC_DATE',
        'modelCompletionTime': 'UTC_DATE',
        'temperature_average': 'temperature_average',
        'temperature_min': 'temperature_min',
        'temperature_max': 'temperature_max',
        'temp': 'temperature_average',
        'temperature': 'temperature_average',
        'Temperature': 'temperature_average',
        'Temp (°C)': 'temperature_average',
        # Old file: 'DEW_POINT_TEMP': 'dew_point_temp',
        'dew_point_temp': 'dew_point_temp',
        'dew_point_temp_flag': 'dew_point_temp_flag', # [0, 1, 2] - The validity of the data
        'wind_speed_average': 'wind_speed_average',
        'wind_speed_min': 'wind_speed_min',
        'wind_speed_max': 'wind_speed_max',
        'wind_speed': 'wind_speed_average',
        'Wind Speed': 'wind_speed_average',
        'Wind Speed (km/h)': 'wind_speed_average',
        # Old file: 'WIND_SPEED': 'wind_speed_average',
        # Old file: 'WINDCHILL': 'windchill',
        # Old file: 'WINDCHILL_FLAG': 'windchill_flag',
        # Old file: 'STATION_PRESSURE': 'station_pressure',
        'station_pressure': 'station_pressure',
        # Old file: 'STATION_PRESSURE_FLAG': 'station_pressure_flag',
        'pressure_average': 'pressure_average',
        'pressure_min': 'pressure_min',
        'pressure_max': 'pressure_max',
        # Old file: 'RELATIVE_HUMIDITY': 'relative_humidity',
        # Old file: 'RELATIVE_HUMIDITY_FLAG': 'relative_humidity_flag',
        # Old file: 'HUMIDEX': 'humidex',
        # Old file: 'HUMIDEX_FLAG': 'humidex_flag',
        # Old file: 'VISIBILITY': 'visibility',
        # Old file: 'VISIBILITY_FLAG': 'visibility_flag',
        # Old file: 'PRECIP_AMOUNT': 'precip_amount',
        # Old file: 'PRECIP_AMOUNT_FLAG': 'precip_amount_flag',
        # Old file: 'WIND_DIRECTION': 'wind_direction',
        # Old file: 'WIND_DIRECTION_FLAG': 'wind_direction_flag',
        # Old file: 'STATION_NAME': 'station_name',
        # Old file: 'PROVINCE_CODE': 'province_code',
        'sourceLocation': 'source_location',
        'location': 'location',
        # Old file: 'ID': 'id',
        'activityID': 'activity_id',
        'CME_ID': 'cme_id',
        'flrID': 'flr_id',
        'gstID': 'gst_id',
        'hssID': 'hss_id',
        'mpcID': 'mpc_id',
        'rbeID': 'rbe_id',
        'sepID': 'sep_id',
        'simulationID': 'simulation_id',
        'messageID': 'message_id',
        'messageType': 'message_type',
        'messageURL': 'message_url',
        'sol': 'sol',
        'catalog': 'catalog',
        'cmeAnalyses': 'cme_analyses',
        'linkedEvents': 'linked_events',
        'note': 'note',
        'versionId': 'version_id',
        'link': 'link',
        'isMostAccurate': 'is_most_accurate',
        'associatedCMEID': 'associated_cme_id',
        'featureCode': 'feature_code',
        'dataLevel': 'data_level',
        'measurementTechnique': 'measurement_technique',
        'imageType': 'image_type',
        'tilt': 'tilt',
        'minorHalfWidth': 'minor_half_width',
        'speedMeasuredAtHeight': 'speed_measured_at_height',
        'latitude': 'latitude',
        'longitude': 'longitude',
        'halfAngle': 'half_angle',
        'speed': 'speed',
        'type': 'type',
        'au': 'au',
        'cmeInputs': 'cme_inputs',
        'estimatedShockArrivalTime': 'estimated_shock_arrival_time',
        'estimatedDuration': 'estimated_duration',
        'rmin_re': 'rmin_re',
        'kp_18': 'kp_18',
        'kp_90': 'kp_90',
        'kp_135': 'kp_135',
        'kp_180': 'kp_180',
        'isEarthGB': 'is_earth_gb',
        'impactList': 'impact_list',
    }
    df.rename(columns=COLUMN_MAPPING, inplace=True)
    logging.debug(f'Renamed columns: {df.columns.tolist()}')
    return df

def verify_numerical_features(df):
    """
    Verifies that all numerical features contain numeric data.
    Excludes any columns that contain non-numeric data from NUMERICAL_FEATURES.

    Parameters:
        df (pd.DataFrame): DataFrame to verify numerical features in.
    """


    global NUMERICAL_FEATURES, FEATURE_COLUMNS
    for col in list(NUMERICAL_FEATURES):
        if col not in df.columns:
            logging.warning(f'Column "{col}" not found in DataFrame and will be excluded from numerical features.')
            NUMERICAL_FEATURES.discard(col)
            FEATURE_COLUMNS.discard(col)
            continue

        missing_percentage = df[col].isnull().mean() * 100
        if missing_percentage > 50:  # Threshold can be adjusted
            logging.warning(f'Column "{col}" has {missing_percentage:.2f}% missing values and will be excluded from numerical features.')
            NUMERICAL_FEATURES.discard(col)
            FEATURE_COLUMNS.discard(col)
            continue

        if not pd.api.types.is_numeric_dtype(df[col]):
            # Attempt to convert to numeric, coercing errors to NaN
            df[col] = pd.to_numeric(df[col], errors='coerce')
            if df[col].isnull().all():
                logging.warning(f'Column "{col}" cannot be converted to numeric and will be excluded from numerical features.')
                NUMERICAL_FEATURES.discard(col)
                FEATURE_COLUMNS.discard(col)
            else:
                num_nans = df[col].isnull().sum()
                logging.info(f'Column "{col}" converted to numeric with {num_nans} NaN values.')

def check_missing_columns(df, required_columns):
    """
    Checks if the required columns are present in the DataFrame.

    Parameters:
        df (pd.DataFrame): The input DataFrame.
        required_columns (set): A set of required column names.

    Returns:
        set: A set of missing column names.
    """
    missing_columns = required_columns - set(df.columns)
    if missing_columns:
        logging.warning(f"Missing columns in DataFrame: {missing_columns}")
    return missing_columns

def log_target_statistics(df):
    """
    Logs the count of non-null values and basic statistics for each target variable.
    Excludes target variables with insufficient data.

    Parameters:
        df (pd.DataFrame): DataFrame to analyze target variables in.
    """
    global TARGET_VARIABLES, FEATURE_COLUMNS
    targets_to_exclude = set()

    for target in TARGET_VARIABLES.copy():
        if target in df.columns:
            non_null = df[target].notnull().sum()
            logging.info(f'Target "{target}" has {non_null} non-null entries.')
            if non_null < MIN_TARGET_DATA_POINTS:
                logging.warning(f'Target "{target}" has insufficient data ({non_null} entries) and will be excluded.')
                targets_to_exclude.add(target)
            else:
                logging.info(f'Target "{target}" statistics:\n{df[target].describe()}')
        else:
            logging.warning(f'Target "{target}" not found in DataFrame.')

    if targets_to_exclude:
        TARGET_VARIABLES -= targets_to_exclude
        FEATURE_COLUMNS -= targets_to_exclude
        logging.info(f'Excluded targets due to insufficient data: {targets_to_exclude}')

def analyze_correlations(df):
    """
    Analyzes and logs the correlation between features and target variables.

    Parameters:
        df (pd.DataFrame): DataFrame to perform correlation analysis on.
    """
    global TARGET_VARIABLES, NUMERICAL_FEATURES
    if TARGET_VARIABLES and NUMERICAL_FEATURES:
        # Select only numerical features and target variables
        relevant_cols = list(TARGET_VARIABLES) + list(NUMERICAL_FEATURES)
        available_cols = df.columns.intersection(relevant_cols)

        if len(available_cols) < 2:
            logging.warning('Not enough columns to perform correlation analysis.')
            return

        # Exclude non-numeric columns
        non_numeric_cols = df[available_cols].select_dtypes(exclude=[np.number]).columns.tolist()
        if non_numeric_cols:
            logging.warning(f'Non-numeric columns excluded from correlation analysis: {non_numeric_cols}')
            available_cols = [col for col in available_cols if col not in non_numeric_cols]

        if len(available_cols) < 2:
            logging.warning('No numeric columns available for correlation analysis after excluding non-numeric columns.')
            return

        # Compute correlation matrix
        correlation_matrix = df[available_cols].corr()

        for target in TARGET_VARIABLES:
            if target in correlation_matrix:
                correlations = correlation_matrix[target].sort_values(ascending=False)
                logging.info(f'Correlations with target "{target}":\n{correlations}')
    else:
        logging.warning('Cannot perform correlation analysis without target and numerical feature variables.')

def load_and_combine_data(): #TODO: Investigate this function
    logging.info('Starting to load and combine data from all sources.')

    # Load Hourly Weather Data
    hourly_data = load_data_from_directory(
        REAL_TIME_DATA_PATH
    )
    clear()
    print(f'Hourly Data: {hourly_data}')

    if hourly_data:
        print('Successfully loaded Hourly Weather Data.')
        logging.info('Loaded Hourly Weather Data.')
        hourly_df = pd.concat(hourly_data.values(), ignore_index=True).copy()
        logging.info(f'Hourly DataFrame shape: {hourly_df.shape}')
        logging.debug(f'Hourly DataFrame type: {type(hourly_df)}')
    else:
        logging.warning('No Hourly Weather Data found.')
        hourly_df = pd.DataFrame()

    # Load and aggregate Space Data
    space_data = load_data_from_directory(
        SPACE_DATA_PATH, expected_pattern='{prefix}_events_{year}.csv'
    )
    if space_data:
        logging.info('Loaded Space Data.')
        space_df = pd.concat(space_data.values(), ignore_index=True).copy()
        space_df = apply_column_mappings(space_df)
        logging.info(f'Space DataFrame shape after mapping: {space_df.shape}')
        logging.debug(f'Space DataFrame type: {type(space_df)}')
    else:
        logging.warning('No Space Data found.')
        space_df = pd.DataFrame()

    # Load and aggregate Near Earth Space Data
    near_earth_data = load_data_from_directory(
        NEAR_EARTH_SPACE_DATA_PATH, expected_pattern='{category}-{year}.csv'
    )
    if near_earth_data:
        logging.info('Loaded Near Earth Space Data.')
        # Combine data by categories
        aggregated_near_earth_dataframes = []
        for (category, year), df in near_earth_data.items():
            if category is None:
                logging.warning(f'Category is None for file loaded for year {year}. Skipping.')
                continue
            df['Category'] = category  # Add category column for clarity
            df = apply_column_mappings(df)
            aggregated_df = aggregate_near_earth_space_data(df, category)
            if not aggregated_df.empty:
                aggregated_near_earth_dataframes.append(aggregated_df)
                logging.info(f'Aggregated {category}-{year} DataFrame with shape: {aggregated_df.shape}')
                logging.debug(f'Aggregated DataFrame type: {type(aggregated_df)}')
            else:
                logging.warning(f'Aggregated DataFrame for {category}-{year} is empty.')

        # Combine all categories into one DataFrame
        if aggregated_near_earth_dataframes:
            aggregated_near_earth_df = pd.concat(aggregated_near_earth_dataframes, ignore_index=True)
            logging.info(f'Combined Near Earth Space DataFrame shape: {aggregated_near_earth_df.shape}')
            logging.debug(f'Combined Near Earth Space DataFrame type: {type(aggregated_near_earth_df)}')
        else:
            logging.warning('No valid Near Earth Space Data found after aggregation.')
            aggregated_near_earth_df = pd.DataFrame()
    else:
        logging.warning('No Near Earth Space Data found.')
        aggregated_near_earth_df = pd.DataFrame()

    # Ensure 'UTC_DATE' is consistent across all DataFrames
    data_frames_to_process = {
        'hourly_df': hourly_df,
        'space_df': space_df,
        'aggregated_near_earth_df': aggregated_near_earth_df
    }

    for df_name, df in data_frames_to_process.items():
        if not df.empty:
            df = standardize_utc_date(df, df_name=df_name)
            logging.info(f'Processed "UTC_DATE" column in {df_name}.')
            logging.debug(f'{df_name} "UTC_DATE" column type: {df["UTC_DATE"].dtype}')
        else:
            logging.warning(f'"UTC_DATE" column missing or empty in {df_name}.')
            # Attempt to create 'UTC_DATE' column if missing
            if 'UTC_DATE' not in df.columns:
                df['UTC_DATE'] = pd.NaT
            # Convert to datetime, handling different formats
            df['UTC_DATE'] = pd.to_datetime(df['UTC_DATE'], errors='coerce')
            # Drop rows with invalid dates
            df = df.dropna(subset=['UTC_DATE'])
            logging.info(f'Attempted to process "UTC_DATE" column in {df_name}.')
        logging.debug(f'{df_name} "UTC_DATE" column type: {df["UTC_DATE"].dtype}')

    # Merge all DataFrames on 'UTC_DATE'
    if not hourly_df.empty and 'UTC_DATE' in hourly_df.columns:
        combined_data = hourly_df.copy()
        logging.debug(f'Combined DataFrame initial type: {type(combined_data)}')
        for df_name, df in data_frames_to_process.items():
            if df_name != 'hourly_df' and not df.empty and 'UTC_DATE' in df.columns:
                combined_data = pd.merge(combined_data, df, on='UTC_DATE', how='left')
                logging.info(f'Merged {df_name} into combined DataFrame. Current shape: {combined_data.shape}')
                logging.debug(f'Combined DataFrame type after merging {df_name}: {type(combined_data)}')
    else:
        logging.warning('No Hourly Data available to merge or "UTC_DATE" missing. Returning an empty DataFrame.')
        combined_data = pd.DataFrame()

    # Verify numerical features
    verify_numerical_features(combined_data)

    # Log target statistics and exclude insufficient targets
    log_target_statistics(combined_data)

    # Perform correlation analysis
    analyze_correlations(combined_data)

    logging.info('Data loading and combination process completed.')
    logging.debug(f'Final combined_data type: {type(combined_data)}')

    return combined_data

def aggregate_space_data(space_df):
    """
    Aggregates space data to a daily level.

    Parameters:
        space_df (pd.DataFrame): Space data DataFrame.

    Returns:
        pd.DataFrame: Aggregated space data with 'UTC_DATE' as the date key.
    """
    if space_df.empty:
        logging.warning("Input space_df is empty. Returning empty DataFrame.")
        return pd.DataFrame()

    # Make a copy to avoid SettingWithCopyWarning
    space_df = space_df.copy()

    # Ensure 'UTC_DATE' is in datetime format
    space_df['UTC_DATE'] = pd.to_datetime(space_df['UTC_DATE'], errors='coerce')

    # Drop rows with invalid dates
    space_df = space_df.dropna(subset=['UTC_DATE'])

    # Remove timezone information
    space_df['UTC_DATE'] = space_df['UTC_DATE'].dt.tz_localize(None)

    # Set 'UTC_DATE' as the index
    space_df.set_index('UTC_DATE', inplace=True)

    # Drop non-numeric columns before aggregation
    numeric_cols = space_df.select_dtypes(include=[np.number]).columns
    space_df = space_df[numeric_cols] 

    if space_df.empty:
        logging.warning("No numeric data available for aggregation after processing 'UTC_DATE'.")
        return pd.DataFrame()

    # Resample to daily frequency and aggregate
    aggregated_space_df = space_df.resample('D').mean().reset_index()

    return aggregated_space_df

def aggregate_near_earth_space_data(space_df, category):
    """
    Aggregates space data to align with Earth weather data.
    Uses specific time keepers for each type of space data.

    Parameters:
        space_df (pd.DataFrame): Space data DataFrame.
        category (str): Category of the space event (e.g., 'FLR', 'CME').

    Returns:
        pd.DataFrame: Aggregated space data with 'UTC_DATE' as the date key.
    """
    if space_df.empty:
        logging.warning("Input space_df is empty. Returning empty DataFrame.")
        return pd.DataFrame()

    time_keepers = {
        "CME": "UTC_DATE",
        "CMEAnalysis": "UTC_DATE",
        "FLR": "UTC_DATE",
        "GST": "UTC_DATE",
        "HSS": "UTC_DATE",
        "IPS": "UTC_DATE",
        "MPC": "UTC_DATE",
        "notifications": "UTC_DATE",
        "RBE": "UTC_DATE",
        "SEP": "UTC_DATE",
        "WSAEnlilSimulations": "UTC_DATE",
    }

    # Make a copy to avoid SettingWithCopyWarning
    space_df = space_df.copy()

    if category in time_keepers:
        time_col = time_keepers[category]
        if time_col in space_df.columns:
            # Drop the existing 'UTC_DATE' column to prevent duplication
            if 'UTC_DATE' in space_df.columns and time_col != 'UTC_DATE':
                space_df = space_df.drop(columns=['UTC_DATE'])
                logging.info(f"Dropped existing 'UTC_DATE' column to prevent duplication.")

            # Rename the time column to 'UTC_DATE'
            space_df = space_df.rename(columns={time_col: 'UTC_DATE'})
            logging.info(f"Renamed '{time_col}' to 'UTC_DATE' for category '{category}'.")
        else:
            logging.warning(f"Time column '{time_col}' not found in data for category '{category}'.")
            return pd.DataFrame()  # Return empty DataFrame to avoid further errors
    else:
        logging.warning(f"Category '{category}' not found in time keepers.")
        return pd.DataFrame()

    # Remove duplicate columns
    space_df = space_df.loc[:, ~space_df.columns.duplicated()]

    # Ensure 'UTC_DATE' is in datetime format
    space_df['UTC_DATE'] = pd.to_datetime(space_df['UTC_DATE'], errors='coerce')

    # Drop rows with invalid dates
    space_df = space_df.dropna(subset=['UTC_DATE'])

    # Remove timezone information
    space_df['UTC_DATE'] = space_df['UTC_DATE'].dt.tz_localize(None)

    # Set 'UTC_DATE' as the index
    space_df.set_index('UTC_DATE', inplace=True)

    # Drop non-numeric columns before aggregation
    numeric_cols = space_df.select_dtypes(include=[np.number]).columns
    space_df = space_df[numeric_cols]

    if space_df.empty:
        logging.warning("No numeric data available for aggregation after processing 'UTC_DATE'.")
        return pd.DataFrame()

    # Resample to daily frequency and aggregate
    aggregated_space_df = space_df.resample('D').mean().reset_index()

    # Verify 'UTC_DATE' exists
    if 'UTC_DATE' not in aggregated_space_df.columns:
        logging.error("'UTC_DATE' column is missing after aggregation.")
        return pd.DataFrame()
    else:
        logging.info(f"'UTC_DATE' column is present after aggregation with {aggregated_space_df.shape[0]} records.")

    return aggregated_space_df

def generate_forecasts():
    """
    Generates weather forecasts using historical weather and space data.
    Saves the forecasts in distinct logs for each prediction period and returns them.
    """
    global model, preprocessor
    logging.info('Starting forecast generation...')
    
    # Load the model if not loaded
    if model is None:
        model_path = os.path.join(KNOWLEDGE_PATH, 'model.joblib')
        if os.path.exists(model_path):
            try:
                model = joblib.load(model_path)
                logging.info('Model loaded successfully from model.joblib.')
            except Exception as e:
                logging.error(f'Failed to load model: {e}')
                print(f'Failed to load model: {e}')
                return {}
        else:
            logging.error('Model file model.joblib does not exist. Please train the model first.')
            print('Model file model.joblib does not exist. Please train the model first.')
            return {}

    # Load the preprocessor if not loaded
    if preprocessor is None:
        preprocessor_path = os.path.join(KNOWLEDGE_PATH, 'preprocessor.joblib')
        if os.path.exists(preprocessor_path):
            try:
                preprocessor = joblib.load(preprocessor_path)
                logging.info('Preprocessor loaded successfully from preprocessor.joblib.')
            except Exception as e:
                logging.error(f'Failed to load preprocessor: {e}')
                print(f'Failed to load preprocessor: {e}')
                return {}
        else:
            logging.error('Preprocessor file preprocessor.joblib does not exist. Please train the model first.')
            print('Preprocessor file preprocessor.joblib does not exist. Please train the model first.')
            return {}
    
    combined_data = load_and_prepare_data()

    # Define forecast periods 
    forecast_periods = {
        "monthly": 30,
        "daily": 1,
        "weekly": 7,
        "yearly": 365
    }
    
    predictions_collection = {}

    # Iterate through forecast periods
    for period_name, days_ahead in forecast_periods.items():
        # Define the target prediction start date
        today = datetime.now()
        prediction_start_date = (today + timedelta(days=days_ahead)).replace(hour=0, minute=0, second=0, microsecond=0)

        logging.info(f"Preparing data for {period_name} forecast starting from {prediction_start_date.date()}...")

        # Load and combine data up to the prediction date
        combined_data = load_and_combine_data()
        if combined_data.empty:
            logging.warning(f"No data available for {period_name} forecast. Skipping...")
            continue

        # Filter data up to one month before the prediction date
        combined_data = combined_data[combined_data['UTC_DATE'] < prediction_start_date]
        
        # Preprocess data
        X_data, _ = preprocess_data(combined_data, is_training=False)
        if X_data is None:
            logging.error(f"Failed to preprocess data for {period_name} forecast.")
            continue

        # **Debugging Statements:**
        logging.debug(f"Type of X_data: {type(X_data)}")
        logging.debug(f"Shape of X_data: {X_data.shape if hasattr(X_data, 'shape') else 'No shape'}")
        if hasattr(X_data, 'dtype'):
            logging.debug(f"Data type of X_data: {X_data.dtype}")
        else:
            logging.debug("X_data does not have 'dtype' attribute.")

        # Perform predictions
        logging.info(f"Generating predictions for {period_name} forecast...")
        try:
            predictions = model.predict(X_data)
            logging.debug(f"Predictions shape: {predictions.shape}")
            logging.debug(f"Predictions type: {type(predictions)}")
            predictions_df = pd.DataFrame(
                predictions, columns=list(TARGET_VARIABLES)
            )
            predictions_df['forecast_period'] = period_name
            predictions_df['timestamp'] = datetime.now()
            predictions_df['forecast_date'] = prediction_start_date

            # Collect predictions for this forecast period
            predictions_collection[period_name] = predictions_df

            # Save predictions to a log file
            forecast_log_path = os.path.join(
                KNOWLEDGE_PATH, f"{period_name}_forecast_log.csv"
            )
            predictions_df.to_csv(
                forecast_log_path, mode='a', header=not os.path.exists(forecast_log_path), index=False
            )
            logging.info(f"Saved {period_name} forecast predictions to {forecast_log_path}.")
        except Exception as e:
            logging.error(f"Failed to generate predictions for {period_name} forecast: {e}")
            print(f"Failed to generate predictions for {period_name} forecast: {e}")

    # Return predictions as a dictionary of DataFrames
    return predictions_collection

# ============================
# Data Preprocessing
# ============================

def validate_numeric_columns(df):
    """
    Validates that numeric columns in the DataFrame contain only numeric data.

    Parameters:
        df (pd.DataFrame): DataFrame to validate.

    Returns:
        pd.DataFrame: Cleaned DataFrame with invalid rows dropped or coerced.
    """
    for col in df.select_dtypes(include='object').columns:
        try:
            df[col] = pd.to_numeric(df[col], errors='coerce')
            logging.info(f"Column '{col}' successfully coerced to numeric.")
        except Exception as e:      
            logging.warning(f"Column '{col}' contains non-numeric data: {e}")
            handle_non_numeric_columns(df)
    return df

def standardize_utc_date(df, df_name=''):
    """
    Standardizes the 'UTC_DATE' column in the given DataFrame by handling multiple date formats.
    
    Parameters:
        df (pd.DataFrame): The DataFrame containing the 'UTC_DATE' column.
        df_name (str): Optional name of the DataFrame for logging purposes.
    
    Returns:
        pd.DataFrame: The DataFrame with the standardized 'UTC_DATE' column.
    """
    print('Converting data to standard UTC_DATE format...')
    # Ensure 'UTC_DATE' column exists
    if 'UTC_DATE' not in df.columns:
        logging.warning(f"'UTC_DATE' column missing in DataFrame: {df_name}. Attempting to create it.")
        df['UTC_DATE'] = pd.NaT  # Create 'UTC_DATE' column with NaT if missing
    
    # Convert 'UTC_DATE' to string to handle various formats
    df['UTC_DATE'] = df['UTC_DATE'].astype(str)
    
    # Specific preprocessing for 'hourly_df' to remove timezone info if necessary
    if df_name == 'hourly_df':
        # Remove timezone information like " +0000 UTC" or similar patterns
        df['UTC_DATE'] = df['UTC_DATE'].apply(lambda x: re.sub(r'\s*\+\d{4}\s*UTC$', '', x).strip())
    
    # Additional preprocessing steps to handle common date format issues
    # For example, replacing slashes with hyphens, removing extra spaces, etc.
    df['UTC_DATE'] = df['UTC_DATE'].str.replace('/', '-', regex=False).str.strip()
    
    # Comprehensive list of common datetime formats
    common_formats = [
        # ISO 8601 Formats
        '%Y-%m-%dT%H:%M:%S.%fZ',    # e.g., 2023-08-15T13:45:30.000Z
        '%Y-%m-%dT%H:%M:%SZ',       # e.g., 2023-08-15T13:45:30Z
        '%Y-%m-%dT%H:%M:%S.%f%z',   # e.g., 2023-08-15T13:45:30.000+0000
        '%Y-%m-%dT%H:%M:%S%z',      # e.g., 2023-08-15T13:45:30+0000
        '%Y-%m-%d %H:%M:%S%z',      # e.g., 2023-08-15 13:45:30+0000
        '%Y-%m-%d %H:%M:%S.%f%z',   # e.g., 2023-08-15 13:45:30.000+0000

        # Common Formats with Different Separators
        '%Y-%m-%d %H:%M:%S',        # e.g., 2023-08-15 13:45:30
        '%d-%m-%Y %H:%M:%S',        # e.g., 15-08-2023 13:45:30
        '%m-%d-%Y %H:%M:%S',        # e.g., 08-15-2023 13:45:30
        '%m/%d/%Y %H:%M:%S',        # e.g., 08/15/2023 13:45:30
        '%d/%m/%Y %H:%M:%S',        # e.g., 15/08/2023 13:45:30
        '%Y.%m.%d %H:%M:%S',        # e.g., 2023.08.15 13:45:30
        '%d.%m.%Y %H:%M:%S',        # e.g., 15.08.2023 13:45:30
        '%Y%m%d %H:%M:%S',           # e.g., 20230815 13:45:30
        '%d%m%Y %H:%M:%S',           # e.g., 15082023 13:45:30
        '%Y%m%dT%H%M%S',            # e.g., 20230815T134530
        '%Y%m%d %H%M%S',            # e.g., 20230815 134530

        # Extended with Milliseconds
        '%Y-%m-%d %H:%M:%S.%f',     # e.g., 2023-08-15 13:45:30.123456
        '%d-%m-%Y %H:%M:%S.%f',     # e.g., 15-08-2023 13:45:30.123456
        '%m/%d/%Y %H:%M:%S.%f',     # e.g., 08/15/2023 13:45:30.123456
        '%d/%m/%Y %H:%M:%S.%f',     # e.g., 15/08/2023 13:45:30.123456
        '%Y.%m.%d %H:%M:%S.%f',     # e.g., 2023.08.15 13:45:30.123456
        '%d.%m.%Y %H:%M:%S.%f',     # e.g., 15.08.2023 13:45:30.123456

        # 12-Hour Formats with AM/PM
        '%Y-%m-%d %I:%M:%S %p',     # e.g., 2023-08-15 01:45:30 PM
        '%d-%m-%Y %I:%M:%S %p',     # e.g., 15-08-2023 01:45:30 PM
        '%m/%d/%Y %I:%M:%S %p',     # e.g., 08/15/2023 01:45:30 PM
        '%d/%m/%Y %I:%M:%S %p',     # e.g., 15/08/2023 01:45:30 PM
        '%Y.%m.%d %I:%M:%S %p',     # e.g., 2023.08.15 01:45:30 PM
        '%d.%m.%Y %I:%M:%S %p',     # e.g., 15.08.2023 01:45:30 PM

        # Unix Timestamp (seconds since epoch)
        '%s',                        # e.g., 1692107130

        # Other Common Formats
        '%B %d, %Y %H:%M:%S',        # e.g., August 15, 2023 13:45:30
        '%b %d, %Y %H:%M:%S',        # e.g., Aug 15, 2023 13:45:30
        '%d %B %Y %H:%M:%S',         # e.g., 15 August 2023 13:45:30
        '%d %b %Y %H:%M:%S',         # e.g., 15 Aug 2023 13:45:30
        '%Y/%m/%d %H:%M:%S',         # e.g., 2023/08/15 13:45:30
        '%d/%b/%Y %H:%M:%S',         # e.g., 15/Aug/2023 13:45:30
        '%d-%b-%Y %H:%M:%S',         # e.g., 15-Aug-2023 13:45:30
        '%d.%b.%Y %H:%M:%S',         # e.g., 15.Aug.2023 13:45:30
        '%d %b %Y %H:%M:%S',         # e.g., 15 Aug 2023 13:45:30
        '%Y %b %d %H:%M:%S',         # e.g., 2023 Aug 15 13:45:30
        '%Y %B %d %H:%M:%S',         # e.g., 2023 August 15 13:45:30
        '%d %B %Y %H:%M:%S',         # e.g., 15 August 2023 13:45:30

        # Timestamps with Timezone Abbreviations
        '%Y-%m-%d %H:%M:%S %Z',      # e.g., 2023-08-15 13:45:30 UTC
        '%d-%m-%Y %H:%M:%S %Z',      # e.g., 15-08-2023 13:45:30 UTC
        '%m/%d/%Y %H:%M:%S %Z',      # e.g., 08/15/2023 13:45:30 UTC
        '%d/%m/%Y %H:%M:%S %Z',      # e.g., 15/08/2023 13:45:30 UTC
        '%Y.%m.%d %H:%M:%S %Z',      # e.g., 2023.08.15 13:45:30 UTC
        '%d.%m.%Y %H:%M:%S %Z',      # e.g., 15.08.2023 13:45:30 UTC

        # ISO 8601 with milliseconds and timezone
        '%Y-%m-%dT%H:%M:%S.%f%z',    # e.g., 2023-08-15T13:45:30.000+0000
        '%Y-%m-%dT%H:%M:%S%z',       # e.g., 2023-08-15T13:45:30+0000

        # Compact Formats
        '%Y%m%dT%H%M%S%z',           # e.g., 20230815T134530+0000
        '%Y%m%dT%H%M%S.%f%z',        # e.g., 20230815T134530.000+0000

        # Other Variations
        '%Y/%m/%dT%H:%M:%S',         # e.g., 2023/08/15T13:45:30
        '%Y-%m-%dT%H:%M',            # e.g., 2023-08-15T13:45
        '%Y-%m-%d %H:%M',            # e.g., 2023-08-15 13:45
        '%d-%m-%Y %H:%M',            # e.g., 15-08-2023 13:45
        '%m/%d/%Y %H:%M',            # e.g., 08/15/2023 13:45
        '%d/%m/%Y %H:%M',            # e.g., 15/08/2023 13:45
        '%Y.%m.%d %H:%M',            # e.g., 2023.08.15 13:45
        '%d.%m.%Y %H:%M',            # e.g., 15.08.2023 13:45

        # Without Time
        '%Y-%m-%d',                  # e.g., 2023-08-15
        '%d-%m-%Y',                  # e.g., 15-08-2023
        '%m/%d/%Y',                  # e.g., 08/15/2023
        '%d/%m/%Y',                  # e.g., 15/08/2023
        '%Y.%m.%d',                  # e.g., 2023.08.15
        '%d.%m.%Y',                  # e.g., 15.08.2023
        '%Y%m%d',                    # e.g., 20230815
        '%d%m%Y',                    # e.g., 15082023

        # With Day Name
        '%A, %Y-%m-%d %H:%M:%S',    # e.g., Tuesday, 2023-08-15 13:45:30
        '%a, %Y-%m-%d %H:%M:%S',    # e.g., Tue, 2023-08-15 13:45:30
        '%A %d %B %Y %H:%M:%S',      # e.g., Tuesday 15 August 2023 13:45:30
        '%a %d %b %Y %H:%M:%S',      # e.g., Tue 15 Aug 2023 13:45:30

        # With Different Orderings
        '%m-%d-%Y %H:%M:%S',        # e.g., 08-15-2023 13:45:30
        '%b %d %Y %H:%M:%S',         # e.g., Aug 15 2023 13:45:30
        '%B %d %Y %H:%M:%S',         # e.g., August 15 2023 13:45:30

        # Year First with Dots
        '%Y.%m.%dT%H:%M:%S',         # e.g., 2023.08.15T13:45:30
        '%Y.%m.%dT%H:%M:%S.%f',      # e.g., 2023.08.15T13:45:30.000

        # Mixed Separators
        '%Y-%m-%d/%H:%M:%S',         # e.g., 2023-08-15/13:45:30
        '%d-%m-%Y/%H:%M:%S',         # e.g., 15-08-2023/13:45:30

        # Without Seconds
        '%Y-%m-%d %H:%M',            # e.g., 2023-08-15 13:45
        '%d-%m-%Y %H:%M',            # e.g., 15-08-2023 13:45
        '%m/%d/%Y %H:%M',            # e.g., 08/15/2023 13:45
        '%d/%m/%Y %H:%M',            # e.g., 15/08/2023 13:45
        '%Y.%m.%d %H:%M',            # e.g., 2023.08.15 13:45
        '%d.%m.%Y %H:%M',            # e.g., 15.08.2023 13:45
        '%Y%m%d %H:%M',              # e.g., 20230815 13:45
        '%d%m%Y %H:%M',              # e.g., 15082023 13:45

        # ISO 8601 without Time
        '%Y-%m-%dT%H:%M:%S',         # e.g., 2023-08-15T13:45:30
        '%Y-%m-%dT%H:%M',            # e.g., 2023-08-15T13:45
    ]
    
    # Function to attempt parsing with multiple formats
    def parse_date(date_str):
        # Handle Unix timestamp separately
        if re.fullmatch(r'\d+', date_str):
            try:
                return pd.to_datetime(int(date_str), unit='s', utc=True)
            except (ValueError, TypeError):
                pass  # Proceed to other formats
        
        # Iterate through the common_formats
        for fmt in common_formats:
            try:
                # Handle timezone-aware formats
                if '%z' in fmt or '%Z' in fmt:
                    parsed = pd.to_datetime(date_str, format=fmt, utc=True)
                else:
                    parsed = pd.to_datetime(date_str, format=fmt, utc=True)
                return parsed
            except (ValueError, TypeError):
                continue  # Try the next format
        
        # Fallback to pandas' generic parser
        try:
            parsed = pd.to_datetime(date_str, infer_datetime_format=True, utc=True)
            return parsed
        except (ValueError, TypeError):
            return pd.NaT  # Return Not-a-Time for invalid parsing
    print(f"Before: {df['UTC_DATE']}")
    # Apply the parsing function to 'UTC_DATE' column
    df['UTC_DATE'] = df['UTC_DATE'].apply(parse_date)
    print(f"After: {df['UTC_DATE']}")
    # Drop rows with invalid 'UTC_DATE'
    initial_count = df.shape[0]
    df = df.dropna(subset=['UTC_DATE'])
    print(f"Final: {df['UTC_DATE']}")
    final_count = df.shape[0]
    dropped_rows = initial_count - final_count
    if dropped_rows > 0:
        logging.info(f"Dropped {dropped_rows} rows with invalid 'UTC_DATE' in DataFrame: {df_name}.")
    
    # Standardize the datetime format as string in UTC
    df['UTC_DATE'] = df['UTC_DATE'].dt.strftime('%Y-%m-%d %H:%M:%S')
    print(f"Standardized: {df['UTC_DATE']}")
    
    logging.info(f"Standardized 'UTC_DATE' in DataFrame: {df_name}.")
    return df

def handle_non_numeric_columns(df):
    """
    Handles non-numeric columns by encoding or converting them to numeric.
    Input DataFrame is modified in place.
    Output: DataFrame with non-numeric columns encoded or converted to numeric.
    """
    non_numeric_cols = df.select_dtypes(include=['object']).columns
    for col in non_numeric_cols:
        try:
            # Attempt to convert to numeric
            df[col] = pd.to_numeric(df[col], errors='coerce')
            logging.info(f"Column '{col}' successfully converted to numeric.")
        except Exception as e:
            # If conversion fails, apply encoding
            logging.warning(f"Column '{col}' contains non-numeric data and will be encoded: {e}")
            df[col] = df[col].astype('category').cat.codes
            logging.info(f"Column '{col}' successfully encoded as categorical.")
    return df

def parse_json(value):
    """
    Attempts to parse a JSON-like string into a Python object.

    Parameters:
        value (str): The string to parse.

    Returns:
        list or dict or None: Parsed object or None if parsing fails.
    """
    
    if pd.isnull(value):
        return None
    
    try:
        # Clean up the string
        value = str(value).strip()
        # Replace single quotes with double quotes
        value = value.replace("'", '"')
        # Fix common issues with brackets and braces
        if not (value.startswith('[') or value.startswith('{')):
            value = '[' + value + ']'
        if not (value.endswith(']') or value.endswith('}')):
            value = value + ']'

        # Remove any trailing commas
        value = value.rstrip(',').rstrip(';')

        # Parse the JSON string
        return json.loads(value)
    except json.JSONDecodeError as e:
        logging.warning(f"JSON decoding failed for value: {value} with error: {e}")
        return None

def clean_utc_date(date_series):
    def clean_single_date(date_str):
        if pd.isnull(date_str):
            return pd.NaT
        try:
            # Remove any trailing non-datetime information (e.g., '-CME-001')
            clean_str = str(date_str).split('-')[0]
            # Handle cases where time is missing
            if 'T' not in clean_str:
                clean_str += 'T00:00:00'
            return pd.to_datetime(clean_str)
        except Exception as e:
            logging.warning(f"Failed to parse date '{date_str}': {e}")
            return pd.NaT
    return date_series.apply(clean_single_date)


def process_text_features(df):
    """
    Identifies text columns and processes them using TF-IDF vectorization.

    Parameters:
        df (pd.DataFrame): The input DataFrame.

    Returns:
        pd.DataFrame: DataFrame with text features processed.
    """
    # Identify text columns
    text_columns = df.select_dtypes(include=['object']).columns
    for col in text_columns:
        df[col] = df[col].astype(str).fillna('')
        tfidf = TfidfVectorizer()
        tfidf_matrix = tfidf.fit_transform(df[col])
        tfidf_df = pd.DataFrame(tfidf_matrix.toarray(), columns=[f"{col}_{word}" for word in tfidf.get_feature_names_out()], index=df.index)
        df = pd.concat([df.drop(col, axis=1), tfidf_df], axis=1)
    return df


def preprocess_data(df, is_training=True):
    global preprocessor, num_imputer, cat_imputer, FEATURE_COLUMNS, TARGET_VARIABLES, CATEGORICAL_FEATURES, NUMERICAL_FEATURES
    logging.info('Preprocessing data...')
    
    # Make a copy to prevent errors
    df = df.copy()
    
    # Return none if data is empty
    if df.empty:
        logging.error('Input DataFrame is empty.')
        return None, None

    # Apply column mappings
    df = apply_column_mappings(df) # Good: No issues

    # Integrate the text processing into the preprocessing pipeline
    df = process_text_features(df)

    # Step 1: Validate 'UTC_DATE' and extract time-based features
    df = validate_and_extract_time_features(df)
    if df is None:
        return None, None

    # Step 2: Parse JSON-like strings into actual lists/dicts
    df = parse_json_columns(df)

    # Step 3: Extract features from parsed JSON columns (if necessary)
    df = extract_features_from_json(df)

    # Step 4: Handle missing values
    df = handle_missing_values(df, is_training)

    # Step 5: Check required features and add default values for missing columns
    df = add_default_values_for_missing_columns(df)

    # Step 6: Extract features and targets
    X, y = extract_features_and_targets(df, is_training, TARGET_VARIABLES)
    if X is None or (is_training and y is None):
        return None, None

    # Ensure 'UTC_DATE' is properly formatted before applying the preprocessor
    if 'UTC_DATE' in X.columns:
        X['UTC_DATE'] = pd.to_datetime(X['UTC_DATE'], errors='coerce')
        X['UTC_DATE'] = X['UTC_DATE'].dt.strftime('%Y-%m-%d %H:%M:%S')

    # Drop columns with all null values
    X = X.dropna(axis=1, how='all')
    logging.debug(f"X columns after dropping all-null columns: {X.columns}")

    # Handle links and string formats
    for col in X.select_dtypes(include='object').columns:
        if X[col].str.contains('http', na=False).any():
            X[col] = X[col].apply(lambda x: 1 if isinstance(x, str) and 'http' in x else 0)
            logging.info(f"Processed column '{col}' for links.")
        else:
            X.loc[:, col] = X[col].astype(str).str.extract(r'(\d+)', expand=False).astype(float)
            logging.info(f"Processed column '{col}' for numeric extraction.")

    # Check for invalid data (e.g., NaN values) and handle them using forward fill and backward fill
    if X.isnull().any(axis=None):
        logging.warning("Data contains NaN values. Filling NaNs using forward fill and backward fill.")
        X = X.ffill().bfill()
        X = X.infer_objects(copy=False)
        logging.debug("Filled NaN values.")

    # Ensure columns in NUMERICAL_FEATURES and CATEGORICAL_FEATURES are present
    missing_numerical_features = [col for col in NUMERICAL_FEATURES if col not in X.columns]
    missing_categorical_features = [col for col in CATEGORICAL_FEATURES if col not in X.columns]
    if missing_numerical_features or missing_categorical_features:
        logging.warning(f"Missing numerical features: {missing_numerical_features}")
        logging.warning(f"Missing categorical features: {missing_categorical_features}")

        # Fill missing numerical features with the mean of the column
        for feature in missing_numerical_features:
            X[feature] = 0  # Default value for missing numerical features

        # Fill missing categorical features with a default category
        for feature in missing_categorical_features:
            X[feature] = 'missing'  # Default value for missing categorical features

        # Update NUMERICAL_FEATURES and CATEGORICAL_FEATURES to include the filled features
        NUMERICAL_FEATURES = [col for col in NUMERICAL_FEATURES if col in X.columns]
        CATEGORICAL_FEATURES = [col for col in CATEGORICAL_FEATURES if col in X.columns]

    # Simulate missing values in rows
    for col in X.columns:
        if X[col].isnull().any():
            # Use a simple model to predict missing values
            model = LinearRegression()
            not_null = X[X[col].notnull()]
            null = X[X[col].isnull()]
            if not_null.empty:
                continue
            X_train = not_null.drop(columns=[col])
            y_train = not_null[col]
            model.fit(X_train, y_train)
            X.loc[X[col].isnull(), col] = model.predict(null.drop(columns=[col]))
            logging.info(f"Simulated missing values in column '{col}'.")

    NUMERICAL_FEATURES = [col for col in NUMERICAL_FEATURES if col in X.columns]
    CATEGORICAL_FEATURES = [col for col in CATEGORICAL_FEATURES if col in X.columns]
    logging.debug(f"Numerical features: {NUMERICAL_FEATURES}")
    logging.debug(f"Categorical features: {CATEGORICAL_FEATURES}")

    # Check y for NaN values and fill them with the mean of the column
    if is_training:
        if y.isnull().any().any():
            logging.warning("y contains NaN values. Filling NaNs with the mean of the column.")
            y = y.fillna(y.mean())
    
    # Step 7: Initialize and apply the preprocessor
    X_processed, y = apply_preprocessor(X, y, is_training)
    if X_processed is None:
        return None, None

    return (X_processed, y) if is_training else (X_processed, None)

def validate_and_extract_time_features(df):
    global FEATURE_COLUMNS, NUMERICAL_FEATURES
    df = standardize_utc_date(df, df_name='validate_and_extract_time_features')
    
    # Proceed with extracting time-based features
    try:
        df['year'] = pd.to_datetime(df['UTC_DATE']).dt.year
        df['month'] = pd.to_datetime(df['UTC_DATE']).dt.month
        df['day'] = pd.to_datetime(df['UTC_DATE']).dt.day
        df['hour'] = pd.to_datetime(df['UTC_DATE']).dt.hour
        df['day_of_week'] = pd.to_datetime(df['UTC_DATE']).dt.dayofweek
    except Exception as e:
        logging.error(f"Failed to extract time features: {e}")
        return None

    # Ensure all required columns are present
    required_columns = ['year', 'month', 'day', 'hour', 'day_of_week']
    missing_columns = [col for col in required_columns if col not in df.columns]
    if missing_columns:
        logging.warning(f"Missing columns after extraction: {missing_columns}")
        return None

    return df

def encode_strings(df):
    """
    Converts string columns in the DataFrame to categorical integer codes.

    Parameters:
        df (pd.DataFrame): The input DataFrame.

    Returns:
        pd.DataFrame: DataFrame with encoded string columns.
    """
    categorical_features = set()
    object_cols = df.select_dtypes(include=['object']).columns
    for col in object_cols:
        df[col] = df[col].astype('category').cat.codes
        categorical_features.add(col)
        logging.info(f'Encoded column "{col}" as categorical.')
    return df

def encode_strings_one_hot(df):
    """
    One-hot encodes string columns in the DataFrame.

    Parameters:
        df (pd.DataFrame): The input DataFrame.

    Returns:
        pd.DataFrame: DataFrame with one-hot encoded string columns.
    """
    string_columns = df.select_dtypes(include=['object']).columns
    if not string_columns.empty:
        df = pd.get_dummies(df, columns=string_columns, drop_first=True)
        logging.info(f'Applied One-Hot Encoding to columns: {list(string_columns)}')
    return df

def parse_json_columns(df):
    """
    Parses JSON-like string columns into actual lists or dictionaries.

    Parameters:
        df (pd.DataFrame): The input DataFrame.

    Returns:
        pd.DataFrame: DataFrame with parsed JSON columns.
    """
    try:
        json_columns = [col for col in df.columns if df[col].dtypes == 'object']
    except Exception as e:
        logging.error(f'Error accessing column data types: {e}')
        return df
    for col in json_columns:
        try:
            df[col] = df[col].apply(json.loads)
            logging.info(f'Parsed JSON-like column: {col}')
        except Exception as e:
            logging.error(f'Failed to parse column {col}: {e}')
    return df

def extract_features_from_json(df):
    """
    Extracts features from parsed JSON columns.

    Parameters:
        df (pd.DataFrame): The input DataFrame.

    Returns:
        pd.DataFrame: DataFrame with extracted features.
    """
    global FEATURE_COLUMNS, NUMERICAL_FEATURES
    if 'impactList' in df.columns:
        df['impact_latitude'] = df['impactList'].apply(
            lambda x: x[0].get('latitude') if isinstance(x, list) and len(x) > 0 and isinstance(x[0], dict) else np.nan)
        df['impact_longitude'] = df['impactList'].apply(
            lambda x: x[0].get('longitude') if isinstance(x, list) and len(x) > 0 and isinstance(x[0], dict) else np.nan)
        NUMERICAL_FEATURES.update(['impact_latitude', 'impact_longitude'])
        FEATURE_COLUMNS.update(['impact_latitude', 'impact_longitude'])
        logging.info('Extracted features from "impactList".')

    if 'cme_analyses' in df.columns:
        df['cme_speed'] = df['cme_analyses'].apply(
            lambda x: x[0].get('speed') if isinstance(x, list) and len(x) > 0 and isinstance(x[0], dict) else np.nan)
        NUMERICAL_FEATURES.add('cme_speed')
        FEATURE_COLUMNS.add('cme_speed')
        logging.info('Extracted features from "cme_analyses".')
    return df

def handle_missing_values(df, is_training):
    """
    Handles missing values in the DataFrame by predicting them using a simple model.

    Parameters:
        df (pd.DataFrame): DataFrame to handle missing values.
        is_training (bool): Flag indicating if the data is for training.
    """
    global TARGET_VARIABLES, NUMERICAL_FEATURES, CATEGORICAL_FEATURES

    # Check if all target variables exist in the DataFrame
    missing_cols = [col for col in TARGET_VARIABLES if col not in df.columns]
    if missing_cols:
        logging.warning(f'The following target columns are missing from the DataFrame: {missing_cols}')
        return df

    # Drop rows with missing values in target variables
    df = df.dropna(subset=list(TARGET_VARIABLES))

    # Handle numerical features
    if NUMERICAL_FEATURES:
        for feature in NUMERICAL_FEATURES:
            if df[feature].isnull().any():
                # Train a simple model to predict missing values
                model = LinearRegression()
                not_null = df[df[feature].notnull()]
                null = df[df[feature].isnull()]
                if not_null.empty:
                    continue
                X_train = not_null.drop(columns=[feature])
                y_train = not_null[feature]
                model.fit(X_train, y_train)
                df.loc[df[feature].isnull(), feature] = model.predict(null.drop(columns=[feature]))

    # Handle categorical features
    if CATEGORICAL_FEATURES:
        for feature in CATEGORICAL_FEATURES:
            if df[feature].isnull().any():
                # Use the most frequent value to fill missing values
                imputer = SimpleImputer(strategy='most_frequent')
                df[feature] = imputer.fit_transform(df[[feature]])

    return df

def add_default_values_for_missing_columns(df):
    """
    Adds default values for missing columns.

    Parameters:
        df (pd.DataFrame): The input DataFrame.

    Returns:
        pd.DataFrame: DataFrame with default values for missing columns.
    """
    global NUMERICAL_FEATURES, CATEGORICAL_FEATURES
    required_features = list(NUMERICAL_FEATURES) + list(CATEGORICAL_FEATURES)
    missing_features = set(required_features) - set(df.columns)
    if missing_features:
        logging.warning(f"Missing features in DataFrame: {missing_features}")
        for feature in missing_features:
            if feature in NUMERICAL_FEATURES:
                df[feature] = 0  # Default value for numerical features
            elif feature in CATEGORICAL_FEATURES:
                df[feature] = 'missing'  # Default value for categorical features
    return df



def fill_missing_columns(df, target_columns):
    """
    Fills missing columns with predicted or default values, excluding target columns.
    
    Parameters:
        df (pd.DataFrame): The input DataFrame.
        target_columns (list): List of target column names.
    
    Returns:
        pd.DataFrame: DataFrame with predicted or default values for missing feature columns.
    """
    global NUMERICAL_FEATURES, CATEGORICAL_FEATURES
    # Exclude target columns from required features
    feature_features = list(NUMERICAL_FEATURES) + list(CATEGORICAL_FEATURES)
    feature_features = [feature for feature in feature_features if feature not in target_columns]
    
    missing_features = set(feature_features) - set(df.columns)
    if missing_features:
        logging.warning(f"Missing features in DataFrame: {missing_features}")
        for feature in missing_features:
            if feature in NUMERICAL_FEATURES:
                # Use mean imputation for numerical features
                imputer = SimpleImputer(strategy='mean')
                df[feature] = imputer.fit_transform(df[[feature]])
            elif feature in CATEGORICAL_FEATURES:
                # Use most frequent value imputation for categorical features
                imputer = SimpleImputer(strategy='most_frequent')
                df[feature] = imputer.fit_transform(df[[feature]])
    
    # Predict missing values for existing columns
    for feature in NUMERICAL_FEATURES:
        if df[feature].isnull().any():
            model = LinearRegression()
            not_null = df[df[feature].notnull()]
            null = df[df[feature].isnull()]
            if not_null.empty:
                continue
            X_train = not_null.drop(columns=[feature])
            y_train = not_null[feature]
            model.fit(X_train, y_train)
            df.loc[df[feature].isnull(), feature] = model.predict(null.drop(columns=[feature]))
    
    for feature in CATEGORICAL_FEATURES:
        if df[feature].isnull().any():
            imputer = SimpleImputer(strategy='most_frequent')
            try:
                df[feature] = imputer.fit_transform(df[[feature]])
            except Exception as e:
                print(f'Error: {e}')
                print(f'Failed to impute missing values for {feature}.')
                df[feature] = 'missing'
    
    return df

def extract_features_and_targets(df, is_training, target_columns):
    """
    Extracts features and target variables from the DataFrame.

    Parameters:
        df (pd.DataFrame): DataFrame to extract features and targets.
        is_training (bool): Flag indicating if the data is for training.
        target_columns (list): List of target columns to extract.

    Returns:
        X (pd.DataFrame): DataFrame containing features.
        y (pd.DataFrame): DataFrame containing target variables.
    """
    # Print DataFrame columns for debugging
    logging.info(f'DataFrame columns before extraction: {df.columns.tolist()}')

    # Check if target columns exist in the DataFrame
    existing_target_columns = [col for col in target_columns if col in df.columns]
    missing_target_columns = [col for col in target_columns if col not in df.columns]

    if missing_target_columns:
        logging.warning(f'The following target columns are missing from the DataFrame: {missing_target_columns}')

    # Drop only the existing target columns
    X = df.drop(columns=existing_target_columns, errors='ignore')  # Use errors='ignore' to avoid KeyError
    y = df[existing_target_columns]

    # Print DataFrame columns after extraction for debugging
    logging.info(f'DataFrame columns after extraction: {X.columns.tolist()}')

    return X, y

def apply_preprocessor(X, y, is_training):
    """
    Applies the preprocessor to the feature matrix.

    Parameters:
        X (pd.DataFrame): The feature matrix.
        y (np.ndarray): The target matrix.
        is_training (bool): Flag indicating whether the preprocessing is for training.

    Returns:
        tuple: (X_processed, y) if is_training is True, else (X_processed, None)
    """
    global preprocessor, NUMERICAL_FEATURES, CATEGORICAL_FEATURES

    try:
        logging.debug(f"Initial X columns: {X.columns}")

        # Ensure 'UTC_DATE' column is present
        if 'UTC_DATE' not in X.columns:
            logging.error("'UTC_DATE' column is missing from DataFrame.")
            return None, None

        # Drop columns with all null values
        X = X.dropna(axis=1, how='all')
        logging.debug(f"X columns after dropping all-null columns: {X.columns}")

        # Handle links and string formats
        for col in X.select_dtypes(include='object').columns:
            if X[col].str.contains('http', na=False).any():
                X[col] = X[col].apply(lambda x: 1 if isinstance(x, str) and 'http' in x else 0)
                logging.info(f"Processed column '{col}' for links.")
            else:
                X.loc[:, col] = X[col].astype(str).str.extract(r'(\d+)', expand=False).astype(float)
                logging.info(f"Processed column '{col}' for numeric extraction.")

            # Check for invalid data (e.g., NaN values) and handle them using forward fill and backward fill
            if X.isnull().any(axis=None):
                logging.warning("Data contains NaN values. Filling NaNs using forward fill and backward fill.")
                X = X.ffill().bfill()
                X = X.infer_objects(copy=False)  # Use infer_objects to handle downcasting
                logging.debug("Filled NaN values.")

        # Ensure columns in NUMERICAL_FEATURES and CATEGORICAL_FEATURES are present
        missing_numerical_features = [col for col in NUMERICAL_FEATURES if col not in X.columns]
        missing_categorical_features = [col for col in CATEGORICAL_FEATURES if col not in X.columns]
        if missing_numerical_features or missing_categorical_features:
            logging.warning(f"Missing numerical features: {missing_numerical_features}")
            logging.warning(f"Missing categorical features: {missing_categorical_features}")
            # Remove columns with all null values
            X = X.dropna(axis=1, how='all')
            logging.info(f"Removed columns with all null values. Remaining columns: {X.columns.tolist()}")

            # Simulate missing values in rows
            for col in X.columns:
                if X[col].isnull().any():
                    # Use a simple model to predict missing values
                    model = LinearRegression()
                    not_null = X[X[col].notnull()]
                    null = X[X[col].isnull()]
                    if not_null.empty:
                        continue
                    X_train = not_null.drop(columns=[col])
                    y_train = not_null[col]
                    model.fit(X_train, y_train)
                    X.loc[X[col].isnull(), col] = model.predict(null.drop(columns=[col]))
                    logging.info(f"Simulated missing values in column '{col}'.")

        NUMERICAL_FEATURES = [col for col in NUMERICAL_FEATURES if col in X.columns]
        CATEGORICAL_FEATURES = [col for col in CATEGORICAL_FEATURES if col in X.columns]
        logging.debug(f"Numerical features: {NUMERICAL_FEATURES}")
        logging.debug(f"Categorical features: {CATEGORICAL_FEATURES}")

        # Initialize preprocessor if None
        if preprocessor is None:
            preprocessor = ColumnTransformer(
                transformers=[
                    ('num', StandardScaler(), NUMERICAL_FEATURES),
                    ('cat', OneHotEncoder(handle_unknown='ignore'), CATEGORICAL_FEATURES)
                ]
            )
            logging.info('Initialized preprocessor with StandardScaler and OneHotEncoder.')

        # Apply preprocessing
        if is_training:
            try:
                X_processed = preprocessor.fit_transform(X)
            except Exception as e:
                logging.error(f'Error during preprocessing: {e}', exc_info=True)
                # Alternative approach: handle missing values and retry
                X = X.fillna(0)  # Example: fill missing values with 0
                try:
                    X_processed = preprocessor.fit_transform(X)
                except Exception as e:
                    logging.error(f'Alternative preprocessing failed: {e}', exc_info=True)
                    X_processed = np.zeros((X.shape[0], len(NUMERICAL_FEATURES) + len(CATEGORICAL_FEATURES)))
                    logging.warning("Preprocessing failed. Filled X_processed with default values.")
            logging.info("Preprocessing completed with fit_transform.")
        else:
            if hasattr(preprocessor, 'transform'):
                X_processed = preprocessor.transform(X)
                logging.info("Preprocessing completed with transform.")
            else:
                try:
                    X_processed = preprocessor.fit_transform(X)
                    logging.info("Preprocessor fitted and transformed successfully.")
                except Exception as e:
                    logging.error(f"Preprocessor fitting failed: {e}")
                    return None, None

        # Convert to NumPy array if necessary
        if isinstance(X_processed, (pd.DataFrame, pd.Series)):
            X_processed = X_processed.values
            logging.debug("Converted X_processed to NumPy array.")

        # Ensure all columns are properly transformed to numeric types
        X_processed = pd.DataFrame(X_processed)
        X_processed = X_processed.apply(pd.to_numeric, errors='coerce')

        # Encode dates and strings using other encoders
        for col in X_processed.select_dtypes(include=['object']).columns:
            if col == 'UTC_DATE':
                X_processed[col] = pd.to_datetime(X_processed[col], errors='coerce').astype(int) / 10**9  # Convert to Unix timestamp
                logging.info(f"Encoded date column '{col}' as Unix timestamp.")
            else:
                X_processed[col] = X_processed[col].astype('category').cat.codes
                logging.info(f"Encoded string column '{col}' as categorical codes.")

        # Check for NaN or infinite values in X_processed
        if np.isnan(X_processed.values).any():
            logging.error("X_processed contains NaN.")
            print('Trying to fill NaN values in X_processed.')
            X_processed = X_processed.fillna(0).replace(0)
            try:
                if np.isnan(X_processed).any() or np.isinf(X_processed).any():
                    logging.error("Failed to fill NaN or infinite values in X_processed.")
                    return None, None
            except Exception as e:
                print(f'Error: {e} stopped from checking NaN or infinite values in X_processed.')
                try:
                    X_processed = X_processed.fillna(0).replace([np.inf, -np.inf], 0)
                except Exception as e:
                    return None, None
            logging.info("Filled NaN or infinite values in X_processed.")

        # Check y for NaN or infinite values if training
        if is_training:
            try:
                print('Checking y for NaN or infinite values.')
                if np.isnan(y).values.any() or np.isinf(y).values.any():
                    logging.error("y contains NaN or infinite values.")
                    return None, None
            except Exception as e:
                print(f'Error: {e} stopped from checking NaN or infinite values in y.')
                try:
                    y = y.fillna(0).replace([np.inf, -np.inf], 0)
                    print('Filled NaN or infinite values in y instead. Continuing...')
                except Exception as e:
                    return None, None
                
            if X_processed.shape[0] != y.shape[0]:
                logging.error("Mismatch in number of samples between X and y.")
                return None, None
        try:
            logging.debug(f"X_processed shape: {X_processed.shape}")
            logging.debug(f"X_processed dtypes: {X_processed.dtypes}")
            logging.debug(f"X_processed dtype: {X_processed.dtype}")
        except Exception as e:
            print(f'Error logging X_processed: {e}. LOL')

        return (X_processed, y) if is_training else (X_processed, None)

    except Exception as e:
        logging.error(f"Preprocessing pipeline error: {e}", exc_info=True)
        return None, None
# ============================
# Performance Analysis
# ============================

def evaluate_model_accuracy(model, X_train, y_train, target_variables):
    roc_auc_scores = {}
    y_pred = model.predict(X_train)
    pred_col_index = 0  # Index for y_pred columns

    for target in target_variables:
        if target in y_train.columns:
            y_score = y_pred[:, pred_col_index]
            y_true = y_train[target].values

            try:
                if len(np.unique(y_true)) > 2:
                    roc_auc = roc_auc_score(y_true, y_score, multi_class='ovr')
                else:
                    roc_auc = roc_auc_score(y_true, y_score)
                roc_auc_scores[target] = roc_auc
                logging.info(f'ROC AUC for {target}: {roc_auc}')
            except ValueError as e:
                logging.error(f"Error calculating ROC AUC for {target}: {e}")

            pred_col_index += 1  # Increment index only when target is present
        else:
            logging.warning(f"Target variable {target} not found in y_train columns")

    return {'roc_auc_scores': roc_auc_scores}

def plot_roc_auc_scores(roc_auc_scores, X_train, y_train, model):
    """
    Plots the ROC AUC scores for each target variable.

    Parameters:
        roc_auc_scores (dict): Dictionary containing ROC AUC scores for each target variable.
    """
    if not roc_auc_scores:
        logging.warning("No ROC AUC scores to plot.")
        return

    targets = list(roc_auc_scores.keys())
    scores = list(roc_auc_scores.values())

    plt.figure(figsize=(10, 6))
    plt.barh(targets, scores, color='skyblue')
    plt.xlabel('ROC AUC Score')
    plt.ylabel('Target Variable')
    plt.title('ROC AUC Scores for Target Variables')
    plt.xlim(0, 1)
    plt.grid(axis='x', linestyle='--', alpha=0.7)
    plt.show()

    # Evaluate model accuracy
    print('Evaluating model accuracy...')
    
    accuracy = evaluate_model_accuracy(model, X_train, y_train, list(TARGET_VARIABLES))
    print(f"Model accuracy: {accuracy}")
    

    def plot_roc_auc_scores(roc_auc_scores):
        """
        Plots the ROC AUC scores for each target variable.

        Parameters:
            roc_auc_scores (dict): Dictionary containing ROC AUC scores for each target variable.
        """
        if not roc_auc_scores:
            logging.warning("No ROC AUC scores to plot.")
            return

        targets = list(roc_auc_scores.keys())
        scores = list(roc_auc_scores.values())

        plt.figure(figsize=(10, 6))
        plt.barh(targets, scores, color='skyblue')
        plt.xlabel('ROC AUC Score')
        plt.ylabel('Target Variable')
        plt.title('ROC AUC Scores for Target Variables')
        plt.xlim(0, 1)
        plt.grid(axis='x', linestyle='--', alpha=0.7)
        plt.show()


# ============================
# Model Training and Updating
# ============================

def train_initial_model():
    global model, preprocessor
    logging.info('Starting initial model training...')
    print('Starting initial model training...')

    # Collect all possible columns
    collect_all_columns()
    if not TARGET_VARIABLES:
        logging.error('No target variables identified. Cannot train the model.')
        print('No target variables identified. Cannot train the model.')
        return

    # Load and combine data from all sources
    combined_data = load_and_combine_data()
    logging.debug(f'Combined data type: {type(combined_data)}')
    if combined_data.empty:
        logging.error('Cannot train model without data.')
        print('Cannot train model without data.')
        return

    # Preprocess data for training
    X_train, y_train = preprocess_data(combined_data, is_training=True)
    logging.debug(f'X_train type: {type(X_train)}, shape: {getattr(X_train, "shape", "No shape")}')
    logging.debug(f'y_train type: {type(y_train)}, shape: {getattr(y_train, "shape", "No shape")}')
    if X_train is None or y_train is None:
        logging.error('Preprocessing failed. Cannot train the model.')
        print('Preprocessing failed. Cannot train the model.')
        return

    # Initialize the model with SGDRegressor for incremental learning
    base_model = SGDRegressor(max_iter=1000, tol=1e-3)
    model = MultiOutputRegressor(base_model)
    logging.info('Model initialized successfully.')

    # Train the model
    logging.info('Training the initial model...')
    print('Training the initial model...')
    try:
        model.fit(X_train, y_train)
        logging.info('Initial model trained successfully.')
        print('Model trained successfully.')
    except Exception as e:
        logging.error(f'Model training failed: {e}', exc_info=True)
        print(f'Model training failed: {e}')
        return

    # Save the trained model and preprocessor
    model_path = os.path.join(KNOWLEDGE_PATH, 'model.joblib')
    preprocessor_path = os.path.join(KNOWLEDGE_PATH, 'preprocessor.joblib')

    try:
        joblib.dump(model, model_path)
        joblib.dump(preprocessor, preprocessor_path)
        logging.info(f'Initial model and preprocessor saved at {model_path} and {preprocessor_path}.')
        print(f'Model and preprocessor saved successfully.')
    except Exception as e:
        logging.error(f'Failed to save model or preprocessor: {e}', exc_info=True)
        print(f'Failed to save model or preprocessor: {e}')

    # Save feature metadata
    feature_metadata = {
        'FEATURE_COLUMNS': list(FEATURE_COLUMNS),
        'NUMERICAL_FEATURES': list(NUMERICAL_FEATURES),
        'CATEGORICAL_FEATURES': list(CATEGORICAL_FEATURES)
    }
    feature_metadata_path = os.path.join(KNOWLEDGE_PATH, 'feature_metadata.json')
    try:
        with open(feature_metadata_path, 'w') as f:
            json.dump(feature_metadata, f)
        logging.info(f'Feature metadata saved successfully at {feature_metadata_path}.')
        print(f'Feature metadata saved successfully.')
    except Exception as e:
        logging.error(f'Failed to save feature metadata: {e}', exc_info=True)
        print(f'Failed to save feature metadata: {e}')

    # Call the function to evaluate model accuracy
    evaluate_model_accuracy(combined_data, X_train, y_train)

def load_and_prepare_data():
    collect_all_columns() # Good

    combined_data = load_and_combine_data()
    logging.debug(f'Combined data type: {type(combined_data)}')
    if combined_data.empty:
        print("Combined data is empty.")
        logging.error("Combined data is empty.")
        return False
    return combined_data

def train_model_for_duration(input_seconds):
    global model, preprocessor, target_scaler

    try:
        input_seconds = int(input_seconds)
    except ValueError:
        logging.error("Invalid input for training duration. Must be an integer.")
        print('Invalid input. Please enter an integer value for the training duration.')
        return False

    if input_seconds <= 0:
        logging.error("Training duration must be greater than 0 seconds.")
        print('Training duration must be a positive integer greater than 0.')
        return False

    start_time = time.time()
    logging.info(f"Training will run for {input_seconds} seconds.")
    print(f"Starting training for {input_seconds} seconds...")

    combined_data = load_and_prepare_data()

    # Initialize or load the preprocessor
    if preprocessor is None:
        preprocessor_path = os.path.join(KNOWLEDGE_PATH, 'preprocessor.joblib')
        if os.path.exists(preprocessor_path):
            try:
                preprocessor = joblib.load(preprocessor_path)
                logging.info('Preprocessor loaded from file.')
            except Exception as e:
                logging.error(f'Failed to load preprocessor: {e}', exc_info=True)
                return False
        else:
            # Initialize the preprocessor if not available
            preprocessor = ColumnTransformer(
                transformers=[
                    ('num', StandardScaler(), list(NUMERICAL_FEATURES)),
                    ('cat', OneHotEncoder(handle_unknown='ignore'), list(CATEGORICAL_FEATURES))
                ]
            )
            logging.info('Preprocessor initialized successfully.')

    # Initialize or load the target scaler
    if 'target_scaler' not in globals():
        target_scaler = None

    if target_scaler is None:
        target_scaler_path = os.path.join(KNOWLEDGE_PATH, 'target_scaler.joblib')
        if os.path.exists(target_scaler_path):
            try:
                target_scaler = joblib.load(target_scaler_path)
                logging.info('Target scaler loaded from file.')
            except Exception as e:
                logging.error(f'Failed to load target scaler: {e}', exc_info=True)
                return False
        else:
            target_scaler = StandardScaler()
            logging.info('Target scaler initialized successfully.')

    # Preprocess data
    X_train, y_train = preprocess_data(combined_data, is_training=True)
    logging.debug(f'X_train type: {type(X_train)}, shape: {getattr(X_train, "shape", "No shape")}')
    logging.debug(f'y_train type: {type(y_train)}, shape: {getattr(y_train, "shape", "No shape")}')
    if X_train is None or y_train is None:
        print("Preprocessing failed.")
        logging.error("Preprocessing failed. Cannot train the model.")
        return False

    # Verify column consistency before fitting the preprocessor
    numeric_features = [col for col in X_train.columns if X_train[col].dtype in [np.float64, np.int64]]
    categorical_features = [col for col in X_train.columns if X_train[col].dtype == 'object']

    # Update or re-initialize the preprocessor with the correct columns
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', StandardScaler(), numeric_features),
            ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_features)
        ]
    )

    # Fit the preprocessor if not already fitted
    if not hasattr(preprocessor, 'transformers_'):
        try:
            preprocessor.fit(X_train)
        except Exception as e:
            logging.error(f'Error during preprocessor fitting: {e}', exc_info=True)
            return False
            
        logging.info('Preprocessor fitted successfully.')

    # Scale target variables
    y_train_scaled = target_scaler.fit_transform(y_train)
    logging.debug(f'y_train_scaled shape: {y_train_scaled.shape}')

    # Initialize the model if not already initialized
    if model is None:
        base_model = SGDRegressor(max_iter=1, warm_start=True, tol=None)
        model = MultiOutputRegressor(base_model)
        logging.info('Model initialized successfully.')

    iteration_count = 0

    # Training loop
    while True:
        elapsed_time = time.time() - start_time
        if elapsed_time >= input_seconds:
            print(f"Training completed after {elapsed_time:.2f} seconds.")
            break

        try:
            print(f"Iteration {iteration_count + 1}: Starting fit...")
            logging.info(f"Starting fit for iteration {iteration_count + 1}.")
            model.partial_fit(X_train, y_train_scaled)
            iteration_count += 1
            logging.info(f"Iteration {iteration_count} completed. Elapsed time: {elapsed_time:.2f} seconds.")
            print(f"Iteration {iteration_count} completed. Elapsed time: {elapsed_time:.2f} seconds.")
        except Exception as e:
            logging.error(f"Error during training iteration {iteration_count}: {e}", exc_info=True)
            return False

    # Save the model and scalers after training
    model_path = os.path.join(KNOWLEDGE_PATH, "model.joblib")
    preprocessor_path = os.path.join(KNOWLEDGE_PATH, 'preprocessor.joblib')
    target_scaler_path = os.path.join(KNOWLEDGE_PATH, 'target_scaler.joblib')
    try:
        joblib.dump(model, model_path)
        joblib.dump(preprocessor, preprocessor_path)
        joblib.dump(target_scaler, target_scaler_path)
        logging.info(f"Model and scalers saved successfully at {model_path}.")
    except Exception as e:
        logging.error(f"Failed to save model or scalers: {e}", exc_info=True)
        return False

    logging.info(f"Training completed. Total iterations: {iteration_count}")
    print(f"Training completed. Total iterations: {iteration_count}")

    # Generate predictions for the next 365 days every hour
    try:
        # Generate timestamps for the next 365 days every hour
        future_data = pd.DataFrame({
            'datetime': pd.date_range(start=datetime.now(), periods=365*24, freq='h')
        })

        # Extract time-based features
        future_data['hour'] = future_data['datetime'].dt.hour
        future_data['day'] = future_data['datetime'].dt.day
        future_data['month'] = future_data['datetime'].dt.month
        future_data['weekday'] = future_data['datetime'].dt.weekday

        # Retain 'datetime' column and exclude it from features
        features_future = future_data.drop(columns=['datetime'])

        # Ensure all columns in the preprocessor are present in the features
        required_columns = set(preprocessor.transformers_[0][2]) | set(preprocessor.transformers_[1][2])
        missing_cols = required_columns - set(features_future.columns)
        for col in missing_cols:
            features_future[col] = 0  # Fill missing columns with default values

        # Convert all column names to strings
        features_future.columns = features_future.columns.astype(str)

        # Ensure features_future has the same number of features as expected by the preprocessor
        features_future = features_future.reindex(columns=list(required_columns), fill_value=0)

        # Always transform without fitting
        X_future = preprocessor.transform(features_future)

        # Ensure the model is fitted before making predictions
        if not hasattr(model, 'estimators_'):
            model.partial_fit(X_train, y_train_scaled)
        else:
            model.fit(X_train, y_train_scaled)

        # Make predictions
        predictions_scaled = model.predict(X_future)

        # Check for NaN or infinite values in predictions_scaled
        if np.isnan(predictions_scaled).any() or np.isinf(predictions_scaled).any():
            logging.error("Predictions contain NaN or infinite values. Cannot apply inverse transform.")
            return False

        # Inverse transform to get original scale
        try:
            predictions = target_scaler.inverse_transform(predictions_scaled)
        except Exception as e:
            logging.error(f"Failed to apply inverse transform: {e}")
            return False

        # Ensure the number of columns in predictions matches the number of target variables
        if predictions.shape[1] != len(TARGET_VARIABLES):
            logging.error(f"Shape of predictions {predictions.shape} does not match number of target variables {len(TARGET_VARIABLES)}.")
            # Adjust the shape of predictions to match TARGET_VARIABLES
            adjusted_predictions = np.zeros((predictions.shape[0], len(TARGET_VARIABLES)))
            adjusted_predictions[:, :predictions.shape[1]] = predictions
            predictions = adjusted_predictions

        # Convert predictions to DataFrame
        predictions_df = pd.DataFrame(predictions, columns=list(TARGET_VARIABLES))
        
        # Combine with datetime
        predictions_df['datetime'] = future_data['datetime'].reset_index(drop=True)

        # Replace null values with zero
        predictions_df.fillna(0, inplace=True)

        # Convert DataFrame to JSON format
        predictions_df.set_index('datetime', inplace=True)
        predictions_json = predictions_df.to_json(orient='index', date_format='iso')

        # Save to a JSON file
        predictions_path = os.path.join(KNOWLEDGE_PATH, 'Predictions', str(datetime.now().date()))
        os.makedirs(predictions_path, exist_ok=True)
        predictions_json_path = os.path.join(predictions_path, 'future_predictions.json')

        with open(predictions_json_path, 'w') as json_file:
            json_file.write(predictions_json)

        logging.info(f"Future predictions saved to {predictions_json_path}.")
        print(f"Future predictions saved to {predictions_json_path}.")

    except Exception as e:
        logging.error(f"Failed to generate future predictions: {e}", exc_info=True)
        return False

    return True

# ============================
# Process new data
# ============================
def process_new_data(file_path):
    """
    Processes new data by making predictions and updating the model incrementally.

    Parameters:
        file_path (str): Path to the new CSV data file.
    """
    global model, preprocessor
    logging.info(f'Processing new data from {file_path}...')

    # Load new data
    try:
        df_new = pd.read_csv(file_path).copy()
        df_new = apply_column_mappings(df_new)
    except Exception as e:
        logging.error(f'Failed to read new data: {e}')
        return

    # Check for 'UTC_DATE' presence
    if 'UTC_DATE' not in df_new.columns:
        logging.error(f"'UTC_DATE' column missing in new data file {file_path}. Available columns: {df_new.columns.tolist()}")
        return

    # Preprocess new data
    X_new, y_new = preprocess_data(df_new, is_training=False)
    if X_new is None:
        logging.error('Preprocessing failed. Cannot make predictions.')
        return

    # Load the model if not already loaded
    if model is None:
        model_path = os.path.join(KNOWLEDGE_PATH, 'model.joblib')
        if os.path.exists(model_path):
            try:
                model = joblib.load(model_path)
                logging.info('Model loaded from file.')
            except Exception as e:
                logging.error(f'Failed to load model: {e}')
                return
        else:
            logging.error('Model not found. Please train the initial model first.')
            return

    # Load the preprocessor if not already loaded
    if preprocessor is None:
        preprocessor_path = os.path.join(KNOWLEDGE_PATH, 'preprocessor.joblib')
        if os.path.exists(preprocessor_path):
            try:
                preprocessor = joblib.load(preprocessor_path)
                logging.info('Preprocessor loaded from file.')
            except Exception as e:
                logging.error(f'Failed to load preprocessor: {e}')
                return
        else:
            logging.error('Preprocessor not found. Please ensure it is saved during training.')
            return

    # Make predictions
    try:
        predictions = model.predict(X_new)
        logging.info('Predictions made successfully.')
    except Exception as e:
        logging.error(f'Failed to make predictions: {e}')
        return

    # Save predictions
    predictions_df = pd.DataFrame(predictions, columns=list(TARGET_VARIABLES))
    predictions_df['timestamp'] = datetime.now()
    predictions_df['source_file'] = os.path.basename(file_path)
    
    # Create a new directory with the current date
    date_str = datetime.now().strftime('%Y-%m-%d')
    predictions_dir = os.path.join(KNOWLEDGE_PATH, date_str)
    os.makedirs(predictions_dir, exist_ok=True)
    
    predictions_path = os.path.join(predictions_dir, 'predictions.csv')
    try:
        # Append to CSV, write header only if file does not exist
        predictions_df.to_csv(predictions_path, mode='a', header=not os.path.exists(predictions_path), index=False)
        logging.info(f'Predictions saved to {predictions_path}.')
    except Exception as e:
        logging.error(f'Failed to save predictions: {e}')
        return

    # Update the model incrementally if target variables are available
    if not y_new.empty and not y_new.isnull().all().all():
        logging.info('Updating the model with new data...')
        try:
            model.partial_fit(X_new, y_new)
            # Save the updated model
            model_path = os.path.join(predictions_dir, 'model.joblib')
            joblib.dump(model, model_path)
            logging.info(f'Model updated and saved at {model_path}.')
        except AttributeError:
            logging.error('Current model does not support partial_fit. Consider using a different model.')
        except Exception as e:
            logging.error(f'Failed to update the model: {e}')
    else:
        logging.info('No valid target data available for model updating.')

# ============================
# Prediction Function
# ============================

def predict_weather_for_date(target_date):
    """
    Predicts weather for a specific date using the trained model and corresponding space data.

    Parameters:
        target_date (str or datetime): The date for which to make the prediction.

    Returns:
        pd.Series or None: Predicted values for the target variables or None if prediction fails.
    """
    global model, preprocessor
    logging.info(f'Predicting weather for date: {target_date}')

    # Convert target_date to datetime
    try:
        date_obj = pd.to_datetime(target_date)
    except Exception as e:
        logging.error(f'Invalid date format: {e}')
        return None

    # Load data for the specific date
    try:
        df_date = load_data_for_date(date_obj)
    except Exception as e:
        logging.error(f'Failed to load data for {target_date}: {e}')
        return None

    if df_date.empty:
        logging.warning(f'No data found for date: {target_date}')
        return None

    # Preprocess data
    X_date, _ = preprocess_data(df_date, is_training=False)
    if X_date is None:
        logging.error('Preprocessing failed. Cannot make predictions.')
        return None

    # Load the model if not already loaded
    if model is None:
        model_path = os.path.join(KNOWLEDGE_PATH, 'model.joblib')
        if os.path.exists(model_path):
            try:
                model = joblib.load(model_path)
                logging.info('Model loaded from file.')
            except Exception as e:
                logging.error(f'Failed to load model: {e}')
                return None
        else:
            logging.error('Model not found. Please train the initial model first.')
            return None

    # Load the preprocessor if not already loaded
    if preprocessor is None:
        preprocessor_path = os.path.join(KNOWLEDGE_PATH, 'preprocessor.joblib')
        if os.path.exists(preprocessor_path):
            try:
                preprocessor = joblib.load(preprocessor_path)
                logging.info('Preprocessor loaded from file.')
            except Exception as e:
                logging.error(f'Failed to load preprocessor: {e}')
                return None
        else:
            logging.error('Preprocessor not found. Please ensure it is saved during training.')
            return None

    # Make prediction
    try:
        prediction = model.predict(X_date)
        logging.info('Prediction made successfully.')
    except Exception as e:
        logging.error(f'Failed to make prediction: {e}')
        return None

    # Convert prediction to Series with target variable names
    prediction_series = pd.Series(prediction[0], index=list(TARGET_VARIABLES))
    return prediction_series

def load_data_for_date(date_obj):
    """
    Loads and combines data from all sources for a specific date.

    Parameters:
        date_obj (pd.Timestamp): The date for which to load data.

    Returns:
        pd.DataFrame: Combined DataFrame containing data for the specified date.
    """
    # Load Hourly Data for the date
    hourly_data = load_data_for_specific_date(REAL_TIME_DATA_PATH, date_obj)

    # Load Space Data for the date
    space_data = load_data_for_specific_date(SPACE_DATA_PATH, date_obj, aggregate_func=aggregate_space_data)

    # Load Near Earth Space Data for the date
    near_earth_data = load_data_for_specific_date(NEAR_EARTH_SPACE_DATA_PATH, date_obj, aggregate_func=aggregate_near_earth_space_data)

    # Merge data on 'UTC_DATE'
    if not hourly_data.empty:
        combined_data = hourly_data.copy()
        for df in [space_data, near_earth_data]:
            if not df.empty:
                combined_data = pd.merge(combined_data, df, on='UTC_DATE', how='left')
        logging.info(f'Data loaded and combined for date {date_obj.date()}. Combined DataFrame shape: {combined_data.shape}')
    else:
        combined_data = pd.DataFrame()
        logging.warning(f'No Hourly Data available for date {date_obj.date()}. Combined DataFrame is empty.')

    return combined_data

def load_data_for_specific_date(directory_path, date_obj, aggregate_func=None):
    """
    Loads data for a specific date from a directory.

    Parameters:
        directory_path (str): Path to the data directory.
        date_obj (pd.Timestamp): The date for which to load data.
        aggregate_func (function, optional): Function to aggregate data if needed.

    Returns:
        pd.DataFrame: DataFrame containing data for the specified date.
    """
    data_frames = load_data_from_directory(directory_path)
    if not data_frames:
        return pd.DataFrame()

    combined_df = pd.concat(data_frames, ignore_index=True).copy()

    # Apply aggregation if function is provided
    if aggregate_func:
        combined_df = aggregate_func(combined_df)

    # Ensure 'UTC_DATE' is in datetime format
    combined_df['UTC_DATE'] = pd.to_datetime(combined_df['UTC_DATE'], errors='coerce')

    # Filter data for the specified date
    filtered_df = combined_df[combined_df['UTC_DATE'].dt.date == date_obj.date()].copy()

    return filtered_df

# ============================
# Main Execution
# ============================

def clear_console():
    """
    Clears the console screen.
    """
    if os.name == 'nt':  # For Windows
        os.system('cls')
    else:  # For Unix/Linux/Mac
        os.system('clear')

def display_menu():
    """
    Displays a menu for the Weather Forecasting System and gets user input.

    The menu options are:
    1. Train AI Model
    2. Generate Forecasts
    3. Exit

    Returns:
        str: The user's input command.
    """
    menu = """
    Weather Forecasting System
    -------------------------
    1. Train AI Model
    2. Generate Forecasts
    3. Decode Last Predictions
    4. Exit
    -------------------------
    """
    return input(menu)

def get_user_choice():
    """
    Call the display_menu function and get the user's choice. 
    Then filter the data
    """
        
    # Return 1, 2, 3, or none
    choice = display_menu()
    if choice in ['1', '2', '3', '4']:
        return int(choice)
    else:
        return

def train():
    """
    Function to handle the training of the AI model.
    """
    duration = input("Enter the training duration in seconds (or type 'exit' to go back): ").strip()
    
    if duration.lower() == 'exit':
        return
    
    if not duration.isdigit():
        print(f'Invalid input {duration}. Please enter a valid number.')
        train()
        return


    if duration.isdigit():
        success = train_model_for_duration(int(duration))
        if success:
            print("Training completed successfully.")
        else:
            print("Training failed.")
    else:
        print("Invalid input. Please enter a valid number.")

def get_last_prediction():
    predictions_path = os.path.join(KNOWLEDGE_PATH, 'Predictions')
    if not os.path.exists(predictions_path):
        print("No predictions directory found.")
        return

    predictions_files = [f for f in os.listdir(predictions_path) if f.endswith('.csv')]
    if not predictions_files:
        print("No predictions files found.")
        return

    latest_file = max(predictions_files, key=lambda f: os.path.getmtime(os.path.join(predictions_path, f)))
    predictions_csv_path = os.path.join(predictions_path, latest_file)

    # Load the predictions from the CSV file
    predictions_df = pd.read_csv(predictions_csv_path)

    # Now you can view the decoded predictions
    print(predictions_df.head())
    if not predictions_files:
        print("No predictions files found.")
        return

    latest_file = max(predictions_files, key=lambda f: os.path.getmtime(os.path.join(predictions_path, f)))
    predictions_csv_path = os.path.join(predictions_path, latest_file)

    # Load the predictions from the CSV file
    predictions_df = pd.read_csv(predictions_csv_path)

    # Now you can view the decoded predictions
    print(predictions_df.head())

def forecast():
    """
    Function to handle the generation of forecasts.
    """
    predictions = generate_forecasts()
    if predictions:
        for period, df in predictions.items():
            print(f"\nPredictions for {period} forecast period:")
            print(df.head())  # Display top rows of predictions
    else:
        print("Please train the model.")
    input("Press Enter to continue...")

def action(command):
    """
    Executes the action based on user choice.
    """
    if command == 4:
        print("Exiting...")
        sys.exit(0)

    update()

    {1: train, 2: forecast, 3: get_last_prediction}.get(command, lambda: print(f"Invalid choice {command}. Please select a valid option."))()

def update():
    """
    Checks the last update time from a file and updates the data if more than an hour has passed since the last update.

    The function performs the following steps:
    1. Reads the last update time from last_update.txt located in the KNOWLEDGE_PATH directory.
    2. If the file does not exist or an error occurs while reading, it assumes the data needs to be updated.
    3. Compares the last update time with the current time.
    4. If more than an hour has passed since the last update, it calls the `update_data` function to update the data.
    5. Writes the current time to 'last_update.txt' after a successful update.
    6. Prints appropriate messages to indicate the status of the update process.

    Exceptions:
    - FileNotFoundError: If 'last_update.txt' does not exist.
    - Exception: For any other errors encountered while reading the last update time.
    """
    print("Checking last updated time...")
    last_update_file = os.path.join(KNOWLEDGE_PATH, 'last_update.txt')
    try:
        with open(last_update_file, 'r') as file:
            last_update = file.read().strip()
            last_update_time = datetime.strptime(last_update, '%Y-%m-%d %H:%M:%S')
    except FileNotFoundError:
        print("No last update file found. Assuming data needs to be updated.")
        last_update_time = datetime.min
    except Exception as e:
        print(f"Error reading last update time: {e}")
        last_update_time = datetime.min

    if datetime.now() - last_update_time > timedelta(hours=1):
        print("Updating data...")
        update_data()
        with open(last_update_file, 'w') as file:
            file.write(datetime.now().strftime('%Y-%m-%d %H:%M:%S'))
        print("Data updated successfully.")
    else:
        print("Data is up to date.")

def initialize_model():
    """
    Initializes the model by loading the model, preprocessor, and feature metadata from the files if they exist.
    """
    global model, preprocessor, FEATURE_COLUMNS, NUMERICAL_FEATURES, CATEGORICAL_FEATURES, KNOWLEDGE_PATH

    if model is not None and preprocessor is not None and FEATURE_COLUMNS:
        return model, preprocessor  # Return the existing model and preprocessor

    model_path = os.path.join(KNOWLEDGE_PATH, 'model.joblib')
    preprocessor_path = os.path.join(KNOWLEDGE_PATH, 'preprocessor.joblib')
    feature_metadata_path = os.path.join(KNOWLEDGE_PATH, 'feature_metadata.json')

    if os.path.exists(model_path) and os.path.exists(preprocessor_path) and os.path.exists(feature_metadata_path):
        try:
            model = joblib.load(model_path)
            preprocessor = joblib.load(preprocessor_path)
            with open(feature_metadata_path, 'r') as f:
                metadata = json.load(f)
                FEATURE_COLUMNS = set(metadata['FEATURE_COLUMNS'])
                NUMERICAL_FEATURES = set(metadata['NUMERICAL_FEATURES'])
                CATEGORICAL_FEATURES = set(metadata['CATEGORICAL_FEATURES'])
            logging.info('Model, preprocessor, and feature metadata loaded successfully.')
        except Exception as e:
            logging.error(f'Failed to load model, preprocessor, or feature metadata: {e}', exc_info=True)
            print(f'Failed to load model, preprocessor, or feature metadata: {e}')
    else:
        logging.warning('Model, preprocessor, or feature metadata files not found. Initiating training.')
        # Decide whether to attempt to train a new model or exit
        choice = input("Would you like to train a new model? (y/n): ").strip().lower()
        if choice == 'y':
            train_initial_model()
        else:
            print("Cannot proceed without a trained model.")
            print('Exiting...')
            sys.exit(0)

def main():
    """
    Main function to update data, train the initial model, and generate forecasts.
    """
    while True:
        clear_console()
        update()
        action(get_user_choice())
        input('Press Enter to continue...')

# Run the main function
if __name__ == '__main__':
    main()
