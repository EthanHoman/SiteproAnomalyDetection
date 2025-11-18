"""
Data Processing Module for Pump Anomaly Detection System

This module handles:
- Loading baseline and operational data
- Calculating deviations from baseline
- Data validation
- Data quality checks

CRITICAL: This module works with EXACT column names from user data,
including spaces and units in parentheses.
"""

import pandas as pd
import numpy as np
from typing import Dict, Optional, List
from pathlib import Path
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def load_baseline_data(filepath: str) -> Dict:
    """
    Load baseline pump specifications from CSV file.

    Args:
        filepath: Path to baseline CSV file

    Returns:
        Dictionary with baseline pump parameters:
        - well_id: str
        - pump_type: str
        - horsepower: float
        - application: str
        - baseline_flow_gpm: float
        - baseline_discharge_pressure_psi: float
        - baseline_power_hp: float
        - baseline_efficiency_percent: float

    Raises:
        FileNotFoundError: If file doesn't exist
        ValueError: If required columns are missing or data is invalid
    """
    try:
        # Load CSV
        df = pd.read_csv(filepath)
        logger.info(f"Loaded baseline data from {filepath}")

        # Required columns
        required_cols = [
            'Well ID', 'pump_type', 'horsepower', 'application',
            'baseline_flow_gpm', 'baseline_discharge_pressure_psi',
            'baseline_power_hp', 'baseline_efficiency_percent'
        ]

        # Validate required columns exist
        missing_cols = [col for col in required_cols if col not in df.columns]
        if missing_cols:
            raise ValueError(f"Missing required columns: {missing_cols}")

        # Get first row (assuming single pump baseline)
        if len(df) == 0:
            raise ValueError("Baseline file is empty")

        row = df.iloc[0]

        # Build baseline dictionary
        baseline = {
            'well_id': str(row['Well ID']),
            'pump_type': str(row['pump_type']),
            'horsepower': float(row['horsepower']),
            'application': str(row['application']),
            'baseline_flow_gpm': float(row['baseline_flow_gpm']),
            'baseline_discharge_pressure_psi': float(row['baseline_discharge_pressure_psi']),
            'baseline_power_hp': float(row['baseline_power_hp']),
            'baseline_efficiency_percent': float(row['baseline_efficiency_percent'])
        }

        # Validate baseline values
        if baseline['horsepower'] <= 0:
            raise ValueError("Horsepower must be positive")
        if baseline['baseline_flow_gpm'] <= 0:
            raise ValueError("Baseline flow must be positive")
        if baseline['baseline_discharge_pressure_psi'] <= 0:
            raise ValueError("Baseline discharge pressure must be positive")
        if baseline['baseline_power_hp'] <= 0:
            raise ValueError("Baseline power must be positive")
        if not (0 <= baseline['baseline_efficiency_percent'] <= 100):
            raise ValueError("Baseline efficiency must be between 0 and 100")

        logger.info(f"Baseline loaded for {baseline['well_id']} - Category: {baseline['application']}")

        return baseline

    except FileNotFoundError:
        logger.error(f"Baseline file not found: {filepath}")
        raise
    except Exception as e:
        logger.error(f"Error loading baseline data: {str(e)}")
        raise


def load_operational_data(filepath: str, well_id: Optional[str] = None) -> pd.DataFrame:
    """
    Load operational sensor logs from CSV file.

    CRITICAL: This function handles the EXACT column names from user data,
    including spaces and units in parentheses:
    - "Flow (gpm)" NOT "Flow"
    - "Discharge Pressure (psi)" NOT "Head"
    - "Motor Power (hp)" NOT "Power"
    - "Pump Efficiency (%)" NOT "Efficiency"

    Args:
        filepath: Path to operational CSV file
        well_id: Optional filter for specific well (e.g., "Well 1")

    Returns:
        DataFrame with columns:
        - timestamp (datetime)
        - Well ID (str)
        - Flow (gpm) (float)
        - Discharge Pressure (psi) (float)
        - Suction Pressure (psi) (float)
        - Motor Power (hp) (float)
        - Pump Efficiency (%) (float)
        - Motor Speed (rpm) (float)

    Raises:
        FileNotFoundError: If file doesn't exist
        ValueError: If required columns are missing
    """
    try:
        # Load CSV
        df = pd.read_csv(filepath)
        logger.info(f"Loaded operational data from {filepath}")
        logger.info(f"Initial shape: {df.shape}")

        # Required columns (EXACT names with spaces and units)
        required_cols = [
            'timestamp',
            'Well ID',
            'Flow (gpm)',
            'Discharge Pressure (psi)',
            'Suction Pressure (psi)',
            'Motor Power (hp)',
            'Pump Efficiency (%)',
            'Motor Speed (rpm)'
        ]

        # Validate required columns exist
        missing_cols = [col for col in required_cols if col not in df.columns]
        if missing_cols:
            raise ValueError(
                f"Missing required columns: {missing_cols}\n"
                f"Available columns: {list(df.columns)}\n"
                f"IMPORTANT: Column names must include spaces and units in parentheses!\n"
                f"Example: 'Flow (gpm)' NOT 'Flow' or 'Flow_gpm'"
            )

        # Parse timestamps (handle non-zero-padded format like "5/1/2024 0:00")
        df['timestamp'] = pd.to_datetime(df['timestamp'], format='mixed')
        logger.info(f"Parsed timestamps: {df['timestamp'].min()} to {df['timestamp'].max()}")

        # Filter by Well ID if specified
        if well_id is not None:
            df = df[df['Well ID'] == well_id].copy()
            logger.info(f"Filtered to Well ID '{well_id}': {len(df)} records")

            if len(df) == 0:
                raise ValueError(
                    f"No data found for Well ID '{well_id}'.\n"
                    f"Available Well IDs: {df['Well ID'].unique()}\n"
                    f"IMPORTANT: Well ID must match exactly (including spaces)!"
                )

        # Sort by timestamp
        df = df.sort_values('timestamp').reset_index(drop=True)

        logger.info(f"Operational data loaded successfully: {len(df)} records")

        return df

    except FileNotFoundError:
        logger.error(f"Operational file not found: {filepath}")
        raise
    except Exception as e:
        logger.error(f"Error loading operational data: {str(e)}")
        raise


def calculate_deviations(
    operational_df: pd.DataFrame,
    baseline: Dict,
    column_mapping: Optional[Dict[str, str]] = None
) -> pd.DataFrame:
    """
    Calculate percentage deviations from baseline for each parameter.

    Formula: deviation_pct = ((current - baseline) / baseline) * 100

    Args:
        operational_df: DataFrame with operational sensor readings
        baseline: Dictionary with baseline values
        column_mapping: Maps parameter names to column names (optional)
                       Default mapping uses exact column names with spaces

    Returns:
        DataFrame with original columns plus deviation columns:
        - flow_deviation_pct
        - head_deviation_pct
        - power_deviation_pct
        - efficiency_deviation_pct
    """
    # Default column mapping (EXACT names with spaces and units)
    if column_mapping is None:
        column_mapping = {
            'flow': 'Flow (gpm)',
            'head': 'Discharge Pressure (psi)',
            'power': 'Motor Power (hp)',
            'efficiency': 'Pump Efficiency (%)'
        }

    # Create copy to avoid modifying original
    df = operational_df.copy()

    # Calculate deviations for each parameter
    params = {
        'flow': ('baseline_flow_gpm', 'flow_deviation_pct'),
        'head': ('baseline_discharge_pressure_psi', 'head_deviation_pct'),
        'power': ('baseline_power_hp', 'power_deviation_pct'),
        'efficiency': ('baseline_efficiency_percent', 'efficiency_deviation_pct')
    }

    for param, (baseline_key, deviation_col) in params.items():
        col_name = column_mapping[param]
        baseline_value = baseline[baseline_key]

        # Calculate percentage deviation
        df[deviation_col] = ((df[col_name] - baseline_value) / baseline_value) * 100

        logger.info(
            f"{param.capitalize()} deviation calculated - "
            f"Baseline: {baseline_value}, "
            f"Mean deviation: {df[deviation_col].mean():.2f}%"
        )

    return df


def validate_data(df: pd.DataFrame, required_columns: Optional[List[str]] = None) -> None:
    """
    Validate data quality and raise errors if issues found.

    Checks:
    - All required columns present (EXACT names with spaces)
    - No negative values (except suction pressure can be negative)
    - Efficiency between 0-100%
    - Timestamps are valid and chronological
    - No duplicate timestamps

    Args:
        df: DataFrame to validate
        required_columns: List of required column names (optional)

    Raises:
        ValueError: If validation fails
    """
    # Default required columns (EXACT names)
    if required_columns is None:
        required_columns = [
            'timestamp',
            'Well ID',
            'Flow (gpm)',
            'Discharge Pressure (psi)',
            'Suction Pressure (psi)',
            'Motor Power (hp)',
            'Pump Efficiency (%)',
            'Motor Speed (rpm)'
        ]

    # Check required columns exist
    missing_cols = [col for col in required_columns if col not in df.columns]
    if missing_cols:
        raise ValueError(
            f"Missing required columns: {missing_cols}\n"
            f"IMPORTANT: Column names must be EXACT including spaces and units!"
        )

    # Check for empty dataframe
    if len(df) == 0:
        raise ValueError("DataFrame is empty")

    # Check timestamps are valid datetime
    if not pd.api.types.is_datetime64_any_dtype(df['timestamp']):
        raise ValueError("Timestamp column must be datetime type")

    # Check for duplicate timestamps
    duplicates = df['timestamp'].duplicated().sum()
    if duplicates > 0:
        logger.warning(f"Found {duplicates} duplicate timestamps")

    # Check timestamps are chronological
    if not df['timestamp'].is_monotonic_increasing:
        logger.warning("Timestamps are not in chronological order")

    # Check for negative values (flow, discharge pressure, power, motor speed should be positive)
    negative_checks = {
        'Flow (gpm)': df['Flow (gpm)'] < 0,
        'Discharge Pressure (psi)': df['Discharge Pressure (psi)'] < 0,
        'Motor Power (hp)': df['Motor Power (hp)'] < 0,
        'Motor Speed (rpm)': df['Motor Speed (rpm)'] < 0
    }

    for col, condition in negative_checks.items():
        if condition.any():
            count = condition.sum()
            raise ValueError(f"Found {count} negative values in {col}")

    # Check efficiency is between 0 and 100%
    invalid_efficiency = (df['Pump Efficiency (%)'] < 0) | (df['Pump Efficiency (%)'] > 100)
    if invalid_efficiency.any():
        count = invalid_efficiency.sum()
        raise ValueError(
            f"Found {count} invalid efficiency values "
            f"(must be between 0 and 100%)"
        )

    # Check for missing values
    missing_counts = df[required_columns].isnull().sum()
    if missing_counts.any():
        logger.warning(f"Missing values found:\n{missing_counts[missing_counts > 0]}")

    logger.info("Data validation passed")


def save_processed_data(df: pd.DataFrame, output_path: str) -> None:
    """
    Save processed data to CSV file.

    Args:
        df: DataFrame to save
        output_path: Path to output CSV file
    """
    # Create directory if it doesn't exist
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)

    # Save to CSV
    df.to_csv(output_path, index=False)
    logger.info(f"Processed data saved to {output_path}")


def get_column_mapping() -> Dict[str, str]:
    """
    Get the standard column mapping for parameter names to actual column names.

    Returns:
        Dictionary mapping parameter names to column names with spaces and units
    """
    return {
        'flow': 'Flow (gpm)',
        'head': 'Discharge Pressure (psi)',
        'power': 'Motor Power (hp)',
        'efficiency': 'Pump Efficiency (%)'
    }


if __name__ == "__main__":
    # Example usage
    print("Data Processing Module")
    print("=" * 50)
    print("\nThis module provides functions for:")
    print("- Loading baseline pump specifications")
    print("- Loading operational sensor data (with exact column names!)")
    print("- Calculating deviations from baseline")
    print("- Validating data quality")
    print("\nIMPORTANT: Column names must be EXACT:")
    print("  - 'Flow (gpm)' NOT 'Flow'")
    print("  - 'Discharge Pressure (psi)' NOT 'Head'")
    print("  - 'Motor Power (hp)' NOT 'Power'")
    print("  - 'Pump Efficiency (%)' NOT 'Efficiency'")
    print("\nTimestamps format: 'M/D/YYYY H:MM' (e.g., '5/1/2024 0:00')")
    print("Well ID format: 'Well 1' (with space)")
