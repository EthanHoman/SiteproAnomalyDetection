"""
Tolerance Checker Module for Pump Anomaly Detection System

This module handles:
- Tolerance category selection based on application and horsepower
- Loading tolerance specifications from config
- Checking if deviations exceed tolerance thresholds
- Classifying pump status (Normal/Warning/Critical/Failure)
- Finding first timestamp when tolerances exceeded

CRITICAL: Uses "Discharge Pressure (psi)" as HEAD measurement,
NOT "Suction Pressure (psi)"
"""

import json
import pandas as pd
import numpy as np
from typing import Dict, Tuple, Optional
from pathlib import Path
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def select_tolerance_category(application: str, horsepower: float) -> str:
    """
    Select tolerance category based on pump application and horsepower.

    Mapping:
    - Municipal Water and Wastewater → 1U (all HP)
    - API → 1B (all HP)
    - Energy Conservation → 1E (all HP)
    - Cooling Tower → 2B (all HP)
    - General Industry → 3B (HP < 134) or 2B (HP >= 134)
    - Dewatering, drainage, and irrigation → 3B (HP < 134) or 2B (HP >= 134)

    Args:
        application: Pump application type
        horsepower: Pump horsepower rating

    Returns:
        Tolerance category: "1B", "1E", "1U", "2B", "2U", or "3B"

    Raises:
        ValueError: If application is not recognized
    """
    # Simple mapping for applications that don't depend on HP
    simple_mapping = {
        "Municipal Water and Wastewater": "1U",
        "API": "1B",
        "Energy Conservation": "1E",
        "Cooling Tower": "2B"
    }

    if application in simple_mapping:
        category = simple_mapping[application]
        logger.info(f"Selected tolerance category {category} for {application}")
        return category

    # HP-dependent mapping
    hp_dependent = ["General Industry", "Dewatering, drainage, and irrigation"]

    if application in hp_dependent:
        if horsepower < 134:
            category = "3B"
        else:
            category = "2B"
        logger.info(
            f"Selected tolerance category {category} for {application} "
            f"with {horsepower} HP"
        )
        return category

    # Unknown application
    raise ValueError(
        f"Unknown application: {application}\n"
        f"Valid applications: Municipal Water and Wastewater, API, "
        f"Energy Conservation, Cooling Tower, General Industry, "
        f"Dewatering, drainage, and irrigation"
    )


def load_tolerances(category: str, config_path: Optional[str] = None) -> Dict:
    """
    Load tolerance specifications for a category from tolerances.json.

    Args:
        category: Tolerance category (e.g., "1U")
        config_path: Path to tolerances.json (optional, uses default if None)

    Returns:
        Dictionary with tolerance specs:
        {
            "flow": {
                "mandatory": True,
                "max_deviation": 10,
                "min_deviation": -999,
                "bidirectional": False
            },
            ...
        }

    Raises:
        FileNotFoundError: If config file doesn't exist
        ValueError: If category not found in config
    """
    # Default config path
    if config_path is None:
        config_path = Path(__file__).parent.parent / "config" / "tolerances.json"

    try:
        # Load config file
        with open(config_path, 'r') as f:
            config = json.load(f)

        # Get category tolerances
        if category not in config['categories']:
            raise ValueError(
                f"Category {category} not found in config.\n"
                f"Available categories: {list(config['categories'].keys())}"
            )

        tolerances = config['categories'][category]['tolerances']
        logger.info(f"Loaded tolerances for category {category}")

        return tolerances

    except FileNotFoundError:
        logger.error(f"Tolerance config file not found: {config_path}")
        raise
    except Exception as e:
        logger.error(f"Error loading tolerances: {str(e)}")
        raise


def check_tolerances(
    deviations: pd.Series,
    tolerances: Dict,
    parameter_names: Optional[Dict[str, str]] = None
) -> Dict:
    """
    Check if deviations exceed tolerance thresholds.

    Args:
        deviations: Series with deviation values (e.g., from a single row)
                   Expected keys: flow_deviation_pct, head_deviation_pct, etc.
        tolerances: Dictionary with tolerance specifications
        parameter_names: Mapping of deviation column names to parameter names

    Returns:
        Dictionary with results for each parameter:
        {
            "flow": {
                "exceeded": True/False,
                "deviation": 12.5,
                "max_threshold": 10,
                "min_threshold": -999,
                "mandatory": True
            },
            ...
        }
    """
    # Default parameter names mapping
    if parameter_names is None:
        parameter_names = {
            'flow_deviation_pct': 'flow',
            'head_deviation_pct': 'head',
            'power_deviation_pct': 'power',
            'efficiency_deviation_pct': 'efficiency'
        }

    results = {}

    for dev_col, param in parameter_names.items():
        if dev_col not in deviations:
            logger.warning(f"Deviation column {dev_col} not found in data")
            continue

        deviation = deviations[dev_col]
        tol = tolerances[param]

        # Check if deviation exceeds thresholds
        max_exceeded = deviation > tol['max_deviation'] if tol['max_deviation'] != 999 else False
        min_exceeded = deviation < tol['min_deviation'] if tol['min_deviation'] != -999 else False

        exceeded = max_exceeded or min_exceeded

        results[param] = {
            'exceeded': exceeded,
            'deviation': float(deviation),
            'max_threshold': tol['max_deviation'],
            'min_threshold': tol['min_deviation'],
            'mandatory': tol['mandatory'],
            'max_exceeded': max_exceeded,
            'min_exceeded': min_exceeded
        }

    return results


def classify_status(
    tolerance_check: Dict,
    history: Optional[pd.DataFrame] = None,
    history_window: int = 24
) -> str:
    """
    Classify pump status based on tolerance violations.

    Classification rules:
    - Normal: All parameters within tolerance
    - Warning:
        * One or more optional parameters exceed tolerance OR
        * One mandatory parameter slightly exceeds (< 1.5x threshold)
    - Critical:
        * Multiple parameters exceed tolerance OR
        * One mandatory parameter significantly exceeds (>= 1.5x threshold) OR
        * Sustained degradation trend (if history provided)
    - Failure:
        * Multiple mandatory parameters far exceed threshold OR
        * Severe sustained degradation

    Args:
        tolerance_check: Dictionary from check_tolerances()
        history: Optional DataFrame with historical deviation data
        history_window: Number of recent records to consider for trends

    Returns:
        Status: "Normal", "Warning", "Critical", or "Failure"
    """
    # Count violations
    mandatory_violations = []
    optional_violations = []
    severe_violations = []

    for param, result in tolerance_check.items():
        if not result['exceeded']:
            continue

        # Calculate severity (how many times over threshold)
        if result['max_exceeded']:
            severity = abs(result['deviation']) / result['max_threshold']
        elif result['min_exceeded']:
            # For min threshold, calculate how far below
            # Handle -0% case (min_threshold = 0)
            if result['min_threshold'] == 0:
                severity = abs(result['deviation']) / 10  # Normalize to 10% for -0% case
            else:
                severity = abs(result['deviation'] - result['min_threshold']) / abs(result['min_threshold'])
        else:
            severity = 0

        if result['mandatory']:
            mandatory_violations.append((param, severity))
        else:
            optional_violations.append((param, severity))

        # Severe violation: > 1.5x threshold
        if severity >= 1.5:
            severe_violations.append((param, severity))

    # Classification logic
    total_violations = len(mandatory_violations) + len(optional_violations)

    # Normal: No violations
    if total_violations == 0:
        return "Normal"

    # Failure: Multiple mandatory violations or multiple severe violations
    if len(mandatory_violations) >= 2 or len(severe_violations) >= 2:
        return "Failure"

    # Critical: Any severe violation or multiple total violations
    if len(severe_violations) >= 1 or total_violations >= 3:
        return "Critical"

    # Critical: Sustained mandatory violation (if history provided)
    if history is not None and len(mandatory_violations) >= 1:
        # Check if mandatory violation sustained over time
        param = mandatory_violations[0][0]
        dev_col = f"{param}_deviation_pct"

        if dev_col in history.columns and len(history) >= history_window:
            recent = history.tail(history_window)
            # If deviation consistently high over window, it's critical
            if (recent[dev_col] > tolerance_check[param]['max_threshold']).sum() >= history_window * 0.7:
                return "Critical"

    # Warning: Only optional violations or minor mandatory violation
    if len(mandatory_violations) >= 1:
        return "Warning"

    # Warning: Only optional violations
    if len(optional_violations) >= 1:
        return "Warning"

    # Default to normal (shouldn't reach here)
    return "Normal"


def find_first_exceedance(
    deviations_df: pd.DataFrame,
    tolerances: Dict,
    parameter_names: Optional[Dict[str, str]] = None
) -> Dict:
    """
    Find first timestamp when each parameter exceeded tolerance.

    Args:
        deviations_df: DataFrame with timestamp and deviation columns
        tolerances: Dictionary with tolerance specifications
        parameter_names: Mapping of deviation column names to parameter names

    Returns:
        Dictionary with first exceedance timestamp for each parameter:
        {
            "flow": "2024-06-15 14:23:00" or None,
            "head": "2024-06-20 08:45:00" or None,
            ...
        }
    """
    # Default parameter names mapping
    if parameter_names is None:
        parameter_names = {
            'flow_deviation_pct': 'flow',
            'head_deviation_pct': 'head',
            'power_deviation_pct': 'power',
            'efficiency_deviation_pct': 'efficiency'
        }

    first_exceedances = {}

    for dev_col, param in parameter_names.items():
        if dev_col not in deviations_df.columns:
            logger.warning(f"Deviation column {dev_col} not found")
            first_exceedances[param] = None
            continue

        tol = tolerances[param]

        # Find where tolerance exceeded
        max_exceeded = deviations_df[dev_col] > tol['max_deviation'] if tol['max_deviation'] != 999 else False
        min_exceeded = deviations_df[dev_col] < tol['min_deviation'] if tol['min_deviation'] != -999 else False
        exceeded = max_exceeded | min_exceeded

        if exceeded.any():
            # Get first timestamp where exceeded
            first_idx = exceeded.idxmax() if exceeded.any() else None
            if first_idx is not None:
                first_timestamp = deviations_df.loc[first_idx, 'timestamp']
                first_exceedances[param] = str(first_timestamp)
                logger.info(f"{param.capitalize()} first exceeded tolerance at {first_timestamp}")
            else:
                first_exceedances[param] = None
        else:
            first_exceedances[param] = None

    return first_exceedances


def get_tolerance_summary(category: str, config_path: Optional[str] = None) -> str:
    """
    Get a human-readable summary of tolerance specifications.

    Args:
        category: Tolerance category
        config_path: Path to tolerances.json (optional)

    Returns:
        Formatted string with tolerance summary
    """
    tolerances = load_tolerances(category, config_path)

    summary = f"\nTolerance Category: {category}\n"
    summary += "=" * 50 + "\n\n"

    for param, tol in tolerances.items():
        summary += f"{param.capitalize()}:\n"
        summary += f"  Mandatory: {tol['mandatory']}\n"

        # Format threshold display
        if tol['bidirectional']:
            summary += f"  Tolerance: +/- {tol['max_deviation']}%\n"
        else:
            if tol['max_deviation'] != 999:
                summary += f"  Max Deviation: +{tol['max_deviation']}%\n"
            if tol['min_deviation'] != -999:
                if tol['min_deviation'] == 0:
                    summary += f"  Min Deviation: -0% (no negative deviation allowed)\n"
                else:
                    summary += f"  Min Deviation: {tol['min_deviation']}%\n"

        summary += "\n"

    return summary


if __name__ == "__main__":
    # Example usage
    print("Tolerance Checker Module")
    print("=" * 50)

    # Example: Well 1 (Municipal Water and Wastewater, 100 HP)
    category = select_tolerance_category("Municipal Water and Wastewater", 100)
    print(f"\nWell 1 Category: {category}")

    # Load and display tolerances
    try:
        summary = get_tolerance_summary(category)
        print(summary)
    except FileNotFoundError:
        print("\nNote: Run this from the project root to load tolerances.json")

    print("\nKey Rules:")
    print("- Flow and Head are MANDATORY parameters")
    print("- Power and Efficiency are OPTIONAL parameters")
    print("- HEAD = 'Discharge Pressure (psi)' NOT 'Suction Pressure (psi)'")
    print("- Category 1U: Flow +10%, Head +6%, Power +10%, Efficiency -0%")
