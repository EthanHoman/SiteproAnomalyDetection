"""
Visualization Module for Pump Anomaly Detection System

This module provides functions for:
- Time-series plots with tolerance bands
- Multi-parameter dashboards
- Degradation timeline charts
- Deviation trend analysis

All plots include proper units (gpm, psi, hp, %) and color-coded status regions.
"""

import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import seaborn as sns
import pandas as pd
import numpy as np
from typing import Dict, Tuple, Optional, List
from pathlib import Path
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Set style
sns.set_style("whitegrid")
plt.rcParams['figure.dpi'] = 300  # High quality for reports


def plot_parameter_timeseries(
    df: pd.DataFrame,
    parameter: str,
    baseline: float,
    tolerance_band: Tuple[float, float],
    output_path: Optional[str] = None,
    title: Optional[str] = None,
    figsize: Tuple[int, int] = (12, 6)
) -> None:
    """
    Plot time-series for one parameter with baseline and tolerance bands.

    Args:
        df: DataFrame with 'timestamp' and parameter columns
        parameter: Column name (e.g., "Flow (gpm)")
        baseline: Baseline value for the parameter
        tolerance_band: (lower_limit, upper_limit) in absolute values
        output_path: Where to save plot (if None, just display)
        title: Plot title (optional, auto-generated if None)
        figsize: Figure size (width, height)
    """
    fig, ax = plt.subplots(figsize=figsize)

    # Extract unit from parameter name
    unit = parameter.split('(')[-1].rstrip(')') if '(' in parameter else ''

    # Plot actual values
    ax.plot(df['timestamp'], df[parameter], 'b-', linewidth=2, label='Actual', alpha=0.8)

    # Plot baseline
    ax.axhline(y=baseline, color='green', linestyle='--', linewidth=2, label='Baseline')

    # Plot tolerance bands
    lower, upper = tolerance_band
    ax.axhline(y=upper, color='orange', linestyle=':', linewidth=1.5, label='Upper Tolerance')
    if lower > 0:  # Only plot lower if it's meaningful
        ax.axhline(y=lower, color='orange', linestyle=':', linewidth=1.5, label='Lower Tolerance')

    # Shade tolerance region
    ax.fill_between(df['timestamp'], lower, upper, alpha=0.2, color='green', label='Normal Range')

    # Shade regions where tolerance exceeded
    if 'timestamp' in df.columns:
        exceeded_upper = df[parameter] > upper
        exceeded_lower = df[parameter] < lower if lower > 0 else False

        if exceeded_upper.any():
            ax.fill_between(
                df['timestamp'],
                upper,
                df[parameter].clip(lower=upper),
                where=exceeded_upper,
                alpha=0.3,
                color='red',
                label='Exceeded'
            )

        if isinstance(exceeded_lower, pd.Series) and exceeded_lower.any():
            ax.fill_between(
                df['timestamp'],
                lower,
                df[parameter].clip(upper=lower),
                where=exceeded_lower,
                alpha=0.3,
                color='red'
            )

    # Format axes
    ax.set_xlabel('Date', fontsize=12, fontweight='bold')
    ax.set_ylabel(f'{parameter}', fontsize=12, fontweight='bold')

    if title is None:
        param_name = parameter.split('(')[0].strip()
        title = f'{param_name} Over Time'

    ax.set_title(title, fontsize=14, fontweight='bold')

    # Format x-axis dates
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%m/%d/%Y'))
    ax.xaxis.set_major_locator(mdates.AutoDateLocator())
    plt.xticks(rotation=45, ha='right')

    # Legend
    ax.legend(loc='best', framealpha=0.9)

    # Grid
    ax.grid(True, alpha=0.3)

    # Tight layout
    plt.tight_layout()

    # Save or show
    if output_path:
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        logger.info(f"Plot saved to {output_path}")
    else:
        plt.show()

    plt.close()


def plot_multi_parameter_dashboard(
    df: pd.DataFrame,
    baseline: Dict,
    tolerances: Dict,
    output_path: Optional[str] = None,
    title: Optional[str] = None,
    figsize: Tuple[int, int] = (16, 12)
) -> None:
    """
    Create 2x2 subplot dashboard showing all four parameters.

    Subplots:
    1. Flow (gpm)
    2. Discharge Pressure / Head (psi)
    3. Motor Power (hp)
    4. Pump Efficiency (%)

    Args:
        df: DataFrame with timestamp and all parameter columns
        baseline: Dictionary with baseline values
        tolerances: Dictionary with tolerance specifications
        output_path: Where to save plot (if None, just display)
        title: Overall title (optional)
        figsize: Figure size (width, height)
    """
    fig, axes = plt.subplots(2, 2, figsize=figsize)
    axes = axes.flatten()

    # Parameter configurations
    params = [
        {
            'column': 'Flow (gpm)',
            'baseline_key': 'baseline_flow_gpm',
            'tolerance_key': 'flow',
            'title': 'Flow Rate'
        },
        {
            'column': 'Discharge Pressure (psi)',
            'baseline_key': 'baseline_discharge_pressure_psi',
            'tolerance_key': 'head',
            'title': 'Head (Discharge Pressure)'
        },
        {
            'column': 'Motor Power (hp)',
            'baseline_key': 'baseline_power_hp',
            'tolerance_key': 'power',
            'title': 'Motor Power'
        },
        {
            'column': 'Pump Efficiency (%)',
            'baseline_key': 'baseline_efficiency_percent',
            'tolerance_key': 'efficiency',
            'title': 'Pump Efficiency'
        }
    ]

    for idx, param_config in enumerate(params):
        ax = axes[idx]
        column = param_config['column']
        baseline_value = baseline[param_config['baseline_key']]
        tol = tolerances[param_config['tolerance_key']]

        # Calculate tolerance band
        if tol['max_deviation'] != 999:
            upper = baseline_value * (1 + tol['max_deviation'] / 100)
        else:
            upper = df[column].max() * 1.1  # 10% above max

        if tol['min_deviation'] != -999:
            lower = baseline_value * (1 + tol['min_deviation'] / 100)
        else:
            lower = 0

        # Plot actual values
        ax.plot(df['timestamp'], df[column], 'b-', linewidth=1.5, label='Actual', alpha=0.8)

        # Plot baseline
        ax.axhline(y=baseline_value, color='green', linestyle='--', linewidth=2, label='Baseline')

        # Plot tolerance bands
        ax.axhline(y=upper, color='orange', linestyle=':', linewidth=1.5, label='Upper Tolerance')
        if lower > 0:
            ax.axhline(y=lower, color='orange', linestyle=':', linewidth=1.5, label='Lower Tolerance')

        # Shade tolerance region
        ax.fill_between(df['timestamp'], lower, upper, alpha=0.2, color='green')

        # Shade exceeded regions
        exceeded_upper = df[column] > upper
        if exceeded_upper.any():
            ax.fill_between(
                df['timestamp'],
                upper,
                df[column].clip(lower=upper),
                where=exceeded_upper,
                alpha=0.3,
                color='red'
            )

        # Format axes
        ax.set_xlabel('Date', fontsize=10)
        ax.set_ylabel(column, fontsize=10)
        ax.set_title(param_config['title'], fontsize=12, fontweight='bold')

        # Format x-axis dates
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%m/%d'))
        plt.setp(ax.xaxis.get_majorticklabels(), rotation=45, ha='right')

        # Legend (only for first subplot to save space)
        if idx == 0:
            ax.legend(loc='best', fontsize=8, framealpha=0.9)

        # Grid
        ax.grid(True, alpha=0.3)

    # Overall title
    if title:
        fig.suptitle(title, fontsize=16, fontweight='bold', y=0.995)

    # Tight layout
    plt.tight_layout()

    # Save or show
    if output_path:
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        logger.info(f"Dashboard saved to {output_path}")
    else:
        plt.show()

    plt.close()


def plot_degradation_timeline(
    first_exceedances: Dict,
    failure_date: Optional[str] = None,
    output_path: Optional[str] = None,
    title: str = "Degradation Timeline",
    figsize: Tuple[int, int] = (12, 6)
) -> None:
    """
    Create timeline showing when each parameter first exceeded tolerance.

    Args:
        first_exceedances: Dictionary with first exceedance timestamps
                          {"flow": "2024-06-15 14:23:00", ...}
        failure_date: Optional failure date to mark on timeline
        output_path: Where to save plot (if None, just display)
        title: Plot title
        figsize: Figure size (width, height)
    """
    fig, ax = plt.subplots(figsize=figsize)

    # Filter out None values and convert to datetime
    events = []
    for param, timestamp in first_exceedances.items():
        if timestamp is not None:
            events.append({
                'parameter': param.capitalize(),
                'timestamp': pd.to_datetime(timestamp)
            })

    if not events:
        logger.warning("No tolerance exceedances to plot")
        plt.close()
        return

    # Sort by timestamp
    events_df = pd.DataFrame(events).sort_values('timestamp')

    # Get earliest event time
    start_time = events_df['timestamp'].min()

    # Create horizontal bar chart
    colors = {'Flow': 'blue', 'Head': 'red', 'Power': 'orange', 'Efficiency': 'green'}

    for idx, row in events_df.iterrows():
        param = row['parameter']
        timestamp = row['timestamp']
        color = colors.get(param, 'gray')

        # Plot marker
        ax.scatter(timestamp, param, s=200, marker='o', color=color, zorder=3, alpha=0.8)

        # Add horizontal line from earliest event
        ax.plot([start_time, timestamp], [param, param], 'k-', linewidth=2, alpha=0.5)

    # Add failure date if provided
    if failure_date:
        failure_dt = pd.to_datetime(failure_date)
        ax.axvline(x=failure_dt, color='red', linestyle='--', linewidth=2, label='Failure Date', zorder=2)

        # Calculate and display time to failure for each parameter
        for idx, row in events_df.iterrows():
            days_to_failure = (failure_dt - row['timestamp']).days
            ax.text(
                row['timestamp'],
                row['parameter'],
                f"  {days_to_failure}d to failure",
                verticalalignment='center',
                fontsize=9,
                style='italic'
            )

    # Format axes
    ax.set_xlabel('Date', fontsize=12, fontweight='bold')
    ax.set_ylabel('Parameter', fontsize=12, fontweight='bold')
    ax.set_title(title, fontsize=14, fontweight='bold')

    # Format x-axis dates
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%m/%d/%Y'))
    ax.xaxis.set_major_locator(mdates.AutoDateLocator())
    plt.xticks(rotation=45, ha='right')

    # Grid
    ax.grid(True, alpha=0.3, axis='x')

    # Legend
    if failure_date:
        ax.legend(loc='best')

    # Tight layout
    plt.tight_layout()

    # Save or show
    if output_path:
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        logger.info(f"Timeline saved to {output_path}")
    else:
        plt.show()

    plt.close()


def plot_deviation_trends(
    df: pd.DataFrame,
    window_size: int = 24,
    output_path: Optional[str] = None,
    title: str = "Deviation Trends (Rate of Change)",
    figsize: Tuple[int, int] = (12, 8)
) -> None:
    """
    Plot rate of change (slope) for each parameter's deviation.

    Shows which parameters are degrading fastest.

    Args:
        df: DataFrame with timestamp and deviation columns
        window_size: Rolling window size for calculating slope (hours)
        output_path: Where to save plot (if None, just display)
        title: Plot title
        figsize: Figure size (width, height)
    """
    fig, ax = plt.subplots(figsize=figsize)

    # Deviation columns
    dev_columns = {
        'flow_deviation_pct': 'Flow',
        'head_deviation_pct': 'Head',
        'power_deviation_pct': 'Power',
        'efficiency_deviation_pct': 'Efficiency'
    }

    colors = {'Flow': 'blue', 'Head': 'red', 'Power': 'orange', 'Efficiency': 'green'}

    for dev_col, label in dev_columns.items():
        if dev_col not in df.columns:
            continue

        # Calculate rolling slope (rate of change)
        # Use simple difference over window
        slope = df[dev_col].diff(window_size) / window_size

        # Plot
        ax.plot(
            df['timestamp'],
            slope,
            label=label,
            color=colors.get(label, 'gray'),
            linewidth=1.5,
            alpha=0.8
        )

    # Zero line
    ax.axhline(y=0, color='black', linestyle='-', linewidth=0.5, alpha=0.5)

    # Format axes
    ax.set_xlabel('Date', fontsize=12, fontweight='bold')
    ax.set_ylabel('Rate of Change (%/hour)', fontsize=12, fontweight='bold')
    ax.set_title(title, fontsize=14, fontweight='bold')

    # Format x-axis dates
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%m/%d/%Y'))
    ax.xaxis.set_major_locator(mdates.AutoDateLocator())
    plt.xticks(rotation=45, ha='right')

    # Legend
    ax.legend(loc='best', framealpha=0.9)

    # Grid
    ax.grid(True, alpha=0.3)

    # Tight layout
    plt.tight_layout()

    # Save or show
    if output_path:
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        logger.info(f"Trend plot saved to {output_path}")
    else:
        plt.show()

    plt.close()


def plot_status_timeline(
    df: pd.DataFrame,
    status_column: str = 'status',
    output_path: Optional[str] = None,
    title: str = "Pump Status Over Time",
    figsize: Tuple[int, int] = (12, 4)
) -> None:
    """
    Plot pump status classification over time.

    Args:
        df: DataFrame with 'timestamp' and status column
        status_column: Name of status column
        output_path: Where to save plot (if None, just display)
        title: Plot title
        figsize: Figure size (width, height)
    """
    if status_column not in df.columns:
        logger.warning(f"Status column '{status_column}' not found")
        return

    fig, ax = plt.subplots(figsize=figsize)

    # Status to numeric mapping
    status_map = {'Normal': 0, 'Warning': 1, 'Critical': 2, 'Failure': 3}
    colors = {'Normal': 'green', 'Warning': 'yellow', 'Critical': 'orange', 'Failure': 'red'}

    df_plot = df.copy()
    df_plot['status_num'] = df_plot[status_column].map(status_map)

    # Fill areas for each status
    for status, num in status_map.items():
        mask = df_plot['status_num'] == num
        if mask.any():
            ax.fill_between(
                df_plot['timestamp'],
                0,
                1,
                where=mask,
                alpha=0.7,
                color=colors[status],
                label=status
            )

    # Format axes
    ax.set_xlabel('Date', fontsize=12, fontweight='bold')
    ax.set_ylabel('Status', fontsize=12, fontweight='bold')
    ax.set_title(title, fontsize=14, fontweight='bold')
    ax.set_ylim([0, 1])
    ax.set_yticks([])

    # Format x-axis dates
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%m/%d/%Y'))
    ax.xaxis.set_major_locator(mdates.AutoDateLocator())
    plt.xticks(rotation=45, ha='right')

    # Legend
    ax.legend(loc='upper right', framealpha=0.9, ncol=4)

    # Grid
    ax.grid(True, alpha=0.3, axis='x')

    # Tight layout
    plt.tight_layout()

    # Save or show
    if output_path:
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        logger.info(f"Status timeline saved to {output_path}")
    else:
        plt.show()

    plt.close()


if __name__ == "__main__":
    print("Visualization Module")
    print("=" * 50)
    print("\nAvailable plotting functions:")
    print("- plot_parameter_timeseries(): Single parameter with tolerance bands")
    print("- plot_multi_parameter_dashboard(): 2x2 dashboard for all parameters")
    print("- plot_degradation_timeline(): Timeline of tolerance exceedances")
    print("- plot_deviation_trends(): Rate of change analysis")
    print("- plot_status_timeline(): Status classification over time")
    print("\nAll plots include:")
    print("- Proper units (gpm, psi, hp, %)")
    print("- Color-coded status regions")
    print("- High-resolution output (300 DPI)")
