"""
Pump Monitor Module - Main Class for Pump Anomaly Detection

This module provides the PumpMonitor class that integrates all functionality:
- Data loading and processing
- Tolerance-based anomaly detection
- Visualization
- Predictive modeling
- Comprehensive reporting

This is the main entry point for users of the system.
"""

import pandas as pd
import numpy as np
from typing import Dict, Optional, List
from pathlib import Path
import logging
from datetime import datetime

# Import internal modules
from . import data_processing
from . import tolerance_checker
from . import visualization
from . import predictive_model

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class PumpMonitor:
    """
    Main class for pump monitoring and anomaly detection.

    Encapsulates all functionality for analyzing a single pump.

    Example usage:
        monitor = PumpMonitor(
            baseline_file="data/raw/baseline/well1_baseline.csv"
        )
        monitor.load_operational_data("data/raw/operational/well1_operational.csv")
        monitor.analyze()
        monitor.generate_report("outputs/reports/well1_analysis.md")
    """

    def __init__(
        self,
        baseline_file: str,
        tolerance_category: Optional[str] = None,
        config_path: Optional[str] = None
    ):
        """
        Initialize PumpMonitor with baseline data.

        Args:
            baseline_file: Path to baseline CSV file
            tolerance_category: If None, auto-detect from application+HP
            config_path: Path to tolerances.json (optional)
        """
        logger.info("Initializing PumpMonitor...")

        # Load baseline data
        self.baseline = data_processing.load_baseline_data(baseline_file)
        logger.info(f"Loaded baseline for {self.baseline['well_id']}")

        # Determine tolerance category
        if tolerance_category is None:
            self.tolerance_category = tolerance_checker.select_tolerance_category(
                self.baseline['application'],
                self.baseline['horsepower']
            )
        else:
            self.tolerance_category = tolerance_category

        logger.info(f"Tolerance category: {self.tolerance_category}")

        # Load tolerances
        self.tolerances = tolerance_checker.load_tolerances(
            self.tolerance_category,
            config_path
        )

        # Initialize data containers
        self.operational_data = None
        self.processed_data = None
        self.anomaly_timeline = None
        self.first_exceedances = None
        self.ml_model = None
        self.ml_scaler = None
        self.ml_features = None

        logger.info("PumpMonitor initialized successfully")

    def load_operational_data(self, filepath: str) -> None:
        """
        Load operational sensor logs.

        Args:
            filepath: Path to operational CSV file
        """
        logger.info(f"Loading operational data from {filepath}...")

        # Load data, filtering by this pump's Well ID
        self.operational_data = data_processing.load_operational_data(
            filepath,
            well_id=self.baseline['well_id']
        )

        # Validate data
        data_processing.validate_data(self.operational_data)

        logger.info(f"Loaded {len(self.operational_data)} operational records")

    def analyze(
        self,
        train_model: bool = True,
        failure_date: Optional[str] = None,
        save_processed_data: bool = True
    ) -> None:
        """
        Run complete analysis pipeline.

        Steps:
        1. Calculate deviations from baseline
        2. Check tolerances
        3. Classify status over time
        4. Identify first exceedances
        5. Generate visualizations
        6. Train predictive model (if enough data and failure_date provided)

        Args:
            train_model: Whether to train ML model
            failure_date: When pump failed (for ML training)
            save_processed_data: Whether to save processed data to file
        """
        if self.operational_data is None:
            raise ValueError("No operational data loaded. Call load_operational_data() first.")

        logger.info("Starting analysis pipeline...")

        # 1. Calculate deviations
        logger.info("Calculating deviations from baseline...")
        self.processed_data = data_processing.calculate_deviations(
            self.operational_data,
            self.baseline
        )

        # 2. Check tolerances and classify status for each timestamp
        logger.info("Checking tolerances and classifying status...")
        statuses = []
        tolerance_results = []

        for idx, row in self.processed_data.iterrows():
            # Check tolerances
            check_result = tolerance_checker.check_tolerances(row, self.tolerances)
            tolerance_results.append(check_result)

            # Classify status (pass history for trend analysis)
            history = self.processed_data.loc[:idx] if idx > 24 else None
            status = tolerance_checker.classify_status(check_result, history)
            statuses.append(status)

        self.processed_data['status'] = statuses

        # 3. Find first exceedances
        logger.info("Identifying first tolerance exceedances...")
        self.first_exceedances = tolerance_checker.find_first_exceedance(
            self.processed_data,
            self.tolerances
        )

        # Log first exceedances
        for param, timestamp in self.first_exceedances.items():
            if timestamp:
                logger.info(f"  {param.capitalize()}: {timestamp}")

        # 4. Create anomaly timeline
        self.anomaly_timeline = self.processed_data[[
            'timestamp', 'status',
            'flow_deviation_pct', 'head_deviation_pct',
            'power_deviation_pct', 'efficiency_deviation_pct'
        ]].copy()

        # 5. Generate visualizations
        logger.info("Generating visualizations...")
        self._generate_visualizations()

        # 6. Train ML model (if requested and failure date provided)
        if train_model and failure_date is not None:
            logger.info("Training predictive model...")
            self._train_model(failure_date)

        # Save processed data
        if save_processed_data:
            output_path = f"data/processed/{self.baseline['well_id']}_processed.csv"
            data_processing.save_processed_data(self.processed_data, output_path)

        logger.info("Analysis complete!")

    def _generate_visualizations(self) -> None:
        """Generate all visualizations for the pump."""
        well_id = self.baseline['well_id']
        viz_dir = Path("outputs/visualizations")
        viz_dir.mkdir(parents=True, exist_ok=True)

        # 1. Multi-parameter dashboard
        logger.info("Creating multi-parameter dashboard...")
        visualization.plot_multi_parameter_dashboard(
            self.processed_data,
            self.baseline,
            self.tolerances,
            output_path=str(viz_dir / f"{well_id}_dashboard.png"),
            title=f"{well_id} - Performance Dashboard"
        )

        # 2. Individual parameter plots
        params = [
            ('Flow (gpm)', 'baseline_flow_gpm', 'flow'),
            ('Discharge Pressure (psi)', 'baseline_discharge_pressure_psi', 'head'),
            ('Motor Power (hp)', 'baseline_power_hp', 'power'),
            ('Pump Efficiency (%)', 'baseline_efficiency_percent', 'efficiency')
        ]

        for column, baseline_key, tol_key in params:
            baseline_value = self.baseline[baseline_key]
            tol = self.tolerances[tol_key]

            # Calculate tolerance band
            if tol['max_deviation'] != 999:
                upper = baseline_value * (1 + tol['max_deviation'] / 100)
            else:
                upper = self.processed_data[column].max() * 1.1

            if tol['min_deviation'] != -999:
                lower = baseline_value * (1 + tol['min_deviation'] / 100)
            else:
                lower = 0

            visualization.plot_parameter_timeseries(
                self.processed_data,
                column,
                baseline_value,
                (lower, upper),
                output_path=str(viz_dir / f"{well_id}_{tol_key}.png"),
                title=f"{well_id} - {column.split('(')[0].strip()}"
            )

        # 3. Degradation timeline
        if self.first_exceedances:
            logger.info("Creating degradation timeline...")
            visualization.plot_degradation_timeline(
                self.first_exceedances,
                output_path=str(viz_dir / f"{well_id}_timeline.png"),
                title=f"{well_id} - Degradation Timeline"
            )

        # 4. Deviation trends
        logger.info("Creating deviation trends plot...")
        visualization.plot_deviation_trends(
            self.processed_data,
            output_path=str(viz_dir / f"{well_id}_trends.png"),
            title=f"{well_id} - Deviation Trends"
        )

        # 5. Status timeline
        logger.info("Creating status timeline...")
        visualization.plot_status_timeline(
            self.processed_data,
            output_path=str(viz_dir / f"{well_id}_status_timeline.png"),
            title=f"{well_id} - Status Classification Over Time"
        )

    def _train_model(self, failure_date: str) -> None:
        """Train ML model for failure prediction."""
        logger.info("Engineering features for ML...")

        # Engineer features
        features_df = predictive_model.engineer_features(self.processed_data)

        # Create labels
        labels = predictive_model.create_failure_labels(
            features_df,
            failure_date,
            mode="regression"
        )

        # Select features (exclude non-feature columns)
        exclude_cols = [
            'timestamp', 'Well ID', 'status',
            'Flow (gpm)', 'Discharge Pressure (psi)', 'Suction Pressure (psi)',
            'Motor Power (hp)', 'Pump Efficiency (%)', 'Motor Speed (rpm)'
        ]

        feature_cols = [col for col in features_df.columns if col not in exclude_cols]
        X = features_df[feature_cols]

        # Train model
        self.ml_model, self.ml_scaler, self.ml_features, metrics = (
            predictive_model.train_failure_predictor(
                X,
                labels,
                model_type="random_forest",
                save_path=f"models/trained_models/{self.baseline['well_id']}"
            )
        )

        logger.info("Model training complete")

    def get_current_status(self) -> str:
        """
        Return current pump status.

        Returns:
            Status: "Normal", "Warning", "Critical", or "Failure"
        """
        if self.processed_data is None or 'status' not in self.processed_data.columns:
            return "Unknown (No analysis performed)"

        return self.processed_data['status'].iloc[-1]

    def get_anomaly_timeline(self) -> pd.DataFrame:
        """
        Return DataFrame with timestamp, status, and violations.

        Returns:
            DataFrame with anomaly timeline
        """
        if self.anomaly_timeline is None:
            raise ValueError("No analysis performed. Call analyze() first.")

        return self.anomaly_timeline

    def predict_failure(self, current_data: Optional[pd.DataFrame] = None) -> Dict:
        """
        Return failure prediction from ML model.

        Args:
            current_data: Optional specific data to predict on (uses latest if None)

        Returns:
            Dictionary with prediction results
        """
        if self.ml_model is None:
            raise ValueError("No ML model trained. Call analyze() with train_model=True.")

        # Use latest data if not specified
        if current_data is None:
            # Engineer features from processed data
            features_df = predictive_model.engineer_features(self.processed_data)

            # Select feature columns
            current_data = features_df[self.ml_features].tail(1)

        # Make prediction
        prediction = predictive_model.predict_failure(
            self.ml_model,
            self.ml_scaler,
            current_data,
            self.ml_features
        )

        return prediction

    def generate_report(self, output_path: str) -> None:
        """
        Generate comprehensive markdown report.

        The report answers key questions:
        1. When did pump first exceed tolerances?
        2. What was the degradation timeline?
        3. How long in degraded state before failure?
        4. What were leading indicators?
        5. Could failure be predicted? How far in advance?
        6. Recommendations for monitoring and maintenance

        Args:
            output_path: Path to save markdown report
        """
        if self.processed_data is None:
            raise ValueError("No analysis performed. Call analyze() first.")

        logger.info("Generating comprehensive report...")

        report = self._build_report()

        # Save report
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, 'w') as f:
            f.write(report)

        logger.info(f"Report saved to {output_path}")

    def _build_report(self) -> str:
        """Build the markdown report content."""
        well_id = self.baseline['well_id']

        # Start report
        report = f"# Pump Failure Analysis Report: {well_id}\n\n"
        report += f"**Generated:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n"
        report += "---\n\n"

        # Executive Summary
        report += "## Executive Summary\n\n"
        report += f"This report presents a comprehensive analysis of {well_id}, "
        report += f"a {self.baseline['pump_type']} pump rated at {self.baseline['horsepower']} HP "
        report += f"in {self.baseline['application']} service.\n\n"

        report += f"**Tolerance Category:** {self.tolerance_category}\n\n"

        # Current status
        current_status = self.get_current_status()
        report += f"**Current Status:** {current_status}\n\n"

        # Timeline of Events
        report += "## Timeline of Events\n\n"
        report += "### First Tolerance Exceedances\n\n"

        if self.first_exceedances:
            any_exceeded = False
            for param, timestamp in self.first_exceedances.items():
                if timestamp:
                    any_exceeded = True
                    tol = self.tolerances[param]
                    baseline_keys = {
                        'flow': 'baseline_flow_gpm',
                        'head': 'baseline_discharge_pressure_psi',
                        'power': 'baseline_power_hp',
                        'efficiency': 'baseline_efficiency_percent'
                    }
                    baseline_value = self.baseline[baseline_keys[param]]

                    # Get actual value at first exceedance
                    exc_row = self.processed_data[
                        self.processed_data['timestamp'] == pd.to_datetime(timestamp)
                    ].iloc[0]
                    dev = exc_row[f'{param}_deviation_pct']

                    report += f"**{param.capitalize()}:**\n"
                    report += f"- First exceeded on: {timestamp}\n"
                    report += f"- Baseline: {baseline_value:.2f}\n"

                    if tol['bidirectional']:
                        report += f"- Threshold: ±{tol['max_deviation']}%\n"
                    else:
                        if tol['max_deviation'] != 999:
                            report += f"- Threshold: +{tol['max_deviation']}%\n"
                        if tol['min_deviation'] != -999:
                            report += f"- Threshold: {tol['min_deviation']}%\n"

                    report += f"- Deviation at first exceedance: {dev:+.2f}%\n\n"

            if not any_exceeded:
                report += "*No tolerance exceedances detected in the operational data.*\n\n"
        else:
            report += "*First exceedance analysis not available.*\n\n"

        # Status breakdown
        report += "### Status Distribution\n\n"
        status_counts = self.processed_data['status'].value_counts()
        total_records = len(self.processed_data)

        report += "| Status | Count | Percentage |\n"
        report += "|--------|-------|------------|\n"
        for status in ['Normal', 'Warning', 'Critical', 'Failure']:
            count = status_counts.get(status, 0)
            pct = (count / total_records * 100) if total_records > 0 else 0
            report += f"| {status} | {count} | {pct:.1f}% |\n"

        report += "\n"

        # Time period
        if 'timestamp' in self.processed_data.columns:
            start_date = self.processed_data['timestamp'].min()
            end_date = self.processed_data['timestamp'].max()
            duration = (end_date - start_date).days

            report += f"**Analysis Period:** {start_date.strftime('%Y-%m-%d')} to {end_date.strftime('%Y-%m-%d')} ({duration} days)\n\n"

        # Visualizations
        report += "## Visualizations\n\n"
        report += f"![Performance Dashboard](../visualizations/{well_id}_dashboard.png)\n\n"
        report += f"![Degradation Timeline](../visualizations/{well_id}_timeline.png)\n\n"
        report += f"![Deviation Trends](../visualizations/{well_id}_trends.png)\n\n"
        report += f"![Status Timeline](../visualizations/{well_id}_status_timeline.png)\n\n"

        # ML Predictions (if available)
        if self.ml_model is not None:
            report += "## Predictive Analysis\n\n"
            report += "### ML Model Performance\n\n"
            report += "A machine learning model was trained to predict pump failures.\n\n"
            report += "*Model metrics are available in `models/trained_models/{}/metrics.txt`*\n\n".format(well_id)

        # Recommendations
        report += "## Recommendations\n\n"

        report += "### 1. Monitoring Frequency\n"
        if current_status in ['Critical', 'Failure']:
            report += "- **Current Status:** Critical - Monitor **hourly**\n"
            report += "- Increase monitoring frequency to detect rapid changes\n\n"
        elif current_status == 'Warning':
            report += "- **Current Status:** Warning - Monitor **every 4 hours**\n"
            report += "- Watch for trend toward Critical status\n\n"
        else:
            report += "- **Current Status:** Normal - Monitor **daily**\n"
            report += "- Continue routine monitoring\n\n"

        report += "### 2. Early Warning Thresholds\n"
        report += "Set alerts at 50% of tolerance limits to enable earlier intervention:\n\n"
        for param, tol in self.tolerances.items():
            if tol['mandatory']:
                report += f"- **{param.capitalize()}** (Mandatory): Alert at "
                if tol['max_deviation'] != 999:
                    report += f"+{tol['max_deviation']/2:.1f}%\n"
                elif tol['min_deviation'] != -999:
                    report += f"{tol['min_deviation']/2:.1f}%\n"
                else:
                    report += "N/A\n"

        report += "\n### 3. Maintenance Actions\n"
        report += "Based on current status:\n\n"

        if current_status == 'Normal':
            report += "- Continue scheduled maintenance\n"
            report += "- No immediate action required\n\n"
        elif current_status == 'Warning':
            report += "- Schedule inspection within 1 week\n"
            report += "- Check for wear, alignment, and seal condition\n"
            report += "- Review recent operational changes\n\n"
        elif current_status == 'Critical':
            report += "- Schedule inspection **immediately**\n"
            report += "- Prepare for potential pump replacement\n"
            report += "- Investigate root cause of degradation\n\n"
        else:  # Failure
            report += "- **IMMEDIATE ACTION REQUIRED**\n"
            report += "- Pump replacement or major repair needed\n"
            report += "- Conduct failure analysis\n\n"

        report += "### 4. Data Collection\n"
        report += "To improve future predictions:\n\n"
        report += "- Continue collecting all sensor data\n"
        report += "- Document all maintenance activities\n"
        report += "- Record environmental conditions (temperature, etc.)\n"
        report += "- Track failure modes when they occur\n\n"

        # Conclusion
        report += "## Conclusion\n\n"
        report += f"This analysis of {well_id} provides insights into pump performance and degradation patterns. "
        report += "Continue monitoring and follow the recommendations above to optimize pump reliability.\n\n"

        report += "---\n\n"
        report += "*Report generated by Pump Anomaly Detection System v1.0*\n"

        return report


if __name__ == "__main__":
    print("Pump Monitor Module")
    print("=" * 50)
    print("\nMain class for pump anomaly detection and analysis.")
    print("\nExample usage:")
    print("""
    from src.pump_monitor import PumpMonitor

    # Initialize monitor
    monitor = PumpMonitor(
        baseline_file="data/raw/baseline/well1_baseline.csv"
    )

    # Load operational data
    monitor.load_operational_data(
        "data/raw/operational/well1_operational.csv"
    )

    # Run analysis
    monitor.analyze()

    # Generate report
    monitor.generate_report("outputs/reports/well1_analysis.md")

    # Get current status
    print(f"Status: {monitor.get_current_status()}")
    """)
