"""
Edge Inference Script for Pump Anomaly Detection

This script runs on edge devices to:
1. Load packaged model and configuration
2. Process incoming sensor data
3. Detect anomalies based on tolerances
4. Make ML predictions (if model available)
5. Report anomalies to central API
6. Save results locally

Usage:
    python inference.py <input_csv> <output_json>

Example:
    python inference.py sensor_data.csv results.json
"""

import os
import sys
import json
import logging
import joblib
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, Optional, Any, List
from pathlib import Path

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class EdgeInference:
    """
    Edge inference engine for pump anomaly detection.
    """

    def __init__(self, artifact_dir: str = "."):
        """
        Initialize the inference engine.

        Args:
            artifact_dir: Directory containing packaged artifact files
        """
        self.artifact_dir = Path(artifact_dir)
        self.model = None
        self.scaler = None
        self.model_metadata = {}
        self.baseline_parameters = {}
        self.tolerances = {}
        self.column_mapping = {}
        self.deployment_config = {}
        self.tolerance_category = None
        self.anomaly_client = None
        self.last_reported = {}  # For debouncing: {parameter: timestamp}

        # Load all configurations and models
        self.load_configs()
        self.load_model()
        self.setup_api_client()

    def load_configs(self):
        """Load all configuration files."""
        try:
            # Load deployment config
            config_path = self.artifact_dir / "config" / "deployment_config.json"
            with open(config_path, 'r') as f:
                self.deployment_config = json.load(f)
            logger.info(f"✓ Loaded deployment config")

            # Load baseline parameters
            baseline_path = self.artifact_dir / "config" / "baseline.json"
            with open(baseline_path, 'r') as f:
                self.baseline_parameters = json.load(f)
            logger.info(f"✓ Loaded baseline parameters")

            # Load tolerances
            tolerances_path = self.artifact_dir / "config" / "tolerances.json"
            with open(tolerances_path, 'r') as f:
                all_tolerances = json.load(f)
                # Get the specific tolerance category
                self.tolerance_category = self.baseline_parameters.get("tolerance_category", "1U")
                self.tolerances = all_tolerances["categories"][self.tolerance_category]
            logger.info(f"✓ Loaded tolerances (category: {self.tolerance_category})")

            # Load column mapping
            mapping_path = self.artifact_dir / "config" / "column_mapping.json"
            with open(mapping_path, 'r') as f:
                self.column_mapping = json.load(f)
            logger.info(f"✓ Loaded column mapping")

        except Exception as e:
            logger.error(f"Failed to load configurations: {e}")
            raise

    def load_model(self):
        """Load trained model and scaler."""
        try:
            # Load model if available
            model_path = self.artifact_dir / "model" / "anomaly_detector.pkl"
            if model_path.exists():
                self.model = joblib.load(model_path)
                logger.info(f"✓ Loaded ML model")
            else:
                logger.warning("No ML model found, will use tolerance-based detection only")

            # Load scaler if available
            scaler_path = self.artifact_dir / "model" / "scaler.pkl"
            if scaler_path.exists():
                self.scaler = joblib.load(scaler_path)
                logger.info(f"✓ Loaded scaler")

            # Load model metadata
            metadata_path = self.artifact_dir / "model" / "model_metadata.json"
            if metadata_path.exists():
                with open(metadata_path, 'r') as f:
                    self.model_metadata = json.load(f)
                logger.info(f"✓ Loaded model metadata (version: {self.model_metadata.get('version', 'unknown')})")

        except Exception as e:
            logger.error(f"Failed to load model: {e}")
            # Continue without model - tolerance-based detection will still work
            self.model = None
            self.scaler = None

    def setup_api_client(self):
        """Setup API client for anomaly reporting."""
        try:
            # Only setup if anomaly reporting is enabled
            if not self.deployment_config.get("anomaly_reporting", {}).get("enabled", False):
                logger.info("Anomaly reporting is disabled")
                return

            # Import here to avoid dependency if not used
            from anomaly_client import AnomalyAPIClient

            api_config = self.deployment_config["api"]
            self.anomaly_client = AnomalyAPIClient(
                base_url=api_config["base_url"],
                bearer_token=api_config["bearer_token"],
                retry_attempts=api_config.get("retry_attempts", 3),
                retry_delay=api_config.get("retry_delay_seconds", 5)
            )
            logger.info(f"✓ API client configured")

        except Exception as e:
            logger.warning(f"Failed to setup API client: {e}. Anomalies will only be saved locally.")
            self.anomaly_client = None

    def process_sensor_data(self, input_csv: str) -> pd.DataFrame:
        """
        Load and process sensor data from CSV.

        Args:
            input_csv: Path to input CSV file

        Returns:
            Processed DataFrame
        """
        try:
            df = pd.read_csv(input_csv)
            logger.info(f"✓ Loaded {len(df)} records from {input_csv}")

            # Parse timestamp
            df['timestamp'] = pd.to_datetime(df['timestamp'], format='mixed')

            # Sort by timestamp
            df = df.sort_values('timestamp').reset_index(drop=True)

            return df

        except Exception as e:
            logger.error(f"Failed to process sensor data: {e}")
            raise

    def calculate_deviations(self, row: pd.Series) -> Dict[str, float]:
        """
        Calculate percentage deviations from baseline for a single row.

        Args:
            row: DataFrame row with sensor readings

        Returns:
            Dict of deviations by parameter
        """
        deviations = {}

        for param, col_name in self.column_mapping.items():
            baseline_key = f"baseline_{param}"
            if baseline_key in self.baseline_parameters:
                baseline = self.baseline_parameters[baseline_key]
                current = float(row[col_name])

                # Calculate deviation: ((current - baseline) / baseline) * 100
                if baseline != 0:
                    deviation = ((current - baseline) / baseline) * 100
                else:
                    deviation = 0.0

                deviations[param] = round(deviation, 2)

        return deviations

    def check_tolerances(self, deviations: Dict[str, float]) -> Dict[str, Any]:
        """
        Check if deviations exceed tolerance thresholds.

        Args:
            deviations: Dict of parameter deviations

        Returns:
            Dict with tolerance check results
        """
        violations = {}
        mandatory_exceeded = False

        for param, deviation in deviations.items():
            if param not in self.tolerances:
                continue

            tolerance = self.tolerances[param]
            max_threshold = tolerance["max_deviation"]
            min_threshold = tolerance["min_deviation"]
            is_mandatory = tolerance["mandatory"]

            exceeded = False
            threshold_type = None

            # Check if deviation exceeds thresholds
            if max_threshold < 999 and deviation > max_threshold:
                exceeded = True
                threshold_type = "max"
            elif min_threshold > -999 and deviation < min_threshold:
                exceeded = True
                threshold_type = "min"

            if exceeded:
                violations[param] = {
                    "deviation": deviation,
                    "threshold": max_threshold if threshold_type == "max" else min_threshold,
                    "threshold_type": threshold_type,
                    "mandatory": is_mandatory
                }

                if is_mandatory:
                    mandatory_exceeded = True

        # Determine status
        if not violations:
            status = "Normal"
        elif mandatory_exceeded:
            # Check severity
            max_violation = max(
                [abs(v["deviation"]) for v in violations.values()]
            )
            max_threshold = max(
                [abs(v["threshold"]) for v in violations.values() if v["threshold"] != 0]
            ) if any(v["threshold"] != 0 for v in violations.values()) else 10

            severity = max_violation / max_threshold if max_threshold != 0 else 1

            if severity > 2.0:
                status = "Failure"
            elif severity > 1.5:
                status = "Critical"
            else:
                status = "Warning"
        else:
            # Only optional parameters exceeded
            status = "Warning"

        return {
            "status": status,
            "violations": violations,
            "mandatory_exceeded": mandatory_exceeded,
            "deviations": deviations
        }

    def extract_features(self, df: pd.DataFrame) -> Optional[np.ndarray]:
        """
        Extract features for ML model prediction.

        This is a simplified version - full feature engineering would match training.

        Args:
            df: DataFrame with historical data

        Returns:
            Feature array or None if not enough data
        """
        if len(df) < 168:  # Need at least 1 week of data
            logger.warning("Not enough historical data for ML prediction")
            return None

        try:
            # Calculate basic features from deviations
            features = []

            # Latest deviation values
            last_row = df.iloc[-1]
            deviations = self.calculate_deviations(last_row)

            for param in ["flow", "head", "power", "efficiency"]:
                features.append(deviations.get(f"{param}", 0.0))

            # Add rolling statistics (simplified)
            for param in ["flow", "head", "power", "efficiency"]:
                col = self.column_mapping.get(param)
                if col and col in df.columns:
                    values = df[col].tail(24)  # Last 24 hours
                    features.extend([
                        values.mean(),
                        values.std(),
                        values.max(),
                        values.min()
                    ])

            # Convert to array
            feature_array = np.array([features])

            # Scale if scaler available
            if self.scaler:
                feature_array = self.scaler.transform(feature_array)

            return feature_array

        except Exception as e:
            logger.error(f"Failed to extract features: {e}")
            return None

    def predict_failure(self, features: np.ndarray) -> Optional[Dict[str, Any]]:
        """
        Predict failure using ML model.

        Args:
            features: Feature array

        Returns:
            Prediction dict or None
        """
        if self.model is None or features is None:
            return None

        try:
            # Make prediction (RUL in days)
            prediction = self.model.predict(features)[0]

            # Calculate confidence (simplified - could use prediction intervals)
            confidence = min(1.0, max(0.0, 1.0 - (abs(prediction) / 30)))  # Higher confidence for near-term predictions

            # Convert negative predictions to positive (RUL should be positive)
            rul_days = max(0.0, prediction)

            # Classify probability based on RUL
            if rul_days < 7:
                probability = 0.9
            elif rul_days < 14:
                probability = 0.7
            elif rul_days < 30:
                probability = 0.5
            else:
                probability = 0.3

            return {
                "rul_days": round(rul_days, 2),
                "probability": round(probability, 2),
                "confidence": round(confidence, 2)
            }

        except Exception as e:
            logger.error(f"Failed to make prediction: {e}")
            return None

    def should_report_anomaly(
        self,
        tolerance_results: Dict[str, Any],
        prediction: Optional[Dict[str, Any]]
    ) -> tuple[bool, str]:
        """
        Determine if anomaly should be reported to API.

        Args:
            tolerance_results: Results from tolerance checking
            prediction: ML prediction results (optional)

        Returns:
            (should_report, reason)
        """
        # Check if reporting is enabled
        if not self.anomaly_client:
            return False, "API client not configured"

        status = tolerance_results["status"]

        # Don't report Normal status
        if status == "Normal":
            return False, "Status is Normal"

        # Check debouncing - don't report same anomaly within configured time
        debounce_minutes = self.deployment_config.get("anomaly_reporting", {}).get("debounce_minutes", 60)
        now = datetime.utcnow()

        for param in tolerance_results["violations"].keys():
            last_reported = self.last_reported.get(param)
            if last_reported:
                elapsed = (now - last_reported).total_seconds() / 60
                if elapsed < debounce_minutes:
                    return False, f"Recently reported {param} anomaly ({elapsed:.1f} min ago)"

        # Report if mandatory parameters exceeded
        if tolerance_results["mandatory_exceeded"]:
            return True, "Mandatory parameter exceeded"

        # Report if status is Critical or Failure
        if status in ["Critical", "Failure"]:
            return True, f"Status is {status}"

        # Report if ML predicts failure soon with high confidence
        if prediction:
            if prediction["probability"] > 0.7 and prediction["rul_days"] < 7:
                return True, "High confidence failure prediction"

        # Report Warning status (optional parameters)
        if status == "Warning":
            return True, "Warning status"

        return False, "No reporting criteria met"

    def format_anomaly_payload(
        self,
        tolerance_results: Dict[str, Any],
        prediction: Optional[Dict[str, Any]],
        current_row: pd.Series
    ) -> Dict[str, Any]:
        """
        Format anomaly data for API submission.

        Args:
            tolerance_results: Tolerance check results
            prediction: ML prediction (optional)
            current_row: Current sensor reading row

        Returns:
            Formatted payload dict
        """
        # Build description
        violations = tolerance_results["violations"]
        exceeded = [
            f"{p}: {v['deviation']:.1f}% (threshold: {v['threshold']:.1f}%)"
            for p, v in violations.items()
        ]
        description = f"Pump anomaly detected - {', '.join(exceeded)}"

        site_info = self.deployment_config.get("site_info", {})
        current_values = {
            param: float(current_row[col])
            for param, col in self.column_mapping.items()
            if col in current_row
        }

        # Build payload
        payload = {
            "sourceType": "log",
            "description": description,
            "siteId": site_info.get("site_id"),
            "pumpId": site_info.get("pump_id"),
            "timestamp": current_row["timestamp"].isoformat() + "Z",
            "additionalContext": {
                "parameter": list(violations.keys()),
                "status": tolerance_results["status"],
                "tolerance_category": self.tolerance_category,
                "all_deviations": tolerance_results["deviations"],
                "baseline_values": {
                    param: self.baseline_parameters.get(f"baseline_{param}")
                    for param in ["flow", "head", "power", "efficiency"]
                },
                "current_values": current_values
            },
            "metadata": {
                "modelName": "pump-anomaly-detector",
                "modelVersion": self.model_metadata.get("version", "1.0.0"),
                "framework": "sklearn",
                "tolerance_category": self.tolerance_category
            }
        }

        # Add ML prediction if available
        if prediction:
            payload["metadata"]["confidence"] = prediction["confidence"]
            payload["metadata"]["prediction_rul_days"] = prediction["rul_days"]
            payload["metadata"]["prediction_probability"] = prediction["probability"]
            payload["metadata"]["model_type"] = self.model_metadata.get("model_type", "unknown")

        # Add sensor-specific info (use worst violation)
        if violations:
            worst_param = max(violations.items(), key=lambda x: abs(x[1]["deviation"]))[0]
            sensor_ids = site_info.get("sensor_ids", {})
            if worst_param in sensor_ids:
                payload["sensorId"] = sensor_ids[worst_param]
                payload["logValue"] = current_values.get(worst_param)

        return payload

    def save_unsent_anomaly(self, payload: Dict[str, Any]):
        """
        Save anomaly locally for later retry if API submission failed.

        Args:
            payload: Anomaly payload
        """
        try:
            unsent_dir = self.artifact_dir / "unsent_anomalies"
            unsent_dir.mkdir(exist_ok=True)

            timestamp = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
            filepath = unsent_dir / f"anomaly_{timestamp}.json"

            with open(filepath, 'w') as f:
                json.dump(payload, f, indent=2)

            logger.info(f"Saved unsent anomaly to {filepath}")

        except Exception as e:
            logger.error(f"Failed to save unsent anomaly: {e}")

    def run_inference(self, input_csv: str, output_json: str):
        """
        Run complete inference pipeline.

        Args:
            input_csv: Path to input sensor data CSV
            output_json: Path to output results JSON
        """
        try:
            logger.info("="*70)
            logger.info("PUMP ANOMALY DETECTION - EDGE INFERENCE")
            logger.info("="*70)

            # 1. Load and process data
            logger.info("Step 1: Loading sensor data...")
            df = self.process_sensor_data(input_csv)

            # 2. Process latest reading
            current_row = df.iloc[-1]
            logger.info(f"Step 2: Processing reading from {current_row['timestamp']}")

            # 3. Calculate deviations
            logger.info("Step 3: Calculating deviations...")
            deviations = self.calculate_deviations(current_row)
            logger.info(f"Deviations: {deviations}")

            # 4. Check tolerances
            logger.info("Step 4: Checking tolerances...")
            tolerance_results = self.check_tolerances(deviations)
            logger.info(f"Status: {tolerance_results['status']}")
            if tolerance_results['violations']:
                logger.warning(f"Violations: {tolerance_results['violations']}")

            # 5. ML prediction (if available and enough data)
            prediction = None
            if self.model is not None:
                logger.info("Step 5: Running ML prediction...")
                features = self.extract_features(df)
                if features is not None:
                    prediction = self.predict_failure(features)
                    if prediction:
                        logger.info(f"Prediction: RUL={prediction['rul_days']:.1f} days, "
                                  f"Probability={prediction['probability']:.2f}")

            # 6. Report anomaly if needed
            should_report, reason = self.should_report_anomaly(tolerance_results, prediction)
            logger.info(f"Step 6: Anomaly reporting - {reason}")

            if should_report and self.anomaly_client:
                try:
                    payload = self.format_anomaly_payload(tolerance_results, prediction, current_row)
                    response = self.anomaly_client.submit_anomaly(payload)
                    logger.info(f"✓ Anomaly reported to API: ID {response.get('id')}")

                    # Update debounce tracking
                    for param in tolerance_results["violations"].keys():
                        self.last_reported[param] = datetime.utcnow()

                except Exception as e:
                    logger.error(f"✗ Failed to report anomaly: {e}")
                    # Save locally for retry
                    self.save_unsent_anomaly(payload)

            # 7. Save results locally
            logger.info(f"Step 7: Saving results to {output_json}...")
            results = {
                "timestamp": current_row['timestamp'].isoformat(),
                "status": tolerance_results["status"],
                "deviations": deviations,
                "violations": tolerance_results["violations"],
                "prediction": prediction,
                "reported_to_api": should_report and self.anomaly_client is not None
            }

            with open(output_json, 'w') as f:
                json.dump(results, f, indent=2)

            logger.info("="*70)
            logger.info(f"✓ INFERENCE COMPLETE - Status: {tolerance_results['status']}")
            logger.info("="*70)

        except Exception as e:
            logger.error(f"Inference failed: {e}")
            raise


def main():
    """Main entry point."""
    if len(sys.argv) != 3:
        print("Usage: python inference.py <input_csv> <output_json>")
        print("Example: python inference.py sensor_data.csv results.json")
        sys.exit(1)

    input_csv = sys.argv[1]
    output_json = sys.argv[2]

    # Initialize inference engine
    engine = EdgeInference(artifact_dir=".")

    # Run inference
    engine.run_inference(input_csv, output_json)


if __name__ == "__main__":
    main()
