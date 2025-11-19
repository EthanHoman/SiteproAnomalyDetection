"""
Artifact Packager for Edge Deployment

Packages trained models and configurations into deployable artifacts for edge devices.

Usage:
    python tools/package_artifact.py --pump "Well 1" --output artifacts/well1_v1.0.0.zip
"""

import os
import sys
import json
import shutil
import argparse
import joblib
from pathlib import Path
from datetime import datetime
from zipfile import ZipFile, ZIP_DEFLATED

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src import data_processing, tolerance_checker


class ArtifactPackager:
    """
    Package trained models and configurations for edge deployment.
    """

    def __init__(self, pump_name: str, version: str = "1.0.0"):
        """
        Initialize packager.

        Args:
            pump_name: Name of pump (e.g., "Well 1")
            version: Artifact version (semantic versioning)
        """
        self.pump_name = pump_name
        self.version = version
        self.project_root = Path(__file__).parent.parent
        self.model_dir = self.project_root / "models" / "trained_models" / pump_name
        self.baseline_file = None
        self.baseline_data = None

    def load_baseline(self, baseline_file: str):
        """
        Load baseline data for the pump.

        Args:
            baseline_file: Path to baseline CSV
        """
        self.baseline_file = baseline_file
        self.baseline_data = data_processing.load_baseline_data(baseline_file)
        print(f"✓ Loaded baseline for {self.baseline_data['well_id']}")

    def create_artifact_structure(self, temp_dir: Path):
        """
        Create artifact directory structure.

        Args:
            temp_dir: Temporary directory for artifact creation
        """
        # Create directories
        (temp_dir / "model").mkdir(parents=True)
        (temp_dir / "config").mkdir(parents=True)
        print(f"✓ Created artifact structure")

    def copy_model_files(self, temp_dir: Path):
        """
        Copy model files to artifact.

        Args:
            temp_dir: Temporary artifact directory
        """
        model_dest = temp_dir / "model"

        # Copy model if exists
        model_file = self.model_dir / "random_forest_model.pkl"
        if model_file.exists():
            shutil.copy(model_file, model_dest / "anomaly_detector.pkl")
            print(f"✓ Copied ML model")
        else:
            print(f"⚠ No ML model found - will use tolerance-based detection only")

        # Copy scaler if exists
        scaler_file = self.model_dir / "scaler.pkl"
        if scaler_file.exists():
            shutil.copy(scaler_file, model_dest / "scaler.pkl")
            print(f"✓ Copied scaler")

        # Create model metadata
        metadata = {
            "version": self.version,
            "pump_name": self.pump_name,
            "created_at": datetime.utcnow().isoformat() + "Z",
            "model_type": "random_forest",
            "framework": "sklearn"
        }

        # Add metrics if available
        metrics_file = self.model_dir / "metrics.txt"
        if metrics_file.exists():
            with open(metrics_file, 'r') as f:
                metrics_text = f.read()
                # Parse metrics (simplified)
                metadata["metrics_summary"] = metrics_text

        with open(model_dest / "model_metadata.json", 'w') as f:
            json.dump(metadata, f, indent=2)
        print(f"✓ Created model metadata")

    def create_config_files(self, temp_dir: Path):
        """
        Create configuration files for edge deployment.

        Args:
            temp_dir: Temporary artifact directory
        """
        config_dest = temp_dir / "config"

        # 1. Baseline parameters
        baseline_params = {
            "well_id": self.baseline_data['well_id'],
            "pump_type": self.baseline_data['pump_type'],
            "horsepower": self.baseline_data['horsepower'],
            "application": self.baseline_data['application'],
            "tolerance_category": tolerance_checker.select_tolerance_category(
                self.baseline_data['application'],
                self.baseline_data['horsepower']
            ),
            "baseline_flow": self.baseline_data['baseline_flow_gpm'],
            "baseline_head": self.baseline_data['baseline_discharge_pressure_psi'],
            "baseline_power": self.baseline_data['baseline_power_hp'],
            "baseline_efficiency": self.baseline_data['baseline_efficiency_percent']
        }

        with open(config_dest / "baseline.json", 'w') as f:
            json.dump(baseline_params, f, indent=2)
        print(f"✓ Created baseline configuration")

        # 2. Tolerances
        tolerances_src = self.project_root / "config" / "tolerances.json"
        shutil.copy(tolerances_src, config_dest / "tolerances.json")
        print(f"✓ Copied tolerances configuration")

        # 3. Column mapping
        column_mapping = {
            "flow": "Flow (gpm)",
            "head": "Discharge Pressure (psi)",
            "power": "Motor Power (hp)",
            "efficiency": "Pump Efficiency (%)"
        }

        with open(config_dest / "column_mapping.json", 'w') as f:
            json.dump(column_mapping, f, indent=2)
        print(f"✓ Created column mapping")

        # 4. Deployment config (template - user must customize)
        deployment_config_src = self.project_root / "config" / "deployment_config.json"
        shutil.copy(deployment_config_src, config_dest / "deployment_config.json")
        print(f"✓ Copied deployment configuration template")

    def copy_inference_script(self, temp_dir: Path):
        """
        Copy inference script to artifact.

        Args:
            temp_dir: Temporary artifact directory
        """
        inference_src = self.project_root / "src" / "templates" / "inference_template.py"
        inference_dest = temp_dir / "inference.py"
        shutil.copy(inference_src, inference_dest)
        print(f"✓ Copied inference script")

    def copy_api_client(self, temp_dir: Path):
        """
        Copy anomaly API client to artifact.

        Args:
            temp_dir: Temporary artifact directory
        """
        client_src = self.project_root / "src" / "anomaly_client.py"
        client_dest = temp_dir / "anomaly_client.py"
        shutil.copy(client_src, client_dest)
        print(f"✓ Copied API client")

    def create_requirements(self, temp_dir: Path):
        """
        Create minimal requirements.txt for edge deployment.

        Args:
            temp_dir: Temporary artifact directory
        """
        # Minimal dependencies for edge inference
        requirements = [
            "# Edge Inference Dependencies",
            "pandas>=1.3.0",
            "numpy>=1.21.0",
            "scikit-learn>=1.0.0",
            "joblib>=1.0.0",
            "requests>=2.28.0",
            "python-dateutil>=2.8.0"
        ]

        with open(temp_dir / "requirements.txt", 'w') as f:
            f.write('\n'.join(requirements))
        print(f"✓ Created requirements.txt")

    def create_readme(self, temp_dir: Path):
        """
        Create README for edge deployment.

        Args:
            temp_dir: Temporary artifact directory
        """
        readme_content = f"""# Pump Anomaly Detection - Edge Artifact

**Pump:** {self.pump_name}
**Version:** {self.version}
**Created:** {datetime.utcnow().strftime('%Y-%m-%d %H:%M:%S')} UTC

## Quick Start

### 1. Installation

```bash
pip install -r requirements.txt
```

### 2. Configuration

Edit `config/deployment_config.json`:
- Set your API bearer token in `api.bearer_token`
- Update `site_info.site_id` and `site_info.pump_id`
- Update `site_info.sensor_ids` to match your sensor IDs

### 3. Run Inference

```bash
python inference.py input_data.csv results.json
```

**Input CSV format:**
```csv
timestamp,Well ID,Flow (gpm),Discharge Pressure (psi),Suction Pressure (psi),Motor Power (hp),Pump Efficiency (%),Motor Speed (rpm)
2024-11-18 14:30:00,Well 1,505,148,25,76,85.2,1760
```

**Output JSON:**
```json
{{
  "timestamp": "2024-11-18T14:30:00",
  "status": "Normal",
  "deviations": {{...}},
  "violations": {{...}},
  "prediction": {{...}},
  "reported_to_api": true
}}
```

## Artifact Structure

```
.
├── model/
│   ├── anomaly_detector.pkl    # Trained ML model
│   ├── scaler.pkl               # Feature scaler
│   └── model_metadata.json      # Model information
├── config/
│   ├── baseline.json            # Baseline parameters
│   ├── tolerances.json          # Tolerance specifications
│   ├── column_mapping.json      # CSV column mapping
│   └── deployment_config.json   # Deployment settings (CUSTOMIZE THIS!)
├── inference.py                 # Main inference script
├── anomaly_client.py            # API client for reporting
├── requirements.txt             # Python dependencies
└── README.md                    # This file
```

## Anomaly Reporting

When anomalies are detected, they are automatically reported to the central API:
- **API Endpoint:** https://sp-api-sink.azurewebsites.net/api/v1/edge/anomalies
- **Authentication:** Bearer token (configure in deployment_config.json)

**Reporting Criteria:**
- Mandatory parameter (Flow/Head) exceeds tolerance
- Status = Warning/Critical/Failure
- ML prediction indicates failure within 7 days (confidence > 0.7)

**Debouncing:**
- Anomalies are not reported more than once per hour (configurable)

## Tolerance Category

**Category:** {tolerance_checker.select_tolerance_category(self.baseline_data['application'], self.baseline_data['horsepower'])}
**Application:** {self.baseline_data['application']}

**Thresholds:**
- Flow: +10% (mandatory)
- Head: +6% (mandatory)
- Power: +10% (optional)
- Efficiency: -0% (optional)

## Troubleshooting

**Error: "Missing required field: sourceType"**
- Check that API client is properly configured

**Error: "Failed to load model"**
- Model files may be missing or corrupted
- Tolerance-based detection will still work

**Warning: "Not enough historical data for ML prediction"**
- Need at least 168 data points (1 week hourly) for ML predictions

## Support

For issues or questions, contact the system administrator.

---

*Generated by Pump Anomaly Detection System v{self.version}*
"""

        with open(temp_dir / "README.md", 'w') as f:
            f.write(readme_content)
        print(f"✓ Created README.md")

    def package(self, output_path: str):
        """
        Create packaged artifact ZIP file.

        Args:
            output_path: Path for output ZIP file
        """
        print(f"\n{'='*70}")
        print(f"PACKAGING ARTIFACT: {self.pump_name} v{self.version}")
        print(f"{'='*70}\n")

        # Create temporary directory
        temp_dir = Path("temp_artifact")
        if temp_dir.exists():
            shutil.rmtree(temp_dir)
        temp_dir.mkdir()

        try:
            # Build artifact
            self.create_artifact_structure(temp_dir)
            self.copy_model_files(temp_dir)
            self.create_config_files(temp_dir)
            self.copy_inference_script(temp_dir)
            self.copy_api_client(temp_dir)
            self.create_requirements(temp_dir)
            self.create_readme(temp_dir)

            # Create ZIP archive
            print(f"\nCreating ZIP archive...")
            output_path = Path(output_path)
            output_path.parent.mkdir(parents=True, exist_ok=True)

            with ZipFile(output_path, 'w', ZIP_DEFLATED) as zipf:
                for file in temp_dir.rglob('*'):
                    if file.is_file():
                        arcname = file.relative_to(temp_dir)
                        zipf.write(file, arcname)

            # Get file size
            size_mb = output_path.stat().st_size / (1024 * 1024)

            print(f"\n{'='*70}")
            print(f"✓ ARTIFACT PACKAGED SUCCESSFULLY")
            print(f"{'='*70}")
            print(f"Output: {output_path}")
            print(f"Size: {size_mb:.2f} MB")
            print(f"{'='*70}\n")

        finally:
            # Cleanup
            if temp_dir.exists():
                shutil.rmtree(temp_dir)


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Package pump anomaly detection artifact for edge deployment"
    )
    parser.add_argument(
        "--pump",
        required=True,
        help="Pump name (e.g., 'Well 1')"
    )
    parser.add_argument(
        "--baseline",
        required=True,
        help="Path to baseline CSV file"
    )
    parser.add_argument(
        "--output",
        required=True,
        help="Output ZIP file path (e.g., 'artifacts/well1_v1.0.0.zip')"
    )
    parser.add_argument(
        "--version",
        default="1.0.0",
        help="Artifact version (default: 1.0.0)"
    )

    args = parser.parse_args()

    # Create packager
    packager = ArtifactPackager(args.pump, args.version)

    # Load baseline
    packager.load_baseline(args.baseline)

    # Package artifact
    packager.package(args.output)


if __name__ == "__main__":
    main()
