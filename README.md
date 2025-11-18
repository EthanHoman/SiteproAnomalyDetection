# Pump Anomaly Detection System - Phase 1

## Overview

A comprehensive pump anomaly detection system that processes sensor data, applies industry-standard tolerance thresholds, trains predictive models, and generates detailed failure analysis reports.

**Phase 1 Focus:** Core functionality for data processing, tolerance-based anomaly detection, visualization, and ML-powered failure prediction.

## Installation

### Prerequisites
- Python 3.8 or higher
- pip package manager

### Setup

```bash
# Install required packages
pip install -r requirements.txt
```

## Data Format Requirements

### 📊 Got Excel Data? No Problem!

If your data is in Excel format:
1. **Easy way:** Run `python load_from_excel.py` (interactive converter)
2. **Manual way:** Save each sheet as CSV and place in `data/raw/`
3. **See:** `EXCEL_GUIDE.md` for detailed Excel instructions

### ⚠️ CRITICAL: Column Names Must Be EXACT

This system is designed to work with your data's **exact format**, including spaces and units in parentheses. Do **NOT** rename columns.

### Baseline Data Format

Place baseline pump specifications in `data/raw/baseline/` directory.

**Required columns:**
```csv
Well ID,pump_type,horsepower,application,baseline_flow_gpm,baseline_discharge_pressure_psi,baseline_power_hp,baseline_efficiency_percent
```

**Example:**
```csv
Well ID,pump_type,horsepower,application,baseline_flow_gpm,baseline_discharge_pressure_psi,baseline_power_hp,baseline_efficiency_percent
Well 1,Goulds 3409,100,Municipal Water and Wastewater,500,150,75,85.5
```

**Field descriptions:**
- `Well ID`: Pump identifier (e.g., "Well 1") - **must match operational data exactly**
- `pump_type`: Manufacturer and model
- `horsepower`: HP rating (determines tolerance category)
- `application`: One of:
  - "Municipal Water and Wastewater"
  - "API"
  - "Energy Conservation"
  - "Cooling Tower"
  - "General Industry"
  - "Dewatering, drainage, and irrigation"
- `baseline_flow_gpm`: Optimal flow rate (GPM)
- `baseline_discharge_pressure_psi`: Optimal discharge pressure (PSI)
- `baseline_power_hp`: Optimal power consumption (HP)
- `baseline_efficiency_percent`: Optimal efficiency (e.g., 85.5 for 85.5%)

### Operational Data Format

Place operational sensor logs in `data/raw/operational/` directory.

**Required columns (WITH SPACES AND UNITS):**
```csv
timestamp,Well ID,Flow (gpm),Discharge Pressure (psi),Suction Pressure (psi),Motor Power (hp),Pump Efficiency (%),Motor Speed (rpm)
```

**Example:**
```csv
timestamp,Well ID,Flow (gpm),Discharge Pressure (psi),Suction Pressure (psi),Motor Power (hp),Pump Efficiency (%),Motor Speed (rpm)
5/1/2024 0:00,Well 1,505,148,25,76,85.2,1760
5/1/2024 1:00,Well 1,510,152,26,77,84.8,1765
5/1/2024 2:00,Well 1,498,147,24,75.5,85.3,1758
```

**Critical formatting rules:**

1. **Column names include SPACES and units in parentheses:**
   - ✅ `"Flow (gpm)"` NOT `"Flow"` or `"Flow_gpm"`
   - ✅ `"Discharge Pressure (psi)"` NOT `"Head"` or `"Discharge_Pressure"`
   - ✅ `"Motor Power (hp)"` NOT `"Power"`
   - ✅ `"Pump Efficiency (%)"` NOT `"Efficiency"`

2. **Timestamps are NOT zero-padded:**
   - ✅ `"5/1/2024 0:00"` NOT `"05/01/2024 00:00"`
   - ✅ `"12/25/2024 13:45"` is correct

3. **Well ID has a SPACE:**
   - ✅ `"Well 1"` NOT `"Well_1"` or `"Well1"`

4. **HEAD = Discharge Pressure (NOT Suction Pressure):**
   - The system uses "Discharge Pressure (psi)" for tolerance checking
   - "Suction Pressure (psi)" is auxiliary data for advanced analysis

## Quick Start

```python
from src.pump_monitor import PumpMonitor

# 1. Create monitor with baseline data
monitor = PumpMonitor(
    baseline_file="data/raw/baseline/well1_baseline.csv"
)

# 2. Load operational sensor data
monitor.load_operational_data(
    "data/raw/operational/well1_operational.csv"
)

# 3. Run complete analysis pipeline
monitor.analyze()

# 4. Generate comprehensive report
monitor.generate_report("outputs/reports/well1_analysis.md")

# 5. Get current status
status = monitor.get_current_status()
print(f"Pump Status: {status}")

# 6. Get failure prediction (if model trained)
prediction = monitor.predict_failure()
print(f"Remaining Useful Life: {prediction['remaining_useful_life_days']} days")
```

## Tolerance Categories

The system automatically selects the appropriate tolerance category based on pump application and horsepower:

| Application | HP < 134 | HP ≥ 134 |
|------------|----------|----------|
| Municipal Water and Wastewater | 1U | 1U |
| API | 1B | 1B |
| Energy Conservation | 1E | 1E |
| Cooling Tower | 2B | 2B |
| General Industry | 3B | 2B |
| Dewatering, drainage, and irrigation | 3B | 2B |

### Category 1U Tolerances (Well 1 Example)

**Application:** Municipal Water and Wastewater
**Tolerance thresholds:**

| Parameter | Mandatory | Tolerance |
|-----------|-----------|-----------|
| Flow | Yes | +10% (positive only) |
| Head (Discharge Pressure) | Yes | +6% (positive only) |
| Power | No | +10% (positive only) |
| Efficiency | No | -0% (no negative deviation allowed) |

## Project Structure

```
pump_anomaly_detection/
├── data/
│   ├── raw/
│   │   ├── baseline/          # Place baseline pump specs here
│   │   └── operational/       # Place operational sensor logs here
│   ├── processed/             # Processed data (auto-generated)
│   └── results/               # Analysis results (auto-generated)
├── models/
│   └── trained_models/        # Trained ML models (auto-generated)
├── outputs/
│   ├── visualizations/        # Generated charts (auto-generated)
│   └── reports/               # Analysis reports (auto-generated)
├── src/
│   ├── data_processing.py     # Data loading and processing
│   ├── tolerance_checker.py   # Tolerance checking logic
│   ├── pump_monitor.py        # Main monitoring class
│   ├── predictive_model.py    # ML model training/prediction
│   └── visualization.py       # Plotting functions
├── config/
│   └── tolerances.json        # Tolerance specifications
├── tests/                     # Unit tests
├── requirements.txt           # Python dependencies
└── README.md                  # This file
```

## Troubleshooting

### Error: `KeyError: 'Flow (gpm)'`

**Problem:** Column names don't match exactly.

**Solution:** Make sure your CSV columns have exact names including spaces and units:
- ✅ `"Flow (gpm)"`
- ❌ `"Flow"` or `"Flow_gpm"`

### Error: `Could not parse timestamp`

**Problem:** Timestamp format doesn't match.

**Solution:** Timestamps should be in format `"M/D/YYYY H:MM"`:
- ✅ `"5/1/2024 0:00"`
- ❌ `"05/01/2024 00:00"`

### Error: No data for Well ID

**Problem:** Well ID doesn't match between baseline and operational data.

**Solution:** Ensure Well ID is identical (including spaces):
- ✅ `"Well 1"`
- ❌ `"Well_1"` or `"Well1"`

### Pump status always "Normal" despite obvious issues

**Problem:** Wrong tolerance category or incorrect head measurement.

**Solution:**
1. Verify tolerance category is correct for your application
2. Ensure using "Discharge Pressure (psi)" NOT "Suction Pressure (psi)"

## Features

### ✅ Data Processing
- Load baseline and operational data
- Calculate deviations from baseline
- Validate data quality
- Handle missing values

### ✅ Tolerance-Based Anomaly Detection
- Automatic tolerance category selection
- Real-time anomaly detection
- Status classification (Normal/Warning/Critical/Failure)
- Track first tolerance exceedance per parameter

### ✅ Visualization
- Time-series plots with tolerance bands
- Multi-parameter dashboards
- Degradation timeline charts
- Rate-of-change analysis

### ✅ Predictive ML Model
- Feature engineering from sensor data
- Train models to predict failures
- Remaining Useful Life (RUL) estimation
- Confidence intervals and contributing factors

### ✅ Comprehensive Reporting
- Detailed failure analysis reports
- Answer key questions:
  - When did pump first exceed tolerances?
  - What was the degradation timeline?
  - How long in degraded state?
  - What were leading indicators?
  - Could failure be predicted?
- Actionable maintenance recommendations

## What's NOT in Phase 1 (Coming Later)

- ❌ API integration for real-time monitoring
- ❌ Edge deployment and inference
- ❌ Model packaging for production
- ❌ Authentication/authorization

These features are planned for Phases 2-4.

## Examples

See template files in `data/raw/baseline/` and `data/raw/operational/` for exact data format examples.

## License

[Add your license information here]

## Support

For issues or questions, please refer to the troubleshooting section above or contact the development team.
