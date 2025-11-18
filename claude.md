# Claude Code Project Guide: Pump Anomaly Detection - Phase 1

## ⚠️ PHASE 1 SCOPE ONLY ⚠️

**YOU ARE BUILDING (Phase 1):**
- ✅ Data loading and processing
- ✅ Tolerance-based anomaly detection  
- ✅ Visualization and reporting
- ✅ Predictive ML model
- ✅ Well 1 failure analysis

**YOU ARE NOT BUILDING (Later Phases):**
- ❌ API integration
- ❌ Edge deployment
- ❌ Model packaging for production
- ❌ Authentication/authorization

If you're asked about API integration, edge deployment, or production packaging, respond: "That's Phase 2/3/4. We're focused on Phase 1 core functionality first."

---

## 🚨 CRITICAL: User's EXACT Data Format 🚨

### Operational Data Column Names (MUST USE EXACTLY):
````
timestamp
Well ID
Flow (gpm)
Discharge Pressure (psi)
Suction Pressure (psi)
Motor Power (hp)
Pump Efficiency (%)
Motor Speed (rpm)
````

### Example Data Row:
````csv
5/1/2024 0:00,Well 1,505,148,25,76,85.2,1760
````

### Non-Negotiable Rules:

1. **Column names have SPACES and UNITS in parentheses**
````python
   # ✅ CORRECT
   flow = df["Flow (gpm)"]
   head = df["Discharge Pressure (psi)"]
   power = df["Motor Power (hp)"]
   efficiency = df["Pump Efficiency (%)"]
   
   # ❌ WRONG - Will cause KeyError
   flow = df["Flow"]
   flow = df.Flow_gpm
   head = df["Discharge_Pressure"]
````

2. **Timestamps are NOT zero-padded**
````python
   # ✅ CORRECT
   df['timestamp'] = pd.to_datetime(df['timestamp'], format='mixed')
   
   # Examples that must work:
   # "5/1/2024 0:00"  (not "05/01/2024 00:00")
   # "12/25/2024 13:45"
````

3. **Well ID has a SPACE**
````python
   # ✅ CORRECT
   well_1_data = df[df["Well ID"] == "Well 1"]
   
   # ❌ WRONG
   well_1_data = df[df["Well_ID"] == "Well_1"]
````

4. **HEAD = "Discharge Pressure (psi)"** (NOT Suction Pressure)
````python
   # ✅ CORRECT - Use discharge pressure for tolerance checking
   head_deviation = calculate_deviation(
       current=df["Discharge Pressure (psi)"],
       baseline=baseline_discharge_pressure_psi
   )
   
   # ❌ WRONG - Suction pressure is auxiliary data
   head_deviation = calculate_deviation(
       current=df["Suction Pressure (psi)"],  # NO!
       baseline=baseline_discharge_pressure_psi
   )
````

---

## 📊 Tolerance Specifications (MEMORIZE)

### Well 1 is Category 1U (Municipal Water and Wastewater)
````python
TOLERANCES_1U = {
    "flow": {
        "mandatory": True,
        "max_deviation": 10,      # +10% only (no negative check)
        "min_deviation": -999     # Effectively no minimum
    },
    "head": {
        "mandatory": True,
        "max_deviation": 6,       # +6% only (no negative check)
        "min_deviation": -999
    },
    "power": {
        "mandatory": False,
        "max_deviation": 10,      # +10% only
        "min_deviation": -999
    },
    "efficiency": {
        "mandatory": False,
        "max_deviation": 999,     # No upper limit
        "min_deviation": 0        # NO negative deviation allowed (can't lose efficiency)
    }
}
````

### Complete Tolerance Table:

| Parameter  | Req.      | 1B      | 1E      | 1U    | 2B      | 2U     | 3B      |
|------------|-----------|---------|---------|-------|---------|--------|---------|
| Flow       | Mandatory | +/- 5%  | +/- 5%  | +10%  | +/- 8%  | +16%   | +/- 9%  |
| Head       | Mandatory | +/- 3%  | +/- 3%  | +6%   | +/- 5%  | +10%   | +/- 7%  |
| Power      | Optional  | +4%     | +4%     | +10%  | +8%     | +16%   | +9%     |
| Efficiency | Optional  | -3%     | -0%     | -0%   | -5%     | -5%    | -7%     |

**Key:**
- `+/- X%` = bidirectional (check both positive and negative deviation)
- `+X%` = positive only (only check if it goes above)
- `-X%` = negative only (only check if it drops below)
- `-0%` = NO negative deviation allowed

### Application → Category Mapping:
````python
APPLICATION_MAPPING = {
    "Municipal Water and Wastewater": "1U",  # ← Well 1 is here
    "API": "1B",
    "Cooling Tower": "2B",
    "General Industry": "3B" if hp < 134 else "2B",
    "Dewatering, drainage, and irrigation": "3B" if hp < 134 else "2B"
}
````

---

## 📁 Project Structure
````
pump_anomaly_detection/
├── data/
│   ├── raw/
│   │   ├── baseline/          # ← User puts baseline CSV here
│   │   └── operational/       # ← User puts operational CSV here
│   ├── processed/             # ← Your code saves processed data here
│   └── results/               # ← Intermediate analysis results
├── models/
│   └── trained_models/        # ← Save .pkl models here
├── outputs/
│   ├── visualizations/        # ← Save plots as PNG/PDF here
│   └── reports/               # ← Save analysis reports here
├── src/
│   ├── data_processing.py     # Load data, calculate deviations
│   ├── tolerance_checker.py   # Check tolerances, classify status
│   ├── pump_monitor.py        # Main PumpMonitor class
│   ├── predictive_model.py    # Train/predict with ML
│   └── visualization.py       # Generate plots
├── config/
│   └── tolerances.json        # All tolerance values (auto-generated)
├── tests/                     # Optional unit tests
├── requirements.txt
├── README.md
└── claude.md                  # This file
````

---

## 🎯 Implementation Order

### ✅ Phase 1: Setup (DO THIS FIRST)

**Files to create:**
1. Directory structure (all folders above)
2. `config/tolerances.json` with all 6 categories × 4 parameters
3. `data/raw/baseline/baseline_template.csv`
4. `data/raw/operational/operational_template.csv`
5. `README.md` with data format instructions

**baseline_template.csv:**
````csv
Well ID,pump_type,horsepower,application,baseline_flow_gpm,baseline_discharge_pressure_psi,baseline_power_hp,baseline_efficiency_percent
Well 1,Goulds 3409,100,Municipal Water and Wastewater,500,150,75,85.5
````

**operational_template.csv:**
````csv
timestamp,Well ID,Flow (gpm),Discharge Pressure (psi),Suction Pressure (psi),Motor Power (hp),Pump Efficiency (%),Motor Speed (rpm)
5/1/2024 0:00,Well 1,505,148,25,76,85.2,1760
5/1/2024 1:00,Well 1,510,152,26,77,84.8,1765
5/1/2024 2:00,Well 1,498,147,24,75.5,85.3,1758
````

**Checklist:**
- [ ] All directories exist
- [ ] `tolerances.json` contains all values from tolerance table
- [ ] Template CSV files match user's EXACT format (with spaces!)
- [ ] README explains data format clearly
- [ ] User can immediately see what format is needed

**DO NOT proceed until this is complete and verified.**

---

### ✅ Phase 2: Data Processing

**File:** `src/data_processing.py`

**Functions to implement:**
````python
def load_baseline_data(filepath: str) -> dict:
    """
    Load baseline pump specifications from CSV.
    
    Returns dict with keys:
    - well_id: str
    - pump_type: str
    - horsepower: float
    - application: str
    - baseline_flow_gpm: float
    - baseline_discharge_pressure_psi: float
    - baseline_power_hp: float
    - baseline_efficiency_percent: float
    """
    pass

def load_operational_data(filepath: str, well_id: str = None) -> pd.DataFrame:
    """
    Load operational sensor logs from CSV.
    
    MUST handle:
    - Column names with spaces: "Flow (gpm)", "Discharge Pressure (psi)"
    - Non-zero-padded timestamps: "5/1/2024 0:00"
    - Well ID filtering: filter by well_id if provided
    
    Returns DataFrame with columns:
    - timestamp (datetime)
    - Well ID (str)
    - Flow (gpm) (float)
    - Discharge Pressure (psi) (float)
    - Suction Pressure (psi) (float)
    - Motor Power (hp) (float)
    - Pump Efficiency (%) (float)
    - Motor Speed (rpm) (float)
    """
    pass

def calculate_deviations(
    operational_df: pd.DataFrame,
    baseline: dict,
    column_mapping: dict
) -> pd.DataFrame:
    """
    Calculate percentage deviations from baseline.
    
    Formula: deviation = ((current - baseline) / baseline) * 100
    
    Args:
        operational_df: DataFrame with sensor readings
        baseline: Dict with baseline values
        column_mapping: Maps parameters to column names
            {
                "flow": "Flow (gpm)",
                "head": "Discharge Pressure (psi)",
                "power": "Motor Power (hp)",
                "efficiency": "Pump Efficiency (%)"
            }
    
    Returns DataFrame with original columns plus:
    - flow_deviation_pct
    - head_deviation_pct
    - power_deviation_pct
    - efficiency_deviation_pct
    """
    pass

def validate_data(df: pd.DataFrame, required_columns: list) -> None:
    """
    Validate data quality. Raise ValueError if issues found.
    
    Check:
    - All required columns present (EXACT names with spaces)
    - No negative values (except suction pressure can be negative)
    - Efficiency between 0-100%
    - Timestamps are valid and chronological
    - No duplicate timestamps
    """
    pass
````

**Checklist:**
- [ ] Can load baseline CSV without errors
- [ ] Can load operational CSV with spaces in columns
- [ ] Timestamps parse correctly (test: "5/1/2024 0:00")
- [ ] Filter by "Well 1" works correctly
- [ ] Deviation calculation is correct (test with known values)
- [ ] Data validation catches common errors
- [ ] Can save processed data to `data/processed/`

---

### ✅ Phase 3: Anomaly Detection

**File:** `src/tolerance_checker.py`

**Functions to implement:**
````python
def select_tolerance_category(application: str, horsepower: float) -> str:
    """
    Select tolerance category based on application and HP.
    
    Returns: "1B", "1E", "1U", "2B", "2U", or "3B"
    """
    pass

def load_tolerances(category: str) -> dict:
    """
    Load tolerance specifications for a category from tolerances.json.
    
    Returns dict like:
    {
        "flow": {"mandatory": True, "max_deviation": 10, "min_deviation": -999},
        "head": {"mandatory": True, "max_deviation": 6, "min_deviation": -999},
        ...
    }
    """
    pass

def check_tolerances(
    deviations: pd.Series,
    tolerances: dict
) -> dict:
    """
    Check if deviations exceed tolerance thresholds.
    
    Args:
        deviations: Series with {parameter}_deviation_pct values
        tolerances: Dict with tolerance specs
    
    Returns dict:
    {
        "flow": {"exceeded": True/False, "deviation": 12.5, "threshold": 10},
        "head": {"exceeded": False, "deviation": 3.2, "threshold": 6},
        ...
    }
    """
    pass

def classify_status(
    tolerance_check: dict,
    history: pd.DataFrame = None
) -> str:
    """
    Classify pump status based on tolerance violations.
    
    Rules:
    - Normal: All parameters within tolerance
    - Warning: 
        * Optional parameter(s) exceed tolerance OR
        * One mandatory parameter slightly exceeds (< 1.5x threshold)
    - Critical:
        * Multiple parameters exceed tolerance OR
        * Mandatory parameter significantly exceeds (> 1.5x threshold) OR
        * Sustained degradation trend (if history provided)
    - Failure:
        * Multiple mandatory parameters far exceed threshold OR
        * Severe degradation
    
    Returns: "Normal", "Warning", "Critical", or "Failure"
    """
    pass

def find_first_exceedance(
    deviations_df: pd.DataFrame,
    tolerances: dict
) -> dict:
    """
    Find first timestamp when each parameter exceeded tolerance.
    
    Returns dict:
    {
        "flow": "2024-06-15 14:23:00" or None,
        "head": "2024-06-20 08:45:00" or None,
        ...
    }
    """
    pass
````

**Checklist:**
- [ ] Well 1 correctly identified as category 1U
- [ ] Tolerances loaded correctly from JSON
- [ ] 1U tolerances applied: Flow +10%, Head +6%, Power +10%, Efficiency -0%
- [ ] Uses "Discharge Pressure (psi)" not "Suction Pressure (psi)" for head
- [ ] Correctly handles directional tolerances (+/-, +only, -only)
- [ ] Mandatory vs optional parameter distinction works
- [ ] Status classification logic is reasonable
- [ ] Can identify first timestamp when tolerance exceeded

---

### ✅ Phase 4: Visualization

**File:** `src/visualization.py`

**Functions to implement:**
````python
def plot_parameter_timeseries(
    df: pd.DataFrame,
    parameter: str,
    baseline: float,
    tolerance_band: tuple,
    output_path: str = None
) -> None:
    """
    Plot time-series for one parameter with baseline and tolerance bands.
    
    Args:
        df: DataFrame with timestamp and parameter columns
        parameter: Column name (e.g., "Flow (gpm)")
        baseline: Baseline value
        tolerance_band: (lower_limit, upper_limit) in absolute values
        output_path: Where to save plot (if None, just display)
    
    Visual elements:
    - Line plot of actual values
    - Horizontal line for baseline
    - Shaded region for tolerance band
    - Color-coded regions (green=normal, yellow=warning, red=critical)
    - Proper axis labels with units
    - Clear title and legend
    """
    pass

def plot_multi_parameter_dashboard(
    df: pd.DataFrame,
    baseline: dict,
    tolerances: dict,
    output_path: str = None
) -> None:
    """
    Create 2x2 subplot dashboard showing all four parameters.
    
    Subplots:
    1. Flow (gpm)
    2. Discharge Pressure (psi) - labeled as "Head"
    3. Motor Power (hp)
    4. Pump Efficiency (%)
    
    Each subplot shows:
    - Actual values over time
    - Baseline and tolerance bands
    - Status color coding
    """
    pass

def plot_degradation_timeline(
    first_exceedances: dict,
    failure_date: str,
    output_path: str = None
) -> None:
    """
    Create timeline showing when each parameter first exceeded tolerance.
    
    Visual: Horizontal bar chart or timeline with:
    - Each parameter as a row
    - Bar showing time from first exceedance to failure
    - Markers for key events
    """
    pass

def plot_deviation_trends(
    df: pd.DataFrame,
    output_path: str = None
) -> None:
    """
    Plot rate of change (slope) for each parameter's deviation.
    
    Shows which parameters are degrading fastest.
    """
    pass
````

**Checklist:**
- [ ] Time-series plots display correctly
- [ ] Axis labels include proper units (gpm, psi, hp, %)
- [ ] Tolerance bands are visible and correct
- [ ] Can visually identify when degradation started
- [ ] Multi-parameter dashboard is clear and readable
- [ ] Plots save to `outputs/visualizations/` as PNG
- [ ] Plots are high quality (300 DPI for reports)

---

### ✅ Phase 5: Predictive Model

**File:** `src/predictive_model.py`

**Functions to implement:**
````python
def engineer_features(
    df: pd.DataFrame,
    window_sizes: list = [24, 168]  # 1 day, 1 week in hours
) -> pd.DataFrame:
    """
    Create features for ML model.
    
    Features to create:
    - Current deviations (flow, head, power, efficiency)
    - Rolling mean/std for each deviation
    - Trend (slope over window)
    - Rate of change (acceleration)
    - Time since first tolerance exceedance
    - Cumulative deviation (area under curve)
    - Cross-parameter correlations
    
    Returns DataFrame with original columns plus engineered features
    """
    pass

def create_failure_labels(
    df: pd.DataFrame,
    failure_date: str,
    mode: str = "regression"
) -> pd.Series:
    """
    Create target labels for ML model.
    
    Args:
        df: DataFrame with timestamps
        failure_date: When pump failed
        mode: "regression" for RUL (days until failure)
              "classification" for binary (will fail soon? yes/no)
    
    Returns: Series of labels aligned with df index
    """
    pass

def train_failure_predictor(
    X: pd.DataFrame,
    y: pd.Series,
    model_type: str = "random_forest"
) -> tuple:
    """
    Train ML model to predict pump failures.
    
    Args:
        X: Feature matrix
        y: Target labels
        model_type: "random_forest", "gradient_boosting", or "linear"
    
    Returns: (trained_model, scaler, feature_names, metrics)
    
    Also saves:
    - Model to models/trained_models/model.pkl
    - Scaler to models/trained_models/scaler.pkl
    - Feature importance plot to outputs/visualizations/
    - Performance metrics to outputs/reports/
    """
    pass

def predict_failure(
    model,
    scaler,
    current_data: pd.DataFrame
) -> dict:
    """
    Make prediction for current pump state.
    
    Returns dict:
    {
        "remaining_useful_life_days": 14.2,
        "failure_probability": 0.85,
        "confidence_interval": (10.1, 18.3),
        "contributing_factors": ["efficiency", "flow", "head"]
    }
    """
    pass
````

**Checklist:**
- [ ] Feature engineering creates reasonable features
- [ ] Can create labels from failure date
- [ ] Model trains without errors
- [ ] Predictions are reasonable (not NaN, within expected range)
- [ ] Model can be saved and loaded
- [ ] Feature importance is documented and makes sense
- [ ] Performance metrics are calculated and saved
- [ ] Can make predictions on new data

---

### ✅ Phase 6: Analysis & Reporting

**File:** `src/pump_monitor.py`

**Main class to implement:**
````python
class PumpMonitor:
    """
    Main class for pump monitoring and analysis.
    
    Encapsulates all functionality for a single pump.
    """
    
    def __init__(
        self,
        baseline_file: str,
        tolerance_category: str = None
    ):
        """
        Initialize monitor with baseline data.
        
        Args:
            baseline_file: Path to baseline CSV
            tolerance_category: If None, auto-detect from application+HP
        """
        pass
    
    def load_operational_data(self, filepath: str) -> None:
        """Load operational sensor logs."""
        pass
    
    def analyze(self) -> None:
        """
        Run complete analysis pipeline:
        1. Calculate deviations
        2. Check tolerances
        3. Classify status over time
        4. Identify first exceedances
        5. Generate visualizations
        6. Train predictive model (if enough data)
        """
        pass
    
    def get_current_status(self) -> str:
        """Return current pump status (Normal/Warning/Critical/Failure)."""
        pass
    
    def get_anomaly_timeline(self) -> pd.DataFrame:
        """Return DataFrame with timestamp, status, and violations."""
        pass
    
    def predict_failure(self) -> dict:
        """Return failure prediction from ML model."""
        pass
    
    def generate_report(self, output_path: str) -> None:
        """
        Generate comprehensive markdown report.
        
        Report must answer:
        1. When did pump first exceed tolerances? (each parameter)
        2. What was the degradation timeline?
        3. How long in degraded state before failure?
        4. What were leading indicators?
        5. Could failure be predicted? How far in advance?
        6. Recommendations for monitoring and maintenance
        
        Include embedded visualizations.
        Export as markdown and PDF.
        """
        pass
````

**Report template:**
````markdown
# Pump Failure Analysis Report: Well 1

## Executive Summary
[Brief overview of findings]

## Timeline of Events

### First Tolerance Exceedances
- **Flow (gpm)**: First exceeded on [date] at [time]
  - Baseline: [value], Threshold: +10%, Actual: [value] (+[X]%)
- **Discharge Pressure (psi)**: First exceeded on [date] at [time]
  - Baseline: [value], Threshold: +6%, Actual: [value] (+[X]%)
- **Motor Power (hp)**: First exceeded on [date] at [time]
  - Baseline: [value], Threshold: +10%, Actual: [value] (+[X]%)
- **Pump Efficiency (%)**: First exceeded on [date] at [time]
  - Baseline: [value], Threshold: -0%, Actual: [value] (-[X]%)

[Embedded timeline visualization]

### Degradation Progression

**Time in each status:**
- Normal: [X] days
- Warning: [X] days
- Critical: [X] days
- Total degraded operation: [X] days before failure

[Embedded multi-parameter dashboard]

## Leading Indicators

The following parameters showed the earliest signs of degradation:

1. **[Parameter]**: Degraded [X]% over [Y] days (slope: [Z]%/day)
2. **[Parameter]**: Degraded [X]% over [Y] days (slope: [Z]%/day)

[Embedded rate-of-change plot]

## Predictive Analysis

### Could this failure have been predicted?

**ML Model Performance:**
- Model type: [Random Forest / Gradient Boosting / etc.]
- Accuracy: [X]%
- Precision: [X]%
- Recall: [X]%

**Early Warning Capability:**
- Failure could be predicted [X] days in advance with [Y]% confidence
- Key contributing factors: [list top 3 features]

[Embedded feature importance plot]

## Auxiliary Observations

**Suction Pressure:**
[Analysis of suction pressure trends]

**Motor Speed:**
[Analysis of motor speed variations]

## Recommendations

1. **Monitoring Frequency**: 
   - Current: [frequency]
   - Recommended: [frequency]

2. **Early Warning Thresholds**:
   - Set alerts at [X]% of tolerance limit
   - Focus on [parameters] as leading indicators

3. **Maintenance Schedule**:
   - Inspect pump when [condition]
   - Preventive maintenance every [interval]

4. **Continuous Improvement**:
   - Collect more failure data to improve model
   - Consider adding [sensors/features]

## Conclusion

[Summary of key findings and next steps]

---

*Report generated on [date] using Pump Anomaly Detection System v1.0*
````

**Checklist:**
- [ ] Report answers all key questions
- [ ] Report includes all required visualizations
- [ ] Recommendations are specific and actionable
- [ ] Report exports as markdown and PDF
- [ ] PumpMonitor class works with Well 1 data
- [ ] Can be reused for other pumps

---

### ✅ Phase 7: Documentation

**Update README.md with:**
````markdown
# Pump Anomaly Detection System

## Overview
[Project description]

## Installation
```bash
pip install -r requirements.txt
```

## Data Format Requirements

### Baseline Data
[Show baseline_template.csv format]

### Operational Data
[Show operational_template.csv format]

**CRITICAL: Column names must match exactly, including spaces and units!**

## Quick Start
```python
from src.pump_monitor import PumpMonitor

# 1. Create monitor
monitor = PumpMonitor(
    baseline_file="data/raw/baseline/well1_baseline.csv"
)

# 2. Load operational data
monitor.load_operational_data(
    "data/raw/operational/well1_operational.csv"
)

# 3. Run analysis
monitor.analyze()

# 4. Generate report
monitor.generate_report("outputs/reports/well1_analysis.md")
```

## Troubleshooting

**Error: KeyError: 'Flow (gpm)'**
- Make sure your CSV columns have exact names including spaces and units
- Check that you didn't rename columns

**Error: Could not parse timestamp**
- Timestamps should be in format "M/D/YYYY H:MM"
- Example: "5/1/2024 0:00" not "05/01/2024 00:00"

[More troubleshooting items]

## Examples

See `examples/well1_analysis.ipynb` for complete walkthrough.

## Project Structure
[Show directory tree]

## License
[License info]
````

**Checklist:**
- [ ] README clearly explains installation
- [ ] Data format is crystal clear (show examples!)
- [ ] Quick start works copy-paste
- [ ] Troubleshooting covers common issues
- [ ] Example notebook demonstrates full workflow

---

## 🧪 Testing Checklist

Run through this before claiming phase complete:

### Phase 1: Setup
- [ ] All directories created
- [ ] `tolerances.json` has all 6 categories × 4 parameters
- [ ] Template CSVs match user's exact format (spaces in columns!)
- [ ] README explains data requirements

### Phase 2: Data Processing
- [ ] Loads baseline CSV without errors
- [ ] Loads operational CSV with spaces in column names
- [ ] Parse "5/1/2024 0:00" timestamp correctly
- [ ] Filter by "Well 1" (with space) works
- [ ] Deviation calculation correct (manual spot check)
- [ ] Data validation catches errors

### Phase 3: Anomaly Detection
- [ ] Well 1 → 1U category (auto-detected)
- [ ] 1U tolerances: Flow +10%, Head +6%, Power +10%, Efficiency -0%
- [ ] Uses "Discharge Pressure (psi)" not "Suction Pressure (psi)"
- [ ] Flags first exceedance correctly
- [ ] Status classification reasonable

### Phase 4: Visualization
- [ ] Plots display without errors
- [ ] Labels have units (gpm, psi, hp, %)
- [ ] Tolerance bands visible
- [ ] Can see degradation visually
- [ ] Saves to outputs/visualizations/

### Phase 5: Predictive Model
- [ ] Features engineered successfully
- [ ] Model trains without errors
- [ ] Predictions reasonable
- [ ] Model saves/loads correctly
- [ ] Performance documented

### Phase 6: Analysis
- [ ] Report answers all questions
- [ ] Report includes visualizations
- [ ] Recommendations actionable
- [ ] PumpMonitor class works
- [ ] Exports as markdown and PDF

### Phase 7: Documentation
- [ ] README clear and complete
- [ ] Quick start works
- [ ] Example notebook runs end-to-end

---

## 🚫 Common Mistakes to Avoid

1. **Renaming columns** - NEVER rename columns to remove spaces. Work with them as-is.

2. **Wrong head measurement** - Always use "Discharge Pressure (psi)", never "Suction Pressure (psi)"

3. **Forgetting Well ID space** - It's "Well 1" not "Well_1"

4. **Hard-coding tolerance values** - Always load from tolerances.json

5. **Ignoring mandatory vs optional** - Flow and Head are mandatory, others optional

6. **Wrong efficiency tolerance interpretation** - It's percentage OF baseline, not percentage points

7. **Skipping data validation** - Always validate before processing

8. **Not testing with actual data** - Test with user's real data format early

9. **Building API integration** - Remember: this is Phase 1 only, no API yet!

10. **Overcomplicating** - Start simple, iterate

---

## ✅ Success Criteria

Phase 1 is complete when:

- [ ] User can drop CSV files in data/raw/ and run analysis
- [ ] System loads user's EXACT format without modification
- [ ] Well 1 correctly identified as 1U category
- [ ] Identifies exact timestamp when Well 1 exceeded tolerances
- [ ] Visualizations clear with proper units
- [ ] ML model predicts failure with documented accuracy
- [ ] Report answers all key questions
- [ ] System works for other pumps (not hardcoded to Well 1)
- [ ] Code clean, documented, maintainable
- [ ] Someone new can use the system from README

**If all checked, Phase 1 is done! 🎉**

Ready to move to Phase 2 (model packaging) or Phase 3 (API integration).

---

## 📝 Progress Tracking

**Current Step:** [Step 1 - Setup]

**Completed:**
- [ ] Phase 1: Setup
- [ ] Phase 2: Data Processing
- [ ] Phase 3: Anomaly Detection
- [ ] Phase 4: Visualization
- [ ] Phase 5: Predictive Model
- [ ] Phase 6: Analysis & Reporting
- [ ] Phase 7: Documentation

**Blockers:** [None yet]

**Next Action:** [Create directory structure and template files]

---

**Remember:** You're building Phase 1 ONLY. Focus on core functionality. API integration comes later. Keep it simple, test frequently, and make sure it works with the user's EXACT data format!
````

## 3. How to use these files

**Give to Claude Code:**
````
I need to build a pump anomaly detection system. I'm providing you with:
1. Complete specification for Phase 1 (core functionality only)
2. Development guide (claude.md) to keep you on track

This is Phase 1 of 4. We're focused on:
- Data processing
- Tolerance-based anomaly detection
- Visualization
- Predictive ML model
- Well 1 failure analysis

We are NOT building (these come in later phases):
- API integration
- Edge deployment
- Production packaging

Start by creating the directory structure and template files that match my EXACT data format (column names have spaces!).

[Attach phase1_prompt.xml]
[Attach phase1_claude.md]