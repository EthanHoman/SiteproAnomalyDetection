# Quick Start Guide

Get started with the Pump Anomaly Detection System in 5 minutes!

## Prerequisites

- Python 3.8 or higher
- pip package manager

## Step 1: Install Dependencies (1 minute)

```bash
pip install -r requirements.txt
```

This installs:
- pandas (data processing)
- numpy (numerical operations)
- scikit-learn (machine learning)
- matplotlib & seaborn (visualization)
- scipy (statistics)

## Step 2: Prepare Your Data (2 minutes)

### Baseline Data

Create or copy your baseline CSV to `data/raw/baseline/`:

```csv
Well ID,pump_type,horsepower,application,baseline_flow_gpm,baseline_discharge_pressure_psi,baseline_power_hp,baseline_efficiency_percent
Well 1,Goulds 3409,100,Municipal Water and Wastewater,500,150,75,85.5
```

**Critical:** Column names must be EXACT (no spaces or variations)

### Operational Data

Create or copy your operational CSV to `data/raw/operational/`:

```csv
timestamp,Well ID,Flow (gpm),Discharge Pressure (psi),Suction Pressure (psi),Motor Power (hp),Pump Efficiency (%),Motor Speed (rpm)
5/1/2024 0:00,Well 1,505,148,25,76,85.2,1760
5/1/2024 1:00,Well 1,510,152,26,77,84.8,1765
```

**Critical:** Column names MUST include spaces and units in parentheses!
- ✅ `"Flow (gpm)"` NOT `"Flow"`
- ✅ `"Discharge Pressure (psi)"` NOT `"Head"`
- ✅ Timestamps: `"5/1/2024 0:00"` NOT `"05/01/2024 00:00"`

## Step 3: Run Analysis (1 minute)

### Option A: Use the Example Script (Easiest)

```bash
python example_analysis.py
```

This will:
1. Load baseline and operational data
2. Calculate deviations
3. Check tolerances
4. Generate visualizations
5. Create a comprehensive report

### Option B: Use Python Code (More Control)

```python
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

# Check status
print(f"Status: {monitor.get_current_status()}")
```

## Step 4: Review Results (1 minute)

### Visualizations

Check `outputs/visualizations/` for:
- `Well 1_dashboard.png` - Overall performance dashboard
- `Well 1_flow.png` - Flow analysis
- `Well 1_head.png` - Head/pressure analysis
- `Well 1_power.png` - Power consumption
- `Well 1_efficiency.png` - Efficiency trends
- `Well 1_timeline.png` - Degradation timeline
- `Well 1_trends.png` - Rate of change analysis
- `Well 1_status_timeline.png` - Status over time

### Report

Open `outputs/reports/Well 1_analysis.md` to see:
- When pump first exceeded tolerances
- Degradation timeline
- Status distribution
- Recommendations

### Processed Data

Check `data/processed/Well 1_processed.csv` for:
- All original data
- Calculated deviations
- Status classifications

## Common Issues & Solutions

### ❌ `KeyError: 'Flow (gpm)'`

**Problem:** Column names don't match exactly

**Solution:** Make sure your CSV has:
```csv
Flow (gpm)         ← with space and parentheses
Discharge Pressure (psi)    ← with space and parentheses
Motor Power (hp)   ← with space and parentheses
Pump Efficiency (%)         ← with space and parentheses
```

### ❌ `Could not parse timestamp`

**Problem:** Timestamp format is incorrect

**Solution:** Use non-zero-padded format:
```
✅ 5/1/2024 0:00
❌ 05/01/2024 00:00
```

### ❌ `No data found for Well ID`

**Problem:** Well ID doesn't match between files

**Solution:** Ensure EXACT match including spaces:
```
✅ "Well 1"    (with space)
❌ "Well_1" or "Well1"
```

### ❌ Pump always shows "Normal" despite issues

**Problem:** Wrong tolerance category or head measurement

**Solution:**
1. Check that application matches one of:
   - "Municipal Water and Wastewater"
   - "API"
   - "Energy Conservation"
   - "Cooling Tower"
   - "General Industry"
   - "Dewatering, drainage, and irrigation"

2. Verify using "Discharge Pressure (psi)" for head (NOT "Suction Pressure (psi)")

## Advanced Usage

### Train ML Model for Failure Prediction

```python
monitor.analyze(
    train_model=True,
    failure_date="2024-08-01"  # Date when pump failed
)

# Get prediction
prediction = monitor.predict_failure()
print(f"RUL: {prediction['remaining_useful_life_days']:.1f} days")
print(f"Probability: {prediction['failure_probability']:.1%}")
```

### Analyze Multiple Pumps

```python
pumps = ["Well 1", "Well 2", "Well 3"]

for pump in pumps:
    monitor = PumpMonitor(
        baseline_file=f"data/raw/baseline/{pump}_baseline.csv"
    )
    monitor.load_operational_data(
        f"data/raw/operational/{pump}_operational.csv"
    )
    monitor.analyze()
    monitor.generate_report(f"outputs/reports/{pump}_analysis.md")
```

## Next Steps

1. ✅ **Test with template data** - Make sure the system works
2. ✅ **Load your actual data** - Replace templates with real data
3. ✅ **Review visualizations** - Check if results make sense
4. ✅ **Adjust if needed** - Fine-tune tolerance thresholds if required
5. ✅ **Train ML model** - Once you have failure data
6. ✅ **Generate reports** - Share insights with your team

## Need Help?

- **Documentation:** Check `README.md` for detailed information
- **Examples:** See `example_analysis.py` for working code
- **Templates:** Use files in `data/raw/` as format reference
- **Development Guide:** Read `claude.md` for implementation details
- **Project Status:** See `PROJECT_SUMMARY.md` for what's implemented

## Success Checklist

- [ ] Dependencies installed
- [ ] Baseline data in correct format
- [ ] Operational data with exact column names
- [ ] Analysis runs without errors
- [ ] Visualizations generated
- [ ] Report created
- [ ] Results make sense

If all checked, you're ready to use the system! 🎉

---

**Total Time:** ~5 minutes to get started
**Total Code:** Just a few lines of Python
**Output:** Comprehensive analysis with visualizations and recommendations

*Built for ease of use - Phase 1 Complete*
