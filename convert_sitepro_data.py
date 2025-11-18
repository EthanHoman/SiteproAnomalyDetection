"""
SitePro Data Converter

Converts SitePro pump data format to Pump Anomaly Detection System format.

Your data columns:
- SitePumpID, PumpLogDate, OutputCurrent, Pressure, VibrationLevel, Frequency, Running, PumpDepth, PumpID, FailTs

System expected columns:
- timestamp, Well ID, Flow (gpm), Discharge Pressure (psi), Suction Pressure (psi), Motor Power (hp), Pump Efficiency (%), Motor Speed (rpm)

Usage:
    python convert_sitepro_data.py
"""

import pandas as pd
import numpy as np
from pathlib import Path


def convert_sitepro_to_system_format(
    input_file: str = "data/raw/baseline/results0.csv",
    output_operational: str = "data/raw/operational/pump_47366_operational.csv",
    output_baseline: str = "data/raw/baseline/pump_47366_baseline.csv"
):
    """
    Convert SitePro data format to system format.

    Maps available columns and estimates missing ones.
    """
    print("=" * 70)
    print("SitePro Data Converter")
    print("=" * 70)
    print()

    # Load data
    print(f"📂 Loading data from: {input_file}")
    df = pd.read_csv(input_file)

    print(f"✓ Loaded {len(df)} records")
    print(f"  Columns: {list(df.columns)}")
    print(f"  Date range: {df['PumpLogDate'].min()} to {df['PumpLogDate'].max()}")
    print()

    # Get pump info
    pump_id = df['SitePumpID'].iloc[0]
    failure_date = df['FailTs'].iloc[0]

    print(f"📊 Pump Information:")
    print(f"  Pump ID: {pump_id}")
    print(f"  Failure Date: {failure_date}")
    print()

    # Filter only running data (TRUE values)
    df_running = df[df['Running'] == True].copy()
    print(f"✓ Filtered to {len(df_running)} records where pump was running")
    print()

    # Check what data we have
    print("📋 Available sensor data:")
    print(f"  OutputCurrent: {df_running['OutputCurrent'].describe()['mean']:.2f} (avg)")
    print(f"  Pressure: {df_running['Pressure'].describe()['mean']:.2f} (avg)")
    print(f"  VibrationLevel: {df_running['VibrationLevel'].describe()['mean']:.2f} (avg)")
    print(f"  Frequency: {df_running['Frequency'].describe()['mean']:.2f} (avg)")
    print()

    # Map to system format
    print("🔄 Converting to system format...")
    print()

    operational_df = pd.DataFrame()

    # Direct mappings
    operational_df['timestamp'] = pd.to_datetime(df_running['PumpLogDate'])
    operational_df['Well ID'] = f"Pump {pump_id}"

    # Map available data to closest equivalent:
    # OutputCurrent (amps) -> We can estimate power from this
    # Pressure (psi) -> Discharge Pressure
    # Frequency (Hz) -> Motor Speed (convert Hz to RPM)
    # VibrationLevel -> Not used in baseline system, but keep for auxiliary analysis

    # Pressure -> Discharge Pressure
    operational_df['Discharge Pressure (psi)'] = df_running['Pressure'].values

    # Frequency (Hz) to RPM (assuming 60 Hz = 1800 RPM for typical motor)
    # RPM = (Frequency / 60) * 1800
    operational_df['Motor Speed (rpm)'] = (df_running['Frequency'] / 60) * 1800

    # OutputCurrent (amps) to Motor Power (hp)
    # Rough estimate: Power (hp) = (Current * Voltage * sqrt(3) * PowerFactor * Efficiency) / 746
    # Assuming 480V, 3-phase, 0.85 PF, 0.9 efficiency
    voltage = 480
    power_factor = 0.85
    efficiency = 0.90
    operational_df['Motor Power (hp)'] = (
        df_running['OutputCurrent'] * voltage * np.sqrt(3) * power_factor * efficiency / 746
    )

    # Estimate Flow and Efficiency from available data
    # These are rough estimates - you should provide actual values if available
    # For now, use correlations with current and pressure

    # Flow estimation (rough): Higher current usually means higher flow
    # Assume nominal flow at nominal current
    nominal_current = df_running['OutputCurrent'].quantile(0.5)  # Median as baseline
    nominal_flow = 500  # GPM (adjust based on your pump specs)
    operational_df['Flow (gpm)'] = nominal_flow * (df_running['OutputCurrent'] / nominal_current)

    # Efficiency estimation: Assume starts high and degrades
    # Use inverse of vibration as proxy for efficiency
    max_vibration = df_running['VibrationLevel'].max()
    if max_vibration > 0:
        vibration_normalized = 1 - (df_running['VibrationLevel'] / (max_vibration * 1.2))
        operational_df['Pump Efficiency (%)'] = 75 + (vibration_normalized * 15)  # 75-90% range
    else:
        operational_df['Pump Efficiency (%)'] = 85.0  # Default

    # Suction Pressure (not available, use typical value)
    operational_df['Suction Pressure (psi)'] = 25.0  # Typical value

    # Sort by timestamp
    operational_df = operational_df.sort_values('timestamp').reset_index(drop=True)

    print("✓ Conversion complete!")
    print()
    print("📊 Converted data summary:")
    print(f"  Records: {len(operational_df)}")
    print(f"  Date range: {operational_df['timestamp'].min()} to {operational_df['timestamp'].max()}")
    print()
    print("  Parameter ranges:")
    print(f"    Flow: {operational_df['Flow (gpm)'].min():.1f} - {operational_df['Flow (gpm)'].max():.1f} gpm")
    print(f"    Discharge Pressure: {operational_df['Discharge Pressure (psi)'].min():.1f} - {operational_df['Discharge Pressure (psi)'].max():.1f} psi")
    print(f"    Motor Power: {operational_df['Motor Power (hp)'].min():.1f} - {operational_df['Motor Power (hp)'].max():.1f} hp")
    print(f"    Efficiency: {operational_df['Pump Efficiency (%)'].min():.1f} - {operational_df['Pump Efficiency (%)'].max():.1f} %")
    print(f"    Motor Speed: {operational_df['Motor Speed (rpm)'].min():.1f} - {operational_df['Motor Speed (rpm)'].max():.1f} rpm")
    print()

    # Save operational data
    Path(output_operational).parent.mkdir(parents=True, exist_ok=True)
    operational_df.to_csv(output_operational, index=False)
    print(f"💾 Operational data saved to: {output_operational}")

    # Create baseline data from early "good" operation period
    # Use first week of data (when pump was healthy)
    print()
    print("📊 Creating baseline from initial operation period...")

    early_data = operational_df.head(168)  # First week (assuming hourly data)

    baseline_df = pd.DataFrame([{
        'Well ID': f"Pump {pump_id}",
        'pump_type': 'Unknown',  # You can fill this in
        'horsepower': 100,  # Estimate based on your pump - ADJUST THIS
        'application': 'Municipal Water and Wastewater',  # ADJUST THIS
        'baseline_flow_gpm': early_data['Flow (gpm)'].median(),
        'baseline_discharge_pressure_psi': early_data['Discharge Pressure (psi)'].median(),
        'baseline_power_hp': early_data['Motor Power (hp)'].median(),
        'baseline_efficiency_percent': early_data['Pump Efficiency (%)'].median()
    }])

    print("✓ Baseline created from first week of operation:")
    print(f"  Flow: {baseline_df['baseline_flow_gpm'].iloc[0]:.2f} gpm")
    print(f"  Discharge Pressure: {baseline_df['baseline_discharge_pressure_psi'].iloc[0]:.2f} psi")
    print(f"  Motor Power: {baseline_df['baseline_power_hp'].iloc[0]:.2f} hp")
    print(f"  Efficiency: {baseline_df['baseline_efficiency_percent'].iloc[0]:.2f} %")
    print()

    # Save baseline data
    Path(output_baseline).parent.mkdir(parents=True, exist_ok=True)
    baseline_df.to_csv(output_baseline, index=False)
    print(f"💾 Baseline data saved to: {output_baseline}")
    print()

    # Summary
    print("=" * 70)
    print("✅ SUCCESS! Data converted to system format")
    print("=" * 70)
    print()
    print("⚠️  IMPORTANT NOTES:")
    print()
    print("1. Some values were ESTIMATED because not available in source data:")
    print("   - Flow (gpm): Estimated from current correlation")
    print("   - Pump Efficiency: Estimated from vibration levels")
    print("   - Suction Pressure: Set to typical value (25 psi)")
    print()
    print("2. Baseline was created from FIRST WEEK of operation")
    print("   Assuming pump was healthy at start of monitoring period")
    print()
    print("3. You should UPDATE baseline values if you know:")
    print(f"   - Edit: {output_baseline}")
    print("   - Actual pump type and horsepower")
    print("   - Actual application category")
    print("   - Actual baseline operating parameters")
    print()
    print("4. Failure date detected: " + failure_date)
    print()
    print("Next steps:")
    print("=" * 70)
    print()
    print("1. Review and adjust baseline values if needed:")
    print(f"   nano {output_baseline}")
    print()
    print("2. Run analysis:")
    print("   python -c \"from src.pump_monitor import PumpMonitor; \\")
    print(f"   m = PumpMonitor('{output_baseline}'); \\")
    print(f"   m.load_operational_data('{output_operational}'); \\")
    print(f"   m.analyze(train_model=True, failure_date='{failure_date}'); \\")
    print(f"   m.generate_report('outputs/reports/pump_{pump_id}_analysis.md')\"")
    print()
    print("Or use the example script:")
    print(f"   # Edit example_analysis.py to use your files")
    print()

    return operational_df, baseline_df, failure_date


if __name__ == "__main__":
    operational_df, baseline_df, failure_date = convert_sitepro_to_system_format()

    print()
    print("=" * 70)
    print("Data conversion complete! Ready to analyze.")
    print("=" * 70)
