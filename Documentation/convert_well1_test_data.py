"""
Well 1 Test Data Converter

Converts pump test data to system format.

Your data has:
- Flow in gal/min
- Head in feet
- Pressure in psi
- Current measurements
- Power measurements

This appears to be a PUMP CURVE TEST (baseline test).
"""

import pandas as pd
import numpy as np
from pathlib import Path


def convert_well1_test_data(
    input_file: str = "data/raw/operational/well1.csv",
    output_baseline: str = "data/raw/baseline/well1_baseline.csv"
):
    """
    Convert Well 1 test data to baseline format.

    This is pump test/curve data, so we'll use it to establish baseline.
    """
    print("=" * 70)
    print("Well 1 Test Data Converter")
    print("=" * 70)
    print()

    # Load data
    print(f"📂 Loading test data from: {input_file}")
    df = pd.read_csv(input_file)

    print(f"✓ Loaded {len(df)} test records")
    print(f"  Columns: {list(df.columns)}")
    print()

    # Clean column names (remove extra quotes)
    df.columns = df.columns.str.strip().str.replace('""', '', regex=False).str.rstrip('"')

    print(f"  Cleaned columns: {list(df.columns)}")
    print()

    # Rename columns for easier access
    column_mapping = {
        '4 Line Current Flow (gal/min)': 'Flow_gpm',
        '4 Line FT Head (ft)': 'Head_ft',
        '4 Line Pressure (psi)': 'Pressure_psi',
        'Power Monitor A Current Average (A)': 'Current_A'
    }
    df = df.rename(columns=column_mapping)

    print("📊 Data Summary:")
    print(f"  Flow range: {df['Flow_gpm'].min():.1f} - {df['Flow_gpm'].max():.1f} gpm")
    print(f"  Head range: {df['Head_ft'].min():.1f} - {df['Head_ft'].max():.1f} ft")
    print(f"  Pressure range: {df['Pressure_psi'].min():.1f} - {df['Pressure_psi'].max():.1f} psi")
    print(f"  Current range: {df['Current_A'].min():.1f} - {df['Current_A'].max():.1f} A")
    print()

    # This is a pump STARTUP TEST - flow decreases as head/pressure increases
    # We want to find the OPTIMAL OPERATING POINT (BEP - Best Efficiency Point)

    # For centrifugal pumps, BEP is typically around 60-80% of shutoff head
    # Look for stable operating point with good flow and reasonable head

    # Filter to stable operation (flow > 100 gpm, head is reasonable)
    stable_operation = df[
        (df['Flow_gpm'] > 200) &
        (df['Head_ft'] > 50) &
        (df['Head_ft'] < 200)
    ].copy()

    print(f"📈 Found {len(stable_operation)} records in stable operating range")
    print()

    if len(stable_operation) == 0:
        print("⚠️  WARNING: No stable operating point found!")
        print("   Using overall median values instead.")
        stable_operation = df[df['Flow_gpm'] > 0].copy()

    # Calculate baseline from stable operation
    baseline_flow_gpm = stable_operation['Flow_gpm'].median()
    baseline_head_ft = stable_operation['Head_ft'].median()
    baseline_pressure_psi = stable_operation['Pressure_psi'].median()
    baseline_current_a = stable_operation['Current_A'].median()

    # Estimate motor power from current
    # Assuming 480V 3-phase, power factor 0.85, efficiency 0.90
    voltage = 480
    power_factor = 0.85
    motor_efficiency = 0.90
    baseline_power_hp = (baseline_current_a * voltage * np.sqrt(3) * power_factor * motor_efficiency) / 746

    # Estimate pump efficiency
    # Pump Efficiency = (Flow * Head * SG) / (3960 * Power)
    # Where SG = specific gravity (1.0 for water)
    specific_gravity = 1.0
    hydraulic_power_hp = (baseline_flow_gpm * baseline_head_ft * specific_gravity) / 3960
    baseline_efficiency = (hydraulic_power_hp / baseline_power_hp) * 100 if baseline_power_hp > 0 else 85.0

    # Clamp efficiency to reasonable range
    baseline_efficiency = max(60, min(95, baseline_efficiency))

    print("✓ Calculated baseline operating parameters:")
    print(f"  Flow: {baseline_flow_gpm:.2f} gpm")
    print(f"  Head: {baseline_head_ft:.2f} ft")
    print(f"  Discharge Pressure: {baseline_pressure_psi:.2f} psi")
    print(f"  Motor Power: {baseline_power_hp:.2f} hp")
    print(f"  Pump Efficiency: {baseline_efficiency:.2f} %")
    print()

    # Create baseline DataFrame
    baseline_df = pd.DataFrame([{
        'Well ID': 'Well 1',
        'pump_type': 'Unknown - 4 Line Pump',  # Update with actual model
        'horsepower': int(np.ceil(baseline_power_hp / 0.85)),  # Estimate HP rating (power / typical loading)
        'application': 'Municipal Water and Wastewater',
        'baseline_flow_gpm': baseline_flow_gpm,
        'baseline_discharge_pressure_psi': baseline_pressure_psi,
        'baseline_power_hp': baseline_power_hp,
        'baseline_efficiency_percent': baseline_efficiency
    }])

    print("📋 Baseline record created:")
    print(baseline_df.to_string(index=False))
    print()

    # Save baseline
    Path(output_baseline).parent.mkdir(parents=True, exist_ok=True)
    baseline_df.to_csv(output_baseline, index=False)
    print(f"💾 Baseline saved to: {output_baseline}")
    print()

    # Summary
    print("=" * 70)
    print("✅ SUCCESS! Baseline created from pump test data")
    print("=" * 70)
    print()
    print("⚠️  IMPORTANT NOTES:")
    print()
    print("1. This data appears to be a PUMP CURVE TEST (startup/ramp-up)")
    print("   - Not continuous operational data")
    print("   - Shows pump performance across operating range")
    print()
    print("2. Baseline was created from STABLE OPERATING REGION:")
    print(f"   - Flow > 200 gpm")
    print(f"   - Head between 50-200 ft")
    print()
    print("3. Some values were ESTIMATED:")
    print("   - Motor Power: Calculated from current (assumes 480V, 3-phase)")
    print("   - Pump Efficiency: Calculated from hydraulic formula")
    print("   - HP rating: Estimated from power consumption")
    print()
    print("4. You should UPDATE if you know:")
    print(f"   - Edit: {output_baseline}")
    print("   - Actual pump model and HP rating")
    print("   - Actual motor voltage and configuration")
    print()
    print("5. For OPERATIONAL MONITORING, you need:")
    print("   - Continuous operational data (hours/days/weeks)")
    print("   - Same sensors logging over time")
    print("   - Data showing degradation towards failure")
    print()
    print("Next steps:")
    print("=" * 70)
    print()
    print("This test data establishes your BASELINE.")
    print()
    print("To detect anomalies, you need OPERATIONAL data:")
    print("  - Same measurements taken over time")
    print("  - Logged continuously during normal operation")
    print("  - Showing any degradation or changes")
    print()
    print("Do you have operational logs from Well 1 over time?")
    print("  - Daily/hourly sensor readings")
    print("  - Data leading up to a failure")
    print("  - Or current real-time monitoring data")
    print()

    return baseline_df


if __name__ == "__main__":
    baseline_df = convert_well1_test_data()

    print()
    print("=" * 70)
    print("Baseline created! Next: Get operational monitoring data")
    print("=" * 70)
