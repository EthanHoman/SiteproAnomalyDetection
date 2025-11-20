"""
Create demo operational data for Well 1 from test data.

Since well1.csv is just a pump test, we'll create synthetic operational
data to demonstrate the system.
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta

# Load the test data to get baseline parameters
test_data = pd.read_csv('data/raw/operational/well1.csv')

# Clean column names
test_data.columns = test_data.columns.str.strip().str.replace('""', '', regex=False).str.rstrip('"')

# Get stable operating point from test
stable = test_data[
    (test_data['4 Line Current Flow (gal/min)'] > 200) &
    (test_data['4 Line FT Head (ft)'] > 50) &
    (test_data['4 Line FT Head (ft)'] < 200)
]

baseline_flow = stable['4 Line Current Flow (gal/min)'].median()
baseline_pressure = stable['4 Line Pressure (psi)'].median()
baseline_current = stable['Power Monitor A Current Average (A)'].median()

print(f"Baseline from test:")
print(f"  Flow: {baseline_flow:.1f} gpm")
print(f"  Pressure: {baseline_pressure:.1f} psi")
print(f"  Current: {baseline_current:.1f} A")
print()

# Create 60 days of operational data (hourly readings)
# Simulate gradual degradation leading to failure

start_date = datetime(2024, 6, 1, 0, 0, 0)
hours = 60 * 24  # 60 days
timestamps = [start_date + timedelta(hours=i) for i in range(hours)]

# Create operational data with gradual degradation
operational_data = []

for i, ts in enumerate(timestamps):
    # Progress through time (0 to 1)
    progress = i / hours

    # Add realistic noise
    noise_flow = np.random.normal(0, 5)
    noise_pressure = np.random.normal(0, 2)
    noise_current = np.random.normal(0, 0.5)

    # Simulate gradual degradation in last 20 days
    if progress > 0.67:  # Last 20 days
        degradation_factor = (progress - 0.67) / 0.33  # 0 to 1
        # Flow decreases, pressure increases, current increases (pump working harder)
        flow_degradation = -50 * degradation_factor
        pressure_degradation = 15 * degradation_factor
        current_degradation = 3 * degradation_factor
    else:
        flow_degradation = 0
        pressure_degradation = 0
        current_degradation = 0

    # Calculate values
    flow = baseline_flow + flow_degradation + noise_flow
    pressure = baseline_pressure + pressure_degradation + noise_pressure
    current = baseline_current + current_degradation + noise_current

    # Estimate other parameters
    # Motor power from current (480V, 3-phase)
    power = (current * 480 * np.sqrt(3) * 0.85 * 0.90) / 746

    # Motor speed (assume constant 1800 RPM for 60 Hz)
    motor_speed = 1800

    # Suction pressure (typical value)
    suction_pressure = 25.0

    # Pump efficiency (decreases with degradation)
    head_ft = pressure * 2.31  # Convert PSI to feet
    hydraulic_power = (flow * head_ft * 1.0) / 3960
    efficiency = (hydraulic_power / power) * 100 if power > 0 else 85
    efficiency = max(50, min(95, efficiency))

    operational_data.append({
        'timestamp': ts.strftime('%m/%d/%Y %H:%M'),  # Match expected format
        'Well ID': 'Well 1',
        'Flow (gpm)': round(flow, 2),
        'Discharge Pressure (psi)': round(pressure, 2),
        'Suction Pressure (psi)': suction_pressure,
        'Motor Power (hp)': round(power, 2),
        'Pump Efficiency (%)': round(efficiency, 2),
        'Motor Speed (rpm)': motor_speed
    })

# Create DataFrame
df = pd.DataFrame(operational_data)

# Save
output_file = 'data/raw/operational/well1_operational.csv'
df.to_csv(output_file, index=False)

print(f"Created {len(df)} operational records")
print(f"Saved to: {output_file}")
print()
print("Data summary:")
print(f"  Period: {df['timestamp'].iloc[0]} to {df['timestamp'].iloc[-1]}")
print(f"  Flow: {df['Flow (gpm)'].min():.1f} - {df['Flow (gpm)'].max():.1f} gpm")
print(f"  Pressure: {df['Discharge Pressure (psi)'].min():.1f} - {df['Discharge Pressure (psi)'].max():.1f} psi")
print(f"  Efficiency: {df['Pump Efficiency (%)'].min():.1f} - {df['Pump Efficiency (%)'].max():.1f} %")
print()
print("✅ Demo operational data created!")
print()
print("This demonstrates:")
print("  - Gradual degradation starting at day 40")
print("  - Flow decreasing, pressure increasing")
print("  - Efficiency declining")
print("  - Leading to simulated failure at day 60")
