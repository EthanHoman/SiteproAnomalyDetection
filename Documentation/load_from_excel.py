"""
Excel Data Loader Script

This script helps you load data from Excel files and convert them
to the format needed by the Pump Anomaly Detection System.

Usage:
    python load_from_excel.py
"""

import pandas as pd
from pathlib import Path
import sys


def load_excel_to_csv(
    excel_file: str,
    baseline_sheet: str = "Baseline",
    operational_sheet: str = "Operational",
    output_dir: str = "data/raw"
):
    """
    Load data from Excel file and save as CSV files.

    Args:
        excel_file: Path to Excel file
        baseline_sheet: Name of sheet with baseline data
        operational_sheet: Name of sheet with operational data
        output_dir: Directory to save CSV files
    """
    print("=" * 70)
    print("Excel to CSV Converter for Pump Anomaly Detection System")
    print("=" * 70)
    print()

    # Check if file exists
    if not Path(excel_file).exists():
        print(f"❌ ERROR: File not found: {excel_file}")
        print()
        print("Please provide the correct path to your Excel file.")
        return False

    try:
        # Load Excel file
        print(f"📂 Loading Excel file: {excel_file}")
        excel_data = pd.ExcelFile(excel_file)

        print(f"✓ Found {len(excel_data.sheet_names)} sheets: {excel_data.sheet_names}")
        print()

        # Load baseline sheet
        print(f"📊 Loading baseline data from sheet: '{baseline_sheet}'...")

        if baseline_sheet not in excel_data.sheet_names:
            print(f"❌ ERROR: Sheet '{baseline_sheet}' not found!")
            print(f"Available sheets: {excel_data.sheet_names}")
            print()
            print("Please specify the correct sheet name for baseline data.")
            return False

        baseline_df = pd.read_excel(excel_file, sheet_name=baseline_sheet)
        print(f"✓ Loaded {len(baseline_df)} baseline records")
        print(f"  Columns: {list(baseline_df.columns)}")
        print()

        # Verify baseline columns
        required_baseline_cols = [
            'Well ID', 'pump_type', 'horsepower', 'application',
            'baseline_flow_gpm', 'baseline_discharge_pressure_psi',
            'baseline_power_hp', 'baseline_efficiency_percent'
        ]

        missing_cols = [col for col in required_baseline_cols if col not in baseline_df.columns]
        if missing_cols:
            print(f"⚠️  WARNING: Missing baseline columns: {missing_cols}")
            print(f"   Current columns: {list(baseline_df.columns)}")
            print()
            print("   Required columns (EXACT names):")
            for col in required_baseline_cols:
                print(f"     - {col}")
            print()

        # Load operational sheet
        print(f"📊 Loading operational data from sheet: '{operational_sheet}'...")

        if operational_sheet not in excel_data.sheet_names:
            print(f"❌ ERROR: Sheet '{operational_sheet}' not found!")
            print(f"Available sheets: {excel_data.sheet_names}")
            print()
            print("Please specify the correct sheet name for operational data.")
            return False

        operational_df = pd.read_excel(excel_file, sheet_name=operational_sheet)
        print(f"✓ Loaded {len(operational_df)} operational records")
        print(f"  Columns: {list(operational_df.columns)}")
        print()

        # Verify operational columns
        required_operational_cols = [
            'timestamp', 'Well ID', 'Flow (gpm)', 'Discharge Pressure (psi)',
            'Suction Pressure (psi)', 'Motor Power (hp)',
            'Pump Efficiency (%)', 'Motor Speed (rpm)'
        ]

        missing_cols = [col for col in required_operational_cols if col not in operational_df.columns]
        if missing_cols:
            print(f"⚠️  WARNING: Missing operational columns: {missing_cols}")
            print(f"   Current columns: {list(operational_df.columns)}")
            print()
            print("   Required columns (EXACT names WITH SPACES):")
            for col in required_operational_cols:
                print(f"     - '{col}'")
            print()
            print("   CRITICAL: Column names must include spaces and units in parentheses!")
            print("   Example: 'Flow (gpm)' NOT 'Flow' or 'Flow_gpm'")
            print()

        # Save to CSV
        print("💾 Saving CSV files...")

        # Create output directories
        baseline_dir = Path(output_dir) / "baseline"
        operational_dir = Path(output_dir) / "operational"
        baseline_dir.mkdir(parents=True, exist_ok=True)
        operational_dir.mkdir(parents=True, exist_ok=True)

        # Get Well ID from baseline data
        well_id = baseline_df['Well ID'].iloc[0] if 'Well ID' in baseline_df.columns else "well1"
        well_id_clean = str(well_id).replace(" ", "_").lower()

        # Save baseline
        baseline_path = baseline_dir / f"{well_id_clean}_baseline.csv"
        baseline_df.to_csv(baseline_path, index=False)
        print(f"✓ Baseline saved to: {baseline_path}")

        # Save operational
        operational_path = operational_dir / f"{well_id_clean}_operational.csv"
        operational_df.to_csv(operational_path, index=False)
        print(f"✓ Operational saved to: {operational_path}")
        print()

        # Summary
        print("=" * 70)
        print("✅ SUCCESS! Your data has been converted to CSV format.")
        print("=" * 70)
        print()
        print("Next steps:")
        print(f"1. Verify the CSV files look correct:")
        print(f"   - {baseline_path}")
        print(f"   - {operational_path}")
        print()
        print("2. Run the analysis:")
        print("   python example_analysis.py")
        print()
        print("   Or use in your code:")
        print("   ```python")
        print("   from src.pump_monitor import PumpMonitor")
        print(f"   monitor = PumpMonitor(baseline_file='{baseline_path}')")
        print(f"   monitor.load_operational_data('{operational_path}')")
        print("   monitor.analyze()")
        print("   ```")
        print()

        return True

    except Exception as e:
        print(f"❌ ERROR: {str(e)}")
        print()
        import traceback
        traceback.print_exc()
        return False


def interactive_mode():
    """Run in interactive mode to guide user through conversion."""
    print("=" * 70)
    print("Excel to CSV Converter - Interactive Mode")
    print("=" * 70)
    print()
    print("This tool will help you convert your Excel file to CSV format")
    print("for use with the Pump Anomaly Detection System.")
    print()

    # Get Excel file path
    excel_file = input("Enter path to your Excel file: ").strip().strip('"').strip("'")

    if not excel_file:
        print("❌ No file path provided. Exiting.")
        return

    if not Path(excel_file).exists():
        print(f"❌ File not found: {excel_file}")
        return

    # Show available sheets
    try:
        excel_data = pd.ExcelFile(excel_file)
        print()
        print(f"Found {len(excel_data.sheet_names)} sheets in your Excel file:")
        for i, sheet in enumerate(excel_data.sheet_names, 1):
            print(f"  {i}. {sheet}")
        print()
    except Exception as e:
        print(f"❌ Error reading Excel file: {e}")
        return

    # Get baseline sheet name
    baseline_sheet = input(f"Enter baseline sheet name (default: {excel_data.sheet_names[0]}): ").strip()
    if not baseline_sheet:
        baseline_sheet = excel_data.sheet_names[0]

    # Get operational sheet name
    default_op_sheet = excel_data.sheet_names[1] if len(excel_data.sheet_names) > 1 else excel_data.sheet_names[0]
    operational_sheet = input(f"Enter operational sheet name (default: {default_op_sheet}): ").strip()
    if not operational_sheet:
        operational_sheet = default_op_sheet

    print()

    # Convert
    success = load_excel_to_csv(
        excel_file,
        baseline_sheet=baseline_sheet,
        operational_sheet=operational_sheet
    )

    if not success:
        print()
        print("Conversion failed. Please check the errors above and try again.")
        print()
        print("Common issues:")
        print("  - Column names must match EXACTLY (including spaces and parentheses)")
        print("  - Operational data must have: 'Flow (gpm)', 'Discharge Pressure (psi)', etc.")
        print("  - Timestamps should be in format: '5/1/2024 0:00'")


if __name__ == "__main__":
    if len(sys.argv) > 1:
        # Command-line mode
        excel_file = sys.argv[1]
        baseline_sheet = sys.argv[2] if len(sys.argv) > 2 else "Baseline"
        operational_sheet = sys.argv[3] if len(sys.argv) > 3 else "Operational"

        load_excel_to_csv(excel_file, baseline_sheet, operational_sheet)
    else:
        # Interactive mode
        interactive_mode()
