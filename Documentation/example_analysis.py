"""
Example Analysis Script

This script demonstrates how to use the Pump Anomaly Detection System
to analyze Well 1 operational data.

Usage:
    python example_analysis.py
"""

from src.pump_monitor import PumpMonitor
import logging

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

logger = logging.getLogger(__name__)


def main():
    """Run example pump analysis."""

    print("=" * 70)
    print("Pump Anomaly Detection System - Example Analysis")
    print("=" * 70)
    print()

    # Step 1: Initialize PumpMonitor with baseline data
    print("Step 1: Loading baseline data...")
    monitor = PumpMonitor(
        baseline_file="data/raw/baseline/baseline_template.csv"
    )
    print(f"✓ Baseline loaded for {monitor.baseline['well_id']}")
    print(f"  Pump: {monitor.baseline['pump_type']}")
    print(f"  Application: {monitor.baseline['application']}")
    print(f"  Tolerance Category: {monitor.tolerance_category}")
    print()

    # Step 2: Load operational data
    print("Step 2: Loading operational data...")
    monitor.load_operational_data(
        "data/raw/operational/operational_template.csv"
    )
    print(f"✓ Loaded {len(monitor.operational_data)} operational records")
    print()

    # Step 3: Run analysis
    print("Step 3: Running analysis pipeline...")
    print("  - Calculating deviations from baseline")
    print("  - Checking tolerance thresholds")
    print("  - Classifying pump status")
    print("  - Identifying first exceedances")
    print("  - Generating visualizations")
    print()

    monitor.analyze(
        train_model=False,  # Set to True if you have failure date
        failure_date=None,  # e.g., "2024-08-01" if pump failed
        save_processed_data=True
    )

    print("✓ Analysis complete!")
    print()

    # Step 4: Get current status
    print("Step 4: Checking pump status...")
    status = monitor.get_current_status()
    print(f"✓ Current Status: {status}")
    print()

    # Step 5: View first exceedances
    print("Step 5: First tolerance exceedances...")
    if monitor.first_exceedances:
        any_exceeded = False
        for param, timestamp in monitor.first_exceedances.items():
            if timestamp:
                print(f"  {param.capitalize()}: {timestamp}")
                any_exceeded = True

        if not any_exceeded:
            print("  No exceedances detected - pump operating within tolerances!")
    print()

    # Step 6: Generate report
    print("Step 6: Generating comprehensive report...")
    report_path = f"outputs/reports/{monitor.baseline['well_id']}_analysis.md"
    monitor.generate_report(report_path)
    print(f"✓ Report saved to {report_path}")
    print()

    # Step 7: View anomaly timeline
    print("Step 7: Anomaly timeline summary...")
    timeline = monitor.get_anomaly_timeline()
    print(f"  Total records: {len(timeline)}")
    print(f"  Date range: {timeline['timestamp'].min()} to {timeline['timestamp'].max()}")

    status_counts = timeline['status'].value_counts()
    print("  Status distribution:")
    for status, count in status_counts.items():
        pct = (count / len(timeline)) * 100
        print(f"    {status}: {count} ({pct:.1f}%)")
    print()

    # Summary
    print("=" * 70)
    print("Analysis Complete!")
    print("=" * 70)
    print()
    print("Generated outputs:")
    print(f"  - Visualizations: outputs/visualizations/{monitor.baseline['well_id']}_*.png")
    print(f"  - Report: {report_path}")
    print(f"  - Processed data: data/processed/{monitor.baseline['well_id']}_processed.csv")
    print()
    print("Next steps:")
    print("  1. Review the generated report")
    print("  2. Examine the visualizations")
    print("  3. If you have actual failure data, re-run with train_model=True")
    print("     and failure_date set to enable ML predictions")
    print()


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        logger.error(f"Error during analysis: {str(e)}", exc_info=True)
        print()
        print("ERROR: Analysis failed. Check the error message above.")
        print()
        print("Common issues:")
        print("  - Make sure data files exist in data/raw/baseline/ and data/raw/operational/")
        print("  - Check that column names match exactly (including spaces and units)")
        print("  - Verify timestamps are in correct format (M/D/YYYY H:MM)")
        print("  - Ensure Well ID matches between baseline and operational data")
