"""
Anomaly Query Tool

Query and analyze reported anomalies from the central API.

Usage:
    python tools/query_anomalies.py --pump 1 --days 30
    python tools/query_anomalies.py --site 35482 --start 2024-11-01 --end 2024-11-30
    python tools/query_anomalies.py --pump 1 --export anomalies.csv
"""

import sys
import argparse
import json
import pandas as pd
from datetime import datetime, timedelta
from pathlib import Path
from typing import Optional

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.anomaly_client import AnomalyAPIClient


class AnomalyQueryTool:
    """
    Tool for querying and analyzing reported anomalies.
    """

    def __init__(self, base_url: str, bearer_token: str):
        """
        Initialize query tool.

        Args:
            base_url: API base URL
            bearer_token: Authentication token
        """
        self.client = AnomalyAPIClient(base_url, bearer_token)

    def query(
        self,
        site_id: Optional[int] = None,
        pump_id: Optional[int] = None,
        sensor_id: Optional[int] = None,
        start_date: Optional[str] = None,
        end_date: Optional[str] = None,
        days_back: Optional[int] = None,
        max_results: int = 1000
    ) -> pd.DataFrame:
        """
        Query anomalies and return as DataFrame.

        Args:
            site_id: Filter by site ID
            pump_id: Filter by pump ID
            sensor_id: Filter by sensor ID
            start_date: Start date (ISO-8601 or YYYY-MM-DD)
            end_date: End date (ISO-8601 or YYYY-MM-DD)
            days_back: Query last N days (alternative to start_date)
            max_results: Maximum results to fetch

        Returns:
            DataFrame with anomaly data
        """
        # Calculate date range if days_back specified
        if days_back:
            end_date = datetime.utcnow().isoformat() + "Z"
            start_date = (datetime.utcnow() - timedelta(days=days_back)).isoformat() + "Z"
        elif start_date and not start_date.endswith('Z'):
            # Convert YYYY-MM-DD to ISO-8601
            start_date = datetime.strptime(start_date, '%Y-%m-%d').isoformat() + "Z"
        if end_date and not end_date.endswith('Z'):
            end_date = datetime.strptime(end_date, '%Y-%m-%d').isoformat() + "Z"

        # Fetch all results with pagination
        all_anomalies = []
        page = 1
        page_size = 100

        while len(all_anomalies) < max_results:
            response = self.client.query_anomalies(
                site_id=site_id,
                pump_id=pump_id,
                sensor_id=sensor_id,
                start_date=start_date,
                end_date=end_date,
                page=page,
                page_size=page_size
            )

            items = response.get('items', [])
            if not items:
                break

            all_anomalies.extend(items)

            # Check if we've fetched all available
            total = response.get('total', 0)
            if len(all_anomalies) >= total:
                break

            page += 1

        print(f"✓ Fetched {len(all_anomalies)} anomalies")

        # Convert to DataFrame
        if not all_anomalies:
            return pd.DataFrame()

        df = pd.DataFrame(all_anomalies)

        # Parse timestamp
        if 'timestamp' in df.columns:
            df['timestamp'] = pd.to_datetime(df['timestamp'])

        # Parse JSON fields
        if 'additionalContext' in df.columns:
            df['additionalContext'] = df['additionalContext'].apply(
                lambda x: json.loads(x) if isinstance(x, str) else x
            )
        if 'metadata' in df.columns:
            df['metadata'] = df['metadata'].apply(
                lambda x: json.loads(x) if isinstance(x, str) else x
            )

        # Extract common fields from additionalContext
        if 'additionalContext' in df.columns:
            df['status'] = df['additionalContext'].apply(
                lambda x: x.get('status') if isinstance(x, dict) else None
            )
            df['tolerance_category'] = df['additionalContext'].apply(
                lambda x: x.get('tolerance_category') if isinstance(x, dict) else None
            )

        return df

    def summary_stats(self, df: pd.DataFrame) -> dict:
        """
        Calculate summary statistics.

        Args:
            df: DataFrame with anomaly data

        Returns:
            Dict with statistics
        """
        if df.empty:
            return {}

        stats = {
            "total_anomalies": len(df),
            "date_range": {
                "start": df['timestamp'].min().isoformat() if 'timestamp' in df.columns else None,
                "end": df['timestamp'].max().isoformat() if 'timestamp' in df.columns else None
            },
            "by_status": df['status'].value_counts().to_dict() if 'status' in df.columns else {},
            "by_pump": df['pumpId'].value_counts().to_dict() if 'pumpId' in df.columns else {},
            "by_site": df['siteId'].value_counts().to_dict() if 'siteId' in df.columns else {},
            "by_sensor": df['sensorId'].value_counts().to_dict() if 'sensorId' in df.columns else {}
        }

        return stats

    def export_csv(self, df: pd.DataFrame, output_path: str):
        """
        Export anomalies to CSV.

        Args:
            df: DataFrame with anomaly data
            output_path: Output CSV path
        """
        # Flatten nested JSON for CSV export
        export_df = df.copy()

        # Convert JSON columns to strings
        for col in ['additionalContext', 'metadata']:
            if col in export_df.columns:
                export_df[col] = export_df[col].apply(
                    lambda x: json.dumps(x) if isinstance(x, dict) else x
                )

        export_df.to_csv(output_path, index=False)
        print(f"✓ Exported to {output_path}")

    def visualize_timeline(self, df: pd.DataFrame, output_path: Optional[str] = None):
        """
        Create timeline visualization of anomalies.

        Args:
            df: DataFrame with anomaly data
            output_path: Output image path (optional)
        """
        try:
            import matplotlib.pyplot as plt
            import matplotlib.dates as mdates

            if df.empty or 'timestamp' not in df.columns:
                print("No data to visualize")
                return

            # Group by date and status
            df['date'] = df['timestamp'].dt.date
            timeline = df.groupby(['date', 'status']).size().unstack(fill_value=0)

            # Create plot
            fig, ax = plt.subplots(figsize=(14, 6))

            # Color mapping
            colors = {
                'Normal': 'green',
                'Warning': 'yellow',
                'Critical': 'orange',
                'Failure': 'red'
            }

            # Stacked bar chart
            bottom = None
            for status in ['Warning', 'Critical', 'Failure']:
                if status in timeline.columns:
                    ax.bar(
                        timeline.index,
                        timeline[status],
                        bottom=bottom,
                        label=status,
                        color=colors.get(status, 'gray'),
                        alpha=0.8
                    )
                    if bottom is None:
                        bottom = timeline[status]
                    else:
                        bottom = bottom + timeline[status]

            ax.set_xlabel('Date')
            ax.set_ylabel('Number of Anomalies')
            ax.set_title('Anomaly Timeline')
            ax.legend()
            ax.grid(True, alpha=0.3)

            # Format x-axis
            ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
            plt.xticks(rotation=45)

            plt.tight_layout()

            if output_path:
                plt.savefig(output_path, dpi=300, bbox_inches='tight')
                print(f"✓ Saved timeline to {output_path}")
            else:
                plt.show()

        except ImportError:
            print("⚠ matplotlib not available for visualization")

    def compare_pumps(self, df: pd.DataFrame):
        """
        Compare anomaly rates across multiple pumps.

        Args:
            df: DataFrame with anomaly data
        """
        if df.empty or 'pumpId' not in df.columns:
            print("No pump data to compare")
            return

        print("\n" + "="*70)
        print("PUMP COMPARISON")
        print("="*70 + "\n")

        # Group by pump
        by_pump = df.groupby('pumpId')

        for pump_id, group in by_pump:
            print(f"\nPump {pump_id}:")
            print(f"  Total anomalies: {len(group)}")

            if 'status' in group.columns:
                status_counts = group['status'].value_counts()
                for status, count in status_counts.items():
                    print(f"  {status}: {count}")

            if 'timestamp' in group.columns:
                print(f"  First anomaly: {group['timestamp'].min()}")
                print(f"  Last anomaly: {group['timestamp'].max()}")

        print("\n" + "="*70 + "\n")


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Query and analyze pump anomalies from API"
    )

    # Authentication
    parser.add_argument(
        "--api-url",
        default="https://sp-api-sink.azurewebsites.net/api/v1",
        help="API base URL"
    )
    parser.add_argument(
        "--token",
        help="Bearer token (or set ANOMALY_API_TOKEN env var)"
    )

    # Filters
    parser.add_argument("--site", type=int, help="Site ID")
    parser.add_argument("--pump", type=int, help="Pump ID")
    parser.add_argument("--sensor", type=int, help="Sensor ID")
    parser.add_argument("--start", help="Start date (YYYY-MM-DD or ISO-8601)")
    parser.add_argument("--end", help="End date (YYYY-MM-DD or ISO-8601)")
    parser.add_argument("--days", type=int, help="Query last N days")

    # Output
    parser.add_argument("--export", help="Export to CSV file")
    parser.add_argument("--visualize", help="Create timeline visualization (PNG)")
    parser.add_argument("--compare-pumps", action="store_true", help="Compare anomaly rates across pumps")
    parser.add_argument("--max-results", type=int, default=1000, help="Maximum results to fetch")

    args = parser.parse_args()

    # Get token
    import os
    token = args.token or os.environ.get('ANOMALY_API_TOKEN')
    if not token:
        print("Error: Bearer token required (--token or ANOMALY_API_TOKEN env var)")
        sys.exit(1)

    # Create query tool
    tool = AnomalyQueryTool(args.api_url, token)

    # Query anomalies
    print(f"\nQuerying anomalies from {args.api_url}...")
    df = tool.query(
        site_id=args.site,
        pump_id=args.pump,
        sensor_id=args.sensor,
        start_date=args.start,
        end_date=args.end,
        days_back=args.days,
        max_results=args.max_results
    )

    if df.empty:
        print("\nNo anomalies found matching criteria")
        return

    # Print summary
    print("\n" + "="*70)
    print("SUMMARY STATISTICS")
    print("="*70 + "\n")

    stats = tool.summary_stats(df)
    print(f"Total anomalies: {stats['total_anomalies']}")
    print(f"Date range: {stats['date_range']['start']} to {stats['date_range']['end']}")

    if stats.get('by_status'):
        print("\nBy Status:")
        for status, count in stats['by_status'].items():
            print(f"  {status}: {count}")

    if stats.get('by_pump'):
        print("\nBy Pump:")
        for pump_id, count in stats['by_pump'].items():
            print(f"  Pump {pump_id}: {count}")

    print("\n" + "="*70 + "\n")

    # Export if requested
    if args.export:
        tool.export_csv(df, args.export)

    # Visualize if requested
    if args.visualize:
        tool.visualize_timeline(df, args.visualize)

    # Compare pumps if requested
    if args.compare_pumps:
        tool.compare_pumps(df)


if __name__ == "__main__":
    main()
