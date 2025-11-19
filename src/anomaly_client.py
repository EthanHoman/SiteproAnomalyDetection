"""
Anomaly API Client for Edge Devices

Handles communication with the central API for reporting and querying anomalies.
"""

import requests
import time
import logging
from typing import Dict, Optional, List, Any
from datetime import datetime

logger = logging.getLogger(__name__)


class AnomalyAPIClient:
    """
    Client for submitting and querying pump anomalies via the SitePro API.

    API Endpoint: https://sp-api-sink.azurewebsites.net/api/v1/edge/anomalies
    Authentication: Bearer token
    """

    def __init__(
        self,
        base_url: str,
        bearer_token: str,
        retry_attempts: int = 3,
        retry_delay: int = 5
    ):
        """
        Initialize the API client.

        Args:
            base_url: API base URL (e.g., "https://sp-api-sink.azurewebsites.net/api/v1")
            bearer_token: Bearer token for authentication
            retry_attempts: Number of retry attempts on failure (default: 3)
            retry_delay: Base delay between retries in seconds (default: 5)
        """
        self.base_url = base_url.rstrip('/')
        self.bearer_token = bearer_token
        self.retry_attempts = retry_attempts
        self.retry_delay = retry_delay

        # Create session with authentication headers
        self.session = requests.Session()
        self.session.headers.update({
            "Authorization": f"Bearer {bearer_token}",
            "Content-Type": "application/json"
        })

    def submit_anomaly(self, anomaly_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Submit an anomaly to the API.

        Args:
            anomaly_data: Dict containing anomaly information
                Required fields:
                - sourceType: "log" (for sensor-based anomalies)
                - description: Human-readable description

                Optional fields:
                - siteId: Site identifier
                - pumpId: Pump identifier
                - sensorId: Sensor identifier
                - timestamp: ISO-8601 timestamp
                - logValue: Sensor reading value
                - additionalContext: Object with extra metadata
                - metadata: Object with model information

        Returns:
            API response with anomaly ID and submitted data

        Raises:
            requests.exceptions.RequestException: If all retries fail
            ValueError: If required fields are missing
        """
        # Validate required fields
        if "sourceType" not in anomaly_data:
            raise ValueError("Missing required field: sourceType")
        if "description" not in anomaly_data:
            raise ValueError("Missing required field: description")

        url = f"{self.base_url}/edge/anomalies"

        for attempt in range(self.retry_attempts):
            try:
                logger.info(f"Submitting anomaly (attempt {attempt + 1}/{self.retry_attempts})")
                response = self.session.post(url, json=anomaly_data, timeout=30)
                response.raise_for_status()

                result = response.json()
                logger.info(f"✓ Anomaly submitted successfully: ID {result.get('id')}")
                return result

            except requests.exceptions.RequestException as e:
                logger.warning(
                    f"Attempt {attempt + 1}/{self.retry_attempts} failed: {e}"
                )

                if attempt < self.retry_attempts - 1:
                    # Exponential backoff
                    delay = self.retry_delay * (2 ** attempt)
                    logger.info(f"Retrying in {delay} seconds...")
                    time.sleep(delay)
                else:
                    logger.error("All retry attempts failed for anomaly submission")
                    raise

    def query_anomalies(
        self,
        site_id: Optional[int] = None,
        pump_id: Optional[int] = None,
        sensor_id: Optional[int] = None,
        camera_id: Optional[int] = None,
        start_date: Optional[str] = None,
        end_date: Optional[str] = None,
        page: int = 1,
        page_size: int = 25,
        sort_direction: str = "desc",
        skip: Optional[int] = None,
        take: Optional[int] = None
    ) -> Dict[str, Any]:
        """
        Query anomalies with filtering and pagination.

        Args:
            site_id: Filter by site ID
            pump_id: Filter by pump ID
            sensor_id: Filter by sensor ID
            camera_id: Filter by camera ID (not used for log anomalies)
            start_date: Start date (ISO-8601 format)
            end_date: End date (ISO-8601 format)
            page: Page number (alternative to skip)
            page_size: Items per page (1-100, default 25)
            sort_direction: "asc" or "desc" (default "desc")
            skip: Offset (alternative to page)
            take: Limit (alternative to page_size)

        Returns:
            Dict with:
            - items: List of anomaly objects
            - total: Total count
            - skip: Offset used
            - take: Limit used

        Raises:
            requests.exceptions.RequestException: If request fails
        """
        url = f"{self.base_url}/edge/anomalies"

        # Build query parameters
        params = {
            "sortDirection": sort_direction
        }

        # Use skip/take if provided, otherwise use page/pageSize
        if skip is not None:
            params["skip"] = skip
        else:
            params["page"] = page

        if take is not None:
            params["take"] = take
        else:
            params["pageSize"] = page_size

        # Add filters
        if site_id is not None:
            params["siteId"] = site_id
        if pump_id is not None:
            params["pumpId"] = pump_id
        if sensor_id is not None:
            params["sensorId"] = sensor_id
        if camera_id is not None:
            params["cameraId"] = camera_id
        if start_date:
            params["startDate"] = start_date
        if end_date:
            params["endDate"] = end_date

        try:
            logger.info(f"Querying anomalies with filters: {params}")
            response = self.session.get(url, params=params, timeout=30)
            response.raise_for_status()

            result = response.json()
            logger.info(f"✓ Retrieved {len(result.get('items', []))} anomalies (total: {result.get('total', 0)})")
            return result

        except requests.exceptions.RequestException as e:
            logger.error(f"Failed to query anomalies: {e}")
            raise

    def validate_payload(self, anomaly_data: Dict[str, Any]) -> bool:
        """
        Validate anomaly payload before submission.

        Args:
            anomaly_data: Anomaly data dict

        Returns:
            True if valid

        Raises:
            ValueError: If validation fails
        """
        # Check required fields
        if "sourceType" not in anomaly_data:
            raise ValueError("Missing required field: sourceType")
        if "description" not in anomaly_data:
            raise ValueError("Missing required field: description")

        # Validate sourceType
        valid_source_types = ["log", "vision"]
        if anomaly_data["sourceType"] not in valid_source_types:
            raise ValueError(f"Invalid sourceType. Must be one of: {valid_source_types}")

        # Validate numeric IDs if present
        for field in ["siteId", "pumpId", "sensorId", "cameraId", "logId"]:
            if field in anomaly_data:
                if not isinstance(anomaly_data[field], int):
                    raise ValueError(f"{field} must be an integer")

        # Validate timestamp if present
        if "timestamp" in anomaly_data:
            try:
                datetime.fromisoformat(anomaly_data["timestamp"].replace('Z', '+00:00'))
            except (ValueError, AttributeError):
                raise ValueError("timestamp must be in ISO-8601 format")

        # Validate logValue if present
        if "logValue" in anomaly_data:
            if not isinstance(anomaly_data["logValue"], (int, float)):
                raise ValueError("logValue must be numeric")

        return True


# Convenience function for quick anomaly submission
def submit_pump_anomaly(
    base_url: str,
    bearer_token: str,
    site_id: int,
    pump_id: int,
    description: str,
    status: str,
    deviations: Dict[str, float],
    baseline_values: Dict[str, float],
    current_values: Dict[str, float],
    tolerance_category: str,
    sensor_id: Optional[int] = None,
    log_value: Optional[float] = None,
    timestamp: Optional[str] = None,
    model_metadata: Optional[Dict[str, Any]] = None
) -> Dict[str, Any]:
    """
    Convenience function to submit a pump anomaly with standard formatting.

    Args:
        base_url: API base URL
        bearer_token: Authentication token
        site_id: Site identifier
        pump_id: Pump identifier
        description: Human-readable description
        status: Pump status (Normal/Warning/Critical/Failure)
        deviations: Dict of parameter deviations (%)
        baseline_values: Dict of baseline parameter values
        current_values: Dict of current parameter values
        tolerance_category: Tolerance category (e.g., "1U")
        sensor_id: Sensor ID (optional)
        log_value: Sensor reading value (optional)
        timestamp: ISO-8601 timestamp (optional, defaults to now)
        model_metadata: ML model metadata (optional)

    Returns:
        API response
    """
    client = AnomalyAPIClient(base_url, bearer_token)

    payload = {
        "sourceType": "log",
        "description": description,
        "siteId": site_id,
        "pumpId": pump_id,
        "timestamp": timestamp or datetime.utcnow().isoformat() + "Z",
        "additionalContext": {
            "status": status,
            "tolerance_category": tolerance_category,
            "all_deviations": deviations,
            "baseline_values": baseline_values,
            "current_values": current_values
        }
    }

    if sensor_id is not None:
        payload["sensorId"] = sensor_id
    if log_value is not None:
        payload["logValue"] = log_value
    if model_metadata:
        payload["metadata"] = model_metadata

    return client.submit_anomaly(payload)
