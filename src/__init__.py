"""
Pump Anomaly Detection System

A comprehensive system for detecting and predicting pump failures using
tolerance-based anomaly detection and machine learning.

Phase 1 Modules:
- data_processing: Load and process baseline and operational data
- tolerance_checker: Apply tolerance thresholds and classify status
- visualization: Generate plots and dashboards
- predictive_model: Train and use ML models for failure prediction
- pump_monitor: Main PumpMonitor class integrating all functionality
"""

__version__ = "1.0.0"
__author__ = "Pump Anomaly Detection Team"

from .pump_monitor import PumpMonitor

__all__ = ['PumpMonitor']
