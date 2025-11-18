"""
Predictive Model Module for Pump Anomaly Detection System

This module handles:
- Feature engineering from sensor data
- Training ML models for failure prediction
- Predicting Remaining Useful Life (RUL)
- Model persistence and evaluation

Uses scikit-learn for machine learning tasks.
"""

import pandas as pd
import numpy as np
from typing import Dict, Tuple, Optional, List, Any
from pathlib import Path
import logging
import joblib
from datetime import datetime, timedelta

from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import matplotlib.pyplot as plt
import seaborn as sns

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def engineer_features(
    df: pd.DataFrame,
    window_sizes: List[int] = [24, 168]  # 1 day, 1 week in hours
) -> pd.DataFrame:
    """
    Create features for ML model from operational data.

    Features created:
    - Current deviations (flow, head, power, efficiency)
    - Rolling mean/std for each deviation
    - Trend (slope over window)
    - Rate of change (acceleration)
    - Time since first tolerance exceedance
    - Cumulative deviation (area under curve)
    - Cross-parameter correlations

    Args:
        df: DataFrame with deviation columns
        window_sizes: List of window sizes for rolling statistics (in hours)

    Returns:
        DataFrame with original columns plus engineered features
    """
    logger.info("Engineering features for ML model...")

    # Create copy to avoid modifying original
    features_df = df.copy()

    # Deviation columns
    dev_columns = [
        'flow_deviation_pct',
        'head_deviation_pct',
        'power_deviation_pct',
        'efficiency_deviation_pct'
    ]

    # Check which deviation columns exist
    available_dev_cols = [col for col in dev_columns if col in features_df.columns]

    if not available_dev_cols:
        logger.warning("No deviation columns found. Make sure deviations are calculated first.")
        return features_df

    # 1. Rolling statistics for each deviation
    for col in available_dev_cols:
        param = col.replace('_deviation_pct', '')

        for window in window_sizes:
            # Rolling mean
            features_df[f'{param}_rolling_mean_{window}h'] = (
                features_df[col].rolling(window=window, min_periods=1).mean()
            )

            # Rolling std
            features_df[f'{param}_rolling_std_{window}h'] = (
                features_df[col].rolling(window=window, min_periods=1).std().fillna(0)
            )

            # Rolling max
            features_df[f'{param}_rolling_max_{window}h'] = (
                features_df[col].rolling(window=window, min_periods=1).max()
            )

            # Rolling min
            features_df[f'{param}_rolling_min_{window}h'] = (
                features_df[col].rolling(window=window, min_periods=1).min()
            )

    # 2. Trend (slope) for each deviation
    for col in available_dev_cols:
        param = col.replace('_deviation_pct', '')

        for window in window_sizes:
            # Calculate slope as change over window
            features_df[f'{param}_slope_{window}h'] = (
                features_df[col].diff(window) / window
            ).fillna(0)

    # 3. Acceleration (rate of change of slope)
    for col in available_dev_cols:
        param = col.replace('_deviation_pct', '')

        for window in window_sizes:
            slope_col = f'{param}_slope_{window}h'
            if slope_col in features_df.columns:
                features_df[f'{param}_acceleration_{window}h'] = (
                    features_df[slope_col].diff().fillna(0)
                )

    # 4. Cumulative deviation (area under curve)
    for col in available_dev_cols:
        param = col.replace('_deviation_pct', '')
        features_df[f'{param}_cumulative_deviation'] = (
            features_df[col].cumsum()
        )

    # 5. Absolute deviation magnitude
    for col in available_dev_cols:
        param = col.replace('_deviation_pct', '')
        features_df[f'{param}_abs_deviation'] = features_df[col].abs()

    # 6. Time-based features
    if 'timestamp' in features_df.columns:
        # Hours since start
        features_df['hours_since_start'] = (
            (features_df['timestamp'] - features_df['timestamp'].min()).dt.total_seconds() / 3600
        )

        # Day of week, hour of day (cyclical features)
        features_df['day_of_week'] = features_df['timestamp'].dt.dayofweek
        features_df['hour_of_day'] = features_df['timestamp'].dt.hour

    # 7. Cross-parameter interactions
    if len(available_dev_cols) >= 2:
        # Correlation between flow and head
        if 'flow_deviation_pct' in features_df.columns and 'head_deviation_pct' in features_df.columns:
            features_df['flow_head_product'] = (
                features_df['flow_deviation_pct'] * features_df['head_deviation_pct']
            )

        # Correlation between power and efficiency
        if 'power_deviation_pct' in features_df.columns and 'efficiency_deviation_pct' in features_df.columns:
            features_df['power_efficiency_product'] = (
                features_df['power_deviation_pct'] * features_df['efficiency_deviation_pct']
            )

    logger.info(f"Feature engineering complete. Total features: {len(features_df.columns)}")

    return features_df


def create_failure_labels(
    df: pd.DataFrame,
    failure_date: str,
    mode: str = "regression"
) -> pd.Series:
    """
    Create target labels for ML model.

    Args:
        df: DataFrame with timestamps
        failure_date: When pump failed (ISO format string or datetime)
        mode: "regression" for RUL (days until failure)
              "classification" for binary (will fail soon? yes/no)

    Returns:
        Series of labels aligned with df index
    """
    if 'timestamp' not in df.columns:
        raise ValueError("DataFrame must have 'timestamp' column")

    # Convert failure date to datetime
    failure_dt = pd.to_datetime(failure_date)

    if mode == "regression":
        # Calculate days until failure (Remaining Useful Life)
        labels = (failure_dt - df['timestamp']).dt.total_seconds() / (24 * 3600)

        # Clip negative values to 0 (already failed)
        labels = labels.clip(lower=0)

        logger.info(f"Created regression labels (RUL): {labels.min():.1f} to {labels.max():.1f} days")

    elif mode == "classification":
        # Binary classification: will fail within next 7 days?
        days_until_failure = (failure_dt - df['timestamp']).dt.total_seconds() / (24 * 3600)
        labels = (days_until_failure <= 7).astype(int)

        logger.info(
            f"Created classification labels: "
            f"{labels.sum()} positive samples ({labels.mean()*100:.1f}%)"
        )

    else:
        raise ValueError(f"Unknown mode: {mode}. Use 'regression' or 'classification'")

    return labels


def train_failure_predictor(
    X: pd.DataFrame,
    y: pd.Series,
    model_type: str = "random_forest",
    test_size: float = 0.2,
    random_state: int = 42,
    save_path: Optional[str] = None
) -> Tuple[Any, StandardScaler, List[str], Dict]:
    """
    Train ML model to predict pump failures.

    Args:
        X: Feature matrix
        y: Target labels (RUL or binary classification)
        model_type: "random_forest", "gradient_boosting", or "linear"
        test_size: Fraction of data for testing
        random_state: Random seed for reproducibility
        save_path: Directory to save model (if None, doesn't save)

    Returns:
        Tuple of (trained_model, scaler, feature_names, metrics)
    """
    logger.info(f"Training {model_type} model...")

    # Remove non-numeric columns
    non_numeric_cols = X.select_dtypes(exclude=[np.number]).columns.tolist()
    if non_numeric_cols:
        logger.info(f"Removing non-numeric columns: {non_numeric_cols}")
        X = X.drop(columns=non_numeric_cols)

    # Handle missing values
    if X.isnull().any().any():
        logger.warning("Missing values found. Filling with forward fill then 0.")
        X = X.fillna(method='ffill').fillna(0)

    # Store feature names
    feature_names = X.columns.tolist()

    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, shuffle=False
    )

    logger.info(f"Train size: {len(X_train)}, Test size: {len(X_test)}")

    # Scale features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # Select model
    if model_type == "random_forest":
        model = RandomForestRegressor(
            n_estimators=100,
            max_depth=10,
            min_samples_split=5,
            random_state=random_state,
            n_jobs=-1
        )
    elif model_type == "gradient_boosting":
        model = GradientBoostingRegressor(
            n_estimators=100,
            max_depth=5,
            learning_rate=0.1,
            random_state=random_state
        )
    elif model_type == "linear":
        model = LinearRegression()
    else:
        raise ValueError(f"Unknown model type: {model_type}")

    # Train model
    model.fit(X_train_scaled, y_train)

    # Make predictions
    y_train_pred = model.predict(X_train_scaled)
    y_test_pred = model.predict(X_test_scaled)

    # Calculate metrics
    metrics = {
        'train': {
            'mae': mean_absolute_error(y_train, y_train_pred),
            'rmse': np.sqrt(mean_squared_error(y_train, y_train_pred)),
            'r2': r2_score(y_train, y_train_pred)
        },
        'test': {
            'mae': mean_absolute_error(y_test, y_test_pred),
            'rmse': np.sqrt(mean_squared_error(y_test, y_test_pred)),
            'r2': r2_score(y_test, y_test_pred)
        }
    }

    logger.info(f"Test MAE: {metrics['test']['mae']:.2f}, RMSE: {metrics['test']['rmse']:.2f}, R²: {metrics['test']['r2']:.3f}")

    # Feature importance (if available)
    if hasattr(model, 'feature_importances_'):
        importance_df = pd.DataFrame({
            'feature': feature_names,
            'importance': model.feature_importances_
        }).sort_values('importance', ascending=False)

        logger.info(f"\nTop 10 Most Important Features:")
        for idx, row in importance_df.head(10).iterrows():
            logger.info(f"  {row['feature']}: {row['importance']:.4f}")

        metrics['feature_importance'] = importance_df

    # Save model if path provided
    if save_path:
        save_dir = Path(save_path)
        save_dir.mkdir(parents=True, exist_ok=True)

        # Save model
        model_file = save_dir / f"{model_type}_model.pkl"
        joblib.dump(model, model_file)
        logger.info(f"Model saved to {model_file}")

        # Save scaler
        scaler_file = save_dir / "scaler.pkl"
        joblib.dump(scaler, scaler_file)
        logger.info(f"Scaler saved to {scaler_file}")

        # Save feature names
        features_file = save_dir / "feature_names.txt"
        with open(features_file, 'w') as f:
            f.write('\n'.join(feature_names))
        logger.info(f"Feature names saved to {features_file}")

        # Save metrics
        metrics_file = save_dir / "metrics.txt"
        with open(metrics_file, 'w') as f:
            f.write(f"Model Type: {model_type}\n")
            f.write(f"Training Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
            f.write(f"Train Metrics:\n")
            f.write(f"  MAE: {metrics['train']['mae']:.4f}\n")
            f.write(f"  RMSE: {metrics['train']['rmse']:.4f}\n")
            f.write(f"  R²: {metrics['train']['r2']:.4f}\n\n")
            f.write(f"Test Metrics:\n")
            f.write(f"  MAE: {metrics['test']['mae']:.4f}\n")
            f.write(f"  RMSE: {metrics['test']['rmse']:.4f}\n")
            f.write(f"  R²: {metrics['test']['r2']:.4f}\n")
        logger.info(f"Metrics saved to {metrics_file}")

    return model, scaler, feature_names, metrics


def predict_failure(
    model: Any,
    scaler: StandardScaler,
    current_data: pd.DataFrame,
    feature_names: List[str],
    confidence_level: float = 0.95
) -> Dict:
    """
    Make prediction for current pump state.

    Args:
        model: Trained ML model
        scaler: Fitted scaler
        current_data: DataFrame with current features (single row or multiple)
        feature_names: List of feature names used in training
        confidence_level: Confidence level for interval (default 95%)

    Returns:
        Dictionary with:
        - remaining_useful_life_days: float
        - failure_probability: float (if available)
        - confidence_interval: tuple (lower, upper)
        - contributing_factors: list of top features
    """
    # Select only required features
    X = current_data[feature_names].copy()

    # Handle missing values
    X = X.fillna(method='ffill').fillna(0)

    # Scale features
    X_scaled = scaler.transform(X)

    # Make prediction
    prediction = model.predict(X_scaled)

    # If multiple rows, take the last (most recent)
    if len(prediction) > 1:
        rul = prediction[-1]
        current_features = X.iloc[-1]
    else:
        rul = prediction[0]
        current_features = X.iloc[0]

    # Estimate confidence interval (simplified)
    # For ensemble models, use std of tree predictions
    if hasattr(model, 'estimators_'):
        # Get predictions from each tree
        tree_predictions = np.array([
            tree.predict(X_scaled[-1:]) for tree in model.estimators_
        ]).flatten()

        std = tree_predictions.std()
        margin = 1.96 * std  # 95% confidence interval

        lower = max(0, rul - margin)
        upper = rul + margin
    else:
        # Simple estimate for non-ensemble models
        lower = max(0, rul * 0.8)
        upper = rul * 1.2

    # Get contributing factors (top features by value)
    if hasattr(model, 'feature_importances_'):
        # Weight feature values by importance
        importance = model.feature_importances_
        weighted_values = np.abs(current_features.values) * importance

        top_indices = np.argsort(weighted_values)[-5:][::-1]
        contributing_factors = [feature_names[i] for i in top_indices]
    else:
        # Just use top absolute values
        top_indices = np.argsort(np.abs(current_features.values))[-5:][::-1]
        contributing_factors = [feature_names[i] for i in top_indices]

    # Failure probability (inverse of RUL, normalized)
    failure_probability = max(0, min(1, 1 - (rul / 30)))  # Assuming 30 day max RUL

    result = {
        'remaining_useful_life_days': float(rul),
        'failure_probability': float(failure_probability),
        'confidence_interval': (float(lower), float(upper)),
        'contributing_factors': contributing_factors
    }

    logger.info(f"Prediction: RUL = {rul:.1f} days (95% CI: {lower:.1f} - {upper:.1f})")

    return result


def load_model(model_path: str) -> Tuple[Any, StandardScaler, List[str]]:
    """
    Load trained model, scaler, and feature names.

    Args:
        model_path: Directory containing saved model files

    Returns:
        Tuple of (model, scaler, feature_names)
    """
    model_dir = Path(model_path)

    # Load model (try different model types)
    model_files = list(model_dir.glob("*_model.pkl"))
    if not model_files:
        raise FileNotFoundError(f"No model file found in {model_path}")

    model = joblib.load(model_files[0])
    logger.info(f"Loaded model from {model_files[0]}")

    # Load scaler
    scaler_file = model_dir / "scaler.pkl"
    scaler = joblib.load(scaler_file)
    logger.info(f"Loaded scaler from {scaler_file}")

    # Load feature names
    features_file = model_dir / "feature_names.txt"
    with open(features_file, 'r') as f:
        feature_names = [line.strip() for line in f.readlines()]
    logger.info(f"Loaded {len(feature_names)} feature names")

    return model, scaler, feature_names


if __name__ == "__main__":
    print("Predictive Model Module")
    print("=" * 50)
    print("\nFeatures:")
    print("- Feature engineering from sensor data")
    print("- Train Random Forest, Gradient Boosting, or Linear models")
    print("- Predict Remaining Useful Life (RUL)")
    print("- Model persistence and evaluation")
    print("\nEngineered Features Include:")
    print("- Rolling statistics (mean, std, max, min)")
    print("- Trends and slopes")
    print("- Acceleration (rate of change)")
    print("- Cumulative deviations")
    print("- Cross-parameter interactions")
