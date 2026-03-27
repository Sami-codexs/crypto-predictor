"""
Day 18: Data Drift Detection System

Monitors model input features for distribution drift over time.
Critical for production ML - 70% of organizations face drift within 6 months.

Methods:
- Population Stability Index (PSI): threshold 0.2 for significant drift
- Kolmogorov-Smirnov Test: p-value < 0.05 indicates drift
- Statistical Summary: mean, std, min, max tracking
"""

import numpy as np
import pandas as pd
import logging
from typing import Dict, List, Tuple, Optional
from scipy.stats import ks_2samp
from datetime import datetime, timedelta
import json
import os

logger = logging.getLogger(__name__)


class DriftDetector:
    """
    Production-grade drift detection for ML model monitoring.
    """
    
    def __init__(self, reference_data: pd.DataFrame = None, 
                 psi_threshold: float = 0.2,
                 ks_pvalue_threshold: float = 0.05,
                 storage_path: str = "data/drift_history.json"):
        self.reference_data = reference_data
        self.psi_threshold = psi_threshold
        self.ks_pvalue_threshold = ks_pvalue_threshold
        self.storage_path = storage_path
        self.drift_history = []
        
        os.makedirs(os.path.dirname(storage_path), exist_ok=True)
        
        if reference_data is not None:
            self.reference_stats = self._compute_stats(reference_data)
            logger.info(f"DriftDetector initialized with {len(reference_data)} reference samples")
        else:
            self.reference_stats = None
    
    def set_reference(self, df: pd.DataFrame):
        """Set or update reference data (e.g., after retraining)."""
        self.reference_data = df.copy()
        self.reference_stats = self._compute_stats(df)
        logger.info(f"Reference data updated: {len(df)} samples")
    
    def _compute_stats(self, df: pd.DataFrame) -> Dict:
        """Compute statistical summary for each numeric feature."""
        stats = {}
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        
        for col in numeric_cols:
            if col in ['timestamp', 'target']:
                continue
                
            stats[col] = {
                'mean': float(df[col].mean()),
                'std': float(df[col].std()),
                'min': float(df[col].min()),
                'max': float(df[col].max()),
                'median': float(df[col].median()),
                'q25': float(df[col].quantile(0.25)),
                'q75': float(df[col].quantile(0.75)),
            }
        
        return stats
    
    def calculate_psi(self, reference: pd.Series, current: pd.Series, 
                     bins: int = 10) -> float:
        """
        Population Stability Index (PSI).
        
        PSI < 0.1: No significant change
        PSI 0.1-0.2: Moderate change
        PSI > 0.2: Significant drift (requires attention)
        """
        min_val = reference.min()
        max_val = reference.max()
        
        if min_val == max_val:
            return 0.0 if current.min() == current.max() == min_val else float('inf')
        
        bin_edges = np.linspace(min_val, max_val, bins + 1)
        
        ref_counts, _ = np.histogram(reference, bins=bin_edges)
        curr_counts, _ = np.histogram(current, bins=bin_edges)
        
        epsilon = 0.0001
        ref_pct = (ref_counts / len(reference)) + epsilon
        curr_pct = (curr_counts / len(current)) + epsilon
        
        psi = np.sum((curr_pct - ref_pct) * np.log(curr_pct / ref_pct))
        
        return float(psi)
    
    def calculate_ks_test(self, reference: pd.Series, current: pd.Series) -> Tuple[float, float]:
        """
        Kolmogorov-Smirnov test.
        p-value < 0.05 indicates drift
        """
        statistic, p_value = ks_2samp(reference.dropna(), current.dropna())
        return float(statistic), float(p_value)
    
    def detect_drift(self, current_data: pd.DataFrame) -> Dict:
        """Detect drift between reference and current data."""
        if self.reference_data is None:
            raise ValueError("Reference data not set. Call set_reference() first.")
        
        results = {
            'timestamp': datetime.now().isoformat(),
            'reference_samples': len(self.reference_data),
            'current_samples': len(current_data),
            'features': {},
            'drift_detected': False,
            'drifted_features': [],
        }
        
        numeric_cols = current_data.select_dtypes(include=[np.number]).columns
        
        for col in numeric_cols:
            if col in ['timestamp', 'target']:
                continue
            
            ref_series = self.reference_data[col].dropna()
            curr_series = current_data[col].dropna()
            
            if len(ref_series) == 0 or len(curr_series) == 0:
                continue
            
            psi = self.calculate_psi(ref_series, curr_series)
            ks_stat, ks_pvalue = self.calculate_ks_test(ref_series, curr_series)
            
            curr_stats = {
                'mean': float(curr_series.mean()),
                'std': float(curr_series.std()),
            }
            
            ref_stats = self.reference_stats.get(col, {})
            
            psi_drift = psi > self.psi_threshold
            ks_drift = ks_pvalue < self.ks_pvalue_threshold
            feature_drift = psi_drift or ks_drift
            
            results['features'][col] = {
                'psi': psi,
                'psi_drift': psi_drift,
                'ks_statistic': ks_stat,
                'ks_pvalue': ks_pvalue,
                'ks_drift': ks_drift,
                'drift_detected': feature_drift,
                'reference_stats': ref_stats,
                'current_stats': curr_stats,
                'mean_shift': curr_stats['mean'] - ref_stats.get('mean', 0),
            }
            
            if feature_drift:
                results['drifted_features'].append(col)
        
        results['drift_detected'] = len(results['drifted_features']) > 0
        results['drift_percentage'] = len(results['drifted_features']) / len(results['features']) * 100
        
        if results['drift_detected']:
            if results['drift_percentage'] > 50:
                results['recommendation'] = "CRITICAL: Major drift. Retrain immediately."
            elif results['drift_percentage'] > 20:
                results['recommendation'] = "WARNING: Significant drift. Schedule retraining."
            else:
                results['recommendation'] = "CAUTION: Minor drift. Monitor closely."
        else:
            results['recommendation'] = "OK: No significant drift."
        
        self.drift_history.append(results)
        
        return results
    
    def generate_report(self, results: Dict) -> str:
        """Generate human-readable drift report."""
        lines = [
            "=" * 70,
            "DRIFT DETECTION REPORT",
            "=" * 70,
            f"Timestamp: {results['timestamp']}",
            f"Drifted Features: {len(results['drifted_features'])}/{len(results['features'])}",
            f"Status: {'⚠️ DRIFT' if results['drift_detected'] else '✅ OK'}",
            "",
            f"Recommendation: {results['recommendation']}",
            "",
            "PER-FEATURE DETAILS:",
            "-" * 70
        ]
        
        for feature, metrics in results['features'].items():
            status = "⚠️" if metrics['drift_detected'] else "✅"
            lines.append(f"\n{feature} {status}")
            lines.append(f"  PSI: {metrics['psi']:.4f} (threshold: {self.psi_threshold})")
            lines.append(f"  K-S p: {metrics['ks_pvalue']:.4f} (threshold: {self.ks_pvalue_threshold})")
            lines.append(f"  Mean shift: {metrics['mean_shift']:+.4f}")
        
        lines.extend(["", "=" * 70])
        return "\n".join(lines)


class PredictionDriftMonitor:
    """Monitor drift in model predictions (output drift)."""
    
    def __init__(self, storage_path: str = "data/prediction_drift.json"):
        self.storage_path = storage_path
        self.prediction_history = []
        self.baseline_distribution = None
        
        os.makedirs(os.path.dirname(storage_path), exist_ok=True)
    
    def set_baseline(self, predictions: np.ndarray):
        """Set baseline prediction distribution."""
        self.baseline_distribution = predictions.copy()
    
    def check_prediction_drift(self, recent_predictions: np.ndarray) -> Dict:
        """Check if recent predictions differ from baseline."""
        if self.baseline_distribution is None:
            return {'error': 'Baseline not set'}
        
        stat, pvalue = ks_2samp(self.baseline_distribution, recent_predictions)
        
        baseline_mean = np.mean(self.baseline_distribution)
        recent_mean = np.mean(recent_predictions)
        mean_shift = abs(recent_mean - baseline_mean)
        
        drift_detected = pvalue < 0.05 or mean_shift > 0.1
        
        return {
            'drift_detected': drift_detected,
            'ks_statistic': float(stat),
            'ks_pvalue': float(pvalue),
            'mean_shift': float(mean_shift),
            'interpretation': (
                "Prediction distribution changed" if drift_detected 
                else "Prediction distribution stable"
            )
        }