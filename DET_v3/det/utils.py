"""DET v3 Utilities - Thresholds, validation, and helper functions."""

import pandas as pd
import numpy as np
from typing import Dict, List, Any, Optional

# Default metric thresholds
THRESHOLDS = {
    'URS': {'green': 0.20, 'yellow': 0.10},
    'AOI': {'green': 0.15, 'yellow': 0.25},
    'DMI': {'green': 0.10, 'yellow': 0.20},
    'k_anonymity': {'green': 5, 'yellow': 3},
    'HRS': {'green': 0.25, 'yellow': 0.50},
    'FOI': {'green': 0.85, 'yellow': 0.70},
    'FPC': {'green': 0.80, 'yellow': 0.60},
    'CPA': {'green': 0.50, 'yellow': 0.70},
    'SPA': {'green': 0.70, 'yellow': 0.80},
    'DAI': {'green': 0.85, 'yellow': 0.70}
}

ID_PATTERNS = ['id', 'patient', 'encounter', 'record', 'index', 'key', 'nbr', 'number']


def assign_flag(score: float, metric: str, higher_is_better: bool = True) -> str:
    """Assign GREEN/YELLOW/RED flag based on metric thresholds."""
    t = THRESHOLDS.get(metric, {'green': 0.8, 'yellow': 0.5})
    
    if higher_is_better:
        if score >= t['green']:
            return 'GREEN'
        return 'YELLOW' if score >= t['yellow'] else 'RED'
    else:
        if score <= t['green']:
            return 'GREEN'
        return 'YELLOW' if score <= t['yellow'] else 'RED'


def format_result(score: float, flag: str, interpretation: str, 
                  details: Optional[Dict] = None) -> Dict[str, Any]:
    """Standard format for metric results."""
    return {
        'score': score,
        'flag': flag,
        'interpretation': interpretation,
        'details': details or {}
    }


def is_id_column(col_name: str) -> bool:
    """Check if column name suggests it's an ID field."""
    return any(p in col_name.lower() for p in ID_PATTERNS)


def filter_features(df: pd.DataFrame, exclude: List[str]) -> List[str]:
    """Get non-ID, non-excluded columns."""
    return [c for c in df.columns if c not in exclude and not is_id_column(c)]


def encode_column(series: pd.Series) -> np.ndarray:
    """Encode categorical series to numeric."""
    from sklearn.preprocessing import LabelEncoder
    if series.dtype == 'object' or series.dtype.name == 'category':
        le = LabelEncoder()
        return le.fit_transform(series.fillna('MISSING').astype(str))
    return series.fillna(series.median() if series.notna().any() else 0).values


def small_sample_discount(n_rows: int, base_score: float) -> float:
    """Apply discount for small samples to avoid spurious results."""
    if n_rows < 200:
        return base_score * 0.4
    if n_rows < 500:
        return base_score * 0.6
    return base_score


def validate_inputs(df: pd.DataFrame, required_cols: List[str]) -> bool:
    """Validate that required columns exist."""
    return all(c in df.columns for c in required_cols)
