"""DET v3 Core Metrics - URS, AOI, DMI, k-Anonymity, HRS."""

import pandas as pd
import numpy as np
from typing import Dict, List, Any
from ..utils import format_result, assign_flag, THRESHOLDS


def calculate_urs(df: pd.DataFrame, protected_attrs: List[str]) -> Dict[str, Any]:
    """Underrepresentation Score - minimum group proportion across protected attributes."""
    if not protected_attrs:
        return format_result(1.0, 'YELLOW', "No protected attributes specified", {})
    
    min_proportion = 1.0
    proportions = {}
    
    for attr in protected_attrs:
        if attr not in df.columns:
            continue
        counts = df[attr].value_counts(normalize=True)
        proportions[attr] = counts.to_dict()
        min_proportion = min(min_proportion, counts.min())
    
    flag = assign_flag(min_proportion, 'URS', higher_is_better=True)
    
    return format_result(
        min_proportion, flag,
        f"Min group proportion: {min_proportion:.1%}",
        {'proportions_by_attribute': proportions}
    )


def calculate_aoi(df: pd.DataFrame, target: str, protected_attrs: List[str]) -> Dict[str, Any]:
    """Attribute-Outcome Imbalance - max disparity in outcome rates."""
    if target not in df.columns or not protected_attrs:
        return format_result(0.0, 'YELLOW', "Missing target or protected attributes", {})
    
    df_valid = df.dropna(subset=[target])
    y = pd.to_numeric(df_valid[target], errors='coerce')
    if y.isna().all():
        y = (df_valid[target] == df_valid[target].mode().iloc[0]).astype(int)
    
    max_disparity = 0.0
    outcome_rates = {}
    
    for attr in protected_attrs:
        if attr not in df_valid.columns:
            continue
        rates = df_valid.groupby(attr).apply(lambda g: y.loc[g.index].mean())
        outcome_rates[attr] = rates.to_dict()
        if len(rates) >= 2:
            max_disparity = max(max_disparity, rates.max() - rates.min())
    
    flag = assign_flag(max_disparity, 'AOI', higher_is_better=False)
    
    return format_result(
        max_disparity, flag,
        f"Max outcome disparity: {max_disparity:.1%}",
        {'outcome_rates_by_attribute': outcome_rates}
    )


def calculate_dmi(df: pd.DataFrame, protected_attrs: List[str]) -> Dict[str, Any]:
    """Differential Missingness Index - max difference in missing rates across groups."""
    if not protected_attrs:
        return format_result(0.0, 'YELLOW', "No protected attributes specified", {})
    
    max_diff = 0.0
    missingness = {}
    
    for attr in protected_attrs:
        if attr not in df.columns:
            continue
        
        attr_miss = {}
        for col in df.columns:
            if col == attr:
                continue
            rates = df.groupby(attr)[col].apply(lambda x: x.isna().mean())
            if len(rates) >= 2:
                diff = rates.max() - rates.min()
                max_diff = max(max_diff, diff)
                attr_miss[col] = {'diff': diff, 'rates': rates.to_dict()}
        missingness[attr] = attr_miss
    
    flag = assign_flag(max_diff, 'DMI', higher_is_better=False)
    
    return format_result(
        max_diff, flag,
        f"Max missingness gap: {max_diff:.1%}",
        {'missingness_by_attribute': missingness}
    )


def calculate_k_anonymity(df: pd.DataFrame, quasi_identifiers: List[str]) -> Dict[str, Any]:
    """k-Anonymity Risk - proportion of rows with equivalence class < k threshold."""
    if not quasi_identifiers:
        return format_result(float('inf'), 'YELLOW', "No quasi-identifiers specified", {})
    
    valid_qis = [q for q in quasi_identifiers if q in df.columns]
    if not valid_qis:
        return format_result(float('inf'), 'YELLOW', "No valid quasi-identifiers found", {})
    
    df_qi = df[valid_qis].fillna('MISSING').astype(str)
    group_sizes = df_qi.groupby(valid_qis).size()
    
    min_k = group_sizes.min()
    k_threshold = THRESHOLDS['k_anonymity']['green']
    risk_proportion = (group_sizes < k_threshold).sum() / len(group_sizes)
    
    if min_k >= k_threshold:
        flag = 'GREEN'
    elif min_k >= THRESHOLDS['k_anonymity']['yellow']:
        flag = 'YELLOW'
    else:
        flag = 'RED'
    
    return format_result(
        min_k, flag,
        f"Min k={min_k}, {risk_proportion:.1%} of groups at risk",
        {'min_k': min_k, 'risk_proportion': risk_proportion, 'quasi_identifiers': valid_qis}
    )


def calculate_hrs(df: pd.DataFrame, target: str, protected_attrs: List[str],
                  severity: str = 'medium', vulnerability: str = 'medium') -> Dict[str, Any]:
    """Harm Risk Score - bias likelihood × severity × vulnerability."""
    severity_map = {'low': 0.3, 'medium': 0.5, 'high': 0.7, 'critical': 0.9}
    s = severity_map.get(severity, 0.5)
    v = severity_map.get(vulnerability, 0.5)
    
    # Estimate bias likelihood from URS and AOI
    urs_result = calculate_urs(df, protected_attrs)
    aoi_result = calculate_aoi(df, target, protected_attrs)
    
    urs_score = urs_result.get('score', 1.0)
    aoi_score = aoi_result.get('score', 0.0)
    
    bias_likelihood = (1 - urs_score) * 0.5 + aoi_score * 0.5
    hrs = bias_likelihood * s * v
    
    flag = assign_flag(hrs, 'HRS', higher_is_better=False)
    
    return format_result(
        hrs, flag,
        f"HRS = {bias_likelihood:.2f} × {s:.1f} × {v:.1f} = {hrs:.3f}",
        {'bias_likelihood': bias_likelihood, 'severity_multiplier': s, 'vulnerability_multiplier': v}
    )
