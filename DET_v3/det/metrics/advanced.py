"""DET v3 Advanced Metrics - FOI, FPC, CPA, SPA, DAI."""

import pandas as pd
import numpy as np
from typing import Dict, List, Any, Optional
from sklearn.preprocessing import LabelEncoder
from sklearn.feature_selection import mutual_info_classif
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, accuracy_score
from scipy.stats import entropy

from ..utils import format_result, filter_features, encode_column, small_sample_discount


def calculate_foi(df: pd.DataFrame, target: str, protected_attrs: List[str]) -> Dict[str, Any]:
    """Feature-Outcome Independence - variance in feature-outcome correlations across groups."""
    if target not in df.columns or not protected_attrs:
        return format_result(1.0, 'YELLOW', "Missing configuration", {})
    
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    features = [f for f in numeric_cols if f not in protected_attrs and f != target]
    
    if not features:
        return format_result(1.0, 'YELLOW', "No numeric features", {})
    
    df_valid = df.dropna(subset=[target])
    y = encode_column(df_valid[target])
    
    max_variance = 0.0
    for attr in protected_attrs:
        if attr not in df_valid.columns:
            continue
        groups = df_valid[attr].dropna().unique()
        if len(groups) < 2:
            continue
        
        for feat in features[:15]:
            correlations = []
            for grp in groups:
                mask = df_valid[attr] == grp
                if mask.sum() < 10:
                    continue
                try:
                    corr = df_valid.loc[mask, feat].corr(pd.Series(y[mask.values]))
                    if not np.isnan(corr):
                        correlations.append(corr)
                except:
                    continue
            
            if len(correlations) >= 2:
                max_variance = max(max_variance, np.var(correlations))
    
    foi_score = 1.0 - min(max_variance * 2, 1.0)
    
    if foi_score >= 0.85:
        flag = 'GREEN'
    elif foi_score >= 0.70:
        flag = 'YELLOW'
    else:
        flag = 'RED'
    
    return format_result(foi_score, flag, f"Correlation variance: {max_variance:.3f}", 
                         {'max_variance': max_variance})


def calculate_fpc(df: pd.DataFrame, target: str, protected_attrs: List[str]) -> Dict[str, Any]:
    """Fairness-Performance Convergence - accuracy variance across groups."""
    if target not in df.columns or not protected_attrs or len(df) < 100:
        return format_result(1.0, 'YELLOW', "Insufficient data for FPC", {'diagnostic': True})
    
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    features = [f for f in numeric_cols if f not in protected_attrs and f != target]
    
    if len(features) < 2:
        return format_result(1.0, 'YELLOW', "Insufficient features", {'diagnostic': True})
    
    df_valid = df.dropna(subset=[target]).copy()
    X = df_valid[features].fillna(0)
    y = encode_column(df_valid[target])
    
    try:
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.3, random_state=42,
            stratify=y if len(np.unique(y)) > 1 else None
        )
        model = LogisticRegression(max_iter=500, random_state=42, solver='lbfgs')
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
    except:
        return format_result(1.0, 'YELLOW', "Model training failed", {'diagnostic': True})
    
    test_df = df_valid.iloc[X_test.index].copy()
    accuracies = []
    
    for attr in protected_attrs:
        if attr not in test_df.columns:
            continue
        for grp in test_df[attr].dropna().unique():
            mask = test_df[attr] == grp
            if mask.sum() >= 10:
                acc = accuracy_score(y_test[mask.values], y_pred[mask.values])
                accuracies.append(acc)
    
    if len(accuracies) >= 2:
        variance = np.var(accuracies)
        fpc_score = 1.0 - min(variance * 5, 1.0)
    else:
        fpc_score = 1.0
    
    if len(df) < 200:
        fpc_score = min(fpc_score + 0.1, 1.0)
    
    if fpc_score >= 0.80:
        flag = 'GREEN'
    elif fpc_score >= 0.60:
        flag = 'YELLOW'
    else:
        flag = 'RED'
    
    return format_result(fpc_score, flag, f"Accuracy variance: {variance if 'variance' in dir() else 0:.3f}",
                         {'diagnostic': True})


def calculate_cpa(df: pd.DataFrame, protected_attrs: List[str]) -> Dict[str, Any]:
    """Conditional Proxy Assessment - mutual information between features and protected attrs."""
    if not protected_attrs:
        return format_result(0.0, 'YELLOW', "No protected attributes", {})
    
    features = filter_features(df, protected_attrs)
    if not features:
        return format_result(0.0, 'YELLOW', "No features to analyze", {})
    
    max_mi = 0.0
    proxy_scores = {}
    
    for attr in protected_attrs:
        if attr not in df.columns:
            continue
        
        y = encode_column(df[attr])
        if len(np.unique(y)) < 2:
            continue
        
        for feat in features[:25]:
            if feat not in df.columns:
                continue
            try:
                x = encode_column(df[feat])
                mi = mutual_info_classif(x.reshape(-1, 1), y, discrete_features=True, random_state=42)[0]
                h_y = entropy(np.bincount(y) / len(y))
                normalized = mi / h_y if h_y > 0 else 0.0
                proxy_scores[f"{feat}→{attr}"] = normalized
                max_mi = max(max_mi, normalized)
            except:
                continue
    
    cpa_score = small_sample_discount(len(df), min(max_mi, 1.0))
    
    if cpa_score <= 0.50:
        flag = 'GREEN'
    elif cpa_score <= 0.70:
        flag = 'YELLOW'
    else:
        flag = 'RED'
    
    return format_result(cpa_score, flag, f"Max proxy MI: {max_mi:.3f}",
                         {'proxy_scores': dict(sorted(proxy_scores.items(), key=lambda x: x[1], reverse=True)[:8])})


def calculate_spa(df: pd.DataFrame, protected_attrs: List[str]) -> Dict[str, Any]:
    """Sensitive Predictability Analysis - can protected attrs be predicted from features?"""
    if not protected_attrs:
        return format_result(0.5, 'YELLOW', "No protected attributes", {})
    
    features = filter_features(df, protected_attrs)
    if len(features) < 2:
        return format_result(0.5, 'YELLOW', "Insufficient features", {})
    
    df_enc = df.copy()
    feature_cols = []
    for f in features:
        if f not in df_enc.columns:
            continue
        df_enc[f'_enc_{f}'] = encode_column(df_enc[f])
        feature_cols.append(f'_enc_{f}')
    
    max_auc = 0.5
    
    for attr in protected_attrs:
        if attr not in df_enc.columns:
            continue
        
        y = encode_column(df_enc[attr])
        if len(np.unique(y)) < 2:
            continue
        
        X = df_enc[feature_cols].values
        try:
            X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.3, random_state=42,
                                                       stratify=y if len(np.unique(y)) <= 10 else None)
            model = LogisticRegression(max_iter=500, random_state=42, solver='lbfgs')
            model.fit(X_tr, y_tr)
            
            if len(np.unique(y_te)) == 2:
                auc = roc_auc_score(y_te, model.predict_proba(X_te)[:, 1])
            else:
                auc = roc_auc_score(y_te, model.predict_proba(X_te), multi_class='ovr', average='weighted')
            max_auc = max(max_auc, auc)
        except:
            continue
    
    if len(df) < 200:
        adjusted_auc = 0.5 + (max_auc - 0.5) * 0.7
    else:
        adjusted_auc = max_auc
    
    if adjusted_auc <= 0.70:
        flag = 'GREEN'
    elif adjusted_auc <= 0.80:
        flag = 'YELLOW'
    else:
        flag = 'RED'
    
    return format_result(max_auc, flag, f"Max predictability AUC: {max_auc:.3f}", {})


def calculate_dai(df: pd.DataFrame, protected_attrs: List[str]) -> Dict[str, Any]:
    """Distributional Alignment Index - distance from uniform distribution."""
    if not protected_attrs:
        return format_result(1.0, 'YELLOW', "No protected attributes", {})
    
    min_dai = 1.0
    dai_details = {}
    
    for attr in protected_attrs:
        if attr not in df.columns:
            continue
        
        counts = df[attr].value_counts(dropna=False)
        n = len(counts)
        if n < 2:
            continue
        
        observed = counts.values / counts.sum()
        uniform = np.ones(n) / n
        
        m = (observed + uniform) / 2
        eps = 1e-10
        js = 0.5 * entropy(observed + eps, m + eps) + 0.5 * entropy(uniform + eps, m + eps)
        dai = 1.0 - js / np.log(2)
        
        dai_details[attr] = {'dai': dai, 'distribution': {str(k): v for k, v in zip(counts.index, observed)}}
        min_dai = min(min_dai, dai)
    
    if min_dai >= 0.85:
        flag = 'GREEN'
    elif min_dai >= 0.70:
        flag = 'YELLOW'
    else:
        flag = 'RED'
    
    return format_result(min_dai, flag, f"Distribution alignment: {min_dai:.3f}",
                         {'dai_by_attribute': dai_details})
