from typing import Tuple, Dict, Any, List, Optional
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, recall_score, precision_score, confusion_matrix
from sklearn.preprocessing import LabelEncoder
from sklearn.impute import SimpleImputer

# DET v3 imports
from src.det_metrics import DETMetricsCalculator
from src.det_decision import DETDecisionEngine
from src.det_visualizations import DETVisualizations

class ModelAuditor:
    """
    Dedicated DET v3 Auditor.
    Focuses on strategic dataset ethical triage.
    """
    def __init__(self, df: pd.DataFrame, target_col: str, sensitive_col: str):
        self.df = df.copy()
        self.target_col = target_col
        self.sensitive_col = sensitive_col
        self.encoders = {}
        
        # Prepare data on init (needed for some advanced metrics that use spy models)
        self._prepare_data()

    def _prepare_data(self):
        """
        Internal: Prepares data for modeling (Encoding, Imputing).
        """
        # Drop rows with missing target
        self.df.dropna(subset=[self.target_col], inplace=True)
        
        X = self.df.drop(columns=[self.target_col])
        y = self.df[self.target_col]
        
        # Encode Target if not numeric
        if y.dtype == 'object' or y.dtype.name == 'category':
            le = LabelEncoder()
            y = le.fit_transform(y)
            self.encoders[self.target_col] = le
            
        # Encode Features
        X_encoded = X.copy()
        for col in X_encoded.select_dtypes(include=['object', 'category']).columns:
            if col not in self.encoders:
                self.encoders[col] = LabelEncoder()
            # Handle NaNs in categorical - convert to string 'nan'
            X_encoded[col] = X_encoded[col].astype(str)
            X_encoded[col] = self.encoders[col].fit_transform(X_encoded[col])
            
        # Drop columns that are entirely missing (Imputer would drop these, causing shape mismatch)
        X_encoded = X_encoded.dropna(axis=1, how='all')
            
        # Impute remaining missing values (numeric primarily)
        if not X_encoded.empty:
            imputer = SimpleImputer(strategy='most_frequent')
            X_imputed = imputer.fit_transform(X_encoded)
            X_encoded = pd.DataFrame(X_imputed, columns=X_encoded.columns, index=X_encoded.index)
        
        self.X_encoded = X_encoded
        self.y_encoded = y

    def run_det_audit(self, 
                     quasi_identifiers: Optional[List[str]] = None,
                     harm_severity: float = 0.5,
                     harm_vulnerability: float = 0.5) -> Dict[str, Any]:
        """
        Run Dataset Ethical Triage (DET) v3 assessment.
        
        Computes all 10 DET metrics, makes triage decision, and generates visualizations.
        """
        # Initialize DET components
        det_calculator = DETMetricsCalculator(
            df=self.df,
            sensitive_col=self.sensitive_col,
            target_col=self.target_col
        )
        
        det_decision_engine = DETDecisionEngine()
        
        # Calculate all metrics
        metric_results = det_calculator.calculate_all_metrics(
            quasi_identifiers=quasi_identifiers,
            harm_severity=harm_severity,
            harm_vulnerability=harm_vulnerability
        )
        
        # Make decision
        decision_result = det_decision_engine.make_decision(metric_results)
        
        # Generate visualizations
        visualizations = {
            'radar_chart': DETVisualizations.create_radar_chart(metric_results),
            'decision_matrix': DETVisualizations.create_decision_matrix(metric_results),
            'category_summary': DETVisualizations.create_category_summary(metric_results),
            'decision_flow': DETVisualizations.create_decision_flow(decision_result),
            'comprehensive_dashboard': DETVisualizations.create_comprehensive_dashboard(
                metric_results, decision_result
            )
        }
        
        return {
            'metric_results': metric_results,
            'decision_result': decision_result,
            'visualizations': visualizations
        }
