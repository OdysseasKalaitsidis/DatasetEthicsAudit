from typing import Tuple, Dict, Any, List
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, recall_score, precision_score, confusion_matrix
from sklearn.preprocessing import LabelEncoder
from sklearn.impute import SimpleImputer

class ModelAuditor:
    """
    The Forensic Audit: Simulation, Leakage, and Counterfactuals.
    """
    def __init__(self, df: pd.DataFrame, target_col: str, sensitive_col: str):
        self.df = df.copy()
        self.target_col = target_col
        self.sensitive_col = sensitive_col
        self.model = RandomForestClassifier(n_estimators=50, random_state=42)
        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.y_test = None
        self.feature_names = None
        self.encoders = {}
        
        # Prepare data on init
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
            # Handle unknown labels? Simple LabelEncoder is fragile. 
            # For a robust tool, should use OneHot or Ordinal.
            # However, Random Forest handles ordinal encoded well enough for an audit simulation.
            X_encoded[col] = X_encoded[col].astype(str)
            X_encoded[col] = self.encoders[col].fit_transform(X_encoded[col])
            
        # Impute
        imputer = SimpleImputer(strategy='most_frequent')
        X_encoded = pd.DataFrame(imputer.fit_transform(X_encoded), columns=X_encoded.columns)
        
        self.feature_names = X_encoded.columns
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            X_encoded, y, test_size=0.3, random_state=42, stratify=y
        )

    def train_simulation(self) -> Dict[str, Any]:
        """
        Trains the Random Forest model and checks for Overfitting.
        """
        self.model.fit(self.X_train, self.y_train)
        
        train_pred = self.model.predict(self.X_train)
        test_pred = self.model.predict(self.X_test)
        
        train_acc = accuracy_score(self.y_train, train_pred)
        test_acc = accuracy_score(self.y_test, test_pred)
        
        # Check Overfitting (Gap > 15%)
        is_overfitting = (train_acc - test_acc) > 0.15
        
        return {
            "train_accuracy": train_acc,
            "test_accuracy": test_acc,
            "is_overfitting": is_overfitting
        }

    def get_feature_importance(self) -> Tuple[pd.DataFrame, go.Figure, str]:
        """
        Detects Data Leakage via Feature Importance.
        """
        if not hasattr(self.model, 'feature_importances_'):
            return pd.DataFrame(), go.Figure(), "Model not trained."
            
        importances = self.model.feature_importances_
        fi_df = pd.DataFrame({
            'Feature': self.feature_names,
            'Importance': importances
        }).sort_values('Importance', ascending=False)
        
        fig = px.bar(
            fi_df.head(10),
            x='Importance',
            y='Feature',
            orientation='h',
            title='Feature Importance (Leakage Detection)',
            range_x=[0, 1]
        )
        fig.update_layout(yaxis={'categoryorder':'total ascending'})
        
        # Leakage Check (> 40%)
        top_imp = fi_df.iloc[0]['Importance']
        top_feature = fi_df.iloc[0]['Feature']
        
        if top_imp > 0.40:
            msg = f"ALERT: Potential Data Leakage! Feature '{top_feature}' has {top_imp:.1%} importance. Run without it."
        else:
            msg = "SAFE: No single feature dominates (>40%) the model predictions."
            
        return fi_df, fig, msg

    def run_counterfactual_analysis(self) -> Tuple[int, pd.DataFrame]:
        """
        INNOVATION 3: Checks if flipping the sensitive attribute changes predictions.
        (Fairness through Unawareness violation check).
        """
        # Take a sample from Test set
        sample = self.X_test.copy()
        original_preds = self.model.predict(sample)
        
        # Flip Sensitive Attribute (Assuming it was encoded)
        # We need to find the sensitive col in X (it might be encoded)
        if self.sensitive_col not in sample.columns:
            return 0, pd.DataFrame()
            
        # "Flip" logic: If binary, invert. If multi, shift?
        # Robust basic flip: Roll values.
        # Or simpler: Set everyone to the most common value? 
        # Better: Permute it? No, we want individual counterfactuals.
        # Let's inverse: 0->1, 1->0. If more than 2, (x+1)%N.
        
        flipped_sample = sample.copy()
        # Find max val to wrap around
        max_val = flipped_sample[self.sensitive_col].max()
        flipped_sample[self.sensitive_col] = (flipped_sample[self.sensitive_col] + 1) % (max_val + 1)
        
        new_preds = self.model.predict(flipped_sample)
        
        # Count diffs
        diff_indices = np.where(original_preds != new_preds)[0]
        num_changed = len(diff_indices)
        
        # Return summary
        return num_changed, self.df.iloc[self.X_test.index[diff_indices]]

    def test_fairness_threshold(self, threshold: float) -> Tuple[go.Figure, Dict[str, float]]:
        """
        Allows user to see Recalls vs Precision at a custom threshold.
        """
        if not hasattr(self.model, 'predict_proba'):
            return go.Figure(), {}
            
        # Get Probabilities for positive class (1)
        probs = self.model.predict_proba(self.X_test)[:, 1]
        
        # Apply Threshold
        preds = (probs >= threshold).astype(int)
        
        # Metric
        acc = accuracy_score(self.y_test, preds)
        rec = recall_score(self.y_test, preds, zero_division=0)
        prec = precision_score(self.y_test, preds, zero_division=0)
        
        metrics = {
            "Threshold": threshold,
            "Accuracy": round(acc, 3),
            "Recall (Sensitivity)": round(rec, 3),
            "Precision": round(prec, 3)
        }
        
        # Confusion Matrix
        cm = confusion_matrix(self.y_test, preds)
        fig = px.imshow(cm, text_auto=True, title=f"Confusion Matrix @ {threshold}")
        
        return fig, metrics