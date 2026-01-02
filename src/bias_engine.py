from typing import Tuple, List, Dict, Optional
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from sklearn.feature_selection import mutual_info_classif, mutual_info_regression
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split

class BiasAuditor:
    """
    The Science - Paper Material: Detects Disparate Impact, Proxies, and Simpson's Paradox.
    """
    def __init__(self, df: pd.DataFrame, sensitive_col: str, target_col: str):
        self.df = df.copy()
        self.sensitive_col = sensitive_col
        self.target_col = target_col
        
        # Precompute encoded dataframe for calculations
        self.le = LabelEncoder()
        # Drop rows with missing sensitive or target for bias analysis validity
        self.df.dropna(subset=[sensitive_col, target_col], inplace=True)

    def get_disparate_impact_ratio(self, positive_outcome: Optional[str] = None) -> Tuple[float, go.Figure]:
        """
        Calculates Disparate Impact Ratio (Selection Rate Ratio).
        """
        y = self.df[self.target_col]
        
        # If no positive outcome specified, try to infer it
        if positive_outcome is None:
            unique_vals = y.unique()
            if len(unique_vals) > 2:
                positive_outcome = unique_vals[1] # Naive fallback
            else:
                unique_vals = sorted(unique_vals)
                positive_outcome = unique_vals[-1] # Tends to be 1 or 'Yes'
        
        # Robust comparison: Convert both to string to ensure matching
        # Or just use the raw type if it matches y
        
        # Selection Rate calculation
        # We need to handle type mismatch (e.g. user selects '1' (str) but data is 1 (int))
        if isinstance(positive_outcome, str) and pd.api.types.is_numeric_dtype(y):
            try:
                positive_outcome = float(positive_outcome)
                if positive_outcome.is_integer():
                    positive_outcome = int(positive_outcome)
            except:
                pass # Keep as string if conversion fails
                
        grouped = self.df.groupby(self.sensitive_col)[self.target_col].apply(lambda x: (x == positive_outcome).mean())
        
        if grouped.empty:
            return 0.0, go.Figure()

        min_rate = grouped.min()
        max_rate = grouped.max()
        dir_score = min_rate / max_rate if max_rate > 0 else 0.0
        
        # Plot
        fig = px.bar(
            x=grouped.index, 
            y=grouped.values,
            color=grouped.index,
            title=f"Selection Rates by {self.sensitive_col} (Positive Outcome: {positive_outcome})",
            labels={'x': self.sensitive_col, 'y': 'Selection Rate'},
            color_discrete_sequence=px.colors.qualitative.Pastel
        )
        fig.add_hline(y=max_rate * 0.8, line_dash="dash", line_color="red", annotation_text="80% Threshold")

        return dir_score, fig

    def check_fairness_generalization(self, positive_outcome: Optional[str] = None) -> Tuple[float, float, str]:
        """
        Metric: Fairness Overfitting (Generalization).
        Checks if the Disparate Impact Ratio holds up on unseen data (Test Set) vs seen data (Train Set).
        """
        # Split Data 70/30
        train_df, test_df = train_test_split(self.df, test_size=0.3, random_state=42, stratify=self.df[self.target_col])
        
        # Helper to calc DIR on a subset
        def calc_dir(subset_df):
            y_sub = subset_df[self.target_col]
            # Infer positive outcome if needed (reusing logic or passed val)
            pos_out = positive_outcome
            if pos_out is None:
                unique_vals = y_sub.unique()
                if len(unique_vals) > 2:
                    pos_out = unique_vals[1]
                else:
                    unique_vals = sorted(unique_vals)
                    pos_out = unique_vals[-1]
            
            # Robust type check
            if isinstance(pos_out, str) and pd.api.types.is_numeric_dtype(y_sub):
                try:
                    pos_out = float(pos_out)
                    if pos_out.is_integer(): pos_out = int(pos_out)
                except: pass

            grouped = subset_df.groupby(self.sensitive_col)[self.target_col].apply(lambda x: (x == pos_out).mean())
            if grouped.empty or grouped.max() == 0: return 0.0
            return grouped.min() / grouped.max()

        train_dir = calc_dir(train_df)
        test_dir = calc_dir(test_df)
        
        diff = abs(train_dir - test_dir)
        
        if diff > 0.15:
            status = f"⚠️ FAIRNESS OVERFITTING: Large gap ({diff:.2f}) between Train DIR ({train_dir:.2f}) and Test DIR ({test_dir:.2f}). The fairness you see might be accidental."
        else:
            status = f"✅ STABLE FAIRNESS: Train DIR ({train_dir:.2f}) and Test DIR ({test_dir:.2f}) are consistent."
            
        return train_dir, test_dir, status

    def detect_proxies(self) -> Tuple[pd.DataFrame, go.Figure]:
        """
        INNOVATION 1: Uses Mutual Information to find feature correlations with the sensitive attribute.
        Detects if a feature 'leaks' the sensitive info.
        """
        # Encode Sensitive Attribute
        y_sensitive = self.le.fit_transform(self.df[self.sensitive_col].astype(str))
        
        # Prepare Features (exclude target and sensitive)
        X = self.df.drop(columns=[self.sensitive_col, self.target_col])
        
        # Simple Encoding for Text Features to run MI
        for col in X.select_dtypes(include='object').columns:
            X[col] = LabelEncoder().fit_transform(X[col].astype(str))
            
        # Handle missings for MI calculation
        X = X.fillna(0) # Simple imputation for proxy detection
        
        # Calculate MI
        mi_scores = mutual_info_classif(X, y_sensitive, discrete_features='auto', random_state=42)
        
        proxy_df = pd.DataFrame({
            'Feature': X.columns,
            'MI Score': mi_scores
        }).sort_values('MI Score', ascending=False)
        
        fig = px.bar(
            proxy_df.head(10),
            x='MI Score',
            y='Feature',
            orientation='h',
            title=f"Top Features Leaking '{self.sensitive_col}' Information",
            color='MI Score',
            color_continuous_scale='Viridis'
        )
        fig.update_layout(yaxis={'categoryorder':'total ascending'})
        
        return proxy_df, fig

    def check_simpsons_paradox(self, confounder_col: str) -> go.Figure:
        """
        INNOVATION 2: Checks if trends reverse when splitting by a confounder.
        Plots Average Target vs Sensitive, colored by Confounder.
        """
        temp_df = self.df.copy()
        try:
            temp_df['target_numeric'] = pd.to_numeric(temp_df[self.target_col])
        except:
            temp_df['target_numeric'] = LabelEncoder().fit_transform(temp_df[self.target_col])

        # Group by Sensitive + Confounder
        grouped = temp_df.groupby([self.sensitive_col, confounder_col])['target_numeric'].mean().reset_index()
        
        fig = px.line(
            grouped,
            x=self.sensitive_col,
            y='target_numeric',
            color=confounder_col,
            markers=True,
            title=f"Simpson's Paradox Check: Effect of {self.sensitive_col} split by {confounder_col}",
            labels={'target_numeric': f'Average {self.target_col}'}
        )
        
        # Add Global Trend (dashed line) for comparison
        global_trend = temp_df.groupby(self.sensitive_col)['target_numeric'].mean().reset_index()
        fig.add_scatter(
            x=global_trend[self.sensitive_col], 
            y=global_trend['target_numeric'], 
            mode='lines+markers', 
            name='Global Trend',
            line=dict(color='black', dash='dash', width=3)
        )
        
        return fig

    def check_class_conditional_representation(self) -> go.Figure:
        """
        Metric: Compare distribution of sensitive attribute within Positive vs Negative class.
        Visualization: 100% Stacked Bar Chart.
        """
        # Group by Target + Sensitive
        grouped = self.df.groupby([self.target_col, self.sensitive_col]).size().reset_index(name='Count')
        
        # Calculate percentages within each Target Class
        grouped['Percentage'] = grouped.groupby(self.target_col)['Count'].transform(lambda x: x / x.sum() * 100)
        
        fig = px.bar(
            grouped,
            x=self.target_col,
            y='Percentage',
            color=self.sensitive_col,
            title=f"Class Conditional Representation ({self.sensitive_col} distrib. per Outcome)",
            barmode='stack', # Stacked bar 
            text='Percentage',
            labels={self.target_col: "Target Outcome", "Percentage": f"% of {self.sensitive_col}"}
        )
        fig.update_traces(texttemplate='%{text:.1f}%', textposition='inside')
        
        return fig

    def check_sensitive_attribute_predictability(self) -> Tuple[float, str]:
        """
        Metric: SAP (Sensitive Attribute Predictability).
        Train a model to predict the Sensitive Attribute using all other features.
        If Acc > 0.80, High Proxy Risk.
        """
        from sklearn.model_selection import train_test_split
        from sklearn.ensemble import RandomForestClassifier
        from sklearn.metrics import accuracy_score
        
        # Prepare Data
        X = self.df.drop(columns=[self.sensitive_col])
        
        # Encode features
        for col in X.select_dtypes(include=['object', 'category']).columns:
            X[col] = LabelEncoder().fit_transform(X[col].astype(str))
            
        y = self.le.fit_transform(self.df[self.sensitive_col].astype(str))
        
        # Split
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
        
        # Train Spy Model
        spy_model = RandomForestClassifier(n_estimators=50, random_state=42)
        spy_model.fit(X_train, y_train)
        
        acc = accuracy_score(y_test, spy_model.predict(X_test))
        
        if acc > 0.80:
            status = f"⚠️ HIGH RISK: The data strongly leaks {self.sensitive_col} information (Spy Model Acc: {acc:.1%})."
        else:
            status = f"✅ LOW RISK: The data does not easily reveal {self.sensitive_col} (Spy Model Acc: {acc:.1%})."
            
        return acc, status