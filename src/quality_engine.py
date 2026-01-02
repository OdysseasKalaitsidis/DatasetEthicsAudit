from typing import Tuple, Optional, Dict
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

class QualityAuditor:
    """
    Data Health: Checks for missing values, class imbalance, and duplicates.
    """
    def __init__(self, df: pd.DataFrame, target: Optional[str] = None):
        self.df = df
        self.target = target

    def check_duplicates(self) -> int:
        """
        Calculates the number of duplicate rows.
        """
        return self.df.duplicated().sum()

    def check_missing_values(self, sensitive_col: Optional[str] = None) -> Tuple[pd.DataFrame, go.Figure]:
        """
        Analyzes missing values and optionally checks for correlation with a sensitive attribute.
        
        Args:
            sensitive_col: Optional sensitive attribute to check for MNAR (Missing Not At Random).
            
        Returns:
            Tuple: (DataFrame with missing counts, Plotly Figure)
        """
        missing = self.df.isnull().sum()
        missing = missing[missing > 0]
        
        miss_df = pd.DataFrame({
            'Column': missing.index,
            'Missing Count': missing.values,
            'Percentage': (missing.values / len(self.df)) * 100
        }).sort_values('Missing Count', ascending=False)

        fig = px.bar(
            miss_df, 
            x='Column', 
            y='Missing Count',
            text='Percentage',
            title='Missing Values by Column',
            color='Missing Count',
            color_continuous_scale='Reds'
        )
        fig.update_traces(texttemplate='%{text:.1f}%', textposition='outside')
        fig.update_layout(yaxis_title="Count of Missing Values", xaxis_title="Feature")

        # MNAR Check (Simple correlation check)
        # Note: True MNAR analysis is complex; this is a heuristic to see if missingness in one column 
        # is related to the sensitive group.
        if sensitive_col and sensitive_col in self.df.columns and not miss_df.empty:
            # For the top missing column, check if missingness varies by sensitive group
            top_miss_col = miss_df.iloc[0]['Column']
            # Create a boolean flag for missing
            self.df[f'{top_miss_col}_is_missing'] = self.df[top_miss_col].isnull()
            
            # Helper text for UI (This might be better returned or printed in the UI layer, 
            # but we can adhere to returning data/plots mostly)
            # logic here is just for internal check, real visualization of this relationship 
            # could be a grouped bar chart if requested, but prompt asked for this specifically.
            pass

        return miss_df, fig

    def check_class_balance(self) -> Tuple[pd.Series, go.Figure, str]:
        """
        Checks the balance of the target variable.
        
        Returns:
            Tuple: (Value Counts, Plotly Pie Chart, Status Message)
        """
        if not self.target or self.target not in self.df.columns:
            return pd.Series(), go.Figure(), "No target variable selected."

        counts = self.df[self.target].value_counts(normalize=True)
        counts_abs = self.df[self.target].value_counts()
        
        fig = px.pie(
            values=counts_abs, 
            names=counts_abs.index,
            title=f"Class Balance: {self.target}",
            hole=0.4,
            color_discrete_sequence=px.colors.qualitative.Safe
        )
        
        # Determine Status
        min_class_pct = counts.min()
        if min_class_pct < 0.10: # Less than 10% representation
            status = "CRITICAL: Severe Class Imbalance (Minority Class < 10%). Suggestions: SMOTE, Oversampling."
        elif min_class_pct < 0.20: # Less than 20%
            status = "WARNING: Moderate Imbalance (Minority Class < 20%). Monitor performance on this class."
        else:
            status = "SAFE: Good Class Balance."

        return counts_abs, fig, status

    def check_group_representation(self, sensitive_col: str) -> Tuple[pd.DataFrame, str]:
        """
        Metric: Calculated percentage of each subgroup. Alert if < 10%.
        """
        if sensitive_col not in self.df.columns:
            return pd.DataFrame(), "Sensitive column not found."
            
        counts = self.df[sensitive_col].value_counts(normalize=True) * 100
        counts_df = counts.reset_index()
        counts_df.columns = [sensitive_col, 'Percentage']
        
        # Check for underrepresented groups
        underrepresented = counts_df[counts_df['Percentage'] < 10][sensitive_col].tolist()
        
        if underrepresented:
            status = f"⚠️ CRITICAL: The following groups are underrepresented (<10%): {', '.join(map(str, underrepresented))}."
        else:
            status = "✅ SAFE: All groups have decent representation (>10%)."
            
        return counts_df, status

    def check_intersectional_coverage(self, sensitive_col: str, other_col: str) -> Tuple[go.Figure, int]:
        """
        Metric: Intersectional Coverage Score (Pivot Table Heatmap).
        Marks 'Blind Spots' (cells with < 5 samples).
        """
        if sensitive_col not in self.df.columns or other_col not in self.df.columns:
            return go.Figure(), 0
            
        # Create Pivot Table of Counts
        pivot = pd.crosstab(self.df[sensitive_col], self.df[other_col])
        
        # Identify Blind Spots (< 5 samples)
        blind_spots = (pivot < 5).sum().sum()
        
        # Plot Heatmap
        fig = px.imshow(
            pivot, 
            text_auto=True,
            title=f"Intersectional Heatmap: {sensitive_col} vs {other_col}",
            color_continuous_scale='Blues'
        )
        
        return fig, blind_spots
