"""
Dataset Ethical Triage (DET) v3 - Metrics Engine

Implements 10 metrics for pre-training ethical assessment:
- 5 CORE: URS, AOI, DMI, k-Anon, HRS
- 5 ADVANCED: FOI, FPC, CPA, SPA, DAI

Excludes: iURS, SSB (as per user request)
"""

from typing import Tuple, Dict, List, Optional, Any
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from scipy import stats
from sklearn.preprocessing import LabelEncoder
from sklearn.feature_selection import mutual_info_classif
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score


class DETMetricsCalculator:
    """
    Calculates all 10 DET v3 metrics with threshold-based flagging.
    """
    
    def __init__(self, df: pd.DataFrame, sensitive_col: str, target_col: str):
        """
        Initialize DET metrics calculator.
        
        Args:
            df: Dataset to analyze
            sensitive_col: Protected attribute column name
            target_col: Target/outcome column name
        """
        self.df = df.copy()
        self.sensitive_col = sensitive_col
        self.target_col = target_col
        
        # Robust Cleaning: Handle Whitespace and NaNs
        # (Hidden spaces are a common cause for unintended group fragmentation)
        for col in [sensitive_col, target_col]:
            if self.df[col].dtype == 'object':
                self.df[col] = self.df[col].astype(str).str.strip()
        
        # Clean data for analysis
        self.df_clean = self.df.dropna(subset=[sensitive_col, target_col])
        self.total_n = len(self.df_clean)
        
    def _is_empty(self) -> bool:
        return self.total_n == 0
        
    # ========================================================================
    # CORE METRICS (5)
    # ========================================================================
    
    def calculate_urs(self) -> Dict[str, Any]:
        """
        URS (Underrepresentation Score)
        
        Measures representation of minority classes in both target and sensitive attributes.
        Replaces R-Index as requested.
        
        Formula:
            URS_target = min(class_counts) / total_samples
            URS_sensitive = min(group_counts) / total_samples
            URS = min(URS_target, URS_sensitive)
        
        Thresholds:
            GREEN: URS >= 0.20 (≥20% representation)
            YELLOW: 0.10 <= URS < 0.20
            RED: URS < 0.10 (<10% severe underrepresentation)
        
        Returns:
            Dict with score, flag, visualization, interpretation
        """
        # Target class representation
        target_counts = self.df_clean[self.target_col].value_counts()
        urs_target = target_counts.min() / len(self.df_clean)
        
        # Sensitive attribute representation
        sensitive_counts = self.df_clean[self.sensitive_col].value_counts()
        urs_sensitive = sensitive_counts.min() / len(self.df_clean)
        
        # Overall URS (worst case)
        urs_score = min(urs_target, urs_sensitive)
        
        # Flag determination
        if urs_score >= 0.20:
            flag = 'GREEN'
            interpretation = f"✅ Good representation: Smallest group is {urs_score:.2%} of dataset"
        elif urs_score >= 0.10:
            flag = 'YELLOW'
            interpretation = f"⚠️ Moderate underrepresentation: Smallest group is {urs_score:.2%} (recommend oversampling)"
        else:
            flag = 'RED'
            interpretation = f"🔴 Severe underrepresentation: Smallest group is only {urs_score:.2%} (critical issue)"
        
        # Visualization: Combined bar chart
        fig = go.Figure()
        
        # Target distribution
        fig.add_trace(go.Bar(
            name='Target Class',
            x=target_counts.index.astype(str),
            y=(target_counts / len(self.df_clean) * 100),
            marker_color='lightblue'
        ))
        
        # Sensitive attribute distribution
        fig.add_trace(go.Bar(
            name=self.sensitive_col,
            x=sensitive_counts.index.astype(str),
            y=(sensitive_counts / len(self.df_clean) * 100),
            marker_color='lightcoral'
        ))
        
        fig.update_layout(
            title=f"URS: Representation Analysis (Score: {urs_score:.3f})",
            xaxis_title="Groups",
            yaxis_title="Percentage (%)",
            barmode='group',
            hovermode='x unified'
        )
        
        # Add threshold lines
        fig.add_hline(y=20, line_dash="dash", line_color="green", 
                     annotation_text="20% threshold (GREEN)")
        fig.add_hline(y=10, line_dash="dash", line_color="red", 
                     annotation_text="10% threshold (RED)")
        
        return {
            'metric_name': 'URS (Underrepresentation Score)',
            'score': urs_score,
            'urs_target': urs_target,
            'urs_sensitive': urs_sensitive,
            'flag': flag,
            'thresholds': {'yellow': 0.20, 'red': 0.10},
            'visualization': fig,
            'interpretation': interpretation,
            'details': {
                'target_distribution': target_counts.to_dict(),
                'sensitive_distribution': sensitive_counts.to_dict(),
                'total_samples': self.total_n,
                'sensitive_groups': len(sensitive_counts)
            }
        }
    
    def calculate_aoi(self) -> Dict[str, Any]:
        """
        AOI (Attribute-Outcome Imbalance)
        
        Detects outcome disparities across protected attribute groups.
        
        Formula:
            AOI = max(outcome_rate_by_group) - min(outcome_rate_by_group)
        
        Thresholds:
            GREEN: AOI <= 0.15 (≤15pp difference)
            YELLOW: 0.15 < AOI <= 0.25
            RED: AOI > 0.25 (>25pp disparity)
        
        Returns:
            Dict with score, flag, visualization, interpretation
        """
        # Calculate outcome rates by protected group
        grouped = self.df_clean.groupby(self.sensitive_col)[self.target_col]
        
        # Try to identify positive outcome
        unique_outcomes = self.df_clean[self.target_col].unique()
        if len(unique_outcomes) == 2:
            # Binary outcome - use the "positive" one (1, True, 'Yes', etc.)
            positive_outcome = sorted(unique_outcomes)[-1]
        else:
            # Multi-class - use most common as baseline
            positive_outcome = self.df_clean[self.target_col].mode()[0]
        
        outcome_rates = grouped.apply(lambda x: (x == positive_outcome).mean())
        
        aoi_score = outcome_rates.max() - outcome_rates.min()
        
        # Chi-square test for statistical significance
        contingency_table = pd.crosstab(
            self.df_clean[self.sensitive_col],
            self.df_clean[self.target_col]
        )
        chi2, p_value, dof, expected = stats.chi2_contingency(contingency_table)
        
        # Flag determination
        if aoi_score <= 0.15:
            flag = 'GREEN'
            interpretation = f"✅ Low outcome disparity: {aoi_score:.2%} difference across groups"
        elif aoi_score <= 0.25:
            flag = 'YELLOW'
            interpretation = f"⚠️ Moderate disparity: {aoi_score:.2%} difference (investigate root cause)"
        else:
            flag = 'RED'
            interpretation = f"🔴 High disparity: {aoi_score:.2%} difference (χ² p={p_value:.4f})"
        
        # Visualization
        fig = px.bar(
            x=outcome_rates.index.astype(str),
            y=outcome_rates.values * 100,
            labels={'x': self.sensitive_col, 'y': f'Outcome Rate (%) for {positive_outcome}'},
            title=f"AOI: Outcome Rates by {self.sensitive_col} (Score: {aoi_score:.3f})",
            color=outcome_rates.values,
            color_continuous_scale='RdYlGn_r'
        )
        
        fig.add_hline(y=outcome_rates.mean() * 100, line_dash="dash", 
                     line_color="black", annotation_text="Overall Mean")
        
        return {
            'metric_name': 'AOI (Attribute-Outcome Imbalance)',
            'score': aoi_score,
            'flag': flag,
            'thresholds': {'yellow': 0.15, 'red': 0.25},
            'visualization': fig,
            'interpretation': interpretation,
            'details': {
                'outcome_rates_by_group': outcome_rates.to_dict(),
                'positive_outcome_used': str(positive_outcome),
                'p_value': p_value,
                'total_samples': self.total_n
            }
        }
    
    def calculate_dmi(self) -> Dict[str, Any]:
        """
        DMI (Differential Missingness Index)
        
        Identifies differential missing data patterns across protected groups.
        
        Formula:
            For each feature: DMI_feature = max|missing_rate_group_i - missing_rate_group_j|
            DMI = max(DMI_feature) across all features
        
        Thresholds:
            GREEN: DMI <= 0.10 (≤10pp difference)
            YELLOW: 0.10 < DMI <= 0.20
            RED: DMI > 0.20 (>20pp differential missingness)
        
        Returns:
            Dict with score, flag, visualization, interpretation
        """
        # Calculate missingness rate by feature and group
        features = [col for col in self.df.columns 
                   if col not in [self.sensitive_col, self.target_col]]
        
        dmi_scores = {}
        worst_feature = None
        max_dmi = 0
        
        for feature in features:
            # Missingness rate by group
            missing_by_group = self.df.groupby(self.sensitive_col)[feature].apply(
                lambda x: x.isnull().mean()
            )
            
            if len(missing_by_group) > 1:
                feature_dmi = missing_by_group.max() - missing_by_group.min()
                dmi_scores[feature] = feature_dmi
                
                if feature_dmi > max_dmi:
                    max_dmi = feature_dmi
                    worst_feature = feature
        
        dmi_score = max_dmi
        
        # Flag determination
        if dmi_score <= 0.10:
            flag = 'GREEN'
            interpretation = f"✅ Low differential missingness: Max {dmi_score:.1%} difference"
        elif dmi_score <= 0.20:
            flag = 'YELLOW'
            interpretation = f"⚠️ Moderate differential missingness in '{worst_feature}': {dmi_score:.1%}"
        else:
            flag = 'RED'
            interpretation = f"🔴 High differential missingness in '{worst_feature}': {dmi_score:.1%} (investigate MNAR)"
        
        # Visualization: Heatmap of missingness by feature × group
        if worst_feature:
            missingness_matrix = pd.DataFrame()
            for feature in list(dmi_scores.keys())[:10]:  # Top 10 features
                missingness_matrix[feature] = self.df.groupby(self.sensitive_col)[feature].apply(
                    lambda x: x.isnull().mean() * 100
                )
            
            fig = px.imshow(
                missingness_matrix.T,
                labels=dict(x=self.sensitive_col, y="Feature", color="Missing %"),
                title=f"DMI: Missingness Heatmap (Score: {dmi_score:.3f})",
                color_continuous_scale='Reds',
                text_auto='.1f'
            )
        else:
            fig = go.Figure()
            fig.add_annotation(text="No missing data found", showarrow=False)
        
        return {
            'metric_name': 'DMI (Differential Missingness Index)',
            'score': dmi_score,
            'flag': flag,
            'thresholds': {'yellow': 0.10, 'red': 0.20},
            'visualization': fig,
            'interpretation': interpretation,
            'details': {
                'worst_feature': worst_feature,
                'feature_dmi_scores': dmi_scores
            }
        }
    
    def calculate_k_anonymity(self, quasi_identifiers: List[str]) -> Dict[str, Any]:
        """
        k-Anonymity
        
        Quantifies privacy risk via quasi-identifier uniqueness.
        
        Formula:
            k = min(equivalence_class_size) for all quasi-ID combinations
        
        Thresholds:
            GREEN: k >= 10 (strong privacy)
            YELLOW: 5 <= k < 10
            RED: k < 5 (weak privacy, re-identification risk)
        
        Args:
            quasi_identifiers: List of column names to use as quasi-IDs
        
        Returns:
            Dict with score, flag, visualization, interpretation
        """
        if not quasi_identifiers:
            return {
                'metric_name': 'k-Anonymity',
                'score': None,
                'flag': 'YELLOW',
                'thresholds': {'yellow': 10, 'red': 5},
                'visualization': go.Figure(),
                'interpretation': "⚠️ No quasi-identifiers specified",
                'details': {}
            }
        
        # Filter to valid columns
        valid_qis = [qi for qi in quasi_identifiers if qi in self.df.columns]
        
        if not valid_qis:
            return {
                'metric_name': 'k-Anonymity',
                'score': None,
                'flag': 'YELLOW',
                'thresholds': {'yellow': 10, 'red': 5},
                'visualization': go.Figure(),
                'interpretation': "⚠️ No valid quasi-identifiers found in dataset",
                'details': {}
            }
        
        # Calculate equivalence class sizes
        equivalence_classes = self.df.groupby(valid_qis).size()
        k_score = int(equivalence_classes.min())
        
        # Flag determination
        if k_score >= 10:
            flag = 'GREEN'
            interpretation = f"✅ Strong privacy: k={k_score} (minimum {k_score} indistinguishable records)"
        elif k_score >= 5:
            flag = 'YELLOW'
            interpretation = f"⚠️ Moderate privacy: k={k_score} (consider generalization or suppression)"
        else:
            flag = 'RED'
            interpretation = f"🔴 Weak privacy: k={k_score} (high re-identification risk, apply differential privacy)"
        
        # Visualization: Distribution of equivalence class sizes
        class_size_dist = equivalence_classes.value_counts().sort_index()
        
        fig = px.bar(
            x=class_size_dist.index,
            y=class_size_dist.values,
            labels={'x': 'Equivalence Class Size', 'y': 'Frequency'},
            title=f"k-Anonymity: Distribution of Class Sizes (k={k_score})",
            color=class_size_dist.index,
            color_continuous_scale='RdYlGn'
        )
        
        fig.add_vline(x=k_score, line_dash="dash", line_color="red",
                     annotation_text=f"k={k_score}")
        
        return {
            'metric_name': 'k-Anonymity',
            'score': k_score,
            'flag': flag,
            'thresholds': {'yellow': 10, 'red': 5},
            'visualization': fig,
            'interpretation': interpretation,
            'details': {
                'quasi_identifiers': valid_qis,
                'num_equivalence_classes': len(equivalence_classes),
                'class_size_distribution': class_size_dist.to_dict()
            }
        }
    
    def calculate_hrs(self, severity: float = 0.5, vulnerability: float = 0.5) -> Dict[str, Any]:
        """
        HRS (Harm Risk Score)
        
        Aggregates likelihood × severity × vulnerability of algorithmic harm.
        
        Formula:
            HRS = L_bias × S_harm × V_vulnerable
            where:
                L_bias = f(AIR) - derived from attribute imbalance ratio
                S_harm = user-provided (0.2-1.0)
                V_vulnerable = user-provided (0.3-1.0)
        
        Thresholds:
            GREEN: HRS <= 0.25 (low harm risk)
            YELLOW: 0.25 < HRS <= 0.50
            RED: HRS > 0.50 (high harm risk)
        
        Args:
            severity: Harm severity score (0.2-1.0)
            vulnerability: Population vulnerability score (0.3-1.0)
        
        Returns:
            Dict with score, flag, visualization, interpretation
        """
        # Calculate AIR (Attribute Imbalance Ratio)
        group_sizes = self.df_clean[self.sensitive_col].value_counts()
        air = group_sizes.max() / group_sizes.min()
        
        # Map AIR to likelihood of bias
        if air <= 1.5:
            l_bias = 0.1  # Balanced
        elif air <= 3.0:
            l_bias = 0.5  # Moderate imbalance
        else:
            l_bias = 0.9  # Severe imbalance
        
        # Calculate HRS
        hrs_score = l_bias * severity * vulnerability
        
        # Flag determination
        if hrs_score <= 0.25:
            flag = 'GREEN'
            interpretation = f"✅ Low harm risk: HRS={hrs_score:.3f}"
        elif hrs_score <= 0.50:
            flag = 'YELLOW'
            interpretation = f"⚠️ Moderate harm risk: HRS={hrs_score:.3f} (implement fairness constraints)"
        else:
            flag = 'RED'
            interpretation = f"🔴 High harm risk: HRS={hrs_score:.3f} (major mitigation required or reject dataset)"
        
        # Visualization: Component breakdown
        fig = go.Figure(data=[
            go.Bar(
                x=['Likelihood\n(from AIR)', 'Severity\n(user input)', 'Vulnerability\n(user input)', 'HRS\n(product)'],
                y=[l_bias, severity, vulnerability, hrs_score],
                marker_color=['lightblue', 'lightcoral', 'lightyellow', 'red' if flag == 'RED' else 'yellow' if flag == 'YELLOW' else 'green'],
                text=[f'{l_bias:.2f}', f'{severity:.2f}', f'{vulnerability:.2f}', f'{hrs_score:.3f}'],
                textposition='auto'
            )
        ])
        
        fig.update_layout(
            title=f"HRS: Harm Risk Components (AIR={air:.2f})",
            yaxis_title="Score",
            yaxis_range=[0, 1]
        )
        
        fig.add_hline(y=0.50, line_dash="dash", line_color="red", annotation_text="RED threshold")
        fig.add_hline(y=0.25, line_dash="dash", line_color="orange", annotation_text="YELLOW threshold")
        
        return {
            'metric_name': 'HRS (Harm Risk Score)',
            'score': hrs_score,
            'flag': flag,
            'thresholds': {'yellow': 0.25, 'red': 0.50},
            'visualization': fig,
            'interpretation': interpretation,
            'details': {
                'air': air,
                'l_bias': l_bias,
                'severity': severity,
                'vulnerability': vulnerability,
                'group_sizes': group_sizes.to_dict()
            }
        }
    
    # ========================================================================
    # ADVANCED METRICS (5)
    # ========================================================================
    
    def calculate_foi(self) -> Dict[str, Any]:
        """
        FOI (Feature-Outcome Independence)
        
        Tests independence between sensitive attribute and outcome.
        Uses mutual information to detect non-linear dependencies.
        
        Formula:
            FOI = 1 - MI(sensitive_attr, outcome) / H(outcome)
            where MI = mutual information, H = entropy
        
        Thresholds:
            GREEN: FOI >= 0.85 (strong independence)
            YELLOW: 0.70 <= FOI < 0.85
            RED: FOI < 0.70 (strong dependence)
        
        Returns:
            Dict with score, flag, visualization, interpretation
        """
        # Encode categorical variables
        le_sensitive = LabelEncoder()
        le_outcome = LabelEncoder()
        
        sensitive_encoded = le_sensitive.fit_transform(self.df_clean[self.sensitive_col].astype(str))
        outcome_encoded = le_outcome.fit_transform(self.df_clean[self.target_col].astype(str))
        
        # Calculate mutual information
        mi = mutual_info_classif(
            sensitive_encoded.reshape(-1, 1),
            outcome_encoded,
            discrete_features=True,
            random_state=42
        )[0]
        
        # Calculate outcome entropy
        outcome_probs = pd.Series(outcome_encoded).value_counts(normalize=True)
        h_outcome = stats.entropy(outcome_probs)
        
        # FOI score (normalized MI)
        foi_score = 1 - (mi / h_outcome if h_outcome > 0 else 0)
        
        # Flag determination
        if foi_score >= 0.85:
            flag = 'GREEN'
            interpretation = f"✅ Strong independence: FOI={foi_score:.3f} (sensitive attr weakly predicts outcome)"
        elif foi_score >= 0.70:
            flag = 'YELLOW'
            interpretation = f"⚠️ Moderate dependence: FOI={foi_score:.3f} (some correlation detected)"
        else:
            flag = 'RED'
            interpretation = f"🔴 Strong dependence: FOI={foi_score:.3f} (sensitive attr strongly predicts outcome)"
        
        # Visualization: Confusion matrix style
        contingency = pd.crosstab(
            self.df_clean[self.sensitive_col],
            self.df_clean[self.target_col],
            normalize='index'
        ) * 100
        
        fig = px.imshow(
            contingency,
            labels=dict(x=self.target_col, y=self.sensitive_col, color="Percentage"),
            title=f"FOI: Sensitive × Outcome Distribution (Score: {foi_score:.3f})",
            color_continuous_scale='Blues',
            text_auto='.1f'
        )
        
        return {
            'metric_name': 'FOI (Feature-Outcome Independence)',
            'score': foi_score,
            'flag': flag,
            'thresholds': {'yellow': 0.85, 'red': 0.70},
            'visualization': fig,
            'interpretation': interpretation,
            'details': {
                'mutual_information': mi,
                'outcome_entropy': h_outcome,
                'contingency_table': contingency.to_dict()
            }
        }
    
    def calculate_fpc(self) -> Dict[str, Any]:
        """
        FPC (Fairness-Performance Correlation)
        
        Placeholder for advanced metric - measures correlation between
        fairness metrics and model performance across subgroups.
        
        Note: Requires trained model, so returns placeholder for now.
        
        Returns:
            Dict with placeholder values
        """
        return {
            'metric_name': 'FPC (Fairness-Performance Correlation)',
            'score': 0.5,
            'flag': 'YELLOW',
            'thresholds': {'yellow': 0.70, 'red': 0.50},
            'visualization': go.Figure().add_annotation(
                text="FPC requires trained model<br>(not available in pre-training assessment)",
                showarrow=False,
                font=dict(size=14)
            ),
            'interpretation': "⚠️ FPC metric requires post-training evaluation",
            'details': {'status': 'not_applicable_pre_training'}
        }
    
    def calculate_cpa(self) -> Dict[str, Any]:
        """
        CPA (Confounding Proxy Analysis)
        
        Detects features that act as proxies for sensitive attributes.
        Uses mutual information ranking.
        
        Formula:
            For each feature: CPA_feature = MI(feature, sensitive_attr)
            CPA = max(CPA_feature)
        
        Thresholds:
            GREEN: CPA <= 0.30 (low proxy risk)
            YELLOW: 0.30 < CPA <= 0.50
            RED: CPA > 0.50 (high proxy leakage)
        
        Returns:
            Dict with score, flag, visualization, interpretation
        """
        # Encode sensitive attribute
        le = LabelEncoder()
        y_sensitive = le.fit_transform(self.df_clean[self.sensitive_col].astype(str))
        
        # Prepare features (exclude target and sensitive)
        X = self.df_clean.drop(columns=[self.sensitive_col, self.target_col])
        
        # Encode categorical features
        for col in X.select_dtypes(include=['object', 'category']).columns:
            X[col] = LabelEncoder().fit_transform(X[col].astype(str))
        
        # Handle missing values
        X = X.fillna(0)
        
        # Calculate MI for each feature
        mi_scores = mutual_info_classif(X, y_sensitive, discrete_features='auto', random_state=42)
        
        proxy_scores = pd.DataFrame({
            'Feature': X.columns,
            'MI_Score': mi_scores
        }).sort_values('MI_Score', ascending=False)
        
        cpa_score = proxy_scores['MI_Score'].max()
        worst_proxy = proxy_scores.iloc[0]['Feature']
        
        # Flag determination
        if cpa_score <= 0.30:
            flag = 'GREEN'
            interpretation = f"✅ Low proxy risk: Max MI={cpa_score:.3f}"
        elif cpa_score <= 0.50:
            flag = 'YELLOW'
            interpretation = f"⚠️ Moderate proxy leakage in '{worst_proxy}': MI={cpa_score:.3f}"
        else:
            flag = 'RED'
            interpretation = f"🔴 High proxy leakage in '{worst_proxy}': MI={cpa_score:.3f} (remove or mask feature)"
        
        # Visualization: Top 10 proxy features
        fig = px.bar(
            proxy_scores.head(10),
            x='MI_Score',
            y='Feature',
            orientation='h',
            title=f"CPA: Top Proxy Features (Score: {cpa_score:.3f})",
            color='MI_Score',
            color_continuous_scale='Reds'
        )
        fig.update_layout(yaxis={'categoryorder': 'total ascending'})
        
        return {
            'metric_name': 'CPA (Confounding Proxy Analysis)',
            'score': cpa_score,
            'flag': flag,
            'thresholds': {'yellow': 0.30, 'red': 0.50},
            'visualization': fig,
            'interpretation': interpretation,
            'details': {
                'worst_proxy': worst_proxy,
                'top_proxies': proxy_scores.head(10).to_dict('records')
            }
        }
    
    def calculate_spa(self) -> Dict[str, Any]:
        """
        SPA (Sensitive Predictability Analysis)
        
        Trains a model to predict sensitive attribute from other features.
        High accuracy indicates leakage risk.
        
        Formula:
            SPA = 1 - Accuracy(spy_model)
            (inverted so higher is better)
        
        Thresholds:
            GREEN: SPA >= 0.30 (accuracy <= 70%, low predictability)
            YELLOW: 0.15 <= SPA < 0.30 (accuracy 70-85%)
            RED: SPA < 0.15 (accuracy > 85%, high predictability)
        
        Returns:
            Dict with score, flag, visualization, interpretation
        """
        # Prepare data
        X = self.df_clean.drop(columns=[self.sensitive_col])
        
        # Encode features
        for col in X.select_dtypes(include=['object', 'category']).columns:
            X[col] = LabelEncoder().fit_transform(X[col].astype(str))
        
        X = X.fillna(0)
        
        # Encode target (sensitive attribute)
        le = LabelEncoder()
        y = le.fit_transform(self.df_clean[self.sensitive_col].astype(str))
        
        # Train-test split
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.3, random_state=42, stratify=y
        )
        
        # Train spy model
        spy_model = RandomForestClassifier(n_estimators=50, random_state=42, max_depth=5)
        spy_model.fit(X_train, y_train)
        
        accuracy = accuracy_score(y_test, spy_model.predict(X_test))
        spa_score = 1 - accuracy  # Invert so higher is better
        
        # Flag determination
        if spa_score >= 0.30:
            flag = 'GREEN'
            interpretation = f"✅ Low predictability: Spy model accuracy={accuracy:.1%} (SPA={spa_score:.3f})"
        elif spa_score >= 0.15:
            flag = 'YELLOW'
            interpretation = f"⚠️ Moderate predictability: Spy model accuracy={accuracy:.1%} (SPA={spa_score:.3f})"
        else:
            flag = 'RED'
            interpretation = f"🔴 High predictability: Spy model accuracy={accuracy:.1%} (SPA={spa_score:.3f}, strong leakage)"
        
        # Visualization: Feature importance
        feature_importance = pd.DataFrame({
            'Feature': X.columns,
            'Importance': spy_model.feature_importances_
        }).sort_values('Importance', ascending=False).head(10)
        
        fig = px.bar(
            feature_importance,
            x='Importance',
            y='Feature',
            orientation='h',
            title=f"SPA: Top Predictive Features (Accuracy: {accuracy:.1%})",
            color='Importance',
            color_continuous_scale='Reds'
        )
        fig.update_layout(yaxis={'categoryorder': 'total ascending'})
        
        return {
            'metric_name': 'SPA (Sensitive Predictability Analysis)',
            'score': spa_score,
            'flag': flag,
            'thresholds': {'yellow': 0.30, 'red': 0.15},
            'visualization': fig,
            'interpretation': interpretation,
            'details': {
                'spy_model_accuracy': accuracy,
                'top_predictive_features': feature_importance.to_dict('records')
            }
        }
    
    def calculate_dai(self) -> Dict[str, Any]:
        """
        DAI (Distributional Alignment Index)
        
        Measures how well the dataset's sensitive attribute distribution
        aligns with a reference/target distribution.
        
        Formula:
            DAI = 1 - (L1_distance / 2)
            where L1_distance = sum|observed - target|
        
        Thresholds:
            GREEN: DAI >= 0.85 (good alignment)
            YELLOW: 0.70 <= DAI < 0.85
            RED: DAI < 0.70 (poor alignment)
        
        Args:
            target_distribution: Dict mapping group names to target proportions
        
        Returns:
            Dict with score, flag, visualization, interpretation
        """
        # For now, use uniform distribution as target
        # In production, this should be user-provided (e.g., census data)
        observed_dist = self.df_clean[self.sensitive_col].value_counts(normalize=True)
        
        # Uniform target distribution
        num_groups = len(observed_dist)
        target_dist = pd.Series(1/num_groups, index=observed_dist.index)
        
        # Calculate L1 distance
        l1_distance = (observed_dist - target_dist).abs().sum()
        
        # DAI score
        dai_score = 1 - (l1_distance / 2)
        
        # Flag determination
        if dai_score >= 0.85:
            flag = 'GREEN'
            interpretation = f"✅ Good distributional alignment: DAI={dai_score:.3f}"
        elif dai_score >= 0.70:
            flag = 'YELLOW'
            interpretation = f"⚠️ Moderate misalignment: DAI={dai_score:.3f} (consider resampling)"
        else:
            flag = 'RED'
            interpretation = f"🔴 Poor alignment: DAI={dai_score:.3f} (severe distributional mismatch)"
        
        # Visualization: Observed vs Target
        comparison_df = pd.DataFrame({
            'Group': observed_dist.index.astype(str),
            'Observed': observed_dist.values * 100,
            'Target (Uniform)': target_dist.values * 100
        })
        
        fig = go.Figure()
        fig.add_trace(go.Bar(name='Observed', x=comparison_df['Group'], y=comparison_df['Observed']))
        fig.add_trace(go.Bar(name='Target', x=comparison_df['Group'], y=comparison_df['Target (Uniform)']))
        
        fig.update_layout(
            title=f"DAI: Distribution Comparison (Score: {dai_score:.3f})",
            xaxis_title=self.sensitive_col,
            yaxis_title="Percentage (%)",
            barmode='group'
        )
        
        return {
            'metric_name': 'DAI (Distributional Alignment Index)',
            'score': dai_score,
            'flag': flag,
            'thresholds': {'yellow': 0.85, 'red': 0.70},
            'visualization': fig,
            'interpretation': interpretation,
            'details': {
                'observed_distribution': observed_dist.to_dict(),
                'target_distribution': target_dist.to_dict(),
                'l1_distance': l1_distance
            }
        }
    
    # ========================================================================
    # CONVENIENCE METHOD
    # ========================================================================
    
    def calculate_all_metrics(self, 
                             quasi_identifiers: Optional[List[str]] = None,
                             harm_severity: float = 0.5,
                             harm_vulnerability: float = 0.5) -> Dict[str, Dict[str, Any]]:
        """
        Calculate all 10 DET metrics at once.
        
        Args:
            quasi_identifiers: List of quasi-ID columns for k-anonymity
            harm_severity: Severity parameter for HRS (0.2-1.0)
            harm_vulnerability: Vulnerability parameter for HRS (0.3-1.0)
        
        Returns:
            Dict mapping metric names to their result dicts
        """
        results = {}
        
        # Core metrics
        results['URS'] = self.calculate_urs()
        results['AOI'] = self.calculate_aoi()
        results['DMI'] = self.calculate_dmi()
        results['k-Anon'] = self.calculate_k_anonymity(quasi_identifiers or [])
        results['HRS'] = self.calculate_hrs(harm_severity, harm_vulnerability)
        
        # Advanced metrics
        results['FOI'] = self.calculate_foi()
        results['FPC'] = self.calculate_fpc()
        results['CPA'] = self.calculate_cpa()
        results['SPA'] = self.calculate_spa()
        results['DAI'] = self.calculate_dai()
        
        return results
