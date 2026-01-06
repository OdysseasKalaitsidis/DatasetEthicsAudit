"""
DET v3 (Dataset Ethics Triage) - Streamlit Application

Analyst-friendly UI for pre-training ethical assessment of clinical datasets.

Features:
- CSV upload
- Column selection (target, protected attrs, quasi-IDs)
- Severity selector
- 10-metric analysis with traffic-light indicators
- Bar charts for group representation and outcome rates
- Final decision with explanation
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from typing import List, Dict, Any

# Import DET modules
from .metrics import (
    calculate_urs, calculate_aoi, calculate_dmi,
    calculate_k_anonymity, calculate_hrs,
    calculate_foi, calculate_fpc, calculate_cpa,
    calculate_spa, calculate_dai
)
from .decision import DETDecisionEngine


# =============================================================================
# PAGE CONFIGURATION
# =============================================================================

st.set_page_config(
    page_title="DET v3 - Ethics Auditor",
    page_icon="🔬",
    layout="wide",
    initial_sidebar_state="expanded"
)


# =============================================================================
# CUSTOM STYLING
# =============================================================================

st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');
    
    :root {
        --bg-dark: #0f172a;
        --card-bg: #1e293b;
        --border: #334155;
        --primary: #6366f1;
        --success: #22c55e;
        --warning: #eab308;
        --danger: #ef4444;
        --text: #f8fafc;
        --text-dim: #94a3b8;
    }
    
    * { font-family: 'Inter', sans-serif; }
    
    .stApp { 
        background: linear-gradient(135deg, #0f172a 0%, #1e1b4b 100%);
    }
    
    /* Metric Cards */
    .metric-card {
        background: var(--card-bg);
        border-radius: 12px;
        padding: 20px;
        margin-bottom: 16px;
        border: 1px solid var(--border);
    }
    
    .metric-title {
        font-size: 0.9rem;
        color: var(--text-dim);
        text-transform: uppercase;
        letter-spacing: 1px;
        margin-bottom: 8px;
    }
    
    .metric-value {
        font-size: 2rem;
        font-weight: 700;
        color: var(--primary);
    }
    
    /* Flag Pills */
    .flag-green { 
        background: rgba(34, 197, 94, 0.15);
        color: #4ade80;
        padding: 4px 12px;
        border-radius: 6px;
        font-size: 0.75rem;
        font-weight: 600;
        border: 1px solid rgba(74, 222, 128, 0.2);
    }
    .flag-yellow {
        background: rgba(234, 179, 8, 0.15);
        color: #facc15;
        padding: 4px 12px;
        border-radius: 6px;
        font-size: 0.75rem;
        font-weight: 600;
        border: 1px solid rgba(250, 204, 21, 0.2);
    }
    .flag-red {
        background: rgba(239, 68, 68, 0.15);
        color: #f87171;
        padding: 4px 12px;
        border-radius: 6px;
        font-size: 0.75rem;
        font-weight: 600;
        border: 1px solid rgba(248, 113, 113, 0.2);
    }
    
    /* Decision Box */
    .decision-proceed {
        background: linear-gradient(135deg, rgba(34, 197, 94, 0.2) 0%, rgba(34, 197, 94, 0.1) 100%);
        border: 2px solid #22c55e;
        border-radius: 16px;
        padding: 24px;
        text-align: center;
    }
    .decision-mitigate {
        background: linear-gradient(135deg, rgba(234, 179, 8, 0.2) 0%, rgba(234, 179, 8, 0.1) 100%);
        border: 2px solid #eab308;
        border-radius: 16px;
        padding: 24px;
        text-align: center;
    }
    .decision-reject {
        background: linear-gradient(135deg, rgba(239, 68, 68, 0.2) 0%, rgba(239, 68, 68, 0.1) 100%);
        border: 2px solid #ef4444;
        border-radius: 16px;
        padding: 24px;
        text-align: center;
    }
    
    .decision-text {
        font-size: 2.5rem;
        font-weight: 800;
        margin-bottom: 12px;
    }
    
    /* Sidebar */
    [data-testid="stSidebar"] {
        background: #020617;
        border-right: 1px solid var(--border);
    }
    
    /* Buttons */
    div.stButton > button {
        width: 100%;
        border-radius: 8px;
        height: 3rem;
        font-weight: 600;
        background: linear-gradient(135deg, #6366f1 0%, #8b5cf6 100%);
        color: white;
        border: none;
        transition: all 0.2s;
    }
    div.stButton > button:hover {
        transform: translateY(-1px);
        box-shadow: 0 4px 12px rgba(99, 102, 241, 0.4);
    }
</style>
""", unsafe_allow_html=True)


# =============================================================================
# HELPER FUNCTIONS
# =============================================================================

def create_group_bar_chart(data: Dict[str, float], title: str) -> go.Figure:
    """Create a bar chart for group distribution."""
    groups = list(data.keys())
    values = list(data.values())
    
    colors = ['#6366f1' if v >= 0.1 else '#ef4444' for v in values]
    
    fig = go.Figure(data=[
        go.Bar(
            x=groups,
            y=[v * 100 for v in values],
            marker_color=colors,
            text=[f'{v:.1%}' for v in values],
            textposition='auto'
        )
    ])
    
    fig.update_layout(
        title=title,
        xaxis_title="Group",
        yaxis_title="Percentage (%)",
        template="plotly_dark",
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        font_color='#f8fafc',
        height=350
    )
    
    return fig


def create_outcome_rate_chart(data: Dict[str, float], title: str) -> go.Figure:
    """Create a bar chart for outcome rates by group."""
    groups = list(data.keys())
    values = list(data.values())
    
    fig = go.Figure(data=[
        go.Bar(
            x=groups,
            y=[v * 100 for v in values],
            marker_color='#8b5cf6',
            text=[f'{v:.1%}' for v in values],
            textposition='auto'
        )
    ])
    
    fig.update_layout(
        title=title,
        xaxis_title="Group",
        yaxis_title="Outcome Rate (%)",
        template="plotly_dark",
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        font_color='#f8fafc',
        height=350
    )
    
    return fig


def create_radar_chart(metric_results: Dict[str, Dict[str, Any]]) -> go.Figure:
    """Create radar chart of all metrics."""
    metrics = list(metric_results.keys())
    
    # Normalize scores to 0-1 where higher is better
    scores = []
    for name, result in metric_results.items():
        score = result.get('score', 0.5)
        # Normalize based on metric type
        if name in ['AOI', 'DMI', 'CPA', 'SPA', 'HRS']:
            # Lower is better - invert
            normalized = 1 - min(score, 1.0)
        elif name == 'k_anonymity':
            # Convert k value to 0-1 (higher k is better)
            normalized = min(score / 10, 1.0) if score != float('inf') else 1.0
        else:
            # Higher is better
            normalized = min(score, 1.0)
        scores.append(normalized)
    
    # Create radar chart
    fig = go.Figure()
    
    fig.add_trace(go.Scatterpolar(
        r=scores + [scores[0]],  # Close the polygon
        theta=metrics + [metrics[0]],
        fill='toself',
        fillcolor='rgba(99, 102, 241, 0.3)',
        line_color='#6366f1',
        name='Scores'
    ))
    
    # Add threshold line at 0.7
    fig.add_trace(go.Scatterpolar(
        r=[0.7] * (len(metrics) + 1),
        theta=metrics + [metrics[0]],
        line_color='rgba(234, 179, 8, 0.5)',
        line_dash='dash',
        name='Warning'
    ))
    
    fig.update_layout(
        polar=dict(
            radialaxis=dict(range=[0, 1], showticklabels=False),
            bgcolor='rgba(0,0,0,0)'
        ),
        showlegend=False,
        template="plotly_dark",
        paper_bgcolor='rgba(0,0,0,0)',
        font_color='#f8fafc',
        height=400,
        title="10-Metric Radar Overview"
    )
    
    return fig


def render_metric_card(name: str, result: Dict[str, Any]):
    """Render a metric card with flag and interpretation."""
    score = result.get('score', 'N/A')
    flag = result.get('flag', 'YELLOW')
    interp = result.get('interpretation', 'No interpretation available.')
    
    flag_class = f"flag-{flag.lower()}"
    
    if isinstance(score, float):
        score_display = f"{score:.4f}"
    else:
        score_display = str(score)
    
    st.markdown(f"""
    <div class="metric-card">
        <span class="{flag_class}">{flag}</span>
        <div class="metric-title">{name}</div>
        <div class="metric-value">{score_display}</div>
        <p style="color: #94a3b8; font-size: 0.9rem; margin-top: 12px;">{interp[:200]}{'...' if len(interp) > 200 else ''}</p>
    </div>
    """, unsafe_allow_html=True)


# =============================================================================
# MAIN APPLICATION
# =============================================================================

def main():
    # =================================================================
    # SIDEBAR - Configuration
    # =================================================================
    with st.sidebar:
        st.title("🔬 DET v3")
        st.caption("Dataset Ethics Triage")
        st.markdown("---")
        
        # File upload
        st.header("📂 Data Input")
        uploaded_file = st.file_uploader(
            "Upload CSV Dataset",
            type=['csv'],
            help="Upload a clinical/tabular dataset for ethical assessment"
        )
        
        if uploaded_file:
            # Load data
            try:
                df = pd.read_csv(uploaded_file)
                st.success(f"✓ Loaded {len(df):,} rows, {len(df.columns)} columns")
                st.session_state['df'] = df
            except Exception as e:
                st.error(f"Error loading file: {e}")
                df = None
        else:
            df = st.session_state.get('df', None)
        
        if df is not None:
            st.markdown("---")
            
            # Column selection
            st.header("⚙️ Configuration")
            
            cols = df.columns.tolist()
            
            # Target column
            target_col = st.selectbox(
                "🎯 Target Column",
                cols,
                index=len(cols)-1 if cols else 0,
                help="The outcome/label column to analyze"
            )
            
            # Protected attributes
            sensitive_candidates = ['race', 'gender', 'sex', 'age', 'ethnicity']
            default_protected = [c for c in cols if any(s in c.lower() for s in sensitive_candidates)]
            
            protected_attrs = st.multiselect(
                "🛡️ Protected Attributes",
                cols,
                default=default_protected[:3] if default_protected else [],
                help="Demographic columns to analyze for fairness"
            )
            
            # Quasi-identifiers
            qi_candidates = ['age', 'zip', 'gender', 'race', 'id']
            default_qis = [c for c in cols if any(s in c.lower() for s in qi_candidates)]
            
            quasi_identifiers = st.multiselect(
                "🔐 Quasi-Identifiers",
                cols,
                default=default_qis[:4] if default_qis else [],
                help="Columns that could enable re-identification"
            )
            
            st.markdown("---")
            
            # Harm parameters
            st.header("⚠️ Harm Assessment")
            
            severity = st.select_slider(
                "Clinical Severity",
                options=['low', 'medium', 'high', 'critical'],
                value='medium',
                help="Severity of potential harm from model errors"
            )
            
            vulnerability = st.select_slider(
                "Population Vulnerability", 
                options=['low', 'medium', 'high', 'critical'],
                value='medium',
                help="Vulnerability of affected population"
            )
            
            st.markdown("---")
            
            # Run button
            if st.button("🚀 RUN ETHICS AUDIT", type="primary"):
                st.session_state['run_audit'] = True
                st.session_state['config'] = {
                    'target_col': target_col,
                    'protected_attrs': protected_attrs,
                    'quasi_identifiers': quasi_identifiers,
                    'severity': severity,
                    'vulnerability': vulnerability
                }
        
        st.markdown("---")
        st.caption("DET v3.0 | Pre-training Triage")
    
    # =================================================================
    # MAIN CONTENT
    # =================================================================
    
    if df is None:
        # Landing page
        st.title("🔬 Dataset Ethics Triage v3")
        st.markdown("""
        ### Pre-Training Ethical Assessment for Clinical Datasets
        
        DET v3 analyzes your dataset for **10 ethical dimensions** before model training:
        
        | Core Metrics | Advanced Metrics |
        |-------------|------------------|
        | URS - Underrepresentation Score | FOI - Feature Outcome Disparity |
        | AOI - Attribute-Outcome Imbalance | FPC - Fairness-Performance Convergence |
        | DMI - Differential Missingness Index | CPA - Conditional Proxy Assessment |
        | k-Anonymity Risk | SPA - Sensitive Predictability Analysis |
        | HRS - Harm Risk Score | DAI - Distributional Alignment Index |
        
        **Get started by uploading a CSV dataset in the sidebar →**
        """)
        
        st.info("💡 **Tip**: Select protected attributes like race, gender, age for comprehensive fairness analysis.")
        return
    
    # Check if audit should run
    if not st.session_state.get('run_audit', False):
        st.title("🔬 Dataset Ethics Triage v3")
        st.info("Configure your audit parameters in the sidebar and click **RUN ETHICS AUDIT**")
        
        # Show data preview
        st.subheader("📊 Data Preview")
        st.dataframe(df.head(20), use_container_width=True)
        return
    
    # =================================================================
    # RUN AUDIT
    # =================================================================
    
    config = st.session_state['config']
    target_col = config['target_col']
    protected_attrs = config['protected_attrs']
    quasi_identifiers = config['quasi_identifiers']
    severity = config['severity']
    vulnerability = config['vulnerability']
    
    st.title("🔬 DET v3 Ethics Assessment")
    
    with st.spinner("Computing 10 ethical metrics..."):
        # Compute all metrics
        metric_results = {}
        
        # Core metrics
        metric_results['URS'] = calculate_urs(df, protected_attrs)
        metric_results['AOI'] = calculate_aoi(df, target_col, protected_attrs)
        metric_results['DMI'] = calculate_dmi(df, protected_attrs)
        metric_results['k_anonymity'] = calculate_k_anonymity(df, quasi_identifiers)
        metric_results['HRS'] = calculate_hrs(df, target_col, protected_attrs, severity, vulnerability)
        
        # Advanced metrics
        metric_results['FOI'] = calculate_foi(df, target_col, protected_attrs)
        metric_results['FPC'] = calculate_fpc(df, target_col, protected_attrs)
        metric_results['CPA'] = calculate_cpa(df, protected_attrs)
        metric_results['SPA'] = calculate_spa(df, protected_attrs)
        metric_results['DAI'] = calculate_dai(df, protected_attrs)
        
        # Make decision
        engine = DETDecisionEngine()
        decision_result = engine.make_decision(metric_results)
    
    # =================================================================
    # DISPLAY RESULTS
    # =================================================================
    
    # Decision Banner
    decision = decision_result['decision']
    decision_class = f"decision-{decision.lower()}"
    
    if decision == 'PROCEED':
        decision_color = '#22c55e'
        decision_icon = '✅'
    elif decision == 'MITIGATE':
        decision_color = '#eab308'
        decision_icon = '⚠️'
    else:
        decision_color = '#ef4444'
        decision_icon = '⛔'
    
    st.markdown(f"""
    <div class="{decision_class}">
        <div class="decision-text" style="color: {decision_color};">{decision_icon} {decision}</div>
        <p style="color: #f8fafc; font-size: 1.1rem;">{decision_result['rationale']}</p>
        <p style="color: #94a3b8;">Confidence: {decision_result['confidence']:.0%}</p>
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown("---")
    
    # Metrics Grid
    st.header("📊 Metric Analysis")
    
    cols = st.columns(2)
    metric_names = list(metric_results.keys())
    
    for i, name in enumerate(metric_names):
        with cols[i % 2]:
            render_metric_card(name, metric_results[name])
            with st.expander(f"Details: {name}"):
                st.json(metric_results[name].get('details', {}))
    
    st.markdown("---")
    
    # Visualizations
    st.header("📈 Visual Dashboard")
    
    col1, col2 = st.columns(2)
    
    with col1:
        # Radar chart
        radar = create_radar_chart(metric_results)
        st.plotly_chart(radar, use_container_width=True)
    
    with col2:
        # Group distribution (from URS)
        urs_details = metric_results['URS'].get('details', {})
        proportions = urs_details.get('proportions_by_attribute', {})
        
        if proportions and protected_attrs:
            first_attr = protected_attrs[0] if protected_attrs else None
            if first_attr and first_attr in proportions:
                bar = create_group_bar_chart(
                    proportions[first_attr],
                    f"Group Representation: {first_attr}"
                )
                st.plotly_chart(bar, use_container_width=True)
    
    # Outcome rates chart
    aoi_details = metric_results['AOI'].get('details', {})
    outcome_rates = aoi_details.get('outcome_rates_by_attribute', {})
    
    if outcome_rates and protected_attrs:
        first_attr = protected_attrs[0] if protected_attrs else None
        if first_attr and first_attr in outcome_rates:
            outcome_bar = create_outcome_rate_chart(
                outcome_rates[first_attr],
                f"Outcome Rates by {first_attr}"
            )
            st.plotly_chart(outcome_bar, use_container_width=True)
    
    st.markdown("---")
    
    # Action Items
    st.header("📋 Action Items")
    
    for action in decision_result['action_items']:
        st.markdown(f"- {action}")
    
    # Download memo
    st.markdown("---")
    st.header("📄 Decision Memo")
    
    col1, col2 = st.columns([3, 1])
    
    with col1:
        with st.expander("Preview Memo", expanded=False):
            st.markdown(decision_result['memo'])
    
    with col2:
        st.download_button(
            label="⬇️ Download Memo",
            data=decision_result['memo'],
            file_name=f"DET_Audit_{decision}.md",
            mime="text/markdown"
        )
    
    # Reset button
    st.markdown("---")
    if st.button("🔄 Run New Audit"):
        st.session_state['run_audit'] = False
        st.rerun()


# =============================================================================
# ENTRY POINT
# =============================================================================

if __name__ == "__main__":
    main()
