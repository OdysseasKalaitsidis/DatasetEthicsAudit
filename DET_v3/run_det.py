"""DET v3 Streamlit App - Clean, modular implementation."""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from typing import Dict, Any, List

from det import *

# Page config
st.set_page_config(page_title="DET v3", page_icon="🔬", layout="wide")

# Styling
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;600;700&display=swap');
    * { font-family: 'Inter', sans-serif; }
    .stApp { background: linear-gradient(135deg, #0f172a, #1e1b4b); }
    .metric-card { background: #1e293b; border-radius: 12px; padding: 20px; margin: 8px 0; border: 1px solid #334155; }
    .flag-green { background: rgba(34,197,94,0.15); color: #4ade80; padding: 4px 12px; border-radius: 6px; font-weight: 600; }
    .flag-yellow { background: rgba(234,179,8,0.15); color: #facc15; padding: 4px 12px; border-radius: 6px; font-weight: 600; }
    .flag-red { background: rgba(239,68,68,0.15); color: #f87171; padding: 4px 12px; border-radius: 6px; font-weight: 600; }
    .decision-box { border-radius: 16px; padding: 24px; text-align: center; margin: 20px 0; }
    .no-bias { background: linear-gradient(135deg, rgba(34,197,94,0.2), rgba(34,197,94,0.1)); border: 2px solid #22c55e; }
    .moderate-bias { background: linear-gradient(135deg, rgba(234,179,8,0.2), rgba(234,179,8,0.1)); border: 2px solid #eab308; }
    .significant-bias { background: linear-gradient(135deg, rgba(239,68,68,0.2), rgba(239,68,68,0.1)); border: 2px solid #ef4444; }
</style>
""", unsafe_allow_html=True)


def create_radar(metrics: Dict[str, Dict]) -> go.Figure:
    names = list(metrics.keys())
    scores = []
    for n, r in metrics.items():
        s = r.get('score', 0.5)
        if n in ['AOI', 'DMI', 'CPA', 'SPA', 'HRS']:
            scores.append(1 - min(s, 1.0))
        elif n == 'k_anonymity':
            scores.append(min(s / 10, 1.0) if s != float('inf') else 1.0)
        else:
            scores.append(min(s, 1.0))
    
    fig = go.Figure(go.Scatterpolar(r=scores + [scores[0]], theta=names + [names[0]], 
                                     fill='toself', fillcolor='rgba(99,102,241,0.3)', line_color='#6366f1'))
    fig.update_layout(polar=dict(radialaxis=dict(range=[0,1], showticklabels=False), bgcolor='rgba(0,0,0,0)'),
                      showlegend=False, template="plotly_dark", paper_bgcolor='rgba(0,0,0,0)', 
                      font_color='#f8fafc', height=400, title="Metrics Overview")
    return fig


def create_metric_chart(name: str, result: Dict, df: pd.DataFrame = None, protected: List[str] = None) -> go.Figure:
    chart_creators = {
        'URS': lambda: bar_chart(result.get('details', {}).get('proportions_by_attribute', {}).get(protected[0] if protected else '', {}), "Group Representation"),
        'AOI': lambda: bar_chart(result.get('details', {}).get('outcome_rates_by_attribute', {}).get(protected[0] if protected else '', {}), "Outcome Rates"),
        'DAI': lambda: bar_chart(result.get('details', {}).get('dai_by_attribute', {}).get(protected[0] if protected else '', {}).get('distribution', {}), "Distribution"),
        'k_anonymity': lambda: gauge(result.get('details', {}).get('min_k', 1), 10, "k-Anonymity"),
        'HRS': lambda: gauge(result.get('score', 0), 1, "Harm Risk"),
        'SPA': lambda: gauge(result.get('score', 0.5), 1, "Predictability"),
        'CPA': lambda: gauge(result.get('score', 0), 1, "Proxy Risk"),
        'FPC': lambda: gauge(result.get('score', 1), 1, "Fairness-Perf"),
        'FOI': lambda: gauge(result.get('score', 1), 1, "Feature-Outcome"),
        'DMI': lambda: gauge(result.get('score', 0), 1, "Missingness Gap"),
    }
    try:
        return chart_creators.get(name, lambda: None)()
    except:
        return None


def bar_chart(data: Dict, title: str) -> go.Figure:
    if not data:
        return None
    fig = go.Figure(go.Bar(x=list(data.keys()), y=[v*100 if isinstance(v, float) and v <= 1 else v for v in data.values()], 
                           marker_color='#6366f1'))
    fig.update_layout(title=title, template="plotly_dark", paper_bgcolor='rgba(0,0,0,0)', 
                      plot_bgcolor='rgba(0,0,0,0)', font_color='#f8fafc', height=250)
    return fig


def gauge(value: float, max_val: float, title: str) -> go.Figure:
    fig = go.Figure(go.Indicator(mode="gauge+number", value=value,
        gauge={'axis': {'range': [0, max_val]}, 'bar': {'color': '#6366f1'},
               'steps': [{'range': [0, max_val*0.3], 'color': '#22c55e'},
                        {'range': [max_val*0.3, max_val*0.7], 'color': '#facc15'},
                        {'range': [max_val*0.7, max_val], 'color': '#ef4444'}]},
        title={'text': title}))
    fig.update_layout(template="plotly_dark", paper_bgcolor='rgba(0,0,0,0)', font_color='#f8fafc', height=200)
    return fig


def main():
    with st.sidebar:
        st.title("🔬 DET v3")
        st.caption("Dataset Ethics Triage")
        st.divider()
        
        uploaded = st.file_uploader("Upload CSV", type=['csv'])
        df = None
        
        if uploaded:
            df = pd.read_csv(uploaded)
            st.success(f"✓ {len(df):,} rows")
            st.session_state['df'] = df
        elif 'df' in st.session_state:
            df = st.session_state['df']
        
        if df is not None:
            st.divider()
            cols = df.columns.tolist()
            target = st.selectbox("🎯 Target", cols, index=len(cols)-1)
            
            sens = ['race', 'gender', 'sex', 'age', 'ethnicity']
            defaults = [c for c in cols if any(s in c.lower() for s in sens)]
            protected = st.multiselect("🛡️ Protected Attrs", cols, default=defaults[:3])
            
            qi_hints = ['age', 'zip', 'gender', 'race']
            qis = st.multiselect("🔐 Quasi-IDs", cols, default=[c for c in cols if any(h in c.lower() for h in qi_hints)][:4])
            
            st.divider()
            severity = st.select_slider("Severity", ['low', 'medium', 'high', 'critical'], value='medium')
            vulnerability = st.select_slider("Vulnerability", ['low', 'medium', 'high', 'critical'], value='medium')
            
            st.divider()
            if st.button("🚀 RUN AUDIT", type="primary"):
                st.session_state['config'] = {'target': target, 'protected': protected, 'qis': qis, 
                                               'severity': severity, 'vulnerability': vulnerability}
                st.session_state['run'] = True
    
    if df is None:
        st.title("🔬 Dataset Ethics Triage v3")
        st.info("Upload a CSV to begin")
        return
    
    if not st.session_state.get('run'):
        st.title("🔬 DET v3")
        st.info("Configure and click RUN AUDIT")
        st.dataframe(df.head(15))
        return
    
    cfg = st.session_state['config']
    st.title("🔬 Ethics Assessment")
    
    with st.spinner("Analyzing..."):
        metrics = {
            'URS': calculate_urs(df, cfg['protected']),
            'AOI': calculate_aoi(df, cfg['target'], cfg['protected']),
            'DMI': calculate_dmi(df, cfg['protected']),
            'k_anonymity': calculate_k_anonymity(df, cfg['qis']),
            'HRS': calculate_hrs(df, cfg['target'], cfg['protected'], cfg['severity'], cfg['vulnerability']),
            'FOI': calculate_foi(df, cfg['target'], cfg['protected']),
            'FPC': calculate_fpc(df, cfg['target'], cfg['protected']),
            'CPA': calculate_cpa(df, cfg['protected']),
            'SPA': calculate_spa(df, cfg['protected']),
            'DAI': calculate_dai(df, cfg['protected'])
        }
        result = make_decision(metrics)
    
    # Decision banner
    d = result['decision']
    colors = {'NO_BIAS': '#22c55e', 'MODERATE_BIAS': '#eab308', 'SIGNIFICANT_BIAS': '#ef4444'}
    icons = {'NO_BIAS': '🟢', 'MODERATE_BIAS': '🟡', 'SIGNIFICANT_BIAS': '🔴'}
    css_class = d.lower().replace('_', '-')
    
    st.markdown(f"""
    <div class="decision-box {css_class}">
        <h1 style="color:{colors[d]}; margin:0;">{icons[d]} {d.replace('_', ' ')}</h1>
        <p style="color:#f8fafc; margin:10px 0;">{result['rationale']}</p>
        <p style="color:#94a3b8;">Confidence: {result['confidence']:.0%}</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Metrics tabs
    st.divider()
    core_tab, adv_tab = st.tabs(["🛡️ Core Metrics", "🔬 Advanced Metrics"])
    
    core = ['URS', 'AOI', 'DMI', 'k_anonymity', 'HRS']
    advanced = ['FOI', 'FPC', 'CPA', 'SPA', 'DAI']
    
    for tab, metric_list in [(core_tab, core), (adv_tab, advanced)]:
        with tab:
            for name in metric_list:
                r = metrics[name]
                col1, col2 = st.columns([1, 1])
                with col1:
                    flag = r.get('flag', 'YELLOW')
                    score = r.get('score', 'N/A')
                    score_str = f"{score:.3f}" if isinstance(score, (int, float)) else str(score)
                    interp = r.get('interpretation', '')[:150]
                    st.markdown(f"""
                    <div class="metric-card">
                        <span class="flag-{flag.lower()}">{flag}</span>
                        <h3 style="color:#f8fafc; margin:8px 0;">{name}</h3>
                        <h2 style="color:#6366f1; margin:0;">{score_str}</h2>
                        <p style="color:#94a3b8; font-size:0.9em;">{interp}</p>
                    </div>
                    """, unsafe_allow_html=True)
                with col2:
                    chart = create_metric_chart(name, r, df, cfg['protected'])
                    if chart:
                        st.plotly_chart(chart, use_container_width=True)
    
    # Summary
    st.divider()
    col1, col2 = st.columns(2)
    with col1:
        st.plotly_chart(create_radar(metrics), use_container_width=True)
    with col2:
        st.markdown("### Summary")
        fc = result['flag_counts']
        st.markdown(f"🟢 **{fc['green']}** GREEN | 🟡 **{fc['yellow']}** YELLOW | 🔴 **{fc['red']}** RED")
        st.divider()
        st.markdown("### Actions")
        for a in result['action_items'][:5]:
            st.markdown(f"- {a}")
    
    # Download
    st.divider()
    st.download_button("📄 Download Memo", result['memo'], f"DET_{d}.md", "text/markdown")
    
    if st.button("🔄 New Audit"):
        st.session_state['run'] = False
        st.rerun()


if __name__ == "__main__":
    main()
