"""
DET (Dataset Ethical Triage) v3 - Streamlit UI

User interface for running DET assessments and viewing results.
"""

import streamlit as st
import pandas as pd
from typing import List


def render_det_page(df: pd.DataFrame, auditor):
    """
    Render a clean, simplified DET assessment page in Pro Dark mode.
    """
    # Simple CSS Overrides for Dark Mode
    st.markdown("""
        <style>
        .metric-card {
            background: #1e293b;
            border: 1px solid #334155;
            border-radius: 12px;
            padding: 24px;
            margin-bottom: 20px;
        }
        .flag-pill {
            padding: 4px 12px;
            border-radius: 6px;
            font-size: 0.7rem;
            font-weight: 700;
            text-transform: uppercase;
            display: inline-block;
            margin-bottom: 12px;
            letter-spacing: 0.5px;
        }
        .flag-green { background: rgba(34, 197, 94, 0.15); color: #4ade80; border: 1px solid rgba(74, 222, 128, 0.2); }
        .flag-yellow { background: rgba(234, 179, 8, 0.15); color: #facc15; border: 1px solid rgba(250, 204, 21, 0.2); }
        .flag-red { background: rgba(239, 68, 68, 0.15); color: #f87171; border: 1px solid rgba(248, 113, 113, 0.2); }
        
        .metric-title { font-size: 1.2rem; font-weight: 600; color: #f8fafc; margin-bottom: 8px; }
        .metric-score { font-size: 2rem; font-weight: 800; color: #6366f1; margin-bottom: 12px; }
        .metric-desc { font-size: 0.95rem; color: #94a3b8; line-height: 1.5; }
        
        .stExpander { background: #0f172a !important; border: 1px solid #334155 !important; border-radius: 12px !important; }
        .streamlit-expanderHeader { color: #f8fafc !important; font-weight: 600 !important; }
        </style>
    """, unsafe_allow_html=True)

    st.markdown("<h1 style='color: #f8fafc; font-weight: 800; letter-spacing: -1px;'>Strategic Dataset Ethical Triage</h1>", unsafe_allow_html=True)
    st.markdown("<p style='color: #94a3b8; font-size: 1.1rem;'>Professional 10-metric ethical assessment engine.</p>", unsafe_allow_html=True)
    
    # 1. Configuration
    with st.expander("🛠️ AUDIT PARAMETERS", expanded=st.session_state.get('det_results') is None):
        c1, c2 = st.columns(2)
        with c1:
            all_cols = df.columns.tolist()
            suggested_qis = [col for col in all_cols if any(keyword in col.lower() 
                            for keyword in ['age', 'zip', 'gender', 'race', 'location', 'id'])]
            quasi_ids = st.multiselect("Active Quasi-Identifiers:", options=all_cols, 
                                     default=suggested_qis[:3] if suggested_qis else [], key="qi_select")
        with c2:
            harm_severity = st.select_slider("Assessed Harm Severity", options=[0.2, 0.4, 0.6, 0.8, 1.0], value=0.6)
            harm_vulnerability = st.select_slider("Population Vulnerability", options=[0.3, 0.5, 0.7, 1.0], value=0.5)

        if st.button("EXECUTE AUDIT", type="primary", use_container_width=True):
            with st.spinner("Analyzing dataset ethics..."):
                try:
                    results = auditor.run_det_audit(quasi_identifiers=quasi_ids if quasi_ids else None,
                                                  harm_severity=harm_severity,
                                                  harm_vulnerability=harm_vulnerability)
                    st.session_state['det_results'] = results
                    st.rerun()
                except Exception as e:
                    st.error(f"Audit failure: {e}")

    # 2. Results
    if st.session_state.get('det_results') is not None:
        res = st.session_state['det_results']
        metrics = res['metric_results']
        decision = res['decision_result']
        viz = res['visualizations']

        st.markdown("<hr style='border-top: 1px solid #334155; margin: 3rem 0;'>", unsafe_allow_html=True)
        
        # Phase 1: Metrics
        st.markdown("<h2 style='color: #f8fafc; font-weight: 700;'>🔬 Analysis Insights</h2>", unsafe_allow_html=True)
        m_list = list(metrics.items())
        for i in range(0, len(m_list), 2):
            cols = st.columns(2)
            for j in range(2):
                if i + j < len(m_list):
                    name, data = m_list[i + j]
                    flag = data['flag']
                    f_cls = f"flag-{flag.lower()}"
                    with cols[j]:
                        st.markdown(f"""
                            <div class="metric-card">
                                <span class="flag-pill {f_cls}">{flag} STATUS</span>
                                <div class="metric-title">{name}</div>
                                <div class="metric-score">{data['score']:.3f}</div>
                                <div class="metric-desc">{data['interpretation'][:120]}...</div>
                            </div>
                        """, unsafe_allow_html=True)
                        with st.expander(f"Detailed trace: {name}", expanded=True):
                            st.write(data['interpretation'])
                            if 'visualization' in data:
                                st.plotly_chart(data['visualization'], use_container_width=True)

        st.divider()

        # Phase 2: Visualizations
        st.header("Visual Dashboard")
        v1, v2 = st.columns(2)
        with v1: st.plotly_chart(viz['radar_chart'], use_container_width=True)
        with v2: st.plotly_chart(viz['decision_matrix'], use_container_width=True)
        with st.expander("Decision Flow Analysis", expanded=True):
            st.plotly_chart(viz['decision_flow'], use_container_width=True)

        st.divider()

        # Phase 3: Final Verdict
        st.header("Final Triage Verdict")
        v_status = decision['decision']
        if v_status == 'PROCEED': st.success(f"### Result: {v_status}")
        elif v_status == 'MITIGATE': st.warning(f"### Result: {v_status}")
        else: st.error(f"### Result: {v_status}")
        
        st.info(f"**Rationale**: {decision['rationale']}")
        
        c1, c2 = st.columns(2)
        with c1:
            st.subheader("Action Strategy")
            for action in decision['action_items']:
                st.markdown(f"- {action}")
        with c2:
            st.subheader("Audit Memo")
            st.download_button("Download Report (.md)", decision['memo'], 
                               file_name=f"DET_Audit_{v_status}.md", use_container_width=True)
            with st.expander("Preview Memo", expanded=True):
                st.markdown(decision['memo'])


