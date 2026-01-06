import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from src.data_manager import DataManager
from src.audit_engine import ModelAuditor
from src.ui.det_ui import render_det_page

# --- 1. GLOBAL CONFIGURATION ---
st.set_page_config(
    page_title="EquiScan DET v3",
    page_icon="📋",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Premium Pro Dark: Minimal & Sophisticated
st.markdown("""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');
    
    :root {
        --bg: #0f172a;           /* Deep Slate */
        --card: #1e293b;         /* Slate 800 */
        --border: #334155;       /* Slate 700 */
        --primary: #6366f1;      /* Indigo */
        --text: #f8fafc;
        --text-dim: #94a3b8;
    }

    * { font-family: 'Inter', sans-serif; }
    
    .stApp { background-color: var(--bg); color: var(--text); }

    /* Clean Pro Cards */
    .clinical-card {
        background: var(--card);
        border-radius: 12px;
        padding: 24px;
        border: 1px solid var(--border);
        margin-bottom: 20px;
        color: var(--text);
    }
    
    /* Metrics Refinement */
    [data-testid="stMetricValue"] {
        font-size: 2.2rem;
        font-weight: 700;
        color: var(--primary);
    }
    
    [data-testid="stMetricLabel"] {
        color: var(--text-dim);
        font-weight: 600;
        text-transform: uppercase;
        font-size: 0.7rem;
        letter-spacing: 1px;
    }
    
    .stMetric {
        background: var(--card);
        padding: 20px;
        border-radius: 12px;
        border: 1px solid var(--border);
    }
    
    /* Pro Button */
    div.stButton > button { 
        width: 100%; 
        border-radius: 8px; 
        height: 3rem;
        font-weight: 600;
        background-color: var(--primary);
        color: white;
        border: none;
        transition: filter 0.2s;
    }
    
    div.stButton > button:hover { filter: brightness(1.1); }

    /* Sidebar Refinement */
    [data-testid="stSidebar"] {
        background-color: #020617;
        border-right: 1px solid var(--border);
    }
    
    .stAlert { border-radius: 12px; background: #1e293b; border: 1px solid var(--border); color: var(--text); }

    hr { border-top: 1px solid var(--border); margin: 2rem 0; }
    </style>
    """, unsafe_allow_html=True)

# --- 2. STATE MANAGEMENT ---
if 'det_results' not in st.session_state:
    st.session_state.det_results = None
if 'df' not in st.session_state:
    st.session_state.df = None

def main():
    # --- 3. SIDEBAR ORCHESTRATION ---
    with st.sidebar:
        st.title("📋 DET v3 Auditor")
        st.caption("Strategic Dataset Ethical Triage")
        st.markdown("---")
        
        st.header("📂 Data Ingestion")
        uploaded_file = st.file_uploader("Upload Medical Dataset (CSV/XLSX)", type=["csv", "xlsx"])
        
        if uploaded_file:
            # Reset results if a new file is uploaded
            if st.session_state.get('last_uploaded_name') != uploaded_file.name:
                st.session_state.det_results = None
                st.session_state.last_uploaded_name = uploaded_file.name

            df = DataManager.load_data(uploaded_file, uploaded_file.name)
            if df is not None:
                st.session_state.df = df
                st.success(f"Loaded {len(df)} records")
                
                st.markdown("---")
                st.header("⚙️ Column Mapping")
                cols = df.columns.tolist()
                
                # Auto-detect defaults
                default_target = df.columns[-1]
                target_col = st.selectbox("🎯 Target Outcome", cols, index=list(cols).index(default_target))
                
                sensitive_candidates = ['sex', 'gender', 'race', 'age', 'ethnicity']
                default_sensitive = next((c for c in cols if any(s in c.lower() for s in sensitive_candidates)), cols[0])
                sensitive_col = st.selectbox("🛡️ Sensitive Attribute", cols, index=cols.index(default_sensitive))
                
                st.session_state.target_col = target_col
                st.session_state.sensitive_col = sensitive_col
                
                # Global Action: PDF Export (only if results exist)
                if st.session_state.get('det_results'):
                    st.markdown("---")
                    st.header("📄 Export")
                    if st.button("🖨️ Generate DET Audit Report"):
                        with st.spinner("Generating Professional DET Report..."):
                            try:
                                from src.report_engine import ReportGenerator
                                
                                results = st.session_state.get('det_results')
                                decision = results['decision_result']
                                metrics = results['metric_results']
                                viz = results['visualizations']
                                
                                pdf = ReportGenerator()
                                pdf.header()
                                
                                # 1. Summary
                                pdf.add_det_summary(decision)
                                
                                # 2. Metrics Table
                                pdf.add_section_title("Detailed Metric Analysis")
                                for m_name, m_res in metrics.items():
                                    pdf.add_metric_row(m_name, m_res['score'], m_res['flag'])
                                pdf.ln(10)
                                
                                # 3. Visualizations
                                pdf.add_plot(viz['radar_chart'], "10-Metric Radar Chart")
                                pdf.add_plot(viz['decision_matrix'], "Decision Matrix")
                                
                                # 4. Decision Memo
                                pdf.add_decision_memo(decision['memo'])
                                
                                pdf_bytes = pdf.save_report()
                                
                                st.download_button(
                                    label="⬇️ Download PDF Report",
                                    data=pdf_bytes,
                                    file_name=f"DET_Audit_{st.session_state.target_col}.pdf",
                                    mime="application/pdf"
                                )
                                st.success("PDF Generated Successfully!")
                                
                            except Exception as e:
                                st.error(f"Report failure: {e}")
        
        st.markdown("---")
        st.markdown("""
        **Framework Version**: 3.0.0  
        **Metrics**: 10 (5 Core / 5 Adv)  
        **Mode**: Pre-training Triage
        """)

    # --- 4. MAIN LAYOUT ---
    if st.session_state.df is not None:
        # Create Auditor
        auditor = ModelAuditor(
            st.session_state.df, 
            st.session_state.target_col, 
            st.session_state.sensitive_col
        )
        
        # Pure DET Workflow
        render_det_page(st.session_state.df, auditor)
        
    else:
        # Modern Landing Page
        st.title("Welcome to EquiScan DET v3")
        
        col1, col2 = st.columns([2, 1])
        
        with col1:
            st.markdown("""
            ### The Gold Standard for Clinical Dataset Triage
            
            EquiScan DET v3 implements a rigorous **10-metric ethical assessment** designed specifically for high-stakes medical datasets. 
            
            **What we analyze:**
            *   **🔍 Underrepresentation (URS)**: Detecting hidden data silences.
            *   **⚖️ Outcome Imbalance (AOI)**: Measuring historical disparities.
            *   **🧩 Proxy Leakage (CPA/SPA)**: Identifying feature-level proxies.
            *   **🔐 Privacy Risk (k-Anon)**: Quantifying re-identification thresholds.
            *   **⚠️ Harm Exposure (HRS)**: Evaluating downstream clinical impact.
            
            **Get started by uploading your dataset in the sidebar.**
            """)
            
            st.info("💡 **Tip**: For medical audits, ensure you define your Quasi-Identifiers (like Age or ZIP) to measure privacy accurately.")
            
        with col2:
            st.image("https://img.icons8.com/illustrations/external-faticon-flat-faticon/256/external-algorithm-artificial-intelligence-faticon-flat-faticon.png")

if __name__ == "__main__":
    main()