
import streamlit as st
import pandas as pd
from src.data_manager import DataManager
from src.ui.health_tab import render_data_health_tab
from src.ui.bias_tab import render_bias_tab
from src.ui.audit_tab import render_audit_tab
from src.report_engine import ReportGenerator

# --- 1. GLOBAL CONFIGURATION ---
st.set_page_config(
    page_title="EquiScan Pro",
    page_icon="🧬",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for "Card-like" styling and metrics
st.markdown("""
    <style>
    .main { background-color: #f8f9fa; }
    div.stButton > button { width: 100%; border-radius: 8px; }
    .metric-card { 
        background-color: white; 
        padding: 20px; 
        border-radius: 12px; 
        box-shadow: 0 4px 6px rgba(0,0,0,0.1); 
        margin-bottom: 20px;
    }
    h1, h2, h3 { font-family: 'Inter', sans-serif; color: #2c3e50; }
    .stAlert { border-radius: 10px; }
    </style>
    """, unsafe_allow_html=True)

# --- 2. STATE MANAGEMENT ---
if 'model_trained' not in st.session_state:
    st.session_state.model_trained = False
if 'audit_results' not in st.session_state:
    st.session_state.audit_results = {}
if 'persona' not in st.session_state:
    st.session_state.persona = "Student"

def main():
    # --- 3. SIDEBAR ORCHESTRATION ---
    with st.sidebar:
        st.title("🧬 EquiScan Pro")
        st.caption("Senior-Level Algorithmic Auditor")
        
        # Persona Selection
        st.header("👤 Persona")
        persona = st.radio(
            "Select Role:", 
            ["Student", "Domain Expert", "Analyst"],
            index=0 if st.session_state.persona == "Student" else 1
        )
        st.session_state.persona = persona
        
        if persona == "Student":
            st.info("🎓 **Student Mode**: Simplified explanations and educational tips.")
        else:
            st.success("🔬 **Expert Mode**: Advanced metrics and deep forensics.")
            
        st.markdown("---")
        
        st.header("📂 Data Ingestion")
        uploaded_file = st.file_uploader("Upload Dataset (CSV/XLSX)", type=["csv", "xlsx"])
        
        # --- REPORT GENERATION (Global) ---
        if st.session_state.get('model_trained') or st.session_state.get('audit_results'):
            st.markdown("---")
            st.header("📄 Export")
            if st.button("🖨️ Generate PDF Report"):
                with st.spinner("Compiling PDF Report..."):
                    try:
                        # Instantiate Auditors
                        from src.quality_engine import QualityAuditor
                        from src.bias_engine import BiasAuditor
                        from src.audit_engine import ModelAuditor
                        
                        df = st.session_state.df
                        target = st.session_state.target_col
                        sensitive = st.session_state.sensitive_col
                        
                        q_auditor = QualityAuditor(df, target)
                        b_auditor = BiasAuditor(df, sensitive, target)
                        # We don't train a new model for PDF, just check if metrics exist
                        
                        # Setup Report
                        pdf = ReportGenerator()
                        pdf.header()
                        
                        pdf.add_text(f"Audit Date: {pd.Timestamp.now()}")
                        pdf.add_text(f"Target Variable: {target}")
                        pdf.add_text(f"Sensitive Attribute: {sensitive}")
                        
                        # 1. Data Health
                        pdf.add_section_title("1. Data Health & Representation")
                        _, bal_fig, bal_status = q_auditor.check_class_balance()
                        pdf.add_metric("Class Balance Status", bal_status)
                        pdf.add_plot(bal_fig, "Target Distribution")
                        
                        miss_df, miss_fig = q_auditor.check_missing_values(sensitive)
                        pdf.add_plot(miss_fig, "Missing Values Pattern")
                        
                        # 2. Bias
                        pdf.add_section_title("2. Bias & Fairness")
                        user_pos = st.session_state.get("bias_positive_outcome", None)
                        dir_score, dir_fig = b_auditor.get_disparate_impact_ratio(positive_outcome=str(user_pos) if user_pos else None)
                        pdf.add_metric("Disparate Impact Ratio", f"{dir_score:.2f}")
                        pdf.add_plot(dir_fig, "Selection Rates by Group")
                        
                        if 'bias_overfitting' in st.session_state:
                             pdf.add_metric("Fairness Overfitting", st.session_state.bias_overfitting)

                        # 3. Model
                        pdf.add_section_title("3. Model Forensics")
                        if 'audit_metrics' in st.session_state:
                            m = st.session_state.audit_metrics
                            pdf.add_metric("Train Accuracy", f"{m['train_accuracy']:.1%}")
                            pdf.add_metric("Test Accuracy", f"{m['test_accuracy']:.1%}")
                            pdf.add_metric("Overfitting Detected", "YES" if m['is_overfitting'] else "NO")
                        else:
                            pdf.add_text("Model metrics not available.")
                            
                        # Save
                        pdf_bytes = pdf.save_report()
                        
                        st.download_button(
                            label="⬇️ Download PDF",
                            data=pdf_bytes,
                            file_name="equiscan_audit_report.pdf",
                            mime="application/pdf"
                        )
                        st.success("PDF Ready!")
                        
                    except Exception as e:
                        st.error(f"Failed to generate PDF: {e}")

    # --- 4. MAIN LAYOUT ---
    if uploaded_file:
        df = DataManager.load_data(uploaded_file, uploaded_file.name)
        
        if df is not None:
            # Column Configuration (Top of Main or Sidebar? Prompt implies Sidebar)
            # Keeping it in sidebar for consistency with previous user preference, 
            # but usually config is better in sidebar.
            with st.sidebar:
                st.markdown("---")
                st.header("⚙️ Column Mapping")
                cols = df.columns.tolist()
                
                # Auto-detect defaults
                default_target = df.columns[-1]
                target_col = st.selectbox("🎯 Target Variable (Label)", cols, index=list(cols).index(default_target))
                
                sensitive_candidates = ['sex', 'gender', 'race', 'age', 'ethnicity']
                default_sensitive = next((c for c in cols if any(s in c.lower() for s in sensitive_candidates)), cols[0])
                sensitive_col = st.selectbox("🛡️ Sensitive Attribute", cols, index=cols.index(default_sensitive))
                
                st.session_state.df = df
                st.session_state.target_col = target_col
                st.session_state.sensitive_col = sensitive_col

            st.title(f"Audit Dashboard ({persona} View)")

            # CUSTOM TABS (State-Controlled)
            # We use st.radio to allow "Next" buttons to switch tabs
            if 'active_tab' not in st.session_state:
                st.session_state.active_tab = "🧬 Data DNA"

            tabs_map = ["🧬 Data DNA", "⚖️ Fairness & Bias", "🔎 Model Forensics"]
            
            # Render as a horizontal radio (looks like tabs)
            active = st.radio("", tabs_map, 
                              index=tabs_map.index(st.session_state.active_tab) if st.session_state.active_tab in tabs_map else 0, 
                              horizontal=True,
                              label_visibility="collapsed")
            
            # Sync state if user clicks directly
            if active != st.session_state.active_tab:
                st.session_state.active_tab = active
                st.rerun()

            # --- RENDER TAB CONTENT ---
            
            if st.session_state.active_tab == "🧬 Data DNA":
                render_data_health_tab(df, target_col, sensitive_col, persona)
                
                # Footer Buttons
                st.markdown("---")
                _, col_next = st.columns([4, 1])
                if col_next.button("Next: Bias Check ➡️"):
                    st.session_state.active_tab = "⚖️ Fairness & Bias"
                    st.rerun()
            
            elif st.session_state.active_tab == "⚖️ Fairness & Bias":
                render_bias_tab(df, sensitive_col, target_col, persona)
                
                # Footer Buttons
                st.markdown("---")
                col_prev, _, col_next = st.columns([1, 3, 1])
                if col_prev.button("⬅️ Back"):
                    st.session_state.active_tab = "🧬 Data DNA"
                    st.rerun()
                if col_next.button("Next: Model Forensics ➡️"):
                    st.session_state.active_tab = "🔎 Model Forensics"
                    st.rerun()
                
            elif st.session_state.active_tab == "🔎 Model Forensics":
                render_audit_tab(df, target_col, sensitive_col, persona)
                
                # Footer Buttons
                st.markdown("---")
                col_prev, _ = st.columns([1, 4])
                if col_prev.button("⬅️ Back"):
                    st.session_state.active_tab = "⚖️ Fairness & Bias"
                    st.rerun()


        else:
            st.error("Failed to load data.")
    else:
        st.title("Welcome to EquiScan Pro")
        st.markdown("""
        ### Please upload a dataset to begin.
        
        **Select your Persona in the sidebar:**
        *   **Student**: Learn about fairness concepts.
        *   **Domain Expert**: Analyze data for bias and proxies.
        *   **Analyst**: Train models and generate audit reports.
        """)

if __name__ == "__main__":
    main()