
import streamlit as st
import pandas as pd
import plotly.express as px
from src.bias_engine import BiasAuditor

def render_bias_tab(df: pd.DataFrame, sensitive_col: str, target_col: str, persona: str = "Student"):
    """Renders the Bias Audit Tab (Fairness & Proxies)."""
    st.header("⚖️ Fairness & Bias Intelligence")
    
    auditor = BiasAuditor(df, sensitive_col, target_col)
    
    # --- 1. CONFIGURATION ---
    # Allow user to pick the "Privileged" or "Positive" outcome
    unique_outcomes = sorted(df[target_col].unique())
    default_idx = len(unique_outcomes) - 1
    
    c_sel1, c_sel2 = st.columns(2)
    with c_sel1:
        positive_outcome = st.selectbox(
            f"Select Positive Outcome (for {target_col})", 
            unique_outcomes, 
            index=default_idx,
            key="bias_positive_outcome",
            help="The outcome considered 'Beneficial' (e.g., Hiring someone, Approving a loan)."
        )
    
    # Calculate Disparate Impact
    dir_score, dir_fig = auditor.get_disparate_impact_ratio(positive_outcome=str(positive_outcome))
    is_fair = 0.8 <= dir_score <= 1.25

    # PERSISTENCE FOR RIBBON
    st.session_state.bias_score = dir_score
    st.session_state.bias_status = "Fair" if is_fair else "Biased"

    # --- 2. HERO METRIC: DISPARATE IMPACT ---
    st.markdown("### 📊 Disparate Impact Ratio (4/5ths Rule)")
    
    c1, c2 = st.columns([1, 2])
    with c1:
        st.metric(
            "DIR Score", 
            f"{dir_score:.2f}", 
            delta="Fair" if is_fair else "Biased", 
            delta_color="normal" if is_fair else "inverse",
            help="Legal Threshold: 0.80 - 1.25. < 0.80 means the unprivileged group is selected significantly less often."
        )
    with c2:
        if is_fair:
            st.success(f"✅ **Pass**: Selection rates are comparable across {sensitive_col} groups.")
        elif dir_score < 0.8:
            st.error(f"❌ **Fail**: Significant bias detected against unprivileged subgroups of {sensitive_col}.")
        else:
            st.warning(f"⚠️ **Reverse Bias**: Privileged group is selected less often.")
            
    st.plotly_chart(dir_fig, use_container_width=True)
    
    if persona == "Student":
        st.info("💡 **Concept**: If 50% of Men get hired, but only 20% of Women get hired, the Ratio is 20/50 = 0.40. This is below 0.80, so it's considered unfair.")

    st.divider()

    # --- 3. PROXY DETECTION (Advanced) ---
    st.subheader("2. Proxy Detection & Leakage")
    
    if persona == "Student":
        st.markdown("""
        **What is a Proxy?**  
        Even if you remove 'Race' from your dataset, the AI might still learn it from 'Zip Code' or 'School'. 
        This section checks if your other columns verify the sensitive attribute.
        """)
        
    # INNOVATION: SAP GAUGE (Sensitive Attribute Predictability)
    with st.expander("🕵️ Spy Model Risk (SAP)", expanded=True):
        st.caption("can we predict 'Race/Gender' using only the other variables?")
        
        if st.button("Training Spy Model..."):
            with st.spinner("Training a model to predict your sensitive attribute..."):
                acc, status = auditor.check_sensitive_attribute_predictability()
                
                # Visual Gauge
                st.write(f"**Spy Model Accuracy:** {acc:.1%}")
                st.progress(acc)
                
                if acc > 0.80:
                    st.error(f"🚨 **High Risk**: {status}")
                elif acc > 0.60:
                    st.warning(f"⚠️ **Medium Risk**: {status}")
                else:
                    st.success(f"✅ **Low Risk**: {status}")

    # Mutual Information
    with st.expander("🔗 Correlation Scan (Mutual Information)", expanded=(persona != "Student")):
        if st.button("Scan for Proxies"):
            _, proxy_fig = auditor.detect_proxies()
            st.plotly_chart(proxy_fig, use_container_width=True)

    # --- 4. ADVANCED FORENSICS ---
    if persona != "Student":
        st.subheader("3. Advanced Forensics")
        
        # Simpson's Paradox
        with st.expander("📉 Simpson's Paradox Check", expanded=True):
            cat_cols = [c for c in df.select_dtypes(include=['object', 'category']).columns if c not in [sensitive_col, target_col]]
            if cat_cols:
                confounder = st.selectbox("Select Potential Confounder", cat_cols)
                simpson_fig = auditor.check_simpsons_paradox(confounder)
                st.plotly_chart(simpson_fig, use_container_width=True)
            else:
                st.info("Not enough categorical variables.")
        
        # Fairness Generalization
        with st.expander("🧪 Fairness Generalization (Overfitting)", expanded=True):
            st.markdown("**Metric:** Does the fairness metric (DIR) hold up on new data?")
            if st.button("Check Fairness Overfitting"):
                with st.spinner("Splitting data and re-calculating..."):
                    t_dir, test_dir, msg = auditor.check_fairness_generalization(positive_outcome=str(positive_outcome))
                    
                    is_overfitting = "OVERFITTING" in msg
                    st.session_state.bias_overfitting = "YES" if is_overfitting else "NO"
                    st.session_state.bias_overfitting_msg = msg
                    
                    c1, c2 = st.columns(2)
                    c1.metric("Train DIR", f"{t_dir:.2f}")
                    c2.metric("Test DIR", f"{test_dir:.2f}", delta=f"{test_dir-t_dir:.2f}")
                    
                    if is_overfitting:
                        st.error(msg)
                    else:
                        st.success(msg)
