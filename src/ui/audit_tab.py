
import streamlit as st
import pandas as pd
import json
from src.audit_engine import ModelAuditor

def render_audit_tab(df: pd.DataFrame, target_col: str, sensitive_col: str, persona: str = "Student"):
    """Renders the Model Audit Tab."""
    st.header("🔎 Algorithmic Forensic Audit")
    
    # Initialize Auditor
    if 'model_auditor' not in st.session_state:
        st.session_state.model_auditor = ModelAuditor(df, target_col, sensitive_col)
    
    auditor = st.session_state.model_auditor

    # --- 1. MODEL SIMULATION (Interactive) ---
    st.subheader("1. Model Simulation")
    if persona == "Student":
        st.markdown("To understand how an AI might discriminate, we first need to teaching one! Click below to train a simulator on your data.")
    else:
        st.markdown("Train a Random Forest (Depth=5) to simulate decision boundaries.")

    if st.button("🚀 Train Simulation Model"):
        with st.spinner("Training Simulator..."):
            metrics = auditor.train_simulation()
            st.session_state.audit_metrics = metrics
            st.session_state.model_trained = True
            st.success("Model Trained Successfully!")

    if st.session_state.model_trained and 'audit_metrics' in st.session_state:
        metrics = st.session_state.audit_metrics
        gap = metrics['train_accuracy'] - metrics['test_accuracy']
        
        # Overfitting Monitor
        st.markdown("### 📈 Performance Monitor")
        c1, c2, c3 = st.columns(3)
        c1.metric("Train Accuracy", f"{metrics['train_accuracy']:.1%}")
        c2.metric("Test Accuracy", f"{metrics['test_accuracy']:.1%}")
        
        if metrics['is_overfitting']:
            c3.metric("Generalization Gap", f"{gap:.1%}", delta="High Overfitting", delta_color="inverse")
            st.error(f"⚠️ **Overfitting Detected:** The model performs {gap:.1%} better on training data than new data.")
        else:
            c3.metric("Generalization Gap", f"{gap:.1%}", delta="Good", delta_color="normal")
            st.success(f"✅ **Robust Logic:** The gap is small ({gap:.1%}), meaning the model generalizes well.")
            
        st.divider()

        # --- 2. FORENSIC DEEP DIVE ---
        st.subheader("2. Forensic Deep Dive")
        
        # Feature Importance
        with st.expander("🔍 Data Leakage (Feature Importance)", expanded=True):
            st.caption("Did the model cheat by using a variable that gives away the answer?")
            fi_df, fi_fig, leak_msg = auditor.get_feature_importance()
            st.plotly_chart(fi_fig, use_container_width=True)
            if "ALERT" in leak_msg:
                st.error(leak_msg)
            else:
                st.success(leak_msg)

        # INNOVATION: Counterfactual Explorer
        with st.expander("🔄 Counterfactual Analysis (What-If?)", expanded=(persona != "Student")):
            st.markdown(f"**Goal:** Check if flipping **{sensitive_col}** changes predictions for specific people.")
            
            if st.button("Run Counterfactual Test"):
                num_changed, flipped_indices = auditor.run_counterfactual_analysis()
                if num_changed > 0:
                    st.error(f"❌ **Fail**: The model changed its mind for {num_changed} people just because their {sensitive_col} changed.")
                    
                    # Show the flipped rows if possible
                    if hasattr(auditor, 'X_test') and len(flipped_indices) > 0:
                         # This assumes run_counterfactual_analysis returns indices relative to test set, 
                         # simplifying for now to just show count.
                         pass
                else:
                    st.success("✅ **Pass**: Robust to sensitive attribute flips.")

        # Fairness Threshold
        if persona != "Student":
            with st.expander("🎚️ Fairness Threshold Slider", expanded=True):
                st.markdown("Adjust the decision boundary.")
                thresh = st.slider("Decision Threshold", 0.0, 1.0, 0.5, 0.05)
                t_fig, t_metrics = auditor.test_fairness_threshold(thresh)
                st.plotly_chart(t_fig, use_container_width=True)
                st.json(t_metrics)

        # --- 3. REPORTING ---
        st.divider()
        st.subheader("📋 Ethics Report")
        
        col1, col2 = st.columns(2)
        with col1:
            if st.button("Generate Ethics Card (JSON)"):
                report = {
                    "Dataset": "Uploaded File",
                    "Target": target_col,
                    "Sensitive Attribute": sensitive_col,
                    "Audit Date": str(pd.Timestamp.now()),
                    "Status": "AUDITED",
                    "Metrics": metrics,
                    "Leakage Warning": "YES" if "ALERT" in leak_msg else "NO",
                    "Bias Overfitting": st.session_state.get('bias_overfitting', "Not Checked"),
                }
                st.json(report)
        
        with col2:
             # Link to sidebar PDF download
             st.info("📄 For a full PDF report, use the **'Generate PDF Report'** button in the Sidebar.")
             
    else:
        st.info("👆 Please train the simulation model above to unlock these tools.")
