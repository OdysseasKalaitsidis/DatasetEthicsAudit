
import streamlit as st
import pandas as pd
import plotly.express as px
from src.quality_engine import QualityAuditor

def render_data_health_tab(df: pd.DataFrame, target_col: str, sensitive_col: str, persona: str = "Student"):
    """Renders the Data Quality & Health Tab (Representation)."""
    st.header("🧬 Data DNA & Health")
    
    auditor = QualityAuditor(df, target_col)
    
    # --- 1. HERO METRIC: DATA HEALTH SCORE ---
    # Heuristic: Start at 100. Deduct for missing values, imbalance, and duplicates.
    score = 100
    details = []
    
    # Missing Penalty
    miss_df, miss_fig = auditor.check_missing_values(sensitive_col)
    total_missing = miss_df['Missing Count'].sum()
    if total_missing > 0:
        penalty = min(20, (total_missing / len(df)) * 100)
        score -= penalty
        details.append(f"-{int(penalty)} pts (Missing Data)")
        
    # Duplicate Penalty
    dupes = auditor.check_duplicates()
    if dupes > 0:
        penalty = min(10, (dupes / len(df)) * 100)
        score -= penalty
        details.append(f"-{int(penalty)} pts (Duplicates)")
        
    # Balance Penalty
    _, _, bal_status = auditor.check_class_balance()
    if "Imbalance" in bal_status:
        score -= 15
        details.append("-15 pts (Class Imbalance)")
        
    score = max(0, int(score))
    
    # PERSISTENCE FOR RIBBON
    st.session_state.data_health_score = score
    
    # Display Hero Metric
    st.markdown("### 🏥 Health Score")
    c1, c2 = st.columns([1, 4])
    with c1:
        st.metric("Score", f"{score}/100", delta="-".join(details) if details else "Perfect", delta_color="inverse")
    with c2:
        if score > 80:
            st.success("This dataset is in good shape! Minimal cleaning required.")
        elif score > 50:
            st.warning("⚠️ Requires attention. Check missing values and balance.")
        else:
            st.error("🚨 Critical Issues Detected. Do not proceed without cleaning.")
    
    st.divider()

    # --- 2. BASIC VITALS (Student + Expert) ---
    st.subheader("1. Vital Signs")
    c1, c2, c3 = st.columns(3)
    c1.metric("Rows", df.shape[0])
    c2.metric("Columns", df.shape[1])
    c3.metric("Duplicates", dupes, delta="High" if dupes > 0 else "None", delta_color="inverse")
    
    # Visuals
    c1, c2 = st.columns(2)
    with c1:
        st.markdown("**Missing Values Pattern**")
        st.plotly_chart(miss_fig, use_container_width=True)
    with c2:
        st.markdown("**Target Class Balance**")
        _, bal_fig, _ = auditor.check_class_balance()
        st.plotly_chart(bal_fig, use_container_width=True)

    # --- 3. ADVANCED DNA (Experts Only / Progressive Disclosure) ---
    if persona != "Student":
        st.subheader("2. Advanced DNA Analysis")
    else:
        st.subheader("2. Representation Analysis")
        st.info("💡 **Why this matters:** If we don't have enough examples of specific groups (e.g., 'Age > 60'), the AI won't learn how to treat them fairly.")

    # Subgroup Representation
    with st.expander("🔬 Subgroup Representation", expanded=True):
        st.caption("Are all subgroups represented adequately (>10%)?")
        counts_df, status = auditor.check_group_representation(sensitive_col)
        
        c1, c2 = st.columns([1, 2])
        with c1:
            st.write(status)
            st.dataframe(counts_df, hide_index=True)
        with c2:
            fig = px.pie(counts_df, names=sensitive_col, values='Percentage', title=f"Distribution of {sensitive_col}")
            st.plotly_chart(fig, use_container_width=True)

    # Intersectional Coverage (Innovation)
    with st.expander("🔬 Intersectional Coverage (Blind Spots)", expanded=(persona != "Student")):
        st.caption("Checks for empty intersections (e.g., 'Black Women' < 5 samples).")
        other_cats = [c for c in df.select_dtypes(include=['object', 'category']).columns if c != sensitive_col and c != target_col]
        
        if other_cats:
            other_col = st.selectbox("Select Intersection Variable", other_cats)
            heatmap, blind_spots = auditor.check_intersectional_coverage(sensitive_col, other_col)
            st.plotly_chart(heatmap, use_container_width=True)
            if blind_spots > 0:
                st.error(f"⚠️ Found {blind_spots} blind spots (groups with < 5 samples).")
            else:
                st.success("✅ Good coverage across intersections.")
        else:
            st.info("No other categorical variables found to cross-reference.")
