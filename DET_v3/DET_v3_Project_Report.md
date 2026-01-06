# DET v3: Dataset Ethical Triage System
## Comprehensive Project Report

---

## Executive Summary

**DET v3 (Dataset Ethical Triage)** is a Python-based tool designed to assess the ethical implications of tabular datasets *before* machine learning model training. It provides data scientists and analysts with a systematic framework to detect bias, privacy risks, and fairness concerns in clinical and sensitive datasets.

The system computes **10 ethical metrics** across two categories (Core and Advanced), applies a **rule-based decision engine**, and outputs a bias assessment: **NO_BIAS**, **MODERATE_BIAS**, or **SIGNIFICANT_BIAS**.

---

## Table of Contents

1. [Problem Statement](#1-problem-statement)
2. [System Architecture](#2-system-architecture)
3. [Core Metrics (5)](#3-core-metrics)
4. [Advanced Metrics (5)](#4-advanced-metrics)
5. [Decision Engine](#5-decision-engine)
6. [User Interface](#6-user-interface)
7. [Usage Examples](#7-usage-examples)
8. [Technical Implementation](#8-technical-implementation)

---

## 1. Problem Statement

### The Challenge
Machine learning models trained on biased datasets perpetuate and amplify historical inequities. In clinical settings, this can lead to:
- Misdiagnosis for underrepresented demographic groups
- Discriminatory treatment recommendations
- Privacy violations through re-identification
- Unequal model performance across populations

### The Solution
DET v3 provides a **pre-training audit** to identify these issues before they become embedded in model predictions. By assessing datasets at the data preparation stage, organizations can:
- Detect bias early in the ML pipeline
- Make informed decisions about data collection needs
- Implement targeted mitigation strategies
- Document ethical due diligence

---

## 2. System Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                         DET v3 System                           │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  ┌──────────┐    ┌─────────────┐    ┌──────────────┐           │
│  │  INPUT   │───▶│   METRICS   │───▶│   DECISION   │           │
│  │ CSV Data │    │   ENGINE    │    │    ENGINE    │           │
│  └──────────┘    └─────────────┘    └──────────────┘           │
│       │               │                    │                    │
│       │               ▼                    ▼                    │
│       │        ┌─────────────┐     ┌─────────────┐             │
│       │        │ 5 Core      │     │ Final       │             │
│       │        │ 5 Advanced  │     │ Decision    │             │
│       │        │ = 10 Total  │     │ + Memo      │             │
│       │        └─────────────┘     └─────────────┘             │
│       │                                   │                     │
│       └───────────────────────────────────┼─────────────────────│
│                                           ▼                     │
│                                   ┌─────────────┐               │
│                                   │ STREAMLIT   │               │
│                                   │ USER        │               │
│                                   │ INTERFACE   │               │
│                                   └─────────────┘               │
└─────────────────────────────────────────────────────────────────┘
```

### Package Structure
```
det/
├── __init__.py          # Package exports
├── utils.py             # Shared utilities, thresholds
├── decision.py          # Decision engine (NO_BIAS/MODERATE_BIAS/SIGNIFICANT_BIAS)
├── metrics/
│   ├── __init__.py      # Metric exports
│   ├── core.py          # 5 core metrics
│   └── advanced.py      # 5 advanced metrics
└── app.py               # Streamlit UI module
```

---

## 3. Core Metrics

Core metrics assess fundamental ethical properties of the dataset.

### 3.1 URS - Underrepresentation Score

**Purpose:** Measures whether minority demographic groups are adequately represented in the dataset.

**Ethical Meaning:**
- Models trained on datasets where some groups comprise <10% of data may produce unreliable predictions for those groups
- Underrepresentation can lead to poor model generalization and discriminatory outcomes

**Calculation:**
```
URS = min(group_proportion) across all protected groups

where group_proportion = count(group) / total_rows
```

**Thresholds:**
| Score | Flag | Interpretation |
|-------|------|----------------|
| ≥ 0.20 | 🟢 GREEN | All groups well represented |
| 0.10 - 0.20 | 🟡 YELLOW | Some underrepresentation |
| < 0.10 | 🔴 RED | Severe underrepresentation |

**Example:** If a dataset has 100 rows with 95 Male and 5 Female, URS = 0.05 (RED).

---

### 3.2 AOI - Attribute-Outcome Imbalance

**Purpose:** Detects disparities in outcome rates across protected groups.

**Ethical Meaning:**
- High AOI indicates the outcome (e.g., hospital readmission) is distributed unequally across demographics
- May reflect historical bias encoded in the labels themselves
- Training on such data perpetuates discriminatory patterns

**Calculation:**
```
AOI = max(outcome_rate) - min(outcome_rate) across protected groups

Uses fairlearn.metrics.demographic_parity_difference when available
```

**Thresholds:**
| Score | Flag | Interpretation |
|-------|------|----------------|
| ≤ 0.15 | 🟢 GREEN | Acceptable disparity |
| 0.15 - 0.25 | 🟡 YELLOW | Concerning disparity |
| > 0.25 | 🔴 RED | Critical disparity |

**Example:** If readmission rate is 30% for Group A and 55% for Group B, AOI = 0.25 (RED).

---

### 3.3 DMI - Differential Missingness Index

**Purpose:** Identifies whether missing data patterns differ systematically across demographic groups.

**Ethical Meaning:**
- If data is more often missing for certain groups, imputation algorithms may introduce bias
- Differential missingness may indicate barriers to data collection (e.g., healthcare access disparities)
- Standard imputation methods assume data is Missing Completely At Random (MCAR), which may not hold

**Calculation:**
```
For each feature:
    DMI_feature = max|missingness_rate_group_i - missingness_rate_group_j|
    
DMI = max(DMI_feature) across all features
```

**Thresholds:**
| Score | Flag | Interpretation |
|-------|------|----------------|
| ≤ 0.10 | 🟢 GREEN | Uniform missingness |
| 0.10 - 0.20 | 🟡 YELLOW | Moderate differential |
| > 0.20 | 🔴 RED | Critical differential |

---

### 3.4 k-Anonymity Risk

**Purpose:** Quantifies re-identification risk by analyzing quasi-identifier combinations.

**Ethical Meaning:**
- k-anonymity measures how many individuals share the same quasi-identifier combination
- Low k (especially k=1) means individuals can be uniquely identified
- In clinical contexts, re-identification can lead to discrimination, insurance issues, or privacy violations

**Calculation:**
```
1. Group data by quasi-identifiers (e.g., age, gender, ZIP)
2. Count rows in each equivalence class
3. k = min(equivalence_class_size)
4. risk_proportion = % of rows with k < threshold
```

**Thresholds:**
| k-value | Risk % | Flag | Interpretation |
|---------|--------|------|----------------|
| ≥ 5 | ≤ 10% | 🟢 GREEN | Acceptable privacy |
| 3-5 | ≤ 30% | 🟡 YELLOW | Moderate risk |
| < 3 | > 30% | 🔴 RED | Critical privacy risk |

---

### 3.5 HRS - Harm Risk Score

**Purpose:** Aggregates bias likelihood, clinical severity, and population vulnerability into a composite harm score.

**Ethical Meaning:**
- Not all biases have equal consequences
- A biased model for clinical diagnosis has higher stakes than a movie recommendation
- Vulnerable populations (elderly, children, minorities) deserve extra protection

**Calculation:**
```
HRS = L_bias × S_harm × V_vulnerable

where:
- L_bias = Likelihood of bias (derived from group imbalance)
- S_harm = Severity multiplier (low=0.2, medium=0.5, high=0.8, critical=1.0)
- V_vulnerable = Vulnerability multiplier (low=0.3, medium=0.5, high=0.7, critical=1.0)
```

**Thresholds:**
| Score | Flag | Interpretation |
|-------|------|----------------|
| < 0.25 | 🟢 GREEN | Low harm risk |
| 0.25 - 0.50 | 🟡 YELLOW | Moderate harm risk |
| ≥ 0.50 | 🔴 RED | High harm risk |

---

## 4. Advanced Metrics

Advanced metrics provide deeper analytical insights, often using machine learning techniques.

### 4.1 FOI - Feature Outcome Disparity

**Purpose:** Measures whether features have consistent predictive relationships across demographic groups.

**Ethical Meaning:**
- A feature that predicts outcomes well for one group but not another indicates structural data differences
- Models may learn group-dependent patterns that fail to generalize
- Can indicate data collection biases across populations

**Calculation:**
```
For each feature:
    corr_g = correlation(feature, outcome) for group g
    FOI_feature = variance(corr_g) across groups

FOI = 1 - max(FOI_feature)  [Higher is better]
```

---

### 4.2 FPC - Fairness-Performance Convergence

**Purpose:** Trains a simple diagnostic model and measures accuracy variance across groups.

**Ethical Meaning:**
- If a baseline model has very different accuracy by group, the final model likely will too
- Early warning for performance disparities before full model training
- Helps anticipate fairness issues

**Calculation:**
```
1. Train LogisticRegression on 70% of data
2. Compute accuracy for each protected group on test set
3. FPC = 1 - variance(group_accuracies)  [Higher is better]
```

**Note:** This is a diagnostic metric; actual model performance may differ.

---

### 4.3 CPA - Conditional Proxy Assessment

**Purpose:** Identifies features that may act as proxies for protected attributes.

**Ethical Meaning:**
- Even without using race/gender directly, models can discriminate via proxies
- ZIP code often correlates with race; name with gender
- Proxy usage leads to indirect discrimination that may evade fairness audits

**Calculation:**
```
For each feature:
    CPA_feature = MutualInformation(feature, protected_attribute)
    
CPA = max(CPA_feature) normalized to [0, 1]
```

**Thresholds:**
| Score | Flag | Interpretation |
|-------|------|----------------|
| ≤ 0.30 | 🟢 GREEN | Low proxy risk |
| 0.30 - 0.50 | 🟡 YELLOW | Moderate proxy risk |
| > 0.50 | 🔴 RED | High proxy leakage |

---

### 4.4 SPA - Sensitive Predictability Analysis

**Purpose:** Trains a "spy model" to predict protected attributes from other features.

**Ethical Meaning:**
- If a model can accurately predict race/gender from other features, those features encode protected information
- High predictability means the model can effectively "know" protected attributes even if not given them
- Violates privacy and enables indirect discrimination

**Calculation:**
```
1. Train spy model: features → protected_attribute
2. Compute ROC-AUC on held-out test set
3. SPA = AUC score  [Lower is better - harder to predict = more private]
```

**Model Options:** Logistic Regression or Decision Tree (per specification)

**Thresholds:**
| AUC | Flag | Interpretation |
|-----|------|----------------|
| ≤ 0.65 | 🟢 GREEN | Low predictability (good) |
| 0.65 - 0.75 | 🟡 YELLOW | Moderate predictability |
| > 0.75 | 🔴 RED | High predictability (bad) |

---

### 4.5 DAI - Distributional Alignment Index

**Purpose:** Measures how close the dataset's demographic distribution is to uniform.

**Ethical Meaning:**
- Perfectly uniform distribution gives each group equal statistical power
- Skewed distributions mean some groups drive model behavior more than others
- Complements URS with a holistic distribution view

**Calculation:**
```
observed = distribution of groups
uniform = 1/n_groups for each group

DAI = 1 - JensenShannonDivergence(observed, uniform)
```

**Thresholds:**
| Score | Flag | Interpretation |
|-------|------|----------------|
| ≥ 0.85 | 🟢 GREEN | Near uniform |
| 0.70 - 0.85 | 🟡 YELLOW | Moderate deviation |
| < 0.70 | 🔴 RED | Significant deviation |

---

## 5. Decision Engine

### Decision Logic

The decision engine applies a hierarchical rule-based system:

```
                    ┌────────────────────────────┐
                    │     Collect 10 Metrics     │
                    └────────────┬───────────────┘
                                 ▼
                    ┌────────────────────────────┐
                    │  Check SIGNIFICANT_BIAS    │
                    │  conditions (hard fails)   │
                    └────────────┬───────────────┘
                                 │
              ┌──────────────────┼──────────────────┐
              │                  │                  │
              ▼                  ▼                  ▼
    ┌─────────────────┐  ┌─────────────────┐  ┌─────────────┐
    │ URS < 0.50?     │  │ k-risk > 30%?   │  │ ≥2 RED?     │
    └────────┬────────┘  └────────┬────────┘  └──────┬──────┘
             │                    │                   │
             └────────────────────┼───────────────────┘
                                  │
                           YES ───┴──▶ 🔴 SIGNIFICANT_BIAS
                                  │
                                  NO
                                  ▼
                    ┌────────────────────────────┐
                    │  Check MODERATE_BIAS       │
                    │  conditions (warnings)     │
                    └────────────┬───────────────┘
                                 │
              ┌──────────────────┼──────────────────┐
              │                  │                  │
              ▼                  ▼                  ▼
    ┌─────────────────┐  ┌─────────────────┐
    │ Exactly 1 RED?  │  │ ≥3 YELLOW?      │
    └────────┬────────┘  └────────┬────────┘
             │                    │
             └────────────────────┘
                        │
                 YES ───┴──▶ 🟡 MODERATE_BIAS
                        │
                        NO
                        ▼
                 🟢 NO_BIAS
```

### Decision Thresholds Summary

| Condition | Triggers |
|-----------|----------|
| **SIGNIFICANT_BIAS** | URS < 0.50 OR k-risk > 30% OR HRS ≥ 0.70 OR ≥2 RED flags |
| **MODERATE_BIAS** | Exactly 1 RED flag OR ≥3 YELLOW flags |
| **NO_BIAS** | All other cases (0 RED, ≤2 YELLOW) |

---

## 6. User Interface

### Streamlit Dashboard Features

1. **Data Upload**
   - CSV file upload via drag-and-drop
   - Automatic column detection
   - Data preview

2. **Configuration Panel**
   - Target column selection
   - Protected attributes multi-select
   - Quasi-identifiers selection
   - Clinical severity slider (low/medium/high/critical)
   - Population vulnerability slider

3. **Results Display**
   - 10 metric cards with traffic-light indicators
   - Expandable details for each metric
   - Radar chart visualization
   - Group distribution bar charts
   - Outcome rate comparison charts

4. **Decision Banner**
   - Large colored banner: 🟢/🟡/🔴
   - Confidence percentage
   - Human-readable rationale

5. **Action Items**
   - Specific recommendations per flagged metric
   - Prioritized next steps

6. **Downloadable Memo**
   - Markdown-formatted decision memo
   - Complete audit trail

---

## 7. Usage Examples

### Programmatic Usage

```python
import pandas as pd
from det import (
    calculate_urs, calculate_aoi, calculate_dmi,
    calculate_k_anonymity, calculate_hrs,
    calculate_foi, calculate_fpc, calculate_cpa,
    calculate_spa, calculate_dai,
    make_decision
)

# Load dataset
df = pd.read_csv('clinical_data.csv')

# Configuration
protected_attrs = ['race', 'gender']
target_col = 'readmitted'
quasi_ids = ['age', 'gender', 'zip_code']

# Compute metrics
results = {
    'URS': calculate_urs(df, protected_attrs),
    'AOI': calculate_aoi(df, target_col, protected_attrs),
    'DMI': calculate_dmi(df, protected_attrs),
    'k_anonymity': calculate_k_anonymity(df, quasi_ids),
    'HRS': calculate_hrs(df, target_col, protected_attrs, 'high', 'high'),
    'FOI': calculate_foi(df, target_col, protected_attrs),
    'FPC': calculate_fpc(df, target_col, protected_attrs),
    'CPA': calculate_cpa(df, protected_attrs),
    'SPA': calculate_spa(df, protected_attrs),
    'DAI': calculate_dai(df, protected_attrs)
}

# Get decision
decision = make_decision(results)
print(f"Decision: {decision['decision']}")
print(f"Confidence: {decision['confidence']:.0%}")
print(f"Rationale: {decision['rationale']}")
```

### Streamlit App

```bash
streamlit run run_det.py
```

---

## 8. Technical Implementation

### Dependencies

| Library | Purpose |
|---------|---------|
| `pandas` | Data manipulation |
| `numpy` | Numerical operations |
| `scikit-learn` | ML models, metrics, preprocessing |
| `fairlearn` | Demographic parity calculation |
| `scipy` | Statistical functions (entropy, divergence) |
| `streamlit` | Web interface |
| `plotly` | Interactive visualizations |

### Key Design Decisions

1. **Pre-training Focus**
   - All metrics run on raw data before any model training
   - Enables early intervention in ML pipeline

2. **Interpretability**
   - Traffic-light system (GREEN/YELLOW/RED) for quick assessment
   - Human-readable interpretations for each metric
   - No black-box scoring

3. **Graceful Degradation**
   - Handles missing columns with warnings
   - Defaults for small datasets
   - Fallback calculations when fairlearn unavailable

4. **Simple Spy Models**
   - Only LogisticRegression and DecisionTree
   - No deep learning per specification
   - Fast computation

---

## Conclusion

DET v3 provides a systematic, interpretable framework for ethical dataset assessment. By computing 10 complementary metrics and applying clear decision rules, it helps data scientists make informed choices about dataset suitability for machine learning.

The three-tier decision system (NO_BIAS / MODERATE_BIAS / SIGNIFICANT_BIAS) provides clear guidance while avoiding oversimplified pass/fail judgments. The detailed action items help teams prioritize mitigation efforts effectively.

---

**Document Version:** 3.0  
**Last Updated:** January 2026  
**Author:** DET Development Team
