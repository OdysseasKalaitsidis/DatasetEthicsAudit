# DET v3 - Dataset Ethics Triage

Pre-training ethical assessment tool for tabular datasets in clinical ML.

## Quick Start

```bash
cd DET_v3
pip install -r requirements.txt
streamlit run run_det.py
```

## What It Does

Analyzes datasets for bias, privacy risks, and fairness issues **before** model training.

### 10 Metrics

| Metric | Checks |
|--------|--------|
| **URS** | Group underrepresentation |
| **AOI** | Outcome disparity |
| **DMI** | Differential missingness |
| **k-Anonymity** | Re-identification risk |
| **HRS** | Harm potential |
| **FOI** | Feature-outcome variance |
| **FPC** | Model accuracy variance |
| **CPA** | Proxy feature detection |
| **SPA** | Attribute predictability |
| **DAI** | Distribution alignment |

### Decisions

- 🟢 **NO_BIAS** - Dataset acceptable
- 🟡 **MODERATE_BIAS** - Mitigation recommended  
- 🔴 **SIGNIFICANT_BIAS** - Intervention required

## Usage

1. Upload CSV dataset
2. Select target column
3. Select protected attributes (race, gender, age...)
4. Select quasi-identifiers (for k-anonymity)
5. Set severity/vulnerability levels
6. Click **RUN AUDIT**

## Project Structure

```
DET_v3/
├── run_det.py          # Streamlit app
├── det/
│   ├── __init__.py     # Package exports
│   ├── decision.py     # Decision engine
│   ├── utils.py        # Helpers
│   └── metrics/
│       ├── core.py     # URS, AOI, DMI, k-Anon, HRS
│       └── advanced.py # FOI, FPC, CPA, SPA, DAI
└── requirements.txt
```

## Requirements

- Python 3.9+
- pandas, numpy, scikit-learn
- streamlit, plotly
- fairlearn, scipy

## License

MIT
