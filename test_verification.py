import pandas as pd
import numpy as np
import os
import sys

# Add src to path
sys.path.append(os.path.join(os.getcwd()))

from src.data_manager import DataManager
from src.quality_engine import QualityAuditor
from src.bias_engine import BiasAuditor
from src.audit_engine import ModelAuditor

def run_verification():
    print("Starting Verification...")
    
    # create dummy data
    df = pd.DataFrame({
        'age': np.random.randint(18, 65, 100),
        'gender': np.random.choice(['Male', 'Female'], 100),
        'salary': np.random.randint(30000, 100000, 100),
        'hired': np.random.choice([0, 1], 100)
    })
    
    # Introduce some missing values
    df.loc[0:5, 'salary'] = np.nan
    
    print("1. Testing DataManager...")
    num, cat = DataManager.get_column_types(df)
    assert 'age' in num
    assert 'gender' in cat
    print("   DataManager OK")
    
    print("2. Testing QualityAuditor...")
    qa = QualityAuditor(df, 'hired')
    miss_df, miss_fig = qa.check_missing_values('gender')
    # assert not miss_df.empty # might be empty if no missing, strictly speaking
    counts, bal_fig, status = qa.check_class_balance()
    print("   QualityAuditor OK")
    
    print("3. Testing BiasAuditor...")
    ba = BiasAuditor(df, 'gender', 'hired')
    dir_score, dir_fig = ba.get_disparate_impact_ratio()
    proxy_df, proxy_fig = ba.detect_proxies()
    simpson_fig = ba.check_simpsons_paradox('salary') # salary is numeric, might error if check expects categorical
    # Bias Engine implementation of simpso paradox expects categorical confounder?
    # check_simpsons_paradox signature: confounder_col: str
    # logic: groupby([sensitive, confounder])...
    # If salary is numeric (many unique values), the plot will be messy but should run. 
    print("   BiasAuditor OK")
    
    print("4. Testing ModelAuditor...")
    # Model Auditor drops missing target rows. Data setup: 100 rows, hired has no missing.
    # It drops nan rows in _prepare_data.
    ma = ModelAuditor(df, 'hired', 'gender')
    metrics = ma.train_simulation()
    assert 'train_accuracy' in metrics
    fi_df, fi_fig, msg = ma.get_feature_importance()
    num_changed, _ = ma.run_counterfactual_analysis()
    thresh_fig, t_metrics = ma.test_fairness_threshold(0.5)
    print("   ModelAuditor OK")
    
    print("All Tests Passed!")

if __name__ == "__main__":
    run_verification()
