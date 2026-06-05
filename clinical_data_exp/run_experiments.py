#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jun  4 10:51:23 2026
@author: mihaelacroitor

Runs experiments 1,2,3:
├── run_track1_global.py  # Script for Track 1 (Whole cohort baseline + raw APOE4)
├── run_track2_standard.py# Script for Track 2 (Standard SuStaIn subtype models)
├── run_track3_extended.py# Script for Track 3 (Your Extended SuStaIn models)

"""

# run_experiments.py
import data_processor
import models
import config
import pandas as pd

def execute_validation_pipeline():
    # 0. Load global data
    df = data_processor.load_and_clean_data()
    print(f"Successfully loaded {len(df)} patient records.\n")
    
    # =========================================================================
    # TRACK 1: Global Baseline Model (Answers Challenge 2: The APOE4 High Bar)
    # =========================================================================
    print("--- Running Track 1: Global Baseline (APOE4 Covariate) ---")
    track1_predictors = [config.APOE4_COL, config.STD_STAGE, config.STD_SUBTYPE] + config.COVARIATES
    t1_results = models.fit_cox_model(df, config.TIME_TO_EVENT_COL, config.EVENT_COL, track1_predictors)
    print(f"Track 1 Global C-Index: {t1_results['c_index']:.3f}\n")
    
    # =========================================================================
    # TRACK 2: Standard SuStaIn Subtype Models
    # =========================================================================
    print("--- Running Track 2: Standard SuStaIn Slices ---")
    standard_subtypes = [1, 2, 3]  # Example list of unique subtypes
    t2_p_values = []
    t2_summary_data = []
    
    for sub in standard_subtypes:
        sub_data = data_processor.get_subtype_dataset(df, config.STD_SUBTYPE, config.STD_STAGE, target_subtype=sub)
        predictors = ['Group', config.STD_STAGE] + config.COVARIATES
        res = models.fit_cox_model(sub_data, config.TIME_TO_EVENT_COL, config.EVENT_COL, predictors)
        
        t2_p_values.append(res['p_value'])
        t2_summary_data.append({
            "Subtype": f"Std Subtype {sub}", "HR": res['hr'], "C-Index": res['c_index'], "Unadj_P": res['p_value']
        })
        
    # Apply Hochberg correction across standard subtype runs
    t2_adj_p = models.apply_hochberg_correction(t2_p_values)
    for i, row in enumerate(t2_summary_data):
        row["Hochberg_P"] = t2_adj_p[i]
        
    print(pd.DataFrame(t2_summary_data).to_string(index=False), "\n")

    # =========================================================================
    # TRACK 3: Your Extended Model (Answers Challenge 1: Staging Precision)
    # =========================================================================
    print("--- Running Track 3: Your Extended Model Slices ---")
    extended_subtypes = [1, 2, 3]
    t3_p_values = []
    t3_summary_data = []
    
    for sub in extended_subtypes:
        sub_data = data_processor.get_subtype_dataset(df, config.EXT_SUBTYPE, config.EXT_STAGE, target_subtype=sub)
        predictors = ['Group', config.EXT_STAGE] + config.COVARIATES
        res = models.fit_cox_model(sub_data, config.TIME_TO_EVENT_COL, config.EVENT_COL, predictors)
        
        t3_p_values.append(res['p_value'])
        t3_summary_data.append({
            "Subtype": f"Ext Subtype {sub}", "HR": res['hr'], "C-Index": res['c_index'], "Unadj_P": res['p_value']
        })
        
    # Apply Hochberg correction across extended subtype runs
    t3_adj_p = models.apply_hochberg_correction(t3_p_values)
    for i, row in enumerate(t3_summary_data):
        row["Hochberg_P"] = t3_adj_p[i]
        
    print(pd.DataFrame(t3_summary_data).to_string(index=False))

if __name__ == "__main__":
    execute_validation_pipeline()