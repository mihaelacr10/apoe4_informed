#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jun  4 10:51:35 2026
@author: mihaelacroitor

# Shared statistical functions (fits Cox models, calculates C-index)

"""

# models.py
from lifeslines import CoxPHFitter
from lifelines.utils import concordance_index
import statsmodels.stats.multitest as smm
import pandas as pd
import config

def fit_cox_model(data, duration_col, event_col, predictor_cols):
    """Fits a Cox Proportional Hazards model and returns evaluation metrics."""
    cph = CoxPHFitter()
    
    # Filter dataframe down to only the variables needed for the regression
    model_cols = [duration_col, event_col] + predictor_cols
    model_data = data[model_cols].dropna()
    
    cph.fit(model_data, duration_col=duration_col, event_col=event_col)
    
    # Extract Hazard Ratio, Confidence Intervals, and p-value for the primary predictor
    primary_predictor = predictor_cols[0]
    summary = cph.summary.loc[primary_predictor]
    
    metrics = {
        "model_object": cph,
        "hr": summary["exp(coef)"],
        "ci_lower": summary["exp(coef) lower 95%"],
        "ci_upper": summary["exp(coef) upper 95%"],
        "p_value": summary["p"],
        "c_index": cph.concordance_index_
    }
    return metrics

def apply_hochberg_correction(p_values):
    """Applies Hochberg adjustment to handle multiple subtype comparisons."""
    reject, p_adjusted, _, _ = smm.multipletests(p_values, alpha=0.05, method='hochberg')
    return p_adjusted