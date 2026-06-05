#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jun  4 10:49:50 2026
@author: mihaelacroitor

This file plots kaplan meier curves

"""

# plot_survival.py
import matplotlib.pyplot as plt
from lifelines import KaplanMeierFitter
import data_processor
import config

def plot_subtype_survival(subtype_id):
    df = data_processor.load_and_clean_data()
    
    # Grab the split dataset matching your extended model parameters
    sub_data = data_processor.get_subtype_dataset(df, config.EXT_SUBTYPE, config.EXT_STAGE, target_subtype=subtype_id)
    
    kmf_control = KaplanMeierFitter()
    kmf_patient = KaplanMeierFitter()
    
    fig, ax = plt.subplots(figsize=(8, 5))
    
    # Split into our shared control bucket vs the specific subtype bucket
    c_mask = (sub_data['Group'] == 0)
    p_mask = (sub_data['Group'] == 1)
    
    kmf_control.fit(sub_data.loc[c_mask, config.TIME_TO_EVENT_COL], sub_data.loc[c_mask, config.EVENT_COL], label="Stage 0 Controls")
    kmf_patient.fit(sub_data.loc[p_mask, config.TIME_TO_EVENT_COL], sub_data.loc[p_mask, config.EVENT_COL], label=f"Extended Subtype {subtype_id}")
    
    kmf_control.plot_survival_function(ax=ax, ci_show=True)
    kmf_patient.plot_survival_function(ax=ax, ci_show=True)
    
    plt.title(f"MCI-to-AD Conversion Risk: Extended Subtype {subtype_id} vs Shared Controls")
    plt.xlabel("Years from Baseline Evaluation")
    plt.ylabel("Probability of Remaining MCI (Survival)")
    plt.grid(True, linestyle="--", alpha=0.5)
    plt.tight_layout()
    plt.savefig(f"outputs/km_curve_subtype_{subtype_id}.png", dpi=300)
    plt.show()

if __name__ == "__main__":
    # Example: Build plot for Subtype 1
    plot_subtype_survival(subtype_id=1)