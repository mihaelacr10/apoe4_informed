#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jun  4 10:52:08 2026
@author: mihaelacroitor

Reads ADNI data, prepares baseline, and formats longitudinal files

"""

# data_processor.py
import pandas as pd
import config

def load_and_clean_data():
    """Loads baseline data and handles missing covariate values."""
    df = pd.read_csv(config.BASELINE_DATA_PATH)
    # Drop rows missing critical tracking or survival features
    df = df.dropna(subset=[config.ID_COL, config.TIME_TO_EVENT_COL, config.EVENT_COL])
    return df

def get_subtype_dataset(df, subtype_col, stage_col, target_subtype):
    """
    Creates a dataset containing ALL Stage 0 controls combined with
    ONLY the patients belonging to a specific target subtype.
    """
    # Universal Stage 0 Control Pool
    controls = df[df[stage_col] == 0].copy()
    controls['Group'] = 0  # Reference group
    
    # Specific Subtype Patients (Staged > 0)
    patients = df[(df[subtype_col] == target_subtype) & (df[stage_col] > 0)].copy()
    patients['Group'] = 1  # Target group
    
    # Combine them into a single analytical dataframe
    combined_df = pd.concat([controls, patients], axis=0).reset_index(drop=True)
    return combined_df