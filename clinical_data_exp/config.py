#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jun  4 10:52:29 2026
@author: mihaelacroitor

# Shared file paths, hyperparameter settings, and global variables

"""

# config.py

# File Paths
BASELINE_DATA_PATH = "/Users/mihaelacroitor/project/data/clinical_data/atrophy_data/processed_data/baseline_data_5ROI.csv"
LONGITUDINAL_DATA_PATH = "/Users/mihaelacroitor/project/data/clinical_data/atrophy_data/processed_data/longitudinal_data_5ROI.csv"


# Profile A: Regional Cortical Thickness ROIs 
BIOMARKER_COLS = ['Frontal', 'Parietal', 'Temporal', 'Occipital', 'MedialTemporal']

# # Profile B: Global / Multimodal Biomarkers
# BIOMARKER_COLS = ["Hippocampus_Volume", "WholeBrain_Volume", "ADAS13", "FDG_PET"]

SUSTAIN_CAT_COL = "APOE4" # could be PTGENDER or others

# Global Variable Names
ID_COL = "RID"
TIME_TO_EVENT_COL = "YearsToConversion"
EVENT_COL = "ConvertedToAD"  # 1 = Converted, 0 = Censored
APOE4_COL = "APOE4_Carrier"   # 1 = Yes, 0 = No

COVARIATES = ["Age", "Sex", "Education"]

# SuStaIn Configuration Outputs
STD_SUBTYPE = "Standard_Subtype"
STD_STAGE = "Standard_Stage"

EXT_SUBTYPE = "Extended_Subtype"
EXT_STAGE = "Extended_Stage"