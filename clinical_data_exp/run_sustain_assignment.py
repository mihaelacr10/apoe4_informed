#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jun  4 10:38:12 2026

Fits SuStaIn and APOE-SuStaIn algorithms,
updates baseline_data.csv with stages/subtypes.

@author: mihaelacroitor
"""

# run_sustain_assignment.py
import pandas as pd
import numpy as np
import config
import os
import sys
import shutil
from pathlib import Path


# 1. Hardcode the absolute path to your model folder
APOE4_MODEL_PATH = "/Users/mihaelacroitor/apoe4_informed"

# 2. Add it to Python's system search paths if it isn't already there
if APOE4_MODEL_PATH not in sys.path:
    sys.path.insert(0, APOE4_MODEL_PATH)
    
from apoe4_sustain import ZscoreSustain_APOE4

def run_sustain_and_save_assignments():
    print("Loading raw clinical biomarker data...")
    # Load raw, un-staged baseline dataset (e.g., ADNI cognitive scores, volumes)
    df = pd.read_csv(config.BASELINE_DATA_PATH)
    
    # -------------------------------------------------------------------------
    # 1. Prepare inputs for SuStaIn (Requires Z-scored data matrix)
    # -------------------------------------------------------------------------
    # Select only the columns representing your disease biomarkers
    biomarker_cols = config.BIOMARKER_COLS
    biomarker_data = df[biomarker_cols].to_numpy()
    
    # SuStaIn configuration parameters
    # Array defining the Z-score milestones each biomarker goes through (e.g., Z=1, Z=2, Z=3)
    Z_vals = np.array([[1, 2, 3]] * len(biomarker_cols)) 
    Z_max = np.array([5] * len(biomarker_cols))          # Max expected Z-score
    
    N_subtypes = 2            # Number of target pathways/clusters
    N_startpoints = 25        # Multi-start optimization points
    N_iterations_MCMC = 100000 # Number of MCMC sampling iterations for uncertainty
    N_iterations_MCMC = int(1e4)
    # -------------------------------------------------------------------------
    # 2. Run Track 2: Standard SuStaIn Model
    # -------------------------------------------------------------------------
    bl_output_folder = Path("outputs/sustain_standard")
    if bl_output_folder.exists():
        shutil.rmtree(bl_output_folder)
    bl_output_folder.mkdir(parents=True, exist_ok=True)
    
    
    # print("Fitting Baseline SuStaIn model (this may take a while)...")
    # sustain_bl = pySuStaIn.ZscoreSustain(
    #     biomarker_data,
    #     Z_vals,
    #     Z_max,
    #     biomarker_labels=biomarker_cols,
    #     N_startpoints=N_startpoints,
    #     N_S_max=N_subtypes,
    #     N_iterations_MCMC=N_iterations_MCMC,
    #     output_folder = bl_output_folder,
    #     dataset_name="adni_standard",
    #     use_parallel_startpoints=False,
    #     )
    sustain_bl = ZscoreSustain_APOE4(
        biomarker_data, Z_vals, Z_max, biomarker_cols, N_startpoints, N_subtypes, 
        N_iterations_MCMC, output_folder=bl_output_folder, dataset_name="adni_extended",
        use_parallel_startpoints= False, apoe_flag=False
    )
    
    
    # Run the EM optimization and MCMC sampling
    _ = sustain_bl.run_sustain_algorithm()
    
    # Extract assignments: returns arrays of (subtype_labels, stage_labels)
    # stage_labels range from 0 (healthy control) up to the total number of events
    subtypes_bl, stages_bl, _, _ = sustain_bl.subtype_and_stage_individuals(biomarker_data)
    
    # -------------------------------------------------------------------------
    # 3. Run Track 3: Your Custom Extended SuStaIn Model (APOE4-Informed)
    # -------------------------------------------------------------------------
    
    print("Fitting Your Extended SuStaIn model...")
    # Pass your extra APOE4 data vector into your extended class architecture
    new_output_folder = Path("outputs/sustain_standard")
    if new_output_folder.exists():
        shutil.rmtree(new_output_folder)
    new_output_folder.mkdir(parents=True, exist_ok=True)
    
    categorical_var = df[config.SUSTAIN_CAT_COL].to_numpy() #apoe or whatever
    
    sustain_apoe = ZscoreSustain_APOE4(
        biomarker_data, Z_vals, Z_max, biomarker_cols, N_startpoints, N_subtypes, 
        N_iterations_MCMC, output_folder=new_output_folder, dataset_name="adni_extended",
        apoe4_status=categorical_var, apoe_flag=True, em_loop_type='combined'
    )
    
    
    _ = sustain_apoe.run_sustain_algorithm()
    # ext_subtypes, ext_stages = sustain_ext.subtype_and_stage_individuals(biomarker_data)
    
    # Dummy placeholder data for script completeness:
    subtypes_apoe, stages_apoe,_ , _ = sustain_apoe.subtype_and_stage_individuals(biomarker_data)
    
    # -------------------------------------------------------------------------
    # 4. Save assignments directly back to the dataset
    # -------------------------------------------------------------------------
    print("Writing generated assignments to data file...")
    df["subtype_bl"] = subtypes_bl
    df["stage_bl"] = stages_bl
    
    ext_subtype_col = f"subtype_{config.SUSTAIN_CAT_COL.lower()}"  # -> "subtype_apoe4"
    ext_stage_col = f"stage_{config.SUSTAIN_CAT_COL.lower()}"      # -> "stage_apoe4"
    
    df[ext_subtype_col] = subtypes_apoe
    df[ext_stage_col] = stages_apoe
    
    # Save the new dataframe
    base_path_without_ext, ext = os.path.splitext(config.BASELINE_DATA_PATH)
    staged_output_path = f"{base_path_without_ext}_subtyped_and_staged{ext}"
    df.to_csv(staged_output_path, index=False)
    print("Initialization complete! Assignments are permanently cached.")

if __name__ == "__main__":
    run_sustain_and_save_assignments()