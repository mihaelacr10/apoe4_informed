#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed May 27 13:50:07 2026

This code simulates synthetic data for my APOE-Informed model.
It integrates biomarker progression vectors with categorical APOE values.

@author: mihaelacroitor
"""

import numpy as np
import pandas as pd
import pickle
from pathlib import Path

from genetics_simfuncs import generate_random_Zscore_sustain_model,\
                              generate_data_Zscore_sustain,\
                              generate_data_Zscore_sustain_point


def generate_apoe_status(M, subtypes, W_true):
    """
    Generates categorical APOE status labels for each subject 
    based on their assigned subtype's true genetic weights.
    """
    subject_genetics = np.zeros(M, dtype=int)
    for i in range(M):
        # Sample category 0, 1, or 2 using the specified subtype probabilities
        subject_genetics[i] = np.random.choice([0, 1, 2], p=W_true[subtypes[i]])
        
    return subject_genetics


def generate_ground_truth_genetic_weights(N_S_gt,
                                         N_genetic_categories = 3,
                                         genetic_signal_strength = None,
                                         rng = None):
    """
    Generates the true underlying matrix of genetic weights (W_true) 
    mapping patient subtypes to discrete genetic category probabilities.
    """
    # 1. DEFAULT RANDOM STATE: Uses flat, symmetric Dirichlet distribution
    if genetic_signal_strength is None:
        alpha_vector = [1] * N_genetic_categories
        W_true = rng.dirichlet(alpha=alpha_vector, size=N_S_gt)
        
    # 2. UNIFORM CASE: Subtypes ripple slightly around a random global background
    elif genetic_signal_strength == 'uniform':
        cohort_profile = rng.dirichlet(alpha=[2.0] * N_genetic_categories)
        print(f"📊 Generated Random Global Cohort Baseline: {np.round(cohort_profile, 4)}")
        
        alpha_base = cohort_profile * 150.0
        W_true = np.zeros((N_S_gt, N_genetic_categories))
        for s in range(N_S_gt):
            W_true[s] = rng.dirichlet(alpha=alpha_base)
        
    # 3. MODERATE CASE: Distinct but overlapping profiles across subtypes
    elif genetic_signal_strength == 'moderate':
        W_true = np.zeros((N_S_gt, N_genetic_categories))
        for s in range(N_S_gt):
            alpha_vector = np.ones(N_genetic_categories) * 2.0
            alpha_vector[s % N_genetic_categories] += 5.0  
            W_true[s] = rng.dirichlet(alpha=alpha_vector)
            
    # 4. STRONG CASE: Clear separation with aggressive profile enrichment
    elif genetic_signal_strength == 'strong':
        W_true = np.zeros((N_S_gt, N_genetic_categories))
        for s in range(N_S_gt):
            alpha_vector = np.array([5.0, 5.0, 5.0])
            alpha_vector[s % N_genetic_categories] += 45.0  
            W_true[s] = rng.dirichlet(alpha=alpha_vector)
            
    else:
        raise ValueError(
            f"Unknown genetic_signal_strength '{genetic_signal_strength}'. "
            "Must be None, 'uniform', 'moderate', or 'strong'."
        )
        
    return W_true


def simulate_apoe_sustain_dataset(N = 5,                                         # number of biomarkers
                                 M = 500,                                       # number of observations (e.g. subjects)
                                 M_control = 100,                               # number of these that are control subjects
                                 N_S_gt = 2,                                    # number of ground truth subtypes
                                 gt_f = None,                                   # subtype prevalence
                                 W_true = None,                                 # genetic_weights
                                 genetic_signal_strength = None,                # 'strong', 'moderate' or 'uniform'
                                 seed = None,                                   # seed number to keep sequence fixed
                                 use_midpoints = False,                         # synthetic data function choice
                                 save = False,
                                 output_path = None,
                                 dataset_name = None,
                                 base_filename = None):
    
    SuStaInLabels = []
    for i in range(N):
        SuStaInLabels.append('Biomarker ' + str(i))
            
    Z_vals = np.array([[1, 2, 3]] * N)
    Z_max  = np.array([5] * N)
       
    # Calculate ground truth proportion of individuals belonging to each subtype  
    if gt_f is None:
        gt_f = [1 + 0.5 * x for x in range(N_S_gt)]
        gt_f = [x / sum(gt_f) for x in gt_f][::-1]
        
    # COMMENT: Generate a random number to use as a seed if none is provided
    if seed is None:
        seed = np.random.default_rng().integers(0, 2**32 - 1, dtype=np.uint64)
    # COMMENT: Create the new random generator using our seed to lock down all randomness
    rng = np.random.default_rng(seed)
    
    # COMMENT: Pass the new generator into the sequence builder to keep it reproducible
    gt_sequence = generate_random_Zscore_sustain_model(Z_vals, N_S_gt, rng=rng)

    # Track total number of progression stages
    N_k = np.sum(Z_vals > 0) + 1
    
    # COMMENT: Use the new generator to keep subtype choices identical across identical seeds
    gt_subtypes = rng.choice(range(N_S_gt), M, replace=True, p=gt_f)
    # gt_subtypes = np.random.choice(range(N_S_gt), M, replace=True, p=gt_f)
    
    gt_stages_control = np.zeros((M_control, 1))
    
    # COMMENT: Use the new generator with a tuple shape constraint to safely lock patient stages
    gt_stages = np.concatenate((gt_stages_control,
                                np.ceil(rng.random((M - M_control, 1)) * N_k)),
                               axis=0)
    # gt_stages = np.concatenate((gt_stages_control,
    #                             np.ceil(np.random.rand(M-M_control,1)*N_k)),
    #                            axis=0)
    
    # Generate simulated biomarker data array paths
    if use_midpoints:
        data, gt_data_denoised, gt_stage_value = generate_data_Zscore_sustain_point(gt_subtypes,
                                                                                   gt_stages,
                                                                                   gt_sequence,
                                                                                   Z_vals,
                                                                                   Z_max)
    else:
        data, gt_data_denoised, gt_stage_value = generate_data_Zscore_sustain(gt_subtypes,
                                                                             gt_stages,
                                                                             gt_sequence,
                                                                             Z_vals,
                                                                             Z_max)
        
    # -------------------------------------------------------------------------
    # STEP 3: GENERATE ALIGNED GENETIC CATEGORIES
    # -------------------------------------------------------------------------
    # COMMENT: Pass our generator into the weights function to keep our testing profiles consistent
    if W_true is None:
        W_true = generate_ground_truth_genetic_weights(
                    N_S_gt=N_S_gt, 
                    N_genetic_categories=3, 
                    genetic_signal_strength=genetic_signal_strength, 
                    rng=rng
                )
    apoe = generate_apoe_status(M, gt_subtypes, W_true)

    # -------------------------------------------------------------------------
    # STEP 4: PACK INTO CLEAN STRUCTURES & EXPORT
    # -------------------------------------------------------------------------
    df = pd.DataFrame(data, columns=SuStaInLabels)
    df['gt_subtypes'] = gt_subtypes
    df['gt_stages'] = gt_stages.flatten()
    df['apoe_status'] = apoe

    # Standardize data labeling constraints for legacy plotting compatibility
    df.loc[:, 'gt_subtypes'] = df.gt_subtypes.values + 1
    df.loc[df.gt_stages == 0, 'gt_subtypes'] = 0
    
    if save:
        if output_path is None: 
            output_path = Path("data") / "simulated_data"
        output_folder = Path(output_path) / dataset_name
        output_folder.mkdir(parents=True, exist_ok=True)
        
        df.to_csv(output_folder / f"{base_filename}.csv", index=False)
        
        parameters = {
            'Z_vals': Z_vals, 'Z_max': Z_max, 'gt_ordering': gt_sequence, 
            'W_true': W_true, 'gt_fractions': gt_f
        }
        with open(output_folder / f"{base_filename}.pkl", "wb") as f:
            pickle.dump(parameters, f)
            
            # NEW: Write out a beautiful, human-readable summary sheet
        summary_txt_path = output_folder / f"{base_filename}_ground_truth.txt"
        with open(summary_txt_path, "w", encoding="utf-8") as f:
            f.write("==================================================\n")
            f.write(f" GROUND TRUTH PARAMETERS: {dataset_name}\n")
            f.write("==================================================\n\n")
            
            
            f.write("🔹 SUBTYPE PREVALENCE FRACTIONS (gt_f):\n")
            f.write(f"   {gt_f}\n\n")
            
            f.write("🔹 GENETIC WEIGHTS MATRIX (W_true):\n")
            for i, row in enumerate(W_true):
                f.write(f"   Subtype {i}: {np.round(row, 4).tolist()}\n")
            f.write("\n")
            
            f.write("🔹 BIOMARKER STAGING ORDERINGS (gt_sequence):\n")
            for i, seq in enumerate(gt_sequence):
                f.write(f"   Subtype {i} Sequence: {seq.tolist()}\n")
                
            f.write("📋 COHORT PROPERTIES\n")
            f.write(f"   • Total Sample Size (M)     : {M}\n")
            f.write(f"   • Control Sample Size (M_0) : {M_control}\n")
            f.write(f"   • Number of Biomarkers (N)  : {N}\n")
            f.write(f"   • Total SuStaIn Stages (N_k): {N_k}\n")
            f.write(f"   • True Subtype Count (N_S)  : {N_S_gt}\n\n")
            
            f.write("🔹 STRUCTURAL THRESHOLD MATRIX (Z_vals):\n")
            f.write("   [Pos 1, Pos 2, Pos 3] per biomarker rows\n")
            for i, row in enumerate(Z_vals):
                f.write(f"   Biomarker {i}: {row.tolist()}\n")
            f.write("\n")
                
            f.write("🔹 MAXIMUM PERMISSIBLE Z-SCORES (Z_max):\n")
            f.write(f"   {Z_max.tolist()}\n")
                
        print(f"📄 Human-readable ground truth saved to: {summary_txt_path}")

    return df, Z_vals, Z_max, gt_sequence, gt_f, W_true


# -------------------------------------------------------------------------
# EXECUTION TEST RUN LOOP ENTRY POINT
# -------------------------------------------------------------------------
if __name__ == '__main__':
    df, Z_vals, Z_max, gt_sequence, gt_f, W_true = simulate_apoe_sustain_dataset(
        N_S_gt=2,
        genetic_signal_strength='strong',
        #gt_f = np.array([0.20, 0.80]),
        seed=None
    )
    
    print('Genetic weights:\n', W_true)
    print('True sequences:\n', gt_sequence)
    print('Subtype prevalence:\n', gt_f)