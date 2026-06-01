#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed May 27 13:50:07 2026


This code simulates synthetic data for my APOE-Informed model

It uses pySuStain simfuncs simulation functions (modifed point_value) 
Adds simulation of genetics apoe categorical values
Creates a function to generate both biomarkers and apoe, 


@author: mihaelacroitor
"""

import numpy as np
import pandas as pd
import pickle
from pathlib import Path

from genetics_simfuncs import generate_random_Zscore_sustain_model,\
                              generate_data_Zscore_sustain,\
                              generate_data_Zscore_sustain_point



def generate_apoe_status(M,subtypes, W_true):
    
    ''' 
    M        : number of patients
    subtypes : array containing subtype membership for each patient
    W_true   : Ground-truth genetic weights, shape (N_subtypes, N_categories).
    '''
    # Generate the categorical labels based on ground-truth tracking matrix
    
    subject_genetics = np.zeros(M, dtype=int)
    for i in range(M):
        # Sample category 0, 1, or 2 based on those specific odds.
        subject_genetics[i] = np.random.choice([0, 1, 2], p=W_true[subtypes[i]])
    
    return subject_genetics




def genetics_simulation(
                        N=5, M=1000, N_S_gt=2, f_outlier=0.0, f_control=0.10,
                        gt_fractions=None, W_true=None, use_midpoints=False,
                        dataset_name=None, base_filename=None, output_path=None, save=False
                    ):
    """
    Master Synthetic Data Generation Harness (Adapted for Genetics):
    - Accommodates healthy controls (Stage 0) and unclassified data outliers.
    - Simulates disease-progression z-scores (Discrete or Midpoint Interval).
    - Generates corresponding aligned APOE/categorical data using W_true.
    """
    rng = np.random.default_rng(51)
    
    # -------------------------------------------------------------------------
    # STEP 0: SET UP DISCRETE OR STOCHASTIC PREVALENCE DEFAULTS
    # -------------------------------------------------------------------------
    if gt_fractions is None:
        # Standard balanced setup for a 2-subtype breakdown
        gt_fractions = np.array([0.45, 0.55]) if N_S_gt == 2 else np.ones(N_S_gt)/N_S_gt

    if W_true is None:
        W_true = np.array([
            [0.80, 0.15, 0.05],  # Pathway 0 odds (Low Risk Anchor)
            [0.10, 0.30, 0.60]   # Pathway 1 odds (High Risk Anchor)
        ])

    BiomarkerNames = ['Biomarker ' + str(i) for i in range(N)]
    Z_vals = np.array([[1, 2, 3]] * N)
    Z_max = np.array([5] * N)
    N_stages = np.sum(Z_vals > 0)

    # Calculate sub-cohort numbers
    M_patients = int(np.round((1 - f_outlier) * M))
    M_control = int(np.round(f_control * M_patients))
    M_outlier = M - M_patients

    # -------------------------------------------------------------------------
    # STEP 1: GENERATE STAGE & PATHWAY ASSIGNMENTS
    # -------------------------------------------------------------------------
    ground_truth_subtypes = np.random.choice(range(N_S_gt), M_patients, replace=True, p=gt_fractions).astype(int)
    
    ground_truth_stages_control = np.zeros((M_control, 1))
    ground_truth_stages_other = np.random.randint(1, N_stages + 1, (M_patients - M_control, 1))
    ground_truth_stages = np.vstack((ground_truth_stages_control, ground_truth_stages_other)).astype(int)

    # Generate ground-truth structural trajectory map
   
    gt_ordering = generate_random_Zscore_sustain_model(Z_vals, N_S_gt, rng=rng)

    # -------------------------------------------------------------------------
    # STEP 2: SIMULATE CONTINUOUS BIOMARKER ARRAYS
    # -------------------------------------------------------------------------
    # Dynamically select which generator method to execute based on your preference flag
    if use_midpoints:
        
        data, data_denoised = generate_data_Zscore_sustain_point(
            ground_truth_subtypes, ground_truth_stages, gt_ordering, Z_vals, Z_max, M_outlier
        )
    else:
        
        data, data_denoised, _ = generate_data_Zscore_sustain(
            ground_truth_subtypes, ground_truth_stages, gt_ordering, Z_vals, Z_max
        )
        # Manually append uniform noise outliers if standard generator doesn't build them
        if M_outlier > 0:
            outlier_noise = np.random.uniform(0, 5, size=(M_outlier, N))
            data = np.vstack((data, outlier_noise))

    # Append outlier identifiers to tracking indices if necessary
    if f_outlier != 0:
        gt_st_out = -1 * np.ones((M_outlier, 1))
        ground_truth_stages = np.vstack((ground_truth_stages, gt_st_out))
        gt_sub_out = -1 * np.ones(M_outlier)
        ground_truth_subtypes = np.concatenate((ground_truth_subtypes, gt_sub_out))

    # -------------------------------------------------------------------------
    # STEP 3: GENERATE ALIGNED GENETIC CATEGORIES
    # -------------------------------------------------------------------------
    apoe = generate_apoe_status(M, ground_truth_subtypes, W_true)

    # -------------------------------------------------------------------------
    # STEP 4: PACK INTO CLEAN STRUCTURES & EXPORT
    # -------------------------------------------------------------------------
    df = pd.DataFrame(data, columns=BiomarkerNames)
    df['ground_truth_subtypes'] = ground_truth_subtypes
    df['ground_truth_stages'] = ground_truth_stages.flatten()
    df['apoe_status'] = apoe

    # Standardize data labeling constraints for legacy plotting compatibility
    df.loc[:, 'ground_truth_subtypes'] = df.ground_truth_subtypes.values + 1
    df.loc[df.ground_truth_stages == 0, 'ground_truth_subtypes'] = 0
    df.loc[df.ground_truth_stages == -1, 'ground_truth_subtypes'] = -1

    if save:
        if output_path is None: output_path = Path("data") / "simulated_data"
        output_folder = Path(output_path) / dataset_name
        output_folder.mkdir(parents=True, exist_ok=True)
        
        df.to_csv(output_folder / f"{base_filename}.csv", index=False)
        
        parameters = {
            'Z_vals': Z_vals, 'Z_max': Z_max, 'gt_ordering': gt_ordering, 
            'W_true': W_true, 'gt_fractions': gt_fractions
        }
        with open(output_folder / f"{base_filename}.pkl", "wb") as f:
            pickle.dump(parameters, f)

    return df, Z_vals, Z_max, gt_ordering, W_true

def new_simulation_function(point = False):
    
    N                       = 5         # number of biomarkers
    M                       = 500       # number of observations ( e.g. subjects )
    M_control               = 100       # number of these that are control subjects
    N_S_gt                  = 2         # number of ground truth subtypes
    
    SuStaInLabels           = []
    for i in range(N):
            SuStaInLabels.append( 'Biomarker '+str(i)) # labels of biomarkers for plotting
            
    Z_vals                  = np.array([[1,2,3]]*N)     # Z-scores for each biomarker
    Z_max                   = np.array([5]*N)           # maximum z-score
       
    # ground truth proportion of individuals belonging to each subtype    
    gt_f                    = [1+0.5*x for x in range(N_S_gt)]
    gt_f                    = [x/sum(gt_f) for x in gt_f][::-1]
    
    # ground truth sequence for each subtype
    gt_sequence             = generate_random_Zscore_sustain_model(Z_vals,
                                                            N_S_gt)
    
    # simulate subtypes and stages for individuals, including a control population at stage 0
    N_k                     = np.sum(Z_vals>0)+1
    gt_subtypes             = np.random.choice(range(N_S_gt), M, replace=True, p=gt_f)
    gt_stages_control       = np.zeros((M_control,1))
    gt_stages               = np.concatenate((gt_stages_control,
                                             np.ceil(np.random.rand(M-M_control,1)*N_k)),
                                            axis=0)
    
    # generate simulated data
    if point:
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
    return data, gt_data_denoised, gt_stage_value, gt_stages, gt_subtypes, gt_sequence

# data, gt_data_denoised, gt_stage_value, gt_stages, gt_subtypes, gt_sequence = new_simulation_function(point=False)
# df = pd.DataFrame(data)
# df['gt_subtypes'] = gt_subtypes
# df['gt_stages'] = gt_stages
# print('Gt sequence', gt_sequence)

# import matplotlib.pyplot as plt
# plt.hist(gt_stages)
# plt.title('Stages distribution')


# data1, gt_data_denoised1, gt_stage_valu1, gt_stages1, gt_subtypes1, gt_sequence1 = new_simulation_function(point=True)
# df1 = pd.DataFrame(data1)
# df1['gt_subtypes'] = gt_subtypes1
# df1['gt_stages'] = gt_stages1
# print('Gt sequence', gt_sequence1)

# import matplotlib.pyplot as plt
# plt.hist(gt_stages)
# plt.title('Stages distribution')


# Example of how to simulate genetics 

# M_patients = 100
# N_S_gt = 2
# gt_fractions = [0.5,0.5]
# mock_subtypes = np.random.choice(range(N_S_gt), M_patients, replace=True, p=gt_fractions).astype(int)

# W_true = np.array([
#     [0.80, 0.15, 0.05],  # Pathway 0: 80% chance of Cat 0 (Low Risk Anchor)
#     [0.10, 0.30, 0.60]   # Pathway 1: 60% chance of Cat 2 (High Risk Anchor)
# ])

# apoe = generate_apoe_status(M_patients, mock_subtypes, W_true)

# print('Subtypes',mock_subtypes)
# print('apoe', apoe)



