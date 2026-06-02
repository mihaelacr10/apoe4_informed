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



# old simulation function used for both genetics and outliers at the same time;
# def genetics_simulation(
#                         N=5, M=1000, N_S_gt=2, f_outlier=0.0, f_control=0.10,
#                         gt_fractions=None, W_true=None, use_midpoints=False,
#                         dataset_name=None, base_filename=None, output_path=None, save=False
#                     ):
#     """
#     Master Synthetic Data Generation Harness (Adapted for Genetics):
#     - Accommodates healthy controls (Stage 0) and unclassified data outliers.
#     - Simulates disease-progression z-scores (Discrete or Midpoint Interval).
#     - Generates corresponding aligned APOE/categorical data using W_true.
#     """
#     rng = np.random.default_rng(51)
    
#     # -------------------------------------------------------------------------
#     # STEP 0: SET UP DISCRETE OR STOCHASTIC PREVALENCE DEFAULTS
#     # -------------------------------------------------------------------------
#     if gt_fractions is None:
#         # Standard balanced setup for a 2-subtype breakdown
#         gt_fractions = np.array([0.45, 0.55]) if N_S_gt == 2 else np.ones(N_S_gt)/N_S_gt

#     if W_true is None:
#         W_true = np.array([
#             [0.80, 0.15, 0.05],  # Pathway 0 odds (Low Risk Anchor)
#             [0.10, 0.30, 0.60]   # Pathway 1 odds (High Risk Anchor)
#         ])

#     BiomarkerNames = ['Biomarker ' + str(i) for i in range(N)]
#     Z_vals = np.array([[1, 2, 3]] * N)
#     Z_max = np.array([5] * N)
#     N_stages = np.sum(Z_vals > 0)

#     # Calculate sub-cohort numbers
#     M_patients = int(np.round((1 - f_outlier) * M))
#     M_control = int(np.round(f_control * M_patients))
#     M_outlier = M - M_patients

#     # -------------------------------------------------------------------------
#     # STEP 1: GENERATE STAGE & PATHWAY ASSIGNMENTS
#     # -------------------------------------------------------------------------
#     ground_truth_subtypes = np.random.choice(range(N_S_gt), M_patients, replace=True, p=gt_fractions).astype(int)
    
#     ground_truth_stages_control = np.zeros((M_control, 1))
#     ground_truth_stages_other = np.random.randint(1, N_stages + 1, (M_patients - M_control, 1))
#     ground_truth_stages = np.vstack((ground_truth_stages_control, ground_truth_stages_other)).astype(int)

#     # Generate ground-truth structural trajectory map
   
#     gt_ordering = generate_random_Zscore_sustain_model(Z_vals, N_S_gt, rng=rng)

#     # -------------------------------------------------------------------------
#     # STEP 2: SIMULATE CONTINUOUS BIOMARKER ARRAYS
#     # -------------------------------------------------------------------------
#     # Dynamically select which generator method to execute based on your preference flag
#     if use_midpoints:
        
#         data, data_denoised = generate_data_Zscore_sustain_point(
#             ground_truth_subtypes, ground_truth_stages, gt_ordering, Z_vals, Z_max, M_outlier
#         )
#     else:
        
#         data, data_denoised, _ = generate_data_Zscore_sustain(
#             ground_truth_subtypes, ground_truth_stages, gt_ordering, Z_vals, Z_max
#         )
#         # Manually append uniform noise outliers if standard generator doesn't build them
#         if M_outlier > 0:
#             outlier_noise = np.random.uniform(0, 5, size=(M_outlier, N))
#             data = np.vstack((data, outlier_noise))

#     # Append outlier identifiers to tracking indices if necessary
#     if f_outlier != 0:
#         gt_st_out = -1 * np.ones((M_outlier, 1))
#         ground_truth_stages = np.vstack((ground_truth_stages, gt_st_out))
#         gt_sub_out = -1 * np.ones(M_outlier)
#         ground_truth_subtypes = np.concatenate((ground_truth_subtypes, gt_sub_out))

#     # -------------------------------------------------------------------------
#     # STEP 3: GENERATE ALIGNED GENETIC CATEGORIES
#     # -------------------------------------------------------------------------
#     apoe = generate_apoe_status(M, ground_truth_subtypes, W_true)

#     # -------------------------------------------------------------------------
#     # STEP 4: PACK INTO CLEAN STRUCTURES & EXPORT
#     # -------------------------------------------------------------------------
#     df = pd.DataFrame(data, columns=BiomarkerNames)
#     df['ground_truth_subtypes'] = ground_truth_subtypes
#     df['ground_truth_stages'] = ground_truth_stages.flatten()
#     df['apoe_status'] = apoe

#     # Standardize data labeling constraints for legacy plotting compatibility
#     df.loc[:, 'ground_truth_subtypes'] = df.ground_truth_subtypes.values + 1
#     df.loc[df.ground_truth_stages == 0, 'ground_truth_subtypes'] = 0
#     df.loc[df.ground_truth_stages == -1, 'ground_truth_subtypes'] = -1

#     if save:
#         if output_path is None: output_path = Path("data") / "simulated_data"
#         output_folder = Path(output_path) / dataset_name
#         output_folder.mkdir(parents=True, exist_ok=True)
        
#         df.to_csv(output_folder / f"{base_filename}.csv", index=False)
        
#         parameters = {
#             'Z_vals': Z_vals, 'Z_max': Z_max, 'gt_ordering': gt_ordering, 
#             'W_true': W_true, 'gt_fractions': gt_fractions
#         }
#         with open(output_folder / f"{base_filename}.pkl", "wb") as f:
#             pickle.dump(parameters, f)

#     return df, Z_vals, Z_max, gt_ordering, W_true



def generate_ground_truth_genetic_weights(N_S_gt,
                               N_genetic_categories = 3,
                               genetic_signal_strength = None,
                               rng = None):
  
    
    # 1. DEFAULT / STANDALONE RANDOM STATE (When no specific profile scenario is requested)
    if genetic_signal_strength is None:
        # An alpha of [1, 1, 1] creates a completely flat, symmetric Dirichlet distribution.
        # This generates random rows that are naturally diverse, meaning some subtypes 
        # will randomly get strong signals, and others will get subtle signals.
        alpha_vector = [1] * N_genetic_categories
        W_true = rng.dirichlet(alpha=alpha_vector, size=N_S_gt)
        
    # 2. CONTROLLED TESTING SCENARIOS
    elif genetic_signal_strength == 'uniform':
        # # All subtypes share the same cohort background distribution (Null Signal)
        # alpha_base = np.array([60.0, 30.0, 10.0]) 
        # W_true = rng.dirichlet(alpha=alpha_base, size=N_S_gt)
        
        # 1. Dynamically generate a random, realistic global cohort background profile
        # alpha=[2, 2, 2] creates a smooth random distribution across categories 
        # without forcing them to be perfectly flat or hitting extreme zeros.
        cohort_profile = rng.dirichlet(alpha=[2.0] * N_genetic_categories)
        
        print(f"📊 Generated Random Global Cohort Baseline: {np.round(cohort_profile, 4)}")
        
        # 2. Scale it up by a multiplier of 150 to keep subtype variations tight
        alpha_base = cohort_profile * 150.0
        
        W_true = np.zeros((N_S_gt, N_genetic_categories))
        for s in range(N_S_gt):
            W_true[s] = rng.dirichlet(alpha=alpha_base)
        
    elif genetic_signal_strength == 'moderate':
        # Subtypes have distinct but overlapping profiles (Subtle Signal)
        W_true = np.zeros((N_S_gt, N_genetic_categories))
        for s in range(N_S_gt):
            alpha_vector = np.ones(N_genetic_categories) * 2.0
            alpha_vector[s % N_genetic_categories] += 5.0  
            W_true[s] = rng.dirichlet(alpha=alpha_vector)
            
    elif genetic_signal_strength == 'strong':
        # High Contrast Signal: Heavily enriched, but biologically realistic
        W_true = np.zeros((N_S_gt, N_genetic_categories))
        for s in range(N_S_gt):
            # A baseline floor of 5 allows some presence in non-target categories
            alpha_vector = np.array([5.0, 5.0, 5.0])
            alpha_vector[s % N_genetic_categories] += 45.0  # Aggressive enrichment
            W_true[s] = rng.dirichlet(alpha=alpha_vector)
            
    else:
        # Raise an exception ONLY if they typed a string that doesn't exist
        raise ValueError(
            f"Unknown genetic_signal_strength '{genetic_signal_strength}'. "
            "Must be None, 'uniform', 'moderate', or 'strong'."
        )
        
    
    return W_true

    

# def new_simulation_function(
#                             N = 5,             # number of biomarkers
#                             M = 500,           # number of observations ( e.g. subjects )
#                             M_control = 100,   # number of these that are control subjects
#                             N_S_gt = 2,        # number of ground truth subtypes
#                             W_true = None,     # genetic_weights
#                             genetic_signal_strength = None, #impact genetics has on subtypes 'strong', 'moderate' or 'uniform
#                             seed = None,       # give a seed number if you want to keep randomness fixed (same sequence)
#                             use_midpoints = False,
#                             save = False,
#                             output_path = None,
#                             dataset_name = None,
#                             base_filename = None):    # point value or stage value function
    
#     SuStaInLabels           = []
#     for i in range(N):
#             SuStaInLabels.append( 'Biomarker '+str(i)) # labels of biomarkers for plotting
            
#     Z_vals                  = np.array([[1,2,3]]*N)     # Z-scores for each biomarker
#     Z_max                   = np.array([5]*N)           # maximum z-score
       
#     # ground truth proportion of individuals belonging to each subtype    
#     gt_f                    = [1+0.5*x for x in range(N_S_gt)]
#     gt_f                    = [x/sum(gt_f) for x in gt_f][::-1]
    
#     # define randomness
#     if seed is None:
#         # generate random seed if we dont want to keep the same sequence
#         seed = np.random.default_rng().integers(0, 2**32 - 1, dtype=np.uint64)
#     rng = np.random.default_rng(seed)
    
#     # Generate ground-truth sequence orderings
#     gt_sequence             = generate_random_Zscore_sustain_model(Z_vals,
#                                                             N_S_gt, rng = rng)

    
#     # simulate subtypes and stages for individuals, including a control population at stage 0
#     N_k                     = np.sum(Z_vals>0)+1
#     gt_subtypes             = rng.choice(range(N_S_gt), M, replace=True, p=gt_f)
#     #gt_subtypes             = np.random.choice(range(N_S_gt), M, replace=True, p=gt_f)
#     gt_stages_control       = np.zeros((M_control,1))
#     #gt_stages               = np.concatenate((gt_stages_control,
#     #                                         np.ceil(np.random.rand(M-M_control,1)*N_k)),
#     #                                        axis=0)
#     gt_stages               = np.concatenate((gt_stages_control,
#                                              np.ceil(rng.random((M - M_control, 1)) * N_k)),
#                                             axis=0)
    
#     # generate simulated data
#     if use_midpoints:
#         data, gt_data_denoised, gt_stage_value = generate_data_Zscore_sustain_point(gt_subtypes,
#                                                                    gt_stages,
#                                                                    gt_sequence,
#                                                                    Z_vals,
#                                                                    Z_max)
#     else:
#         data, gt_data_denoised, gt_stage_value = generate_data_Zscore_sustain(gt_subtypes,
#                                                                    gt_stages,
#                                                                    gt_sequence,
#                                                                    Z_vals,
#                                                                    Z_max)
        
    
#     # -------------------------------------------------------------------------
#     # STEP 3: GENERATE ALIGNED GENETIC CATEGORIES
#     # -------------------------------------------------------------------------
#     # 
    
#     if W_true is None:
#         W_true = generate_ground_truth_genetic_weights(
#                     N_S_gt=N_S_gt, 
#                     N_genetic_categories=3, 
#                     genetic_signal_strength=genetic_signal_strength, 
#                     rng=rng
#                 )
#     apoe = generate_apoe_status(M, gt_subtypes, W_true)

#     # -------------------------------------------------------------------------
#     # STEP 4: PACK INTO CLEAN STRUCTURES & EXPORT
#     # -------------------------------------------------------------------------
#     df = pd.DataFrame(data, columns=SuStaInLabels)
#     df['gt_subtypes'] = gt_subtypes
#     df['gt_stages'] = gt_stages.flatten()
#     df['apoe_status'] = apoe

#     # Standardize data labeling constraints for legacy plotting compatibility
#     df.loc[:, 'gt_subtypes'] = df.gt_subtypes.values + 1
#     df.loc[df.gt_stages == 0, 'gt_subtypes'] = 0
    
#     if save:
#         if output_path is None: output_path = Path("data") / "simulated_data"
#         output_folder = Path(output_path) / dataset_name
#         output_folder.mkdir(parents=True, exist_ok=True)
        
#         df.to_csv(output_folder / f"{base_filename}.csv", index=False)
        
#         parameters = {
#             'Z_vals': Z_vals, 'Z_max': Z_max, 'gt_ordering': gt_sequence, 
#             'W_true': W_true, 'gt_fractions': gt_f
#         }
#         with open(output_folder / f"{base_filename}.pkl", "wb") as f:
#             pickle.dump(parameters, f)

#     return df, Z_vals, Z_max, gt_sequence,gt_f, W_true
    #return data, gt_data_denoised, gt_stage_value, gt_stages, gt_subtypes, gt_sequence

def new_simulation_function(
                            N = 5,             # number of biomarkers
                            M = 500,           # number of observations ( e.g. subjects )
                            M_control = 100,   # number of these that are control subjects
                            N_S_gt = 2,        # number of ground truth subtypes
                            W_true = None,     # genetic_weights
                            genetic_signal_strength = None, #impact genetics has on subtypes 'strong', 'moderate' or 'uniform
                            seed = None,       # give a seed number if you want to keep randomness fixed (same sequence)
                            use_midpoints = False,
                            save = False,
                            output_path = None,
                            dataset_name = None,
                            base_filename = None):    # point value or stage value function
    
    SuStaInLabels           = []
    for i in range(N):
            SuStaInLabels.append( 'Biomarker '+str(i)) # labels of biomarkers for plotting
            
    Z_vals                  = np.array([[1,2,3]]*N)     # Z-scores for each biomarker
    Z_max                   = np.array([5]*N)           # maximum z-score
       
    # ground truth proportion of individuals belonging to each subtype    
    gt_f                    = [1+0.5*x for x in range(N_S_gt)]
    gt_f                    = [x/sum(gt_f) for x in gt_f][::-1]
    
    # COMMENT: Generate a random number to use as a seed if none is provided
    if seed is None:
        seed = np.random.default_rng().integers(0, 2**32 - 1, dtype=np.uint64)
    # COMMENT: Create the new random generator using our seed to lock down all randomness
    rng = np.random.default_rng(seed)
    
    # COMMENT: Pass the new generator into the sequence builder to keep it reproducible
    gt_sequence             = generate_random_Zscore_sustain_model(Z_vals,
                                                            N_S_gt, rng = rng)

    
    # simulate subtypes and stages for individuals, including a control population at stage 0
    N_k                     = np.sum(Z_vals>0)+1
    
    # COMMENT: Use the new generator to keep subtype choices identical across identical seeds
    gt_subtypes             = rng.choice(range(N_S_gt), M, replace=True, p=gt_f)
    # gt_subtypes             = np.random.choice(range(N_S_gt), M, replace=True, p=gt_f)
    
    gt_stages_control       = np.zeros((M_control,1))
    
    # COMMENT: Use the new generator with a tuple shape constraint to safely lock patient stages
    gt_stages               = np.concatenate((gt_stages_control,
                                             np.ceil(rng.random((M - M_control, 1)) * N_k)),
                                            axis=0)
    # gt_stages               = np.concatenate((gt_stages_control,
    #                                          np.ceil(np.random.rand(M-M_control,1)*N_k)),
    #                                         axis=0)
    
    # generate simulated data
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
        if output_path is None: output_path = Path("data") / "simulated_data"
        output_folder = Path(output_path) / dataset_name
        output_folder.mkdir(parents=True, exist_ok=True)
        
        df.to_csv(output_folder / f"{base_filename}.csv", index=False)
        
        parameters = {
            'Z_vals': Z_vals, 'Z_max': Z_max, 'gt_ordering': gt_sequence, 
            'W_true': W_true, 'gt_fractions': gt_f
        }
        with open(output_folder / f"{base_filename}.pkl", "wb") as f:
            pickle.dump(parameters, f)

    return df, Z_vals, Z_max, gt_sequence, gt_f, W_true


df, Z_vals, Z_max, gt_sequence, gt_f, W_true = new_simulation_function(N_S_gt= 2,
                                                                       genetic_signal_strength ='moderate' , #'strong'
                                                                       seed = None)
print('Genetic weights',W_true)
print('True sequences',gt_sequence)
print('Subtype prevalence',gt_f)



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



