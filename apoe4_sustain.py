#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon May 11 06:17:43 2026

@author: mihaelacroitor
"""

import pySuStaIn
import numpy as np
from scipy.special import logsumexp

# Change the parent class to pySuStaIn.SuStaIn or the specific implementation class
class APOE4Sustain(pySuStaIn.ZscoreSustain): 
    
    def __init__(self,
                 data,
                 Z_vals,
                 Z_max,
                 SuStaInLabels,
                 N_startpoints,
                 N_S_max,
                 N_iterations_MCMC,
                 output_folder,
                 dataset_name,
                 use_parallel_startpoints=False,
                 apoe4_status=None, # should be numpy not categ
                 model_type='binary'): #binary or categorical
    
        # --- 1. DATA CLEANING ---
        # Find indices where APOE4 is NOT missing (assuming NaNs mark missing data)
        # and where biomarker data is valid
        valid_indices = ~np.isnan(apoe4_status).flatten()
        
        # Drop subjects with missing genetics
        clean_data = data[valid_indices, :]
        clean_apoe = apoe4_status[valid_indices]
        
        print(f"Dropped {np.sum(~valid_indices)} subjects due to missing APOE4 status.")

        # --- 2. PREPROCESSING ---
        if model_type == 'binary':
            # 0 remains 0, (1, 2) become 1
            self.internal_apoe = (clean_apoe > 0).astype(float)
        else:
            # Keep as 0, 1, 2
            self.internal_apoe = clean_apoe.astype(float)

        self.model_type = model_type
        self.apoe4_status = apoe4_status
        
        # Initialize genetic weights
        # We need N_S_max slots because the model will grow 
        self.genetic_alpha = np.zeros(N_S_max)
        self.genetic_beta  = np.zeros(N_S_max)
        
        # We also need a way to store the MCMC samples for these 
        # so we can plot the results later
        self.samples_genetic_alpha = []
        self.samples_genetic_beta  = []
        
        # --- 3. INITIALIZE PARENT ZscoreSustain---
        super().__init__(clean_data,
                         Z_vals,
                         Z_max,
                         SuStaInLabels,
                         N_startpoints,
                         N_S_max,
                         N_iterations_MCMC,
                         output_folder,
                         dataset_name,
                         use_parallel_startpoints)
        

    # # little test
    # def run_sustain_algorithm(self):
    #     print("--- Testing the super class ---")
    #     # In many versions, you call the run method of the parent or a helper
    #     return super().run_sustain_algorithm()
    
    def _calculate_likelihood(self, sustainData, S, f):
        """
        Overrides AbstractSustain._calculate_likelihood to use 
        APOE-informed priors instead of the global 'f'.
        """
        M   = sustainData.getNumSamples()  
        N_S = S.shape[0]
        N   = sustainData.getNumStages()

        # 1. COMPUTE INDIVIDUALIZED PRIOR (f_apoe)
        # We ignore the 'f' passed by the caller and calculate our own
        # based on alpha, beta, and internal_apoe status.
        z = np.zeros((M, N_S))
        for s in range(1, N_S): # Subtype 0 is reference
            z[:, s] = self.genetic_alpha[s] + (self.internal_apoe.flatten() * self.genetic_beta[s])
        
        # Convert log-odds to probabilities (Softmax)
        log_f_apoe = z - logsumexp(z, axis=1, keepdims=True)
        f_apoe = np.exp(log_f_apoe) 

        # 2. CREATE WEIGHT MATRIX
        # Original code: f_val_mat = np.tile(f, (1, N + 1, M))
        # Our version: We use our (M, N_S) matrix and broadcast to (M, N+1, N_S)
        f_val_mat = np.tile(f_apoe[:, np.newaxis, :], (1, N + 1, 1))

        # 3. CALCULATE BIOMARKER PROBABILITIES (p_perm_k)
        p_perm_k = np.zeros((M, N + 1, N_S))
        for s in range(N_S):
            p_perm_k[:, :, s] = self._calculate_likelihood_stage(sustainData, S[s])

        # 4. AGGREGATE RESULTS (Mirroring original logic)
        # total_prob_cluster = probability of subject belonging to subtype
        total_prob_cluster  = np.squeeze(np.sum(p_perm_k * f_val_mat, 1))
        # total_prob_stage = probability of subject being at each stage
        total_prob_stage    = np.sum(p_perm_k * f_val_mat, 2)
        # total_prob_subj = marginalized probability across all subtypes and stages
        total_prob_subj     = np.sum(total_prob_stage, 1)

        # loglike is the sum of log of total probabilities
        loglike             = np.sum(np.log(total_prob_subj + 1e-250))

        return loglike, total_prob_subj, total_prob_stage, total_prob_cluster, p_perm_k
    
    def _compute_log_genetic_prior(self):
        """
        Calculates ln(P(Subtype | APOE4)) using a Multinomial Softmax.
        """
        # Linear predictor: z = alpha + beta * APOE_status
        # Shape: (N_subjects, N_subtypes)
        z = self.genetic_alpha + (self.internal_apoe @ self.genetic_beta)
        
        # Softmax in log space: log_prior = z - log(sum(exp(z)))
        log_prior = z - logsumexp(z, axis=1, keepdims=True)
        
        return log_prior
    
    def _optimise_parameters(self, sustainData, S_init, f_init, rng):
        # 1. Update Biomarkers (Standard SuStaIn)
        # This shuffles the biomarker sequences to find the best fit
        S_opt, f_opt, _ = super()._optimise_parameters(sustainData, S_init, f_init, rng)

        # 2. Get Current Assignments
        # We need the 'responsibilities' to know which subjects belong to which subtype
        _, _, _, total_prob_cluster, _ = self._calculate_likelihood(sustainData, S_opt, f_opt)

        # 3. Update Genetics (The APOE-Informed Step)
        # THIS IS WHERE self.genetic_alpha AND beta ARE ACTUALLY UPDATED
        self._update_genetic_weights(total_prob_cluster)

        # 4. Final Likelihood
        # Recalculate now that alpha and beta have changed
        likelihood_opt, _, _, _, _ = self._calculate_likelihood(sustainData, S_opt, f_opt)

        return S_opt, f_opt, likelihood_opt

    def _update_genetic_weights(self, responsibilities):
        from scipy.optimize import minimize
        
       
        if responsibilities.ndim == 1:
            responsibilities = responsibilities[:, np.newaxis]
        N_S = responsibilities.shape[1]
        if N_S < 2: return 

        # We only optimize alpha/beta for subtypes 1 to N_S-1 (Subtype 0 is reference)
        initial_guess = np.concatenate([self.genetic_alpha[1:N_S], 
                                        self.genetic_beta[1:N_S]])

        # Minimize the negative weighted log-likelihood
        res = minimize(self._genetic_objective, initial_guess, 
                       args=(responsibilities,), method='L-BFGS-B')

        if res.success:
            split = N_S - 1
            self.genetic_alpha[1:N_S] = res.x[:split]
            self.genetic_beta[1:N_S]  = res.x[split:]
            
            # --- ADD THIS FOR TESTING ---
            print(f"\n[EM Optimization] Subtypes: {N_S}")
            print(f"Alpha (Intercepts): {self.genetic_alpha}")
            print(f"Beta (APOE Effect): {self.genetic_beta}")
            
            
    def _genetic_objective(self, params, responsibilities):
        """
        The Negative Weighted Log-Likelihood (Bernoulli/Multinomial).
        """
        M, N_S = responsibilities.shape
        split = N_S - 1
        
        # Reconstruct alpha/beta (fixing Subtype 0 to zero)
        alpha = np.zeros(N_S)
        beta  = np.zeros(N_S)
        alpha[1:N_S] = params[:split]
        beta[1:N_S]  = params[split:]
        
        # 2. Linear predictor calculation
        # Ensure internal_apoe is (M, 1) and beta is (1, N_S)
        # This ensures the result 'z' is (M, N_S)
        apoe_col = self.internal_apoe.reshape(-1, 1)  # Force (M, 1)
        beta_row = beta.reshape(1, -1)                # Force (1, N_S)
        
        # z = alpha (broadcast) + (apoe_col * beta_row)
        # We use * for element-wise broadcasting here which is safer than @ for this shape
        z = alpha + (apoe_col * beta_row) 
        
        # 3. Log-Softmax
        log_p_gen = z - logsumexp(z, axis=1, keepdims=True)
        
        # 4. Weighted Likelihood
        return -np.sum(responsibilities * log_p_gen)
    
    
    
    