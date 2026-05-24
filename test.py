# #!/usr/bin/env python3
# # -*- coding: utf-8 -*-
# """
# Created on Mon May 11 06:17:43 2026

# @author: mihaelacroitor
# """

# import pySuStaIn
# from pySuStaIn import AbstractSustain
# import numpy as np
# from scipy.special import logsumexp
# import tqdm

# from functools import partial, partialmethod
# from tqdm import tqdm


# # Change the parent class to pySuStaIn.SuStaIn or the specific implementation class
# class APOE4Sustain(pySuStaIn.ZscoreSustain): 
    
#     def __init__(self,
#                  data,
#                  Z_vals,
#                  Z_max,
#                  SuStaInLabels,
#                  N_startpoints,
#                  N_S_max,
#                  N_iterations_MCMC,
#                  output_folder,
#                  dataset_name,
#                  use_parallel_startpoints=False,
#                  apoe4_status=None, # should be numpy not categ
#                  model_type='binary'): #binary or categorical
    
#         # --- 1. DATA CLEANING ---
#         # Find indices where APOE4 is NOT missing (assuming NaNs mark missing data)
#         # and where biomarker data is valid
#         valid_indices = ~np.isnan(apoe4_status).flatten()
        
#         # Drop subjects with missing genetics
#         clean_data = data[valid_indices, :]
#         clean_apoe = apoe4_status[valid_indices]
        
#         print(f"Dropped {np.sum(~valid_indices)} subjects due to missing APOE4 status.")

#         # --- 2. PREPROCESSING ---
#         if model_type == 'binary':
#             # 0 remains 0, (1, 2) become 1
#             self.internal_apoe = (clean_apoe > 0).astype(float)
#         else:
#             # Keep as 0, 1, 2
#             self.internal_apoe = clean_apoe.astype(float)

#         self.model_type = model_type
#         self.apoe4_status = apoe4_status
        
#         # Initialize genetic weights
#         # We need N_S_max slots because the model will grow 
#         self.genetic_alpha = np.zeros(N_S_max)
#         self.genetic_beta  = np.zeros(N_S_max)
        
#         # We also need a way to store the MCMC samples for these 
#         # so we can plot the results later
#         self.samples_genetic_alpha = []
#         self.samples_genetic_beta  = []
        
#         # --- 3. INITIALIZE PARENT ZscoreSustain---
#         super().__init__(clean_data,
#                          Z_vals,
#                          Z_max,
#                          SuStaInLabels,
#                          N_startpoints,
#                          N_S_max,
#                          N_iterations_MCMC,
#                          output_folder,
#                          dataset_name,
#                          use_parallel_startpoints)
        

#     # # little test
#     # def run_sustain_algorithm(self):
#     #     print("--- Testing the super class ---")
#     #     # In many versions, you call the run method of the parent or a helper
#     #     return super().run_sustain_algorithm()
    
    
    
#     def _calculate_likelihood(self, sustainData, S, f):
#         """
#         Overrides AbstractSustain._calculate_likelihood to use 
#         APOE-informed priors instead of the global 'f'.
#         """
#         M   = sustainData.getNumSamples()  
#         N_S = S.shape[0]
#         N   = sustainData.getNumStages()

#         # 1. COMPUTE INDIVIDUALIZED PRIOR (f_apoe)
#         # We ignore the 'f' passed by the caller and calculate our own
#         # based on alpha, beta, and internal_apoe status.
#         z = np.zeros((M, N_S))
#         for s in range(1, N_S): # Subtype 0 is reference
#             z[:, s] = self.genetic_alpha[s] + (self.internal_apoe.flatten() * self.genetic_beta[s])
        
#         # Convert log-odds to probabilities (Softmax)
#         log_f_apoe = z - logsumexp(z, axis=1, keepdims=True)
#         f_apoe = np.exp(log_f_apoe) 

#         # 2. CREATE WEIGHT MATRIX
#         # Original code: f_val_mat = np.tile(f, (1, N + 1, M))
#         # Our version: We use our (M, N_S) matrix and broadcast to (M, N+1, N_S)
#         f_val_mat = np.tile(f_apoe[:, np.newaxis, :], (1, N + 1, 1))

#         # 3. CALCULATE BIOMARKER PROBABILITIES (p_perm_k)
#         p_perm_k = np.zeros((M, N + 1, N_S))
#         for s in range(N_S):
#             p_perm_k[:, :, s] = self._calculate_likelihood_stage(sustainData, S[s])

#         # 4. AGGREGATE RESULTS (Mirroring original logic)
#         # total_prob_cluster = probability of subject belonging to subtype
#         total_prob_cluster  = np.squeeze(np.sum(p_perm_k * f_val_mat, 1))
#         # total_prob_stage = probability of subject being at each stage
#         total_prob_stage    = np.sum(p_perm_k * f_val_mat, 2)
#         # total_prob_subj = marginalized probability across all subtypes and stages
#         total_prob_subj     = np.sum(total_prob_stage, 1)

#         # loglike is the sum of log of total probabilities
#         loglike             = np.sum(np.log(total_prob_subj + 1e-250))

#         return loglike, total_prob_subj, total_prob_stage, total_prob_cluster, p_perm_k
    
#     def _compute_log_genetic_prior(self):
#         """
#         Calculates ln(P(Subtype | APOE4)) using a Multinomial Softmax.
#         """
#         # Linear predictor: z = alpha + beta * APOE_status
#         # Shape: (N_subjects, N_subtypes)
#         z = self.genetic_alpha + (self.internal_apoe @ self.genetic_beta)
        
#         # Softmax in log space: log_prior = z - log(sum(exp(z)))
#         log_prior = z - logsumexp(z, axis=1, keepdims=True)
        
#         return log_prior
    
#     def _optimise_parameters(self, sustainData, S_init, f_init, rng):
#         # 1. Update Biomarkers (Standard SuStaIn)
#         # This shuffles the biomarker sequences to find the best fit
#         S_opt, f_opt, _ = super()._optimise_parameters(sustainData, S_init, f_init, rng)

#         # 2. Get Current Assignments
#         # We need the 'responsibilities' to know which subjects belong to which subtype
#         _, _, _, total_prob_cluster, _ = self._calculate_likelihood(sustainData, S_opt, f_opt)

#         # 3. Update Genetics (The APOE-Informed Step)
#         # THIS IS WHERE self.genetic_alpha AND beta ARE ACTUALLY UPDATED
#         self._update_genetic_weights(total_prob_cluster)

#         # 4. Final Likelihood
#         # Recalculate now that alpha and beta have changed
#         likelihood_opt, _, _, _, _ = self._calculate_likelihood(sustainData, S_opt, f_opt)

#         return S_opt, f_opt, likelihood_opt

#     def _update_genetic_weights(self, responsibilities):
#         from scipy.optimize import minimize
        
       
#         if responsibilities.ndim == 1:
#             responsibilities = responsibilities[:, np.newaxis]
#         N_S = responsibilities.shape[1]
#         if N_S < 2: return 

#         # We only optimize alpha/beta for subtypes 1 to N_S-1 (Subtype 0 is reference)
#         initial_guess = np.concatenate([self.genetic_alpha[1:N_S], 
#                                         self.genetic_beta[1:N_S]])

#         # Minimize the negative weighted log-likelihood
#         res = minimize(self._genetic_objective, initial_guess, 
#                        args=(responsibilities,), method='L-BFGS-B')

#         if res.success:
#             split = N_S - 1
#             self.genetic_alpha[1:N_S] = res.x[:split]
#             self.genetic_beta[1:N_S]  = res.x[split:]
            
#             # --- ADD THIS FOR TESTING ---
#             print(f"\n[EM Optimization] Subtypes: {N_S}")
#             print(f"Alpha (Intercepts): {self.genetic_alpha}")
#             print(f"Beta (APOE Effect): {self.genetic_beta}")
            
            
#     def _genetic_objective(self, params, responsibilities):
#         """
#         The Negative Weighted Log-Likelihood (Bernoulli/Multinomial).
#         """
#         M, N_S = responsibilities.shape
#         split = N_S - 1
        
#         # Reconstruct alpha/beta (fixing Subtype 0 to zero)
#         alpha = np.zeros(N_S)
#         beta  = np.zeros(N_S)
#         alpha[1:N_S] = params[:split]
#         beta[1:N_S]  = params[split:]
        
#         # 2. Linear predictor calculation
#         # Ensure internal_apoe is (M, 1) and beta is (1, N_S)
#         # This ensures the result 'z' is (M, N_S)
#         apoe_col = self.internal_apoe.reshape(-1, 1)  # Force (M, 1)
#         beta_row = beta.reshape(1, -1)                # Force (1, N_S)
        
#         # z = alpha (broadcast) + (apoe_col * beta_row)
#         # We use * for element-wise broadcasting here which is safer than @ for this shape
#         z = alpha + (apoe_col * beta_row) 
        
#         # 3. Log-Softmax
#         log_p_gen = z - logsumexp(z, axis=1, keepdims=True)
        
#         # 4. Weighted Likelihood
#         return -np.sum(responsibilities * log_p_gen)
    
#     # def _optimise_mcmc_settings(self, sustainData, seq_init, f_init):
#     #     # Optimise the perturbation size for the MCMC algorithm
#     #     n_iterations_MCMC_optimisation      = int(1e4)  
#     #     n_passes_optimisation               = 3

#     #     # 1. INITIALIZE SIGMAS
#     #     seq_sigma_currentpass               = 1
#     #     f_sigma_currentpass                 = 0.01 
#     #     # Genetic sigmas start small (magic numbers to kickstart the std calculation)
#     #     alpha_sigma_currentpass             = 0.1
#     #     beta_sigma_currentpass              = 0.1

#     #     N_S                                 = seq_init.shape[0]

#     #     for i in range(n_passes_optimisation):
#     #         # 2. RUN MCMC WITH CURRENT SIGMAS
#     #         # We override perform_mcmc below to accept and use these genetic sigmas
#     #         _, _, _, samples_sequence_currentpass, samples_f_currentpass, _, \
#     #         samples_alpha_currentpass, samples_beta_currentpass = self._perform_mcmc(
#     #                                                                 sustainData,
#     #                                                                 seq_init,
#     #                                                                 f_init,
#     #                                                                 n_iterations_MCMC_optimisation,
#     #                                                                 seq_sigma_currentpass,
#     #                                                                 f_sigma_currentpass,
#     #                                                                 alpha_sigma_currentpass,
#     #                                                                 beta_sigma_currentpass)

#     #         # 3. BIOMARKER SIGMA UPDATE (Standard SuStaIn logic)
#     #         samples_position_currentpass    = np.zeros(samples_sequence_currentpass.shape)
#     #         for s in range(N_S):
#     #             for sample in range(n_iterations_MCMC_optimisation):
#     #                 temp_seq                         = samples_sequence_currentpass[s, :, sample]
#     #                 temp_inv                         = np.array([0] * samples_sequence_currentpass.shape[1])
#     #                 temp_inv[temp_seq.astype(int)]   = np.arange(samples_sequence_currentpass.shape[1])
#     #                 samples_position_currentpass[s, :, sample] = temp_inv

#     #         seq_sigma_currentpass            = np.std(samples_position_currentpass, axis=2, ddof=1)
#     #         seq_sigma_currentpass[seq_sigma_currentpass < 0.01] = 0.01

#     #         # 4. FRACTION SIGMA UPDATE
#     #         f_sigma_currentpass              = np.std(samples_f_currentpass, axis=1, ddof=1)

#     #         # 5. GENETIC SIGMA UPDATE (The New Logic)
#     #         # We calculate the standard deviation of the accepted alpha/beta samples
#     #         # This tells the next MCMC pass how much it's allowed to "jitter"
#     #         alpha_sigma_currentpass          = np.std(samples_alpha_currentpass, axis=0, ddof=1)
#     #         beta_sigma_currentpass           = np.std(samples_beta_currentpass, axis=0, ddof=1)
            
#     #         # Floor the sigmas so they don't collapse to zero
#     #         alpha_sigma_currentpass[alpha_sigma_currentpass < 0.001] = 0.001
#     #         beta_sigma_currentpass[beta_sigma_currentpass < 0.001]   = 0.001

#     #     # Final optimized sigmas to be used in the main MCMC
#     #     return seq_sigma_currentpass, f_sigma_currentpass, alpha_sigma_currentpass, beta_sigma_currentpass
    
    
#     # def _perform_mcmc(self, sustainData, seq_init, f_init, n_iterations, seq_sigma, f_sigma, alpha_sigma, beta_sigma):
#     #     # Take MCMC samples of the uncertainty in the SuStaIn model parameters
#     #     N   = self.stage_zscore.shape[1]
#     #     N_S = seq_init.shape[0]

#     #     if isinstance(f_sigma, float):
#     #         f_sigma = np.array([f_sigma])

#     #     # 1. INITIALIZE SAMPLE STORAGE
#     #     samples_sequence   = np.zeros((N_S, N, n_iterations))
#     #     samples_f          = np.zeros((N_S, n_iterations))
#     #     samples_alpha      = np.zeros((N_S, n_iterations))
#     #     samples_beta       = np.zeros((N_S, n_iterations))
#     #     samples_likelihood = np.zeros((n_iterations, 1))

#     #     # 2. SET INITIAL STATE (Iteration 0)
#     #     samples_sequence[:, :, 0] = seq_init
#     #     samples_f[:, 0]           = f_init
#     #     samples_alpha[:, 0]       = self.genetic_alpha[:N_S]
#     #     samples_beta[:, 0]        = self.genetic_beta[:N_S]

#     #     # Calculate the starting likelihood for the initial state
#     #     L_start, _, _, _, _       = self._calculate_likelihood(sustainData, seq_init, f_init)
#     #     samples_likelihood[0]     = L_start

#     #     # Progress bar setup
#     #     tqdm_update_iters = int(n_iterations/1000) if n_iterations > 100000 else None 

#     #     for i in tqdm(range(n_iterations), "MCMC Iteration", n_iterations, miniters=tqdm_update_iters):
#     #         if i == 0: continue

#     #         # --- STEP A: PROPOSE BIOMARKER MOVE ---
#     #         # (Following standard SuStaIn shuffling logic)
#     #         seq_order = self.global_rng.permutation(N_S)
#     #         for s in seq_order:
#     #             move_event_from  = int(np.ceil(N * self.global_rng.random())) - 1
#     #             current_sequence = samples_sequence[s, :, i - 1].copy()

#     #             current_location = np.array([0] * N)
#     #             current_location[current_sequence.astype(int)] = np.arange(N)

#     #             selected_event      = int(current_sequence[move_event_from])
#     #             this_stage_zscore   = self.stage_zscore[0, selected_event]
#     #             selected_biomarker  = self.stage_biomarker_index[0, selected_event]
                
#     #             possible_zscores_biomarker = self.stage_zscore[self.stage_biomarker_index == selected_biomarker]
#     #             min_filter = possible_zscores_biomarker < this_stage_zscore
#     #             max_filter = possible_zscores_biomarker > this_stage_zscore
#     #             events     = np.array(range(N))
                
#     #             if np.any(min_filter):
#     #                 min_z_bound = max(possible_zscores_biomarker[min_filter])
#     #                 min_event   = events[((self.stage_zscore[0] == min_z_bound).astype(int) + (self.stage_biomarker_index[0] == selected_biomarker).astype(int)) == 2]
#     #                 low_bound   = current_location[min_event] + 1
#     #             else: low_bound = 0

#     #             if np.any(max_filter):
#     #                 max_z_bound = min(possible_zscores_biomarker[max_filter])
#     #                 max_event   = events[((self.stage_zscore[0] == max_z_bound).astype(int) + (self.stage_biomarker_index[0] == selected_biomarker).astype(int)) == 2]
#     #                 up_bound    = current_location[max_event]
#     #             else: up_bound = N

#     #             possible_pos = np.arange(low_bound, up_bound) if low_bound != up_bound else np.array([0])
#     #             distance     = possible_pos - move_event_from

#     #             this_seq_sig = seq_sigma if isinstance(seq_sigma, int) else seq_sigma[s, selected_event]
#     #             weight       = AbstractSustain.calc_coeff(this_seq_sig) * AbstractSustain.calc_exp(distance, 0., this_seq_sig)
#     #             weight      /= np.sum(weight)
                
#     #             move_to = possible_pos[self.global_rng.choice(range(len(possible_pos)), 1, p=weight)]
#     #             current_sequence = np.delete(current_sequence, move_event_from, 0)
#     #             samples_sequence[s, :, i] = np.concatenate([current_sequence[np.arange(move_to)], [selected_event], current_sequence[np.arange(move_to, N - 1)]])

#     #         # --- STEP B: PROPOSE FRACTION MOVE ---
#     #         new_f = samples_f[:, i - 1] + f_sigma * self.global_rng.standard_normal()
#     #         samples_f[:, i] = (np.fabs(new_f) / np.sum(np.fabs(new_f)))

#     #         # --- STEP C: PROPOSE GENETIC MOVE ---
#     #         # Jitter Alpha and Beta for all subtypes except the reference (index 0)
#     #         new_alpha = samples_alpha[:, i - 1].copy()
#     #         new_beta  = samples_beta[:, i - 1].copy()
            
#     #         new_alpha[1:N_S] += alpha_sigma[1:N_S] * self.global_rng.standard_normal(N_S - 1)
#     #         new_beta[1:N_S]  += beta_sigma[1:N_S]  * self.global_rng.standard_normal(N_S - 1)
            
#     #         samples_alpha[:, i] = new_alpha
#     #         samples_beta[:, i]  = new_beta
            
#     #         # --- STEP D: UPDATE CLASS & CALCULATE CANDIDATE LIKELIHOOD ---
#     #         # Temporarily set the class attributes to the candidate parameters
#     #         self.genetic_alpha = samples_alpha[:, i]
#     #         self.genetic_beta  = samples_beta[:, i]

#     #         # Calculate likelihood of the proposed state
#     #         candidate_L, _, _, _, _ = self._calculate_likelihood(sustainData, samples_sequence[:, :, i], samples_f[:, i])
#     #         samples_likelihood[i]   = candidate_L

#     #         # --- STEP E: METROPOLIS-HASTINGS REJECT (The Overwrite) ---
#     #         # If the move is rejected, replace current index i with values from index i-1
#     #         ratio = np.exp(samples_likelihood[i] - samples_likelihood[i - 1])
#     #         if ratio < self.global_rng.random():
#     #             samples_likelihood[i]     = samples_likelihood[i - 1]
#     #             samples_sequence[:, :, i] = samples_sequence[:, :, i - 1]
#     #             samples_f[:, i]           = samples_f[:, i - 1]
#     #             samples_alpha[:, i]       = samples_alpha[:, i - 1]
#     #             samples_beta[:, i]        = samples_beta[:, i - 1]
                
#     #             # IMPORTANT: Sync global state back to the accepted (previous) state
#     #             self.genetic_alpha = samples_alpha[:, i]
#     #             self.genetic_beta  = samples_beta[:, i]

#     #     # --- FINAL: SET CLASS TO MAXIMUM LIKELIHOOD STATE ---
#     #     ml_idx = np.argmax(samples_likelihood)
#     #     self.genetic_alpha = samples_alpha[:, ml_idx]
#     #     self.genetic_beta  = samples_beta[:, ml_idx]

#     #     return (samples_sequence[:, :, ml_idx], samples_f[:, ml_idx], samples_likelihood[ml_idx], 
#     #             samples_sequence, samples_f, samples_likelihood, samples_alpha, samples_beta)
    
    
    
#     # def _estimate_uncertainty_sustain_model(self, sustainData, seq_init, f_init):
#     #     # 1. SAVE THE ML STATE FOUND DURING EM
#     #     # We store these so we can "re-center" the model after the tuning passes
#     #     ml_alpha_initial = self.genetic_alpha.copy()
#     #     ml_beta_initial  = self.genetic_beta.copy()

#     #     # 2. Tuning Passes
#     #     # Note: self.genetic_alpha/beta WILL change inside here as it tunes sigmas
#     #     seq_sigma_opt, f_sigma_opt, alpha_sigma_opt, beta_sigma_opt = self._optimise_mcmc_settings(
#     #         sustainData, seq_init, f_init)

#     #     # 3. RESET TO ML STATE BEFORE FULL MCMC
#     #     # This ensures the "Full MCMC" starts exactly at the peak found during EM
#     #     self.genetic_alpha = ml_alpha_initial.copy()
#     #     self.genetic_beta  = ml_beta_initial.copy()

#     #     # 4. Run the full MCMC
#     #     ml_sequence, ml_f, ml_likelihood, \
#     #     samples_sequence, samples_f, samples_likelihood, \
#     #     samples_alpha, samples_beta = self._perform_mcmc(
#     #         sustainData, seq_init, f_init, self.N_iterations_MCMC, 
#     #         seq_sigma_opt, f_sigma_opt, alpha_sigma_opt, beta_sigma_opt)

#     #     # The _perform_mcmc already sets self.genetic_alpha to the best 
#     #     # found in the samples, so we are good to return.
#     #     return ml_sequence, ml_f, ml_likelihood, samples_sequence, samples_f, samples_likelihood, samples_alpha, samples_beta
    
    
#     # def _perform_em(self, sustainData, current_sequence, current_f, rng):
#     #     # Perform an E-M procedure to estimate parameters of SuStaIn model
#     #     MaxIter = 100
#     #     N = sustainData.getNumStages()
#     #     N_S = current_sequence.shape[0]
        
#     #     # 1. INITIAL LIKELIHOOD (Uses current class alpha/beta)
#     #     current_likelihood, _, _, _, _ = self._calculate_likelihood(sustainData, current_sequence, current_f)

#     #     # 2. INITIALIZE TRACKING ARRAYS FOR GENETICS
#     #     # We need to store these for each EM iteration just like sequence/f
#     #     samples_alpha = np.nan * np.ones((MaxIter, N_S))
#     #     samples_beta  = np.nan * np.ones((MaxIter, N_S))
        
#     #     # Store the starting state
#     #     current_alpha = self.genetic_alpha[:N_S].copy()
#     #     current_beta  = self.genetic_beta[:N_S].copy()
#     #     samples_alpha[0, :] = current_alpha
#     #     samples_beta[0, :]  = current_beta

#     #     terminate = 0
#     #     iteration = 0
        
#     #     # (Standard SuStaIn tracking arrays)
#     #     samples_sequence = np.nan * np.ones((MaxIter, N, N_S))
#     #     samples_f = np.nan * np.ones((MaxIter, N_S))
#     #     samples_likelihood = np.nan * np.ones((MaxIter, 1))
#     #     samples_sequence[0, :, :] = current_sequence.T
#     #     samples_f[0, :] = np.array(current_f).flatten()
#     #     samples_likelihood[0] = current_likelihood

#     #     while terminate == 0:
#     #         # 3. OPTIMISE PARAMETERS (This calls your M-step which updates class alpha/beta)
#     #         candidate_sequence, candidate_f, candidate_likelihood = self._optimise_parameters(
#     #             sustainData, current_sequence, current_f, rng)

#     #         # Capture what the class now thinks are the best genetics
#     #         candidate_alpha = self.genetic_alpha[:N_S].copy()
#     #         candidate_beta  = self.genetic_beta[:N_S].copy()

#     #         # Convergence check
#     #         HAS_converged = np.fabs((candidate_likelihood - current_likelihood) / max(candidate_likelihood, current_likelihood)) < 1e-6
            
#     #         if HAS_converged:
#     #             terminate = 1
#     #         else:
#     #             if candidate_likelihood > current_likelihood:
#     #                 # ACCEPT: Move the "current" pointers to the new best
#     #                 current_sequence   = candidate_sequence
#     #                 current_f          = candidate_f
#     #                 current_likelihood = candidate_likelihood
#     #                 current_alpha      = candidate_alpha
#     #                 current_beta       = candidate_beta
#     #             else:
#     #                 # REJECT: If the move was worse, we MUST revert the class genetics
#     #                 self.genetic_alpha[:N_S] = current_alpha
#     #                 self.genetic_beta[:N_S]  = current_beta

#     #         # Store iteration results
#     #         samples_sequence[iteration, :, :] = current_sequence.T
#     #         samples_f[iteration, :] = current_f
#     #         samples_likelihood[iteration] = current_likelihood
#     #         samples_alpha[iteration, :] = current_alpha
#     #         samples_beta[iteration, :] = current_beta

#     #         if iteration == (MaxIter - 1):
#     #             terminate = 1
#     #         iteration += 1

#     #     # 4. FINAL ML PARAMETERS
#     #     self.genetic_alpha[:N_S] = current_alpha
#     #     self.genetic_beta[:N_S]  = current_beta
        
#     #     return current_sequence, current_f, current_likelihood, samples_sequence, samples_f, samples_likelihood, samples_alpha, samples_beta
    
#     # def _find_ml_mixture_iteration(self, sustainData, seq_init, f_init, seed_seq):
#     #     rng = np.random.default_rng(seed_seq)
#     #     # Catch 8 values
#     #     res = self._perform_em(sustainData, seq_init, f_init, rng)
#     #     return res # Now a tuple of 8
    
#     # def _find_ml_mixture(self, sustainData, seq_init, f_init):
#     #     N_S = seq_init.shape[0]
#     #     partial_iter = partial(self._find_ml_mixture_iteration, sustainData, seq_init, f_init)
#     #     seed_sequences = np.random.SeedSequence(self.global_rng.integers(1e10))
#     #     pool_output_list = list(self.pool.map(partial_iter, seed_sequences.spawn(self.N_startpoints)))
    
#     #     # Trackers for the genetics
#     #     ml_alpha_mat = np.zeros((N_S, self.N_startpoints))
#     #     ml_beta_mat  = np.zeros((N_S, self.N_startpoints))
        
#     #     # Standard trackers
#     #     ml_sequence_mat = np.zeros((N_S, sustainData.getNumStages(), self.N_startpoints))
#     #     ml_f_mat = np.zeros((N_S, self.N_startpoints))
#     #     ml_likelihood_mat = np.zeros((self.N_startpoints, 1))
    
#     #     for i in range(self.N_startpoints):
#     #         ml_sequence_mat[:, :, i] = pool_output_list[i][0]
#     #         ml_f_mat[:, i]           = pool_output_list[i][1]
#     #         ml_likelihood_mat[i]     = pool_output_list[i][2]
#     #         # CAPTURE THE GENETICS FROM EACH STARTPOINT
#     #         ml_alpha_mat[:, i]       = pool_output_list[i][6] 
#     #         ml_beta_mat[:, i]        = pool_output_list[i][7]
    
#     #     # Find the absolute winner
#     #     ix = np.argmax(ml_likelihood_mat)
    
#     #     # Sync the main program's class to the winning genetics
#     #     self.genetic_alpha[:N_S] = ml_alpha_mat[:, ix]
#     #     self.genetic_beta[:N_S]  = ml_beta_mat[:, ix]
    
#     #     return (ml_sequence_mat[:, :, ix], ml_f_mat[:, ix], ml_likelihood_mat[ix], 
#     #             ml_sequence_mat, ml_f_mat, ml_likelihood_mat, 
#     #             ml_alpha_mat[:, ix], ml_beta_mat[:, ix]) # Return 8 values


### Issues with the version bellow: apoe4 indexing, and a few other issues

# import os
# import numpy as np
# import pickle
# from scipy.special import logsumexp
# from scipy.optimize import minimize
# from functools import partial
# from tqdm.auto import tqdm
# from pathlib import Path
# import pySuStaIn

# class APOE4Sustain(pySuStaIn.ZscoreSustain): 
    
#     def __init__(self, data, Z_vals, Z_max, biomarker_labels, N_startpoints, N_S_max, 
#                  N_iterations_MCMC, output_folder, dataset_name, use_parallel_startpoints, 
#                  apoe4_status, seed=None):
        
#         valid_indices = ~np.isnan(apoe4_status).flatten()
#         clean_data = data[valid_indices, :]
        
#         self.master_apoe = apoe4_status[valid_indices].reshape(-1, 1).astype(float)
#         self.internal_apoe = self.master_apoe.copy()
        
#         print(f"Dropped {np.sum(~valid_indices)} subjects due to missing APOE4 status.")

#         self.genetic_alpha = np.zeros(N_S_max)
#         self.genetic_beta  = np.zeros(N_S_max)
        
#         super().__init__(clean_data, Z_vals, Z_max, biomarker_labels, N_startpoints, 
#                          N_S_max, N_iterations_MCMC, output_folder, dataset_name, 
#                          use_parallel_startpoints, seed)

#     # ==========================================
#     # 1. CORE LIKELIHOOD
#     # ==========================================
#     def _calculate_likelihood(self, sustainData, S, f):
#         """
#         Computes the likelihood of a mixture of models using individualized 
#         APOE-informed priors (alpha and beta) instead of global fractions (f).
#         """
#         # 1. Get dimensions
#         M   = sustainData.getNumSamples()   # Current number of subjects
#         N_S = S.shape[0]                    # Number of subtypes
#         N   = sustainData.getNumStages()    # Number of stages

#         # 2. DYNAMIC GENETIC SLICING
#         # During hierarchical splitting, SuStaIn passes a subset of subjects.
#         # We must ensure current_apoe has shape (M, 1) to match sustainData.
#         # self.internal_apoe is updated by the calling function to match the context.
#         current_apoe = self.internal_apoe[:M]

#         # 3. COMPUTE INDIVIDUALIZED PRIOR (f_apoe)
#         # Linear predictor: z = alpha + beta * APOE_status
#         # We use the class-level alpha and beta parameters
#         z = self.genetic_alpha[:N_S] + (current_apoe * self.genetic_beta[:N_S])
        
#         # Softmax transformation: pi = exp(z) / sum(exp(z))
#         # Calculated in log-space for numerical stability
#         log_f_apoe = z - logsumexp(z, axis=1, keepdims=True)
#         f_apoe = np.exp(log_f_apoe) 

#         # 4. CREATE WEIGHT MATRIX
#         # Reshape f_apoe from (M, N_S) to (M, N+1, N_S) for broadcasting
#         # This mirrors how the original SuStaIn tiled the global 'f'
#         f_val_mat = np.tile(f_apoe[:, np.newaxis, :], (1, N + 1, 1))

#         # 5. CALCULATE BIOMARKER PROBABILITIES (p_perm_k)
#         # This calculates the probability of the data given a subtype and stage
#         p_perm_k = np.zeros((M, N + 1, N_S))
#         for s in range(N_S):
#             # _calculate_likelihood_stage is the standard SuStaIn biomarker logic
#             p_perm_k[:, :, s] = self._calculate_likelihood_stage(sustainData, S[s])

#         # 6. AGGREGATE RESULTS
#         # total_prob_cluster: P(x_i | Subtype s) * P(Subtype s | Genetics_i)
#         # Shape: (M, N_S)
#         total_prob_cluster  = np.squeeze(np.sum(p_perm_k * f_val_mat, 1))
        
#         # total_prob_stage: Marginalized across subtypes for each stage
#         # Shape: (M, N+1)
#         total_prob_stage    = np.sum(p_perm_k * f_val_mat, 2)
        
#         # total_prob_subj: Total probability for the subject (marginalized over all stages)
#         # Shape: (M,)
#         total_prob_subj     = np.sum(total_prob_stage, 1)

#         # 7. LOG-LIKELIHOOD
#         # Sum of log probabilities across all subjects
#         # We add a small constant (1e-250) to prevent log(0) errors
#         loglike             = np.sum(np.log(total_prob_subj + 1e-250))

#         return loglike, total_prob_subj, total_prob_stage, total_prob_cluster, p_perm_k
#     # ==========================================
#     # 2. OPTIMIZATION & TRACKING
#     # ==========================================
#     def _optimise_parameters(self, sustainData, S_init, f_init, rng):
#         S_opt, f_opt, _ = super()._optimise_parameters(sustainData, S_init, f_init, rng)
#         _, _, _, responsibilities, _ = self._calculate_likelihood(sustainData, S_opt, f_opt)
#         self._update_genetic_weights(responsibilities, sustainData.getNumSamples())
#         likelihood_opt, _, _, _, _ = self._calculate_likelihood(sustainData, S_opt, f_opt)
#         return S_opt, f_opt, likelihood_opt

#     def _update_genetic_weights(self, responsibilities, M_current):
#         if responsibilities.ndim == 1: responsibilities = responsibilities[:, np.newaxis]
#         N_S = responsibilities.shape[1]
#         if N_S < 2: return 
        
#         initial_guess = np.concatenate([self.genetic_alpha[1:N_S], self.genetic_beta[1:N_S]])
#         res = minimize(self._genetic_objective, initial_guess, args=(responsibilities, M_current), method='L-BFGS-B')
        
#         if res.success:
#             self.genetic_alpha[1:N_S] = res.x[:N_S-1]
#             self.genetic_beta[1:N_S]  = res.x[N_S-1:]
#             print(f"\n[EM Update] {N_S} Subtypes | Subtype 1 Beta: {self.genetic_beta[1]:.4f}")

#     def _genetic_objective(self, params, responsibilities, M_current):
#         N_S = responsibilities.shape[1]
#         alpha, beta = np.zeros(N_S), np.zeros(N_S)
#         alpha[1:N_S], beta[1:N_S] = params[:N_S-1], params[N_S-1:]
#         z = alpha + (self.internal_apoe[:M_current] * beta)
#         log_p_gen = z - logsumexp(z, axis=1, keepdims=True)
#         return -np.sum(responsibilities * log_p_gen)

#     # ==========================================
#     # 3. WRAPPERS
#     # ==========================================
#     def _perform_em(self, sustainData, current_sequence, current_f, rng):
#         res = super()._perform_em(sustainData, current_sequence, current_f, rng)
#         N_S = current_sequence.shape[0]
#         return (*res, self.genetic_alpha[:N_S].copy(), self.genetic_beta[:N_S].copy())

#     def _find_ml_iteration(self, sustainData, seed_seq):
#         rng = np.random.default_rng(seed_seq)
#         res = self._perform_em(sustainData, self._initialise_sequence(sustainData, rng), [1], rng)
#         return res[0], res[1], res[2]

#     def _find_ml_split_iteration(self, sustainData, seed_seq):
#         rng = np.random.default_rng(seed_seq)
#         M_local = sustainData.getNumSamples()
        
#         # 1. Generate cluster assignments for this subset of subjects
#         vals = rng.random(M_local)
#         cluster_assignment = np.ceil(2 * vals).astype(int)
#         seq_init = np.zeros((2, sustainData.getNumStages()))
        
#         # Capture the genetics that belong to THIS subset (sustainData)
#         # We slice self.internal_apoe down to M_local to ensure the masks match
#         current_subset_apoe = self.internal_apoe[:M_local].copy()
        
#         # Backup the genetics as they were when we entered this function
#         original_internal_apoe = self.internal_apoe.copy()
        
#         for s in range(2):
#             # idx is a boolean mask for the current M_local subjects
#             idx = cluster_assignment.flatten() == (s + 1)
            
#             # Slice the subset genetics to match the sub-subset data
#             self.internal_apoe = current_subset_apoe[idx]
            
#             # Run EM on the sub-subset
#             res_s = self._perform_em(sustainData.reindex(idx), 
#                                      self._initialise_sequence(sustainData, rng), [1], rng)
#             seq_init[s, :] = res_s[0]
        
#         # Restore the genetics to the state they were for the full subset
#         self.internal_apoe = current_subset_apoe
#         res = self._perform_em(sustainData, seq_init, [0.5, 0.5], rng)
        
#         # Restore completely before exiting
#         self.internal_apoe = original_internal_apoe
        
#         return res[0], res[1], res[2]

#     def _find_ml_mixture_iteration(self, sustainData, seq_init, f_init, seed_seq):
#         rng = np.random.default_rng(seed_seq)
#         return self._perform_em(sustainData, seq_init, f_init, rng)

#     def _find_ml_mixture(self, sustainData, seq_init, f_init):
#         N_S = seq_init.shape[0]
#         partial_iter = partial(self._find_ml_mixture_iteration, sustainData, seq_init, f_init)
#         seed_sequences = np.random.SeedSequence(self.global_rng.integers(1e10))
#         pool_output_list = list(self.pool.map(partial_iter, seed_sequences.spawn(self.N_startpoints)))
#         ml_lik_mat = np.zeros(self.N_startpoints)
#         ml_alpha_mat, ml_beta_mat = np.zeros((N_S, self.N_startpoints)), np.zeros((N_S, self.N_startpoints))
#         ml_sequence_mat = np.zeros((N_S, sustainData.getNumStages(), self.N_startpoints))
#         ml_f_mat = np.zeros((N_S, self.N_startpoints))
#         for i in range(self.N_startpoints):
#             ml_sequence_mat[:, :, i], ml_f_mat[:, i], ml_lik_mat[i] = pool_output_list[i][0:3]
#             ml_alpha_mat[:, i], ml_beta_mat[:, i] = pool_output_list[i][6:8]
#         ix = np.argmax(ml_lik_mat)
#         self.genetic_alpha[:N_S], self.genetic_beta[:N_S] = ml_alpha_mat[:, ix], ml_beta_mat[:, ix]
#         return (ml_sequence_mat[:, :, ix:ix+1], ml_f_mat[:, ix:ix+1], [ml_lik_mat[ix]], ml_sequence_mat, ml_f_mat, ml_lik_mat.reshape(-1,1))

#     # ==========================================
#     # 4. MCMC & SAFE TRACKING
#     # ==========================================
#     # ==========================================
    
    
#     def _perform_mcmc(self, sustainData, seq_init, f_init, n_iterations, seq_sigma, f_sigma, alpha_sigma=None, beta_sigma=None):
#         N_S, N = seq_init.shape[0], sustainData.getNumStages()
        
#         if alpha_sigma is None: alpha_sigma = np.ones(N_S) * 0.05 
#         if beta_sigma is None:  beta_sigma  = np.ones(N_S) * 0.05

#         samples_sequence = np.zeros((N_S, N, n_iterations))
#         samples_f = np.zeros((N_S, n_iterations))
#         samples_alpha, samples_beta = np.zeros((N_S, n_iterations)), np.zeros((N_S, n_iterations))
#         samples_likelihood = np.zeros((n_iterations, 1))

#         samples_sequence[:,:,0], samples_f[:,0] = seq_init, f_init
#         samples_alpha[:,0], samples_beta[:,0] = self.genetic_alpha[:N_S], self.genetic_beta[:N_S]
#         L_start, _, _, _, _ = self._calculate_likelihood(sustainData, seq_init, f_init)
#         samples_likelihood[0] = L_start

#         for i in tqdm(range(1, n_iterations), desc="MCMC Sampling"):
#             # 1. Start with previous state
#             prev_seq = samples_sequence[:, :, i-1].copy()
#             prev_f   = samples_f[:, i-1].copy()
#             prev_alpha = samples_alpha[:, i-1].copy()
#             prev_beta  = samples_beta[:, i-1].copy()

#             # 2. PROPOSE BIOMARKER MOVE
#             # We use the parent's _optimise_parameters but for just 1 iteration
#             # to ensure the sequence remains valid (Monotonicity check)
#             # This is safer than manual shuffling
#             new_seq, _, _ = super()._optimise_parameters(sustainData, prev_seq, prev_f, self.global_rng)

#             # 3. PROPOSE GENETIC MOVE
#             new_alpha = prev_alpha + (alpha_sigma * self.global_rng.standard_normal(N_S))
#             new_beta  = prev_beta  + (beta_sigma  * self.global_rng.standard_normal(N_S))
#             new_alpha[0], new_beta[0] = 0, 0 # Keep Subtype 0 as reference

#             # 4. METROPOLIS-HASTINGS
#             # Sync class params for likelihood calculation
#             self.genetic_alpha[:N_S], self.genetic_beta[:N_S] = new_alpha, new_beta
            
#             L_prop, _, _, _, _ = self._calculate_likelihood(sustainData, new_seq, prev_f)
            
#             # Acceptance logic (Log-space)
#             if (L_prop - samples_likelihood[i-1]) > np.log(self.global_rng.random()):
#                 samples_likelihood[i] = L_prop
#                 samples_sequence[:,:,i] = new_seq
#                 samples_f[:,i] = prev_f
#                 samples_alpha[:,i] = new_alpha
#                 samples_beta[:,i] = new_beta
#             else:
#                 samples_likelihood[i] = samples_likelihood[i-1]
#                 samples_sequence[:,:,i] = samples_sequence[:,:,i-1]
#                 samples_f[:,i] = samples_f[:,i-1]
#                 samples_alpha[:,i] = samples_alpha[:,i-1]
#                 samples_beta[:,i] = samples_beta[:,i-1]

#         # Final best state update
#         ml_idx = np.argmax(samples_likelihood)
#         self.genetic_alpha[:N_S] = samples_alpha[:, ml_idx]
#         self.genetic_beta[:N_S] = samples_beta[:, ml_idx]

#         return (samples_sequence[:,:,ml_idx], samples_f[:,ml_idx], samples_likelihood[ml_idx], 
#                 samples_sequence, samples_f, samples_likelihood, samples_alpha, samples_beta)
    
#     # ==========================================
#     # 5. TUNING WRAPPER
#     # ==========================================
#     def _estimate_uncertainty_sustain_model(self, sustainData, seq_init, f_init):
#         """Runs a short MCMC to tune sigmas, then the main MCMC."""
#         print("Tuning MCMC proposal distributions...")
#         # Start with very small sigmas to ensure we actually move
#         tuned_res = self._perform_mcmc(sustainData, seq_init, f_init, 
#                                        n_iterations=int(self.N_iterations_MCMC * 0.1), 
#                                        seq_sigma=1, f_sigma=0.01, 
#                                        alpha_sigma=np.ones(seq_init.shape[0])*0.01, 
#                                        beta_sigma=np.ones(seq_init.shape[0])*0.01)
        
#         # Extract the standard deviation of the "tuned" samples to use as the final sigmas
#         # This is a standard trick to get the acceptance rate near 23-44%
#         new_alpha_sig = np.std(tuned_res[6], axis=1) + 1e-4
#         new_beta_sig  = np.std(tuned_res[7], axis=1) + 1e-4
        
#         print(f"Tuning complete. Final Beta Sigmas: {new_beta_sig}")
        
#         return self._perform_mcmc(sustainData, seq_init, f_init, 
#                                   self.N_iterations_MCMC, 1, 0.01, 
#                                   new_alpha_sig, new_beta_sig)
#     # ==========================================
#     # 5. RUNNER
#     # ==========================================
#     def run_sustain_algorithm(self, plot=False, plot_format="png", **kwargs):
#         ml_sequence_prev_EM, ml_f_prev_EM = [], []
#         pickle_dir = os.path.join(self.output_folder, 'pickle_files')
#         if not os.path.isdir(pickle_dir): os.mkdir(pickle_dir)
#         sustainData = self._AbstractSustain__sustainData 

#         for s in range(self.N_S_max):
#             pickle_filename = os.path.join(pickle_dir, f"{self.dataset_name}_subtype{s}.pickle")
#             if Path(pickle_filename).exists():
#                 with open(pickle_filename, 'rb') as f: loaded = pickle.load(f)
#                 ml_sequence_EM, ml_f_EM = loaded["ml_sequence_EM"], loaded["ml_f_EM"]
#                 samples_sequence, samples_f = loaded["samples_sequence"], loaded["samples_f"]
#                 samples_alpha, samples_beta, samples_likelihood = loaded.get("samples_alpha"), loaded.get("samples_beta"), loaded.get("samples_likelihood")
#             else:
#                 print(f"\n--- Fitting Subtype {s+1} ---")
#                 res_em = self._estimate_ml_sustain_model_nplus1_clusters(sustainData, ml_sequence_prev_EM, ml_f_prev_EM)
                
#                 # Fix dimensions for EM results
#                 ml_sequence_EM = res_em[0].squeeze(axis=2) if res_em[0].ndim==3 else res_em[0]
#                 ml_f_EM = res_em[1].flatten()
                
#                 uncertainty_res = self._estimate_uncertainty_sustain_model(sustainData, ml_sequence_EM, ml_f_EM)
#                 ml_sequence, ml_f, ml_likelihood, samples_sequence, samples_f, samples_likelihood, samples_alpha, samples_beta = uncertainty_res

#             ml_sequence_prev_EM, ml_f_prev_EM = ml_sequence_EM, ml_f_EM
            
#             # Subtype and stage for individuals (1000 samples)
#             res_indiv = self.subtype_and_stage_individuals(sustainData, samples_sequence, samples_f, 1000)
            
#             if not Path(pickle_filename).exists():
#                 save_dict = {
#                     "samples_sequence": samples_sequence, "samples_f": samples_f, 
#                     "samples_likelihood": samples_likelihood, "samples_alpha": samples_alpha, 
#                     "samples_beta": samples_beta, "ml_sequence_EM": ml_sequence_EM, 
#                     "ml_f_EM": ml_f_EM, "ml_subtype": res_indiv[0], "ml_stage": res_indiv[2], 
#                     "prob_subtype_stage": res_indiv[6]
#                 }
#                 with open(pickle_filename, 'wb') as f: pickle.dump(save_dict, f)

#         # Final returns match the expected signature in your calling code
#         return ml_sequence_EM, ml_f_EM, res_indiv[0], res_indiv[1], res_indiv[2], res_indiv[3], res_indiv[6]
    
    
    
    
    
    
    
    
##### CHAT GPT VERSION

# Corrected APOE4-SuStaIn Extension


# 2. FULL APOE4-SUSTAIN CLASS


import os
import pickle
import numpy as np

from pathlib import Path
from functools import partial

from scipy.special import logsumexp
from scipy.optimize import minimize

from tqdm.auto import tqdm

import pySuStaIn
from pySuStaIn import AbstractSustainData


class ZScoreSustainData(AbstractSustainData):

    def __init__(self, data, numStages, apoe4=None):
        self.data = data
        self.__numStages = numStages
        self.apoe4 = apoe4  # (N,) or (N,1)

    def getNumSamples(self):
        return self.data.shape[0]

    def getNumBiomarkers(self):
        return self.data.shape[1]

    def getNumStages(self):
        return self.__numStages

    def reindex(self, index):
        print('Reindexing sucessful')
        print(f"Patients have apoe: {self.data[index],self.apoe4[index]}")
        return ZScoreSustainData(
            self.data[index,],
            self.__numStages,
            None if self.apoe4 is None else self.apoe4[index]
        )

class APOE4Sustain(pySuStaIn.ZscoreSustain):

    def __init__(self,
                 data,
                 Z_vals,
                 Z_max,
                 biomarker_labels,
                 N_startpoints,
                 N_S_max,
                 N_iterations_MCMC,
                 output_folder,
                 dataset_name,
                 use_parallel_startpoints,
                 apoe4_status,
                 seed=None):

        valid = ~np.isnan(apoe4_status).flatten()

        #clean_data = data[valid, :]
        clean_apoe = apoe4_status[valid].astype(float).reshape(-1, 1)

        self.master_apoe = clean_apoe

        self.genetic_alpha = np.zeros(N_S_max)
        self.genetic_beta = np.zeros(N_S_max)
        
        # We need this for the custom object initialization
        self.lambda_reg = 0.01  # Added this because your objective function references self.lambda_reg but it wasn't initialized!

        # FIX 1: Use a standard local variable name, NOT a double underscore variable!
        sustain_data_instance = ZScoreSustainData(
            clean_data,
            Z_vals.shape[1],
            clean_apoe
        )

        # FIX 2: Pass the local instance directly as the first argument.
        # AbstractSustain will naturally absorb it into its own internal name-mangled tracking.
        super().__init__(
            sustain_data_instance,
            Z_vals,
            Z_max,
            biomarker_labels,
            N_startpoints,
            N_S_max,
            N_iterations_MCMC,
            output_folder,
            dataset_name,
            use_parallel_startpoints,
            seed
        )



    # ==========================================================
    # GENETIC PRIOR
    # ==========================================================

    def _compute_genetic_prior(self, sustainData, N_S):

        current_apoe = self._get_current_apoe(sustainData)

        z = (
            self.genetic_alpha[:N_S][None, :]
            +
            current_apoe * self.genetic_beta[:N_S][None, :]
        )

        log_prior = z - logsumexp(z, axis=1, keepdims=True)

        return np.exp(log_prior)

    # ==========================================================
    # CORE LIKELIHOOD
    # ==========================================================

   
    def _calculate_likelihood(self, sustainData, S, f):
        M = sustainData.getNumSamples()
        N = sustainData.getNumStages()
        N_S = S.shape[0]
    
        apoe = self._get_current_apoe(sustainData)
    
        z = np.zeros((M, N_S))
        for s in range(N_S):
            z[:, s] = (
                self.genetic_alpha[s]
                + apoe[:, 0] * self.genetic_beta[s]
            )
    
        log_f = z - logsumexp(z, axis=1, keepdims=True)
        f_apoe = np.exp(log_f)  # Subtype fractions updated by genetic covariates
    
        f_mat = np.tile(f_apoe[:, None, :], (1, N + 1, 1))
    
        p_perm_k = np.zeros((M, N + 1, N_S))
        for s in range(N_S):
            p_perm_k[:, :, s] = self._calculate_likelihood_stage(
                sustainData,
                S[s]
            )
    
        # Combine probabilities weighted by subtype fractions
        # total_stage shape: (M, N + 1)
        total_stage = np.sum(p_perm_k * f_mat, 2)
        
        # total shape: (M,)
        total = np.sum(total_stage, 1)
    
        loglike = np.sum(np.log(total + 1e-250))
    
        # FIXED: Compute and return full 5-tuple payload required by EM/MCMC loops
        responsibilities = (p_perm_k * f_mat) / (total[:, None, None] + 1e-250)
        subclone_prob = np.sum(p_perm_k * f_mat, 1) / (total[:, None] + 1e-250)
        stage_prob = total_stage / (total[:, None] + 1e-250)
    
        return loglike, total, responsibilities, subclone_prob, stage_prob

    # ==========================================================
    # GENETIC OBJECTIVE
    # ==========================================================

    def _genetic_objective(
        self,
        params,
        responsibilities,
        sustainData
    ):

        N_S = responsibilities.shape[1]

        alpha = np.zeros(N_S)
        beta = np.zeros(N_S)

        alpha[1:] = params[:N_S - 1]
        beta[1:] = params[N_S - 1:]

        current_apoe = self._get_current_apoe(sustainData)

        z = (
            alpha[None, :]
            +
            current_apoe * beta[None, :]
        )

        log_prior = z - logsumexp(z, axis=1, keepdims=True)

        neg_ll = -np.sum(
            responsibilities * log_prior
        )

        penalty = self.lambda_reg * (
            np.sum(alpha[1:] ** 2)
            +
            np.sum(beta[1:] ** 2)
        )

        return neg_ll + penalty

    # ==========================================================
    # UPDATE GENETIC WEIGHTS
    # ==========================================================

    def _update_genetic_weights(
        self,
        responsibilities,
        sustainData
    ):

        if responsibilities.ndim == 1:
            responsibilities = responsibilities[:, None]

        N_S = responsibilities.shape[1]

        if N_S < 2:
            return

        initial_guess = np.concatenate([
            self.genetic_alpha[1:N_S],
            self.genetic_beta[1:N_S]
        ])

        result = minimize(
            self._genetic_objective,
            initial_guess,
            args=(responsibilities, sustainData),
            method='L-BFGS-B'
        )

        if result.success:

            self.genetic_alpha[1:N_S] = result.x[:N_S - 1]
            self.genetic_beta[1:N_S] = result.x[N_S - 1:]

            print("\n=== Genetic Update ===")
            print("Alpha:", self.genetic_alpha[:N_S])
            print("Beta:", self.genetic_beta[:N_S])

    # ==========================================================
    # EM OPTIMIZATION
    # ==========================================================

    def _optimise_parameters(
        self,
        sustainData,
        S_init,
        f_init,
        rng
    ):

        S_opt, f_opt, _ = super()._optimise_parameters(
            sustainData,
            S_init,
            f_init,
            rng
        )

        (
            _,
            _,
            responsibilities,
            _,
            _
        ) = self._calculate_likelihood(
            sustainData,
            S_opt,
            f_opt
        )

        self._update_genetic_weights(
            responsibilities,
            sustainData
        )

        (
            likelihood_opt,
            _,
            _,
            _,
            _
        ) = self._calculate_likelihood(
            sustainData,
            S_opt,
            f_opt
        )

        return S_opt, f_opt, likelihood_opt

    # ==========================================================
    # RESET GENETICS ON SPLITS
    # ==========================================================

    def _reset_genetics(self, N_S):

        self.genetic_alpha[:N_S] = 0
        self.genetic_beta[:N_S] = 0

    # ==========================================================
    # MIXTURE SEARCH
    # ==========================================================

    def _find_ml_mixture_iteration(
        self,
        sustainData,
        seq_init,
        f_init,
        seed_seq
    ):

        rng = np.random.default_rng(seed_seq)

        return self._perform_em(
            sustainData,
            seq_init,
            f_init,
            rng
        )

    def _find_ml_mixture(
        self,
        sustainData,
        seq_init,
        f_init
    ):

        N_S = seq_init.shape[0]

        self._reset_genetics(N_S)

        partial_iter = partial(
            self._find_ml_mixture_iteration,
            sustainData,
            seq_init,
            f_init
        )

        seed_sequences = np.random.SeedSequence(
            self.global_rng.integers(1e10)
        )

        pool_output_list = list(
            self.pool.map(
                partial_iter,
                seed_sequences.spawn(self.N_startpoints)
            )
        )

        ml_likelihood_mat = np.zeros(self.N_startpoints)

        ml_sequence_mat = np.zeros((
            N_S,
            sustainData.getNumStages(),
            self.N_startpoints
        ))

        ml_f_mat = np.zeros((N_S, self.N_startpoints))

        ml_alpha_mat = np.zeros((N_S, self.N_startpoints))
        ml_beta_mat = np.zeros((N_S, self.N_startpoints))

        for i in range(self.N_startpoints):

            ml_sequence_mat[:, :, i] = pool_output_list[i][0]
            ml_f_mat[:, i] = pool_output_list[i][1]
            ml_likelihood_mat[i] = pool_output_list[i][2]

            ml_alpha_mat[:, i] = pool_output_list[i][6]
            ml_beta_mat[:, i] = pool_output_list[i][7]

        ix = np.argmax(ml_likelihood_mat)

        self.genetic_alpha[:N_S] = ml_alpha_mat[:, ix]
        self.genetic_beta[:N_S] = ml_beta_mat[:, ix]

        return (
            ml_sequence_mat[:, :, ix:ix+1],
            ml_f_mat[:, ix:ix+1],
            [ml_likelihood_mat[ix]],
            ml_sequence_mat,
            ml_f_mat,
            ml_likelihood_mat.reshape(-1, 1)
        )

    # ==========================================================
    # MCMC
    # ==========================================================

    def _perform_mcmc(
        self,
        sustainData,
        seq_init,
        f_init,
        n_iterations,
        seq_sigma,
        f_sigma,
        alpha_sigma=None,
        beta_sigma=None
    ):

        # IMPORTANT:
        # genetics are FIXED during MCMC
        # otherwise detailed balance breaks

        N_S = seq_init.shape[0]
        N = sustainData.getNumStages()

        samples_sequence = np.zeros((N_S, N, n_iterations))
        samples_f = np.zeros((N_S, n_iterations))
        samples_likelihood = np.zeros((n_iterations, 1))

        samples_alpha = np.tile(
            self.genetic_alpha[:N_S][:, None],
            (1, n_iterations)
        )

        samples_beta = np.tile(
            self.genetic_beta[:N_S][:, None],
            (1, n_iterations)
        )

        samples_sequence[:, :, 0] = seq_init
        samples_f[:, 0] = f_init

        L0, _, _, _, _ = self._calculate_likelihood(
            sustainData,
            seq_init,
            f_init
        )

        samples_likelihood[0] = L0

        for i in tqdm(range(1, n_iterations)):

            new_seq, _, _ = super()._optimise_parameters(
                sustainData,
                samples_sequence[:, :, i-1],
                samples_f[:, i-1],
                self.global_rng
            )

            L_new, _, _, _, _ = self._calculate_likelihood(
                sustainData,
                new_seq,
                samples_f[:, i-1]
            )

            if (
                L_new - samples_likelihood[i-1]
            ) > np.log(self.global_rng.random()):

                samples_sequence[:, :, i] = new_seq
                samples_f[:, i] = samples_f[:, i-1]
                samples_likelihood[i] = L_new

            else:

                samples_sequence[:, :, i] = samples_sequence[:, :, i-1]
                samples_f[:, i] = samples_f[:, i-1]
                samples_likelihood[i] = samples_likelihood[i-1]

        ml_idx = np.argmax(samples_likelihood)

        return (
            samples_sequence[:, :, ml_idx],
            samples_f[:, ml_idx],
            samples_likelihood[ml_idx],
            samples_sequence,
            samples_f,
            samples_likelihood,
            samples_alpha,
            samples_beta
        )

