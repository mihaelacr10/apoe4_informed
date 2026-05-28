#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed May 27 15:12:16 2026

@author: mihaelacroitor
"""

# In your Jupyter Notebook:
import numpy as np
from genetics_simulation_utils import genetics_simulation

# Define your scenarios cleanly in their own cell
W_SCENARIOS = {
    'STRONG': np.array([[0.80, 0.15, 0.05], [0.10, 0.30, 0.60]]),
    'SUBTLE': np.array([[0.45, 0.35, 0.20], [0.30, 0.40, 0.30]]),
    'NULL':   np.array([[0.33, 0.33, 0.33], [0.33, 0.33, 0.33]])
}

# Run whichever test scenario you want tomorrow morning with one line:
df, Z_vals, Z_max, gt_ordering, W_true = genetics_simulation(W_true=W_SCENARIOS['STRONG'])


# when im ready I want to make simrun very professional in the same way
# simrun.py from pysustain works and run all evaluations from this one script

# #!/usr/bin/env python3
# # -*- coding: utf-8 -*-
# """
# Created on Wed May 27 15:45:00 2026

# Master Evaluation Runner for the Genetic-Informed SuStaIn Model.
# Mimics the official pySuStaIn in-memory simulation philosophy.

# @author: mihaelacroitor
# """

# import os
# import numpy as np
# import pandas as pd
# import pickle
# from pySuStaIn.GeneticsSuStaIn import GeneticsSuStaIn
# from genetics_simulation_utils import genetics_simulation

# def main():
#     # ***************** parameters for generating the ground-truth
#     N                = 5          # number of biomarkers
#     M                = 1000       # number of subjects
#     N_S_ground_truth = 2          # number of ground truth subtypes
#     gt_fractions     = np.array([0.45, 0.55])
    
#     # ***************** parameters for SuStaIn-based inference
#     N_startpoints     = 25
#     N_iterations_MCMC = int(1e4)   # Set to 1e5 or 1e6 for final publication run
    
#     # Define experiment output directory
#     output_folder = os.path.join(os.getcwd(), 'sim_genetics_results')
#     if not os.path.isdir(output_folder):
#         os.mkdir(output_folder)

#     # Define your 3 evaluation scenarios explicitly
#     SCENARIOS = {
#         'STRONG': np.array([[0.80, 0.15, 0.05], [0.10, 0.30, 0.60]]),
#         'SUBTLE': np.array([[0.45, 0.35, 0.20], [0.30, 0.40, 0.30]]),
#         'NULL':   np.array([[0.33, 0.33, 0.33], [0.33, 0.33, 0.33]])
#     }

#     # ***************** Automated Loop Over Scenarios
#     for name, W_matrix in SCENARIOS.items():
#         print(f"\n=====================================================")
#         print(f"LAUNCHING EVALUATION: SCENARIO {name}")
#         print(f"=====================================================")
        
#         # 1. Generate everything IN MEMORY (No CSV file writing)
#         df, Z_vals, Z_max, gt_ordering, W_true = genetics_simulation(
#             N=N, M=M, N_S_gt=N_S_ground_truth, 
#             gt_fractions=gt_fractions, W_true=W_matrix, save=False
#         )
        
#         # Extract the raw matrices from the dataframe cleanly
#         BiomarkerNames = ['Biomarker ' + str(i) for i in range(N)]
#         X_data         = df[BiomarkerNames].values
#         y_genetics     = df['apoe_status'].values
        
#         # 2. Initialize your specialized model
#         sustain_model           = GeneticsSuStaIn(X_data, y_genetics, N_genetic_categories=3)
#         sustain_model.apoe_flag = True
        
#         # 3. Create optimization anchors
#         S_start = np.array([np.random.permutation(sustain_model.N) for _ in range(N_S_ground_truth)])
#         f_start = np.ones(N_S_ground_truth) / N_S_ground_truth
#         W_start = np.ones((N_S_ground_truth, 3)) / 3
#         rng     = np.random.default_rng(42)
        
#         # 4. Fit the model using Combined, No Burn-In unified loop optimization!
#         print(f"Running joint EM parameter estimation for Scenario {name}...")
#         ml_S, ml_f, ml_W, ml_lik, samples_S, samples_f, samples_W, samples_lik = sustain_model._perform_em(
#             sustain_model.data, S_start, f_start, rng, W_start
#         )
        
#         # 5. Save ONLY the final results to disk for your notebook visualizations
#         results = {
#             'scenario_name': name,
#             'ml_S': ml_S, 'ml_f': ml_f, 'ml_W': ml_W, 'ml_lik': ml_lik,
#             'samples_S': samples_S, 'samples_f': samples_f, 'samples_W': samples_W, 'samples_lik': samples_lik,
#             'gt_ordering': gt_ordering, 'W_true': W_true, 'df_true_labels': df[['ground_truth_subtypes', 'ground_truth_stages']]
#         }
        
#         pkl_path = os.path.join(output_folder, f"scenario_{name}_evaluation.pkl")
#         with open(pkl_path, "wb") as f:
#             pickle.dump(results, f)
            
#         print(f"Scenario {name} finalized. Metrics written cleanly to: {pkl_path}")

#     print('\nAll batch simulations completed successfully.')

# if __name__ == '__main__':
#     np.random.seed(42)
#     main()