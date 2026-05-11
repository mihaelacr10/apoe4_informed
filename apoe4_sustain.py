#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon May 11 06:17:43 2026

@author: mihaelacroitor
"""
# import pySuStaIn
# from pySuStaIn.ZscoreSustain import ZscoreSustain

# class Apoe4Sustain(ZscoreSustain):
#     def __init__(self, data, apoe_status, *args, **kwargs):
#         # Initialize the normal z-score model
#         super().__init__(data, *args, **kwargs)
#         self.apoe_status = apoe_status
        
#     def _calculate_likelihood(self, sequence): # (Note: check exact function name in pysustain)
        
#         # 1. Let ZscoreSustain do the hard biomarker math!
#         # This calls the function that already exists in the parent classes
#         biomarker_likelihood = super()._calculate_likelihood(sequence)
        
#         # 2. Calculate your new APOE4 prior (the math you are building)
#         apoe4_prior = self._calculate_apoe4_prior()
        
#         # 3. Combine them 
#         # (Note: if pysustain returns log-likelihoods, you ADD them instead of multiply)
#         total_likelihood = biomarker_likelihood * apoe4_prior
        
#         return total_likelihood

#     def _calculate_apoe4_prior(self):
#         # Your custom logic here for carrier vs non-carrier probabilities
#         pass
    

# from pySuStaIn.ZscoreSustain import ZscoreSustain

# class APOE4Sustain(ZscoreSustain):
#     def __init__(self, *args, **kwargs):
#         # We'll handle the APOE4 status properly later; 
#         # for now, just pass everything to the original SuStaIn
#         super().__init__(*args, **kwargs)

#     # We are temporarily overriding this method just to see if it triggers
#     def _calculate_likelihood(self, sustainData, S, f):
#         print(f"--- Smoke Test: APOE4Sustain is running for subtype {S} ---")
        
#         # Now call the ORIGINAL method so the math still works
#         return super()._calculate_likelihood(self, sustainData, S, f)

# import pySuStaIn
# import numpy as np

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
#                  apoe4_status=None): # Add our new variable at the end
        
#         # Pass the standard SuStaIn arguments up to the parent class
#         super().__init__(data,
#                          Z_vals,
#                          Z_max,
#                          SuStaInLabels,
#                          N_startpoints,
#                          N_S_max,
#                          N_iterations_MCMC,
#                          output_folder,
#                          dataset_name,
#                          use_parallel_startpoints)
        
#         # Store the APOE4 status in our subclass
#         self.apoe4_status = apoe4_status
        
#     def _calculate_sequenced_data_likelihood(self, data, S):
#         # The Smoke Test: verify we are 'intercepting' the logic
#         print(f"--- Smoke Test: APOE4Sustain is running for subtype {S} ---")
        
#         # Call the original math for now so it doesn't crash
#         return super()._calculate_sequenced_data_likelihood(data, S)

import pySuStaIn
import numpy as np

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
                 apoe4_status=None):
        
        # 1. Initialize the parent ZscoreSustain math
        super().__init__(data,
                         Z_vals,
                         Z_max,
                         SuStaInLabels,
                         N_startpoints,
                         N_S_max,
                         N_iterations_MCMC,
                         output_folder,
                         dataset_name,
                         use_parallel_startpoints)
        
        self.apoe4_status = apoe4_status

    # The run() method usually lives in the pySuStaIn implementation. 
    # If it's missing, we can 'borrow' it or call the library's run function.
    def run(self):
        print("--- Intercepting Run: Starting SuStaIn Iterations ---")
        # In many versions, you call the run method of the parent or a helper
        return super().run_sustain_algorithm()

    def _calculate_sequenced_data_likelihood(self, data, S):
        print(f"--- Smoke Test: APOE4Sustain is running for subtype {S} ---")
        return super()._calculate_sequenced_data_likelihood(data, S)