#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon May 11 06:17:43 2026

@author: mihaelacroitor
"""

from pysustain.ZscoreSustain import ZscoreSustain

class Apoe4Sustain(ZscoreSustain):
    def __init__(self, data, apoe_status, *args, **kwargs):
        # Initialize the normal z-score model
        super().__init__(data, *args, **kwargs)
        self.apoe_status = apoe_status
        
    def _calculate_likelihood(self, sequence): # (Note: check exact function name in pysustain)
        
        # 1. Let ZscoreSustain do the hard biomarker math!
        # This calls the function that already exists in the parent classes
        biomarker_likelihood = super()._calculate_likelihood(sequence)
        
        # 2. Calculate your new APOE4 prior (the math you are building)
        apoe4_prior = self._calculate_apoe4_prior()
        
        # 3. Combine them 
        # (Note: if pysustain returns log-likelihoods, you ADD them instead of multiply)
        total_likelihood = biomarker_likelihood * apoe4_prior
        
        return total_likelihood

    def _calculate_apoe4_prior(self):
        # Your custom logic here for carrier vs non-carrier probabilities
        pass