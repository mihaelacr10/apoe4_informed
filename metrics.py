#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jun  5 15:49:41 2026

functions to calculate metrics for sequence and weight recovery
also measures subtyping and staging errors

@author: mihaelacroitor
"""

import numpy as np
from scipy.stats import kendalltau



def calculate_genetic_weight_mae(W_true, W_est, subtype_order):
    ordered_W_est = W_est[subtype_order]
    abs_error_matrix = np.abs(ordered_W_est - W_true)

    subtype_maes = []
    for s in range(abs_error_matrix.shape[0]):
        sub_mae = np.mean(abs_error_matrix[s])
        subtype_maes.append(sub_mae)
        print(f"   Subtype {s+1} MAE: {sub_mae:.4f}")

    global_genetic_mae = np.mean(subtype_maes)
    print(f"   Global MAE: {global_genetic_mae:.4f}\n")

    return global_genetic_mae, abs_error_matrix


def calculate_sequence_recovery(gt_sequences, predicted_sequences):
    # Apply argsort to transform event-to-stage mappings into rank vectors
    gt_ranks = np.argsort(gt_sequences, axis=1)
    pred_ranks = np.argsort(predicted_sequences, axis=1)

    kendal_tau = []
    for s in range(gt_sequences.shape[0]):
        tau, _ = kendalltau(gt_ranks[s], pred_ranks[s])
        tau = np.round(tau, 2)
        kendal_tau.append(tau)
        print(f"   Subtype {s+1} Kendall Tau: {tau}")

    avg_ktau = np.round(np.mean(kendal_tau), 2)
    print(f"   Global Avg Kendall Tau: {avg_ktau}\n")

    return np.array(kendal_tau)



# function for subtyping f1/ARI?

# staging performance



