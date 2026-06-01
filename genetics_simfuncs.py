#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed May 27 15:17:40 2026

@author: mihaelacroitor

This file contains the pysustain simfuncs.py functions needed to simulate zscoresustain biomarker
data; 

I also modified generate_data_Zscore_sustain to generate_data_Zscore_sustain_point

"""


import numpy as np
from scipy.stats import norm


#generate the Z-score based event sequences for the desired number of subtypes (N_S)
# def generate_random_Zscore_sustain_model(Z_vals, N_S):

#     B                                   = Z_vals.shape[0]
#     stage_zscore                        = np.array([y for x in Z_vals.T for y in x])
#     stage_zscore                        = stage_zscore.reshape(1, len(stage_zscore))

#     IX_select                           = stage_zscore > 0
#     stage_zscore                        = stage_zscore[IX_select]
#     stage_zscore                        = stage_zscore.reshape(1, len(stage_zscore))

#     num_zscores                         = Z_vals.shape[1]
#     IX_vals                             = np.array([[x for x in range(B)]] * num_zscores).T
#     stage_biomarker_index               = np.array([y for x in IX_vals.T for y in x])
#     stage_biomarker_index               = stage_biomarker_index.reshape(1, len(stage_biomarker_index))
#     stage_biomarker_index               = stage_biomarker_index[IX_select]
#     stage_biomarker_index               = stage_biomarker_index.reshape(1, len(stage_biomarker_index))

#     N                                   = np.array(stage_zscore).shape[1]
#     S                                   = np.zeros((N_S, N))
#     for s in range(N_S):
#         for i in range(N):
#             IS_min_stage_zscore         = np.array([False] * N)
#             possible_biomarkers         = np.unique(stage_biomarker_index)

#             for j in range(len(possible_biomarkers)):
#                 IS_unselected           = [False] * N

#                 for k in set(range(N)) - set(S[s][:i]):
#                     IS_unselected[k]    = True

#                 this_biomarkers         = np.array([(np.array(stage_biomarker_index)[0] == possible_biomarkers[j]).astype(int) + (np.array(IS_unselected) == 1).astype(int)]) == 2
#                 if not np.any(this_biomarkers):
#                     this_min_stage_zscore = 0
#                 else:
#                     this_min_stage_zscore = min(stage_zscore[this_biomarkers])
#                 if (this_min_stage_zscore):
#                     temp                = ((this_biomarkers.astype(int) + (stage_zscore == this_min_stage_zscore).astype(int)) == 2).T
#                     temp                = temp.reshape(len(temp), )
#                     IS_min_stage_zscore[temp] = True

#             events                      = np.array(range(N))
#             possible_events             = np.array(events[IS_min_stage_zscore])
#             this_index                  = np.ceil(np.random.rand() * ((len(possible_events)))) - 1
#             S[s][i]                     = possible_events[int(this_index)]

#     return S

def generate_random_Zscore_sustain_model(Z_vals, N_S,rng=None):
    print(rng)
    if rng is None:
        rng = np.random.default_rng()
        print('Rng random')
    else:
        print('Rng from before')

    B                                   = Z_vals.shape[0]
    stage_zscore                        = np.array([y for x in Z_vals.T for y in x])
    stage_zscore                        = stage_zscore.reshape(1, len(stage_zscore))

    IX_select                           = stage_zscore > 0
    stage_zscore                        = stage_zscore[IX_select]
    stage_zscore                        = stage_zscore.reshape(1, len(stage_zscore))

    num_zscores                         = Z_vals.shape[1]
    IX_vals                             = np.array([[x for x in range(B)]] * num_zscores).T
    stage_biomarker_index               = np.array([y for x in IX_vals.T for y in x])
    stage_biomarker_index               = stage_biomarker_index.reshape(1, len(stage_biomarker_index))
    stage_biomarker_index               = stage_biomarker_index[IX_select]
    stage_biomarker_index               = stage_biomarker_index.reshape(1, len(stage_biomarker_index))

    N                                   = np.array(stage_zscore).shape[1]
    S                                   = np.zeros((N_S, N))
    for s in range(N_S):
        for i in range(N):
            IS_min_stage_zscore         = np.array([False] * N)
            possible_biomarkers         = np.unique(stage_biomarker_index)

            for j in range(len(possible_biomarkers)):
                IS_unselected           = [False] * N

                for k in set(range(N)) - set(S[s][:i]):
                    IS_unselected[k]    = True

                this_biomarkers         = np.array([(np.array(stage_biomarker_index)[0] == possible_biomarkers[j]).astype(int) + (np.array(IS_unselected) == 1).astype(int)]) == 2
                if not np.any(this_biomarkers):
                    this_min_stage_zscore = 0
                else:
                    this_min_stage_zscore = min(stage_zscore[this_biomarkers])
                if (this_min_stage_zscore):
                    temp                = ((this_biomarkers.astype(int) + (stage_zscore == this_min_stage_zscore).astype(int)) == 2).T
                    temp                = temp.reshape(len(temp), )
                    IS_min_stage_zscore[temp] = True

            events                      = np.array(range(N))
            possible_events             = np.array(events[IS_min_stage_zscore])
            #this_index                  = np.ceil(np.random.rand() * ((len(possible_events)))) - 1
            this_index                  = np.ceil(rng.random() * ((len(possible_events))))-1
            S[s][i]                     = possible_events[int(this_index)]

    return S


# can't remember which one is it, 
# I need to ask Alex why was it that we modified this function: stage_value or point_value; 
# and do i need this modification for my simulation

def generate_data_Zscore_sustain(subtypes, stages, gt_ordering, Z_vals, Z_max):

    B                                   = Z_vals.shape[0]
    stage_zscore                        = np.array([y for x in Z_vals.T for y in x])
    stage_zscore                        = stage_zscore.reshape(1,len(stage_zscore))
    IX_select                           = stage_zscore>0
    stage_zscore                        = stage_zscore[IX_select]
    stage_zscore                        = stage_zscore.reshape(1,len(stage_zscore))

    num_zscores                         = Z_vals.shape[1]
    IX_vals                             = np.array([[x for x in range(B)]] * num_zscores).T
    stage_biomarker_index               = np.array([y for x in IX_vals.T for y in x])
    stage_biomarker_index               = stage_biomarker_index.reshape(1,len(stage_biomarker_index))
    stage_biomarker_index               = stage_biomarker_index[IX_select]
    stage_biomarker_index               = stage_biomarker_index.reshape(1,len(stage_biomarker_index))

    min_biomarker_zscore                = [0]*B
    max_biomarker_zscore                = Z_max
    std_biomarker_zscore                = [1]*B # [0]*B

    N                                   = stage_biomarker_index.shape[1]
    N_S                                 = gt_ordering.shape[0]

    possible_biomarkers                 = np.unique(stage_biomarker_index)
    stage_value                         = np.zeros((B,N+2,N_S))

    for s in range(N_S):
        S                               = gt_ordering[s,:]
        S_inv                           = np.array([0]*N)
        S_inv[S.astype(int)]            = np.arange(N)
        for i in range(B):
            b                           = possible_biomarkers[i]
            event_location              = np.concatenate([[0], S_inv[(stage_biomarker_index == b)[0]], [N]])
            event_value                 = np.concatenate([[min_biomarker_zscore[i]], stage_zscore[stage_biomarker_index == b], [max_biomarker_zscore[i]]])

            for j in range(len(event_location)-1):

                if j == 0: # FIXME: nasty hack to get Matlab indexing to match up - necessary here because indices are used for linspace limits
                    index               = np.arange(event_location[j],event_location[j+1]+2)
                    stage_value[i,index,s] = np.linspace(event_value[j],event_value[j+1],event_location[j+1]-event_location[j]+2)
                else:
                    index               = np.arange(event_location[j] + 1, event_location[j + 1] + 2)
                    stage_value[i,index,s] = np.linspace(event_value[j],event_value[j+1],event_location[j+1]-event_location[j]+1)

    M                                   = stages.shape[0]
    data_denoised                       = np.zeros((M,B))
    for m in range(M):
        data_denoised[m,:]              = stage_value[:,int(stages[m]),subtypes[m]]
    data                                = data_denoised + norm.ppf(np.random.rand(B,M).T)*np.tile(std_biomarker_zscore,(M,1))

    return data, data_denoised, stage_value

def generate_data_Zscore_sustain_point(subtypes, stages, gt_ordering, Z_vals, Z_max):
    B                                   = Z_vals.shape[0]
    stage_zscore                        = np.array([y for x in Z_vals.T for y in x])
    stage_zscore                        = stage_zscore.reshape(1,len(stage_zscore))
    IX_select                           = stage_zscore>0
    stage_zscore                        = stage_zscore[IX_select]
    stage_zscore                        = stage_zscore.reshape(1,len(stage_zscore))

    num_zscores                         = Z_vals.shape[1]
    IX_vals                             = np.array([[x for x in range(B)]] * num_zscores).T
    stage_biomarker_index               = np.array([y for x in IX_vals.T for y in x])
    stage_biomarker_index               = stage_biomarker_index.reshape(1,len(stage_biomarker_index))
    stage_biomarker_index               = stage_biomarker_index[IX_select]
    stage_biomarker_index               = stage_biomarker_index.reshape(1,len(stage_biomarker_index))

    min_biomarker_zscore                = [0]*B
    max_biomarker_zscore                = Z_max
    std_biomarker_zscore                = [1]*B

    N                                   = stage_biomarker_index.shape[1]
    N_S                                 = gt_ordering.shape[0]

    possible_biomarkers                 = np.unique(stage_biomarker_index)
    point_value = np.zeros((B, N + 2, N_S))
    stage_value                         = np.zeros((B,N+2,N_S))

    for s in range(N_S):
        S                               = gt_ordering[s,:]
        S_inv                           = np.array([0]*N)
        S_inv[S.astype(int)]            = np.arange(N)
        for i in range(B):
            b                           = possible_biomarkers[i]
            event_location              = np.concatenate([[0], S_inv[(stage_biomarker_index == b)[0]], [N]])
            event_value                 = np.concatenate([[min_biomarker_zscore[i]], stage_zscore[stage_biomarker_index == b], [max_biomarker_zscore[i]]])

            for j in range(len(event_location)-1):

                if j == 0: # FIXME: nasty hack to get Matlab indexing to match up - necessary here because indices are used for linspace limits
                    index               = np.arange(event_location[j],event_location[j+1]+2)
                    stage_value[i,index,s] = np.linspace(event_value[j],event_value[j+1],event_location[j+1]-event_location[j]+2)
                    point_value[i, index, s] = np.linspace(event_value[j], event_value[j + 1], 
                                                                               event_location[j + 1] - event_location[j] + 2)
                else:
                    index               = np.arange(event_location[j] + 1, event_location[j + 1] + 2)
                    stage_value[i,index,s] = np.linspace(event_value[j],event_value[j+1],event_location[j+1]-event_location[j]+1)
                    point_value[i, index, s] = np.linspace(event_value[j], event_value[j + 1], 
                                                          event_location[j + 1] - event_location[j] + 1)


    M                                   = stages.shape[0]
    data_denoised                       = np.zeros((M,B))
    for m in range(M):
        data_denoised[m,:]              = stage_value[:,int(stages[m]),subtypes[m]]
    data                                = data_denoised + norm.ppf(np.random.rand(B,M).T)*np.tile(std_biomarker_zscore,(M,1))

    return data, data_denoised, stage_value



    


