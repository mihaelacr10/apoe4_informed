#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon May 25 13:58:32 2026

@author: mihaelacroitor
"""



from abc import ABC, abstractmethod

from tqdm.auto import tqdm
import numpy as np
import scipy.stats as stats
from matplotlib import pyplot as plt
import matplotlib.colors as mcolors
from pathlib import Path
import pickle
import csv
import os
import multiprocessing
from functools import partial, partialmethod

import time
import pathos


'''
in this version of the algorithm, we introduce a set of genetic weights, which act as a subtype prior lik
we optimise these weights in training
'''


#*******************************************
#The data structure class for AbstractSustain. It has no data itself - the implementations of AbstractSustain need to define their own implementations of this class.
class AbstractSustainData(ABC):

    @abstractmethod
    def __init__(self):
        pass

    @abstractmethod
    def getNumSamples(self):
        pass

    @abstractmethod
    def getNumBiomarkers(self):
        pass

    @abstractmethod
    def getNumStages(self):
        pass

    @abstractmethod
    def reindex(self, index):
        pass

#*******************************************
class AbstractSustain(ABC):

    def __init__(self,
                 sustainData,
                 N_startpoints,
                 N_S_max,
                 N_iterations_MCMC,
                 output_folder,
                 dataset_name,
                 use_parallel_startpoints,
                 seed=None):
        # The initializer for the abstract class
        # Parameters:
        #   sustainData                 - an instance of an AbstractSustainData implementation
        #   N_startpoints               - number of startpoints to use in maximum likelihood step of SuStaIn, typically 25
        #   N_S_max                     - maximum number of subtypes, should be 1 or more
        #   N_iterations_MCMC           - number of MCMC iterations, typically 1e5 or 1e6 but can be lower for debugging
        #   output_folder               - where to save pickle files, etc.
        #   dataset_name                - for naming pickle files
        #   use_parallel_startpoints    - boolean for whether or not to parallelize the maximum likelihood loop
        #   seed                        - random number seed

        assert(isinstance(sustainData, AbstractSustainData))

        self.__sustainData              = sustainData

        self.N_startpoints              = N_startpoints
        self.N_S_max                    = N_S_max
        self.N_iterations_MCMC          = N_iterations_MCMC

        self.num_cores                  = multiprocessing.cpu_count()

        self.output_folder              = output_folder
        self.dataset_name               = dataset_name

        if isinstance(seed, int):
            self.seed = seed
        elif isinstance(seed, float):
            self.seed = int(seed)
        elif seed is None:
            # Select random seed if none given
            self.seed = np.random.default_rng().integers((2**32)-1)

        # Create global rng to create process-specific rngs
        self.global_rng = np.random.default_rng(self.seed)

        self.use_parallel_startpoints   = use_parallel_startpoints

        if self.use_parallel_startpoints:
            np_version                  = float(np.__version__.split('.')[0] + '.' + np.__version__.split('.')[1])
            assert np_version >= 1.18, "numpy version must be >= 1.18 for parallelization to work properly."

            self.pool                   = pathos.multiprocessing.ProcessingPool() #pathos.multiprocessing.ParallelPool()
            self.pool.ncpus             = multiprocessing.cpu_count()
        else:
            self.pool                   = pathos.serial.SerialPool()
            
        self.apoe_flag = getattr(self, 'apoe_flag', False)
        if self.apoe_flag:
            print('Running genetic weighted SuStaIn ')
        else:
            print('Running baseline SuStain')
        
    #********************* PUBLIC METHODS
    def run_sustain_algorithm(self, plot=False, plot_format="png", **kwargs):
        # Externally called method to start the SuStaIn algorithm after initializing the SuStaIn class object properly

        ml_sequence_prev_EM                 = []
        ml_f_prev_EM                        = []
        if self.apoe_flag:
            ml_genetic_weights_prev_EM          = []

        pickle_dir                          = os.path.join(self.output_folder, 'pickle_files')
        if not os.path.isdir(pickle_dir):
            os.mkdir(pickle_dir)
        if plot:
            fig0, ax0                           = plt.subplots()
        for s in range(self.N_S_max):

            pickle_filename_s               = os.path.join(pickle_dir, self.dataset_name + '_subtype' + str(s) + '.pickle')
            pickle_filepath                 = Path(pickle_filename_s)
            if pickle_filepath.exists():
                print("Found pickle file: " + pickle_filename_s + ". Using pickled variables for " + str(s) + " subtype.")

                pickle_file                 = open(pickle_filename_s, 'rb')

                loaded_variables            = pickle.load(pickle_file)

                #self.stage_zscore           = loaded_variables["stage_zscore"]
                #self.stage_biomarker_index  = loaded_variables["stage_biomarker_index"]
                #self.N_S_max                = loaded_variables["N_S_max"]

                samples_likelihood          = loaded_variables["samples_likelihood"]
                samples_sequence            = loaded_variables["samples_sequence"]
                samples_f                   = loaded_variables["samples_f"]
                if self.apoe_flag:
                    samples_genetic_weights = loaded_variables["samples_genetic_weights"]

                ml_sequence_EM              = loaded_variables["ml_sequence_EM"]
                ml_sequence_prev_EM         = loaded_variables["ml_sequence_prev_EM"]
                ml_f_EM                     = loaded_variables["ml_f_EM"]
                ml_f_prev_EM                = loaded_variables["ml_f_prev_EM"]
                
                if self.apoe_flag:
                    ml_genetic_weights_EM   = loaded_variables["ml_genetic_weights_EM"]
                    ml_genetic_weights_prev_EM = loaded_variables["ml_genetic_weights_prev_EM"]
                

                pickle_file.close()
            else:
                print("Failed to find pickle file: " + pickle_filename_s + ". Running SuStaIn model for " + str(s) + " subtype.")
                
                if self.apoe_flag:
                    ml_sequence_EM,         \
                    ml_f_EM,                   \
                    ml_likelihood_EM,          \
                    ml_sequence_mat_EM,        \
                    ml_f_mat_EM,               \
                    ml_likelihood_mat_EM,      \
                    em_likelihood_histories,   \
                    ml_genetic_weights_EM,     \
                    ml_genetic_weights_mat_EM  = self._estimate_ml_sustain_model_nplus1_clusters(self.__sustainData, ml_sequence_prev_EM, ml_f_prev_EM, ml_genetic_weights_prev_EM) #self.__estimate_ml_sustain_model_nplus1_clusters(self.__data, ml_sequence_prev_EM, ml_f_prev_EM)
                else:
                    
                    ml_sequence_EM,     \
                    ml_f_EM,            \
                    ml_likelihood_EM,   \
                    ml_sequence_mat_EM, \
                    ml_f_mat_EM,        \
                    ml_likelihood_mat_EM,\
                    em_likelihood_histories  = self._estimate_ml_sustain_model_nplus1_clusters(self.__sustainData, ml_sequence_prev_EM, ml_f_prev_EM) #self.__estimate_ml_sustain_model_nplus1_clusters(self.__data, ml_sequence_prev_EM, ml_f_prev_EM)

                seq_init                    = ml_sequence_EM
                f_init                      = ml_f_EM
                
                if self.apoe_flag:
                    genetic_weights_init = ml_genetic_weights_EM
                
                # when i perturb the genetic weights dont forget to return them too
                if self.apoe_flag:
                    ml_sequence,        \
                    ml_f,               \
                    ml_likelihood,      \
                    samples_sequence,   \
                    samples_f,          \
                    samples_likelihood, \
                    ml_genetic_weights, \
                    samples_genetic_weights     = self._estimate_uncertainty_sustain_model(self.__sustainData, seq_init, f_init,genetic_weights_init)           #self.__estimate_uncertainty_sustain_model(self.__data, seq_init, f_init)
                else:
                
                    ml_sequence,        \
                    ml_f,               \
                    ml_likelihood,      \
                    samples_sequence,   \
                    samples_f,          \
                    samples_likelihood          = self._estimate_uncertainty_sustain_model(self.__sustainData, seq_init, f_init)           #self.__estimate_uncertainty_sustain_model(self.__data, seq_init, f_init)
                
                ml_sequence_prev_EM         = ml_sequence_EM
                ml_f_prev_EM                = ml_f_EM
                if self.apoe_flag:
                    ml_genetic_weights_prev_EM = ml_genetic_weights_EM
            
            if self.apoe_flag:
                # max like subtype and stage / subject
                N_samples                       = 1000
                ml_subtype,             \
                prob_ml_subtype,        \
                ml_stage,               \
                prob_ml_stage,          \
                prob_subtype,           \
                prob_stage,             \
                prob_subtype_stage               = self.subtype_and_stage_individuals(self.__sustainData, samples_sequence, samples_f, N_samples, samples_genetic_weights)   #self.subtype_and_stage_individuals(self.__data, samples_sequence, samples_f, N_samples)
            
            else:

                # max like subtype and stage / subject
                N_samples                       = 1000
                ml_subtype,             \
                prob_ml_subtype,        \
                ml_stage,               \
                prob_ml_stage,          \
                prob_subtype,           \
                prob_stage,             \
                prob_subtype_stage               = self.subtype_and_stage_individuals(self.__sustainData, samples_sequence, samples_f, N_samples)   #self.subtype_and_stage_individuals(self.__data, samples_sequence, samples_f, N_samples)
            
            # -----------------------------------------------------------------
            # FINAL SUSTAIN PARAMETER REPORTING
            # -----------------------------------------------------------------
            print("\n" + "="*60)
            print("   FINAL MODEL SUBTYPE COHORT FRACTIONS")
            print("="*60)
            for subtype_idx in range(len(ml_f_EM)):
                print(f" -> Subtype {subtype_idx + 1} Cohort Size: {ml_f_EM[subtype_idx]*100:.2f}% of dataset")
            print("="*60 + "\n")
    
            if self.apoe_flag:
                print("="*60)
                print("   SUSTAIN GENETIC PRIOR PROFILE (EM MAX LIKELIHOOD)")
                print("="*60)
                
                # 1. Print the baseline cohort frequencies for direct comparison
                global_str = ", ".join([f"Cat_{c}: {w:.4f}" for c, w in enumerate(self._global_genetic_frequencies)])
                print(f" -> COHORT BACKGROUND BASELINE : [{global_str}]")
                print("-"*60) # Visual separator
                
                # 2. Loop through each optimized subtype cluster row cleanly
                for subtype_idx in range(ml_genetic_weights_EM.shape[0]):
                    weights_str = ", ".join([f"Cat_{c}: {w:.4f}" for c, w in enumerate(ml_genetic_weights_EM[subtype_idx])])
                    print(f" -> Subtype {subtype_idx + 1} Profile Prior : [{weights_str}]")
                    
                print("="*60 + "\n")
            
            if not pickle_filepath.exists():

                if not os.path.exists(self.output_folder):
                    os.makedirs(self.output_folder)

                save_variables                          = {}
                save_variables["samples_sequence"]      = samples_sequence
                save_variables["samples_f"]             = samples_f
                save_variables["samples_likelihood"]    = samples_likelihood
                if self.apoe_flag:
                    save_variables["samples_genetic_weights"] = samples_genetic_weights

                save_variables["ml_subtype"]            = ml_subtype
                save_variables["prob_ml_subtype"]       = prob_ml_subtype
                save_variables["ml_stage"]              = ml_stage
                save_variables["prob_ml_stage"]         = prob_ml_stage
                save_variables["prob_subtype"]          = prob_subtype
                save_variables["prob_stage"]            = prob_stage
                save_variables["prob_subtype_stage"]    = prob_subtype_stage

                save_variables["ml_sequence_EM"]        = ml_sequence_EM
                save_variables["ml_sequence_prev_EM"]   = ml_sequence_prev_EM
                save_variables["ml_f_EM"]               = ml_f_EM
                save_variables["ml_f_prev_EM"]          = ml_f_prev_EM
                if self.apoe_flag:
                    save_variables["ml_genetic_weights_EM"] = ml_genetic_weights_EM
                    save_variables["ml_genetic_weights_prev_EM"] = ml_genetic_weights_prev_EM
                
                # save lik history of diff EM startingpoints to plot and check convergence
                save_variables["ml_likelihood_mat_EM"] = ml_likelihood_mat_EM
                save_variables["em_likelihood_histories"] = em_likelihood_histories

                pickle_file                 = open(pickle_filename_s, 'wb')
                pickle_output               = pickle.dump(save_variables, pickle_file)
                pickle_file.close()

            n_samples                       = self.__sustainData.getNumSamples() #self.__data.shape[0]

            #order of subtypes displayed in positional variance diagrams plotted by _plot_sustain_model
            self._plot_subtype_order        = np.argsort(ml_f_EM)[::-1]
            #order of biomarkers in each subtypes' positional variance diagram
            self._plot_biomarker_order      = ml_sequence_EM[self._plot_subtype_order[0], :].astype(int)

            # plot results
            if plot:
                figs, ax = self._plot_sustain_model(
                    samples_sequence=samples_sequence,
                    samples_f=samples_f,
                    n_samples=n_samples,
                    biomarker_labels=self.biomarker_labels,
                    subtype_order=self._plot_subtype_order,
                    biomarker_order=self._plot_biomarker_order,
                    save_path=Path(self.output_folder) / f"{self.dataset_name}_subtype{s}_PVD.{plot_format}",
                    **kwargs
                )
                for fig in figs:
                    fig.show()

                ax0.plot(range(self.N_iterations_MCMC), samples_likelihood, label="Subtype " + str(s+1))

        # save and show this figure after all subtypes have been calculcated
        if plot:
            ax0.legend(loc='upper right')
            fig0.tight_layout()
            fig0.savefig(Path(self.output_folder) / f"MCMC_likelihoods.{plot_format}", bbox_inches='tight')
            fig0.show()
        
        
        return samples_sequence, samples_f, ml_subtype, prob_ml_subtype, ml_stage, prob_ml_stage, prob_subtype_stage


    def cross_validate_sustain_model(self, test_idxs, select_fold = [], plot=False):
        # Cross-validate the SuStaIn model by running the SuStaIn algorithm (E-M
        # and MCMC) on a training dataset and evaluating the model likelihood on a test
        # dataset.
        # Parameters:
        #   'test_idxs'     - list of test set indices for each fold
        #   'select_fold'   - allows user to just run for a single fold (allows the cross-validation to be run in parallel).
        #                     leave this variable empty to iterate across folds sequentially.

        if not os.path.exists(self.output_folder):
            os.makedirs(self.output_folder)

        pickle_dir                          = os.path.join(self.output_folder, 'pickle_files')
        if not os.path.isdir(pickle_dir):
            os.mkdir(pickle_dir)

        if select_fold != []:
            if np.isscalar(select_fold):
                select_fold                 = [select_fold]
        else:
            select_fold                     = np.arange(len(test_idxs)) #test_idxs
        Nfolds                              = len(select_fold)

        is_full                             = Nfolds == len(test_idxs)

        loglike_matrix                      = np.zeros((Nfolds, self.N_S_max))

        for fold in tqdm(select_fold, "Folds: ", Nfolds, position=0, leave=True):

            indx_test                       = test_idxs[fold]
            indx_train                      = np.array([x for x in range(self.__sustainData.getNumSamples()) if x not in indx_test])

            sustainData_train               = self.__sustainData.reindex(indx_train)
            sustainData_test                = self.__sustainData.reindex(indx_test)

            ml_sequence_prev_EM             = []
            ml_f_prev_EM                    = []
            
            if self.apoe_flag:
                ml_genetic_weights_prev_EM = []
            

            for s in range(self.N_S_max):

                pickle_filename_fold_s      = os.path.join(pickle_dir, self.dataset_name + '_fold' + str(fold) + '_subtype' + str(s) + '.pickle')
                pickle_filepath             = Path(pickle_filename_fold_s)

                if pickle_filepath.exists():

                    print("Loading " + pickle_filename_fold_s)

                    pickle_file             = open(pickle_filename_fold_s, 'rb')

                    loaded_variables        = pickle.load(pickle_file)

                    ml_sequence_EM          = loaded_variables["ml_sequence_EM"]
                    ml_sequence_prev_EM     = loaded_variables["ml_sequence_prev_EM"]
                    ml_f_EM                 = loaded_variables["ml_f_EM"]
                    ml_f_prev_EM            = loaded_variables["ml_f_prev_EM"]
                    
                    if self.apoe_flag:
                        ml_genetic_weights_EM   = loaded_variables["ml_genetic_weights_EM"]
                        ml_genetic_weights_prev_EM = loaded_variables["ml_genetic_weights_prev_EM"]
                    

                    samples_likelihood      = loaded_variables["samples_likelihood"]
                    samples_sequence        = loaded_variables["samples_sequence"]
                    samples_f               = loaded_variables["samples_f"]
                    if self.apoe_flag:
                        samples_genetic_weights = loaded_variables["samples_genetic_weights"]
                        

                    mean_likelihood_subj_test = loaded_variables["mean_likelihood_subj_test"]
                    pickle_file.close()
                    
                    if self.apoe_flag:
                        samples_likelihood_subj_test = self._evaluate_likelihood_setofsamples(sustainData_test, samples_sequence, samples_f)
                    else:
                        samples_likelihood_subj_test = self._evaluate_likelihood_setofsamples(sustainData_test, samples_sequence, samples_f, samples_genetic_weights)

                else:
                    
                    if self.apoe_flag:
                        ml_sequence_EM,         \
                        ml_f_EM,                \
                        ml_likelihood_EM,       \
                        ml_sequence_mat_EM,     \
                        ml_f_mat_EM,            \
                        ml_likelihood_mat_EM,   \
                        ml_genetic_weights_EM,  \
                        ml_genetic_weights_mat_EM  = self._estimate_ml_sustain_model_nplus1_clusters(sustainData_train, ml_sequence_prev_EM, ml_f_prev_EM, ml_genetic_weights_prev_EM)
                    else:
                        
                        ml_sequence_EM,         \
                        ml_f_EM,                \
                        ml_likelihood_EM,       \
                        ml_sequence_mat_EM,     \
                        ml_f_mat_EM,            \
                        ml_likelihood_mat_EM    = self._estimate_ml_sustain_model_nplus1_clusters(sustainData_train, ml_sequence_prev_EM, ml_f_prev_EM)

                    seq_init                    = ml_sequence_EM
                    f_init                      = ml_f_EM
                    
                    if self.apoe_flag:
                        genetic_weights_init    = ml_genetic_weights_EM
                        
                    
                    
                    if self.apoe_flag:
                        
                        ml_sequence,        \
                        ml_f,               \
                        ml_likelihood,      \
                        samples_sequence,   \
                        samples_f,          \
                        samples_likelihood, \
                        ml_genetic_weights, \
                        samples_genetic_weights     = self._estimate_uncertainty_sustain_model(self.__sustainData, seq_init, f_init,genetic_weights_init) 
                        
                    else:
                    
                        ml_sequence,            \
                        ml_f,                   \
                        ml_likelihood,          \
                        samples_sequence,       \
                        samples_f,              \
                        samples_likelihood           = self._estimate_uncertainty_sustain_model(sustainData_train, seq_init, f_init)
                    
                    if self.apoe_flag:
                        samples_likelihood_subj_test = self._evaluate_likelihood_setofsamples(sustainData_test, samples_sequence, samples_f, samples_genetic_weights)
                    else:  
                        samples_likelihood_subj_test = self._evaluate_likelihood_setofsamples(sustainData_test, samples_sequence, samples_f)

                    mean_likelihood_subj_test    = np.mean(samples_likelihood_subj_test,axis=1)

                    ml_sequence_prev_EM         = ml_sequence_EM
                    ml_f_prev_EM                = ml_f_EM
                    
                    if self.apoe_flag:
                        ml_genetic_weights_prev_EM = ml_genetic_weights_EM

                    save_variables                                      = {}
                    save_variables["ml_sequence_EM"]                    = ml_sequence_EM
                    save_variables["ml_sequence_prev_EM"]               = ml_sequence_prev_EM
                    save_variables["ml_f_EM"]                           = ml_f_EM
                    save_variables["ml_f_prev_EM"]                      = ml_f_prev_EM
                    
                    if self.apoe_flag:
                        save_variables["ml_genetic_weights_EM"]         = ml_genetic_weights_EM
                        save_variables["ml_genetic_weights_prev_EM"]    = ml_genetic_weights_prev_EM
                        

                    save_variables["samples_sequence"]                  = samples_sequence
                    save_variables["samples_f"]                         = samples_f
                    save_variables["samples_likelihood"]                = samples_likelihood
                    if self.apoe_flag:
                        save_variables["samples_genetic_weights"]        = samples_genetic_weights

                    save_variables["mean_likelihood_subj_test"]         = mean_likelihood_subj_test

                    pickle_file                     = open(pickle_filename_fold_s, 'wb')
                    pickle_output                   = pickle.dump(save_variables, pickle_file)
                    pickle_file.close()

                if is_full:
                    loglike_matrix[fold, s]         = np.mean(np.sum(np.log(samples_likelihood_subj_test + 1e-250),axis=0))

        if not is_full:
            print("Cannot calculate CVIC and loglike_matrix without all folds. Rerun cross_validate_sustain_model after all folds calculated.")
            return [], []

        print(f"Average test set log-likelihood for each subtype model: {np.mean(loglike_matrix, 0)}")

        if plot:
            import pandas as pd
            fig, ax = plt.subplots()

            df_loglike = pd.DataFrame(data = loglike_matrix, columns = ["Subtype " + str(i+1) for i in range(self.N_S_max)])
            df_loglike.boxplot(grid=False, ax=ax, fontsize=15)
            for i in range(self.N_S_max):
                y = df_loglike[["Subtype " + str(i+1)]]
                # Add some random "jitter" to the x-axis
                x = np.random.normal(1+i, 0.04, size=len(y))
                ax.plot(x, y.values, 'r.', alpha=0.2)
            fig.savefig(Path(self.output_folder) / 'Log_likelihoods_cv_folds.png')
            fig.show()

        CVIC                            = np.zeros(self.N_S_max)

        for s in range(self.N_S_max):
            for fold in range(Nfolds):
                pickle_filename_fold_s  = os.path.join(pickle_dir, self.dataset_name + '_fold' + str(fold) + '_subtype' + str(s) + '.pickle')
                pickle_filepath         = Path(pickle_filename_fold_s)

                pickle_file             = open(pickle_filename_fold_s, 'rb')
                loaded_variables        = pickle.load(pickle_file)

                mean_likelihood_subj_test = loaded_variables["mean_likelihood_subj_test"]
                pickle_file.close()
    
                if fold == 0:
                    mean_likelihood_subj_test_cval    = mean_likelihood_subj_test
                else:
                    mean_likelihood_subj_test_cval    = np.concatenate((mean_likelihood_subj_test_cval, mean_likelihood_subj_test), axis=0)

            CVIC[s]                     = -2*sum(np.log(mean_likelihood_subj_test_cval))

        print("CVIC for each subtype model: " + str(CVIC))

        return CVIC, loglike_matrix

    
    def combine_cross_validated_sequences_1(self, N_subtypes, N_folds, plot_format="png", **kwargs):
        '''dont know if i need this function or the original one is fine
        it doesnt seem to need the genetic weights; it just combines seq across folds'''
        
        # Combine MCMC sequences across cross-validation folds to get cross-validated positional variance diagrams,
        # so that you get more realistic estimates of variance within event positions within subtypes

        pickle_dir                          = os.path.join(self.output_folder, 'pickle_files')

        #*********** load ML sequence for full model for N_subtypes
        pickle_filename_s                   = os.path.join(pickle_dir, self.dataset_name + '_subtype' + str(N_subtypes-1) + '.pickle')        
        pickle_filepath                     = Path(pickle_filename_s)

        assert pickle_filepath.exists(), "Failed to find pickle file for full model with " + str(N_subtypes) + " subtypes."

        pickle_file                         = open(pickle_filename_s, 'rb')
        loaded_variables_full               = pickle.load(pickle_file)

        ml_sequence_EM_full                 = loaded_variables_full["ml_sequence_EM"]
        ml_f_EM_full                        = loaded_variables_full["ml_f_EM"]
        
        if self.apoe_flag:
            ml_genetic_weights_EM_full      = loaded_variables_full["ml_genetic_weights_EM"]

        pickle_file.close()

        for i in range(N_folds):
            # load the MCMC sequences for this fold's model of N_subtypes
            pickle_filename_fold_s          = os.path.join(pickle_dir, self.dataset_name + '_fold' + str(i) + '_subtype' + str(N_subtypes-1) + '.pickle')        
            pickle_filepath                 = Path(pickle_filename_fold_s)

            assert pickle_filepath.exists(), f"Failed to find pickle file for fold {i}"

            pickle_file                     = open(pickle_filename_fold_s, 'rb')
            loaded_variables_i              = pickle.load(pickle_file)

            ml_sequence_EM_i                = loaded_variables_i["ml_sequence_EM"]
            ml_f_EM_i                       = loaded_variables_i["ml_f_EM"]

            samples_sequence_i              = loaded_variables_i["samples_sequence"]
            samples_f_i                     = loaded_variables_i["samples_f"]
            
            if self.apoe_flag:
                samples_genetic_weights_i   = loaded_variables_i["samples_genetic_weights"]

            pickle_file.close()

            # Calculate Kendall's tau correlation matrix based on anatomical layouts
            corr_mat                        = np.zeros((N_subtypes, N_subtypes))
            for j in range(N_subtypes):
                for k in range(N_subtypes):
                    corr_mat[j,k]            = stats.kendalltau(np.argsort(ml_sequence_EM_full[j,:]), np.argsort(ml_sequence_EM_i[k,:])).correlation
            
            set_full                        = []
            set_fold_i                      = []
            i_i, i_j                        = np.unravel_index(np.argsort(corr_mat.flatten())[::-1], (N_subtypes, N_subtypes))
            for k in range(len(i_i)):
                if not i_i[k] in set_full and not i_j[k] in set_fold_i:
                    set_full.append(i_i[k].astype(int))
                    set_fold_i.append(i_j[k].astype(int))
            index_set_full                  = np.argsort(set_full).astype(int)
            iMax_vec                        = [set_fold_i[m] for m in index_set_full]

            assert(np.all(np.sort(iMax_vec)==np.arange(N_subtypes)))

            # Concatenate arrays along the iteration timeline dimensions
            if i == 0:
                samples_sequence_cval       = samples_sequence_i[iMax_vec,:,:]
                samples_f_cval              = samples_f_i[iMax_vec, :]
                if self.apoe_flag:
                    samples_genetic_cval    = samples_genetic_weights_i[iMax_vec,:,:]
            else:
                samples_sequence_cval       = np.concatenate((samples_sequence_cval,    samples_sequence_i[iMax_vec,:,:]),  axis=2)
                samples_f_cval              = np.concatenate((samples_f_cval,           samples_f_i[iMax_vec,:]),           axis=1)
                if self.apoe_flag:
                    samples_genetic_cval    = np.concatenate((samples_genetic_cval,      samples_genetic_weights_i[iMax_vec,:,:]), axis=2)

        n_samples                           = self.__sustainData.getNumSamples()

        # Order of subtypes displayed in positional variance diagrams plotted by _plot_sustain_model
        plot_subtype_order                  = np.argsort(ml_f_EM_full)[::-1]
        # Order of biomarkers in each subtypes' positional variance diagram
        plot_biomarker_order                = ml_sequence_EM_full[plot_subtype_order[0], :].astype(int)

        figs, ax = self._plot_sustain_model(
            samples_sequence=samples_sequence_cval,
            samples_f=samples_f_cval,
            n_samples=n_samples,
            cval=True,
            biomarker_labels=self.biomarker_labels,
            subtype_order=plot_subtype_order,
            biomarker_order=plot_biomarker_order,
            **kwargs
        )
        
        # ---------------------------------------------------------------------
        # REPORT OUT-OF-FOLD CROSS-VALIDATED GENETIC MATRIX PROFILES
        # ---------------------------------------------------------------------
        if self.apoe_flag:
            print("\n" + "="*60)
            print("    CROSS-VALIDATED GENETIC PRIOR PREVALENCE PROFILE")
            print("="*60)
            # Take the mean across all combined cross-validation iterations safely
            mean_genetic_weights_cval = np.mean(samples_genetic_cval, axis=2)
            for subtype_idx in plot_subtype_order:
                weights_str = ", ".join([f"Cat_{c}: {w:.4f}" for c, w in enumerate(mean_genetic_weights_cval[subtype_idx])])
                print(f" -> CV-Stabilized Subtype {subtype_idx + 1} Prior Profile: [{weights_str}]")
            print("="*60 + "\n")

        if "save_path" not in kwargs:
            if len(figs) > 1:
                for num_subtype, fig in zip(range(N_subtypes), figs):
                    plot_fname = Path(self.output_folder) / f"{self.dataset_name}_subtype{N_subtypes - 1}_subtype{num_subtype}-separated_PVD_{N_folds}fold_CV.{plot_format}"
                    fig.savefig(plot_fname, bbox_inches='tight')
                    fig.show()
            else:
                fig = figs[0]
                plot_fname = Path(self.output_folder) / f"{self.dataset_name}_subtype{N_subtypes - 1}_PVD_{N_folds}fold_CV.{plot_format}"
                fig.savefig(plot_fname, bbox_inches='tight')
                fig.show()
                
        # Optional return expansion for downstream testing wrappers
        # if self.apoe_flag:
        #     return samples_sequence_cval, samples_f_cval, samples_genetic_cval
        # return samples_sequence_cval, samples_f_cval
    

    def subtype_and_stage_individuals(self, sustainData, samples_sequence, samples_f, N_samples, samples_genetic_weights=None):
        # Subtype and stage a set of subjects. Useful for subtyping/staging subjects that were not used to build the model

        nSamples                            = sustainData.getNumSamples()  #data_local.shape[0]
        nStages                             = sustainData.getNumStages()    #self.stage_zscore.shape[1]

        n_iterations_MCMC                   = samples_sequence.shape[2]
        select_samples                      = np.round(np.linspace(0, n_iterations_MCMC - 1, N_samples)) #np.linspace builds an index array that skips through the timeline in increments of 1000 steps:
        N_S                                 = samples_sequence.shape[0]
        temp_mean_f                         = np.mean(samples_f, axis=1)
        ix                                  = np.argsort(temp_mean_f)[::-1] # sorting indexes based on decreasing subtype prevalence

        prob_subtype_stage                  = np.zeros((nSamples, nStages + 1, N_S))
        prob_subtype                        = np.zeros((nSamples, N_S))
        prob_stage                          = np.zeros((nSamples, nStages + 1))
        
        
        # this part calculates individual posterior probabilities by avg the results across mcmc samples
        # but bc of computational constraints its subsamples only a small sample of the whole mcmc array
        # np.linspace builds an index array that skips through the timeline in increments of 1000 steps:
        # and then iterate over samples
        for i in range(N_samples): 
            sample                          = int(select_samples[i])

            this_S                          = samples_sequence[ix, :, sample]
            this_f                          = samples_f[ix, sample]
            
            if self.apoe_flag:
                this_genetic_weights        = samples_genetic_weights[ix, :,sample]
                # in case it throws a shape mismatch for N_S>1
                # this_genetic_weights  = samples_genetic_weights[:, :, sample][ix, :]
                
                _,                  \
                _,                  \
                total_prob_stage,   \
                total_prob_subtype, \
                total_prob_subtype_stage        = self._calculate_likelihood(sustainData, this_S, this_f, this_genetic_weights)
            
            else:
                _,                  \
                _,                  \
                total_prob_stage,   \
                total_prob_subtype, \
                total_prob_subtype_stage        = self._calculate_likelihood(sustainData, this_S, this_f)

            total_prob_subtype              = total_prob_subtype.reshape(len(total_prob_subtype), N_S)
            total_prob_subtype_norm         = total_prob_subtype        / np.tile(np.sum(total_prob_subtype, 1).reshape(len(total_prob_subtype), 1),        (1, N_S))
            total_prob_stage_norm           = total_prob_stage          / np.tile(np.sum(total_prob_stage, 1).reshape(len(total_prob_stage), 1),          (1, nStages + 1)) #removed total_prob_subtype

            #total_prob_subtype_stage_norm   = total_prob_subtype_stage  / np.tile(np.sum(np.sum(total_prob_subtype_stage, 1), 1).reshape(nSamples, 1, 1),   (1, nStages + 1, N_S))
            total_prob_subtype_stage_norm   = total_prob_subtype_stage / np.tile(np.sum(np.sum(total_prob_subtype_stage, 1, keepdims=True), 2).reshape(nSamples, 1, 1),(1, nStages + 1, N_S))

            prob_subtype_stage              = (i / (i + 1.) * prob_subtype_stage)   + (1. / (i + 1.) * total_prob_subtype_stage_norm)
            prob_subtype                    = (i / (i + 1.) * prob_subtype)         + (1. / (i + 1.) * total_prob_subtype_norm)
            prob_stage                      = (i / (i + 1.) * prob_stage)           + (1. / (i + 1.) * total_prob_stage_norm)

        ml_subtype                          = np.nan * np.ones((nSamples, 1))
        prob_ml_subtype                     = np.nan * np.ones((nSamples, 1))
        ml_stage                            = np.nan * np.ones((nSamples, 1))
        prob_ml_stage                       = np.nan * np.ones((nSamples, 1))

        for i in range(nSamples):
            this_prob_subtype               = np.atleast_1d(np.squeeze(prob_subtype[i, :]))
            # if not np.isnan(this_prob_subtype).any()
            if (np.sum(np.isnan(this_prob_subtype)) == 0):
                # this_subtype = this_prob_subtype.argmax(
                this_subtype                = np.where(this_prob_subtype == np.max(this_prob_subtype))

                try:
                    ml_subtype[i]           = this_subtype
                except:
                    ml_subtype[i]           = this_subtype[0][0]
                if this_prob_subtype.size == 1 and this_prob_subtype == 1:
                    prob_ml_subtype[i]      = 1
                else:
                    try:
                        prob_ml_subtype[i]  = this_prob_subtype[this_subtype]
                    except:
                        prob_ml_subtype[i]  = this_prob_subtype[this_subtype[0][0]]

            this_prob_stage                 = np.squeeze(prob_subtype_stage[i, :, int(ml_subtype[i])])
            
            if (np.sum(np.isnan(this_prob_stage)) == 0):
                # this_stage = 
                this_stage                  = np.where(this_prob_stage == np.max(this_prob_stage))
                ml_stage[i]                 = this_stage[0][0]
                prob_ml_stage[i]            = this_prob_stage[this_stage[0][0]]
        # NOTE: The above loop can be replaced with some simpler numpy calls
        # May need to do some masking to avoid NaNs, or use `np.nanargmax` depending on preference
        # E.g. ml_subtype == prob_subtype.argmax(1)
        # E.g. ml_stage == prob_subtype_stage[np.arange(prob_subtype_stage.shape[0]), :, ml_subtype].argmax(1)
        return ml_subtype, prob_ml_subtype, ml_stage, prob_ml_stage, prob_subtype, prob_stage, prob_subtype_stage

    # ********************* PROTECTED METHODS
    def _estimate_ml_sustain_model_nplus1_clusters(self, sustainData, ml_sequence_prev, ml_f_prev, ml_genetic_weights_prev=None):
        # Given the previous SuStaIn model, estimate the next model in the
        # hierarchy (i.e. number of subtypes goes from N to N+1)
        #
        #
        # OUTPUTS:
        # ml_sequence       - the ordering of the stages for each subtype for the next SuStaIn model in the hierarchy
        # ml_f              - the most probable proportion of individuals belonging to each subtype for the next SuStaIn model in the hierarchy
        # ml_likelihood     - the likelihood of the most probable SuStaIn model for the next SuStaIn model in the hierarchy

        N_S = len(ml_sequence_prev) + 1
        if N_S == 1:
            # If the number of subtypes is 1, fit a single linear z-score model
            print('Finding ML solution to 1 cluster problem')
            if self.apoe_flag:
                
                ml_sequence,           \
                ml_f,                  \
                ml_genetic_weights,    \
                ml_likelihood,         \
                ml_sequence_mat,       \
                ml_f_mat,              \
                ml_genetic_weights_mat,\
                ml_likelihood_mat,     \
                em_likelihood_histories  = self._find_ml(sustainData)
            else: 
                ml_sequence,        \
                ml_f,               \
                ml_likelihood,      \
                ml_sequence_mat,    \
                ml_f_mat,           \
                ml_likelihood_mat,  \
                em_likelihood_histories  = self._find_ml(sustainData)
                print('Overall ML likelihood is', ml_likelihood)

        else:
            #em_likelihood_histories = []
            
            # If the number of subtypes is greater than 1, go through each subtype
            # in turn and try splitting into two subtypes
            
            if self.apoe_flag:
                _, _, _, p_sequence, _          = self._calculate_likelihood(sustainData, ml_sequence_prev, ml_f_prev, ml_genetic_weights_prev)
            else:   
                _, _, _, p_sequence, _          = self._calculate_likelihood(sustainData, ml_sequence_prev, ml_f_prev)

            ml_sequence_prev                = ml_sequence_prev.reshape(ml_sequence_prev.shape[0], ml_sequence_prev.shape[1])
            
            p_sequence                      = p_sequence.reshape(p_sequence.shape[0], N_S - 1) # prob cluster
            p_sequence_norm                 = p_sequence / np.tile(np.sum(p_sequence, 1).reshape(len(p_sequence), 1), (N_S - 1))

            # Assign individuals to a subtype (cluster) based on the previous model
            ml_cluster_subj                 = np.zeros((sustainData.getNumSamples(), 1))   #np.zeros((len(data_local), 1))
            for m in range(sustainData.getNumSamples()):                                   #range(len(data_local)):
                ix                          = np.argmax(p_sequence_norm[m, :]) + 1

                #TEMP: MATLAB comparison
                #ml_cluster_subj[m]          = ix*np.ceil(np.random.rand())
                ml_cluster_subj[m]          = ix  # FIXME: should check this always works, as it differs to the Matlab code, which treats ix as an array

            ml_likelihood                   = -np.inf
            for ix_cluster_split in range(N_S - 1):
                this_N_cluster              = sum(ml_cluster_subj == int(ix_cluster_split + 1))

                if this_N_cluster > 1:

                    # Take the data from the individuals belonging to a particular
                    # cluster and fit a two subtype model
                    print('Splitting cluster', ix_cluster_split + 1, 'of', N_S - 1)
                    ix_i                    = (ml_cluster_subj == int(ix_cluster_split + 1)).reshape(sustainData.getNumSamples(), )
                    sustainData_i           = sustainData.reindex(ix_i)
                    print('Reindexing')

                    print(' + Resolving 2 cluster problem')
                    this_ml_sequence_split, _, _, _, _, _ = self._find_ml_split(sustainData_i)

                    # Use the two subtype model combined with the other subtypes to
                    # inititialise the fitting of the next SuStaIn model in the
                    # hierarchy
                    this_seq_init           = ml_sequence_prev.copy()  # have to copy or changes will be passed to ml_sequence_prev

                    #replace the previous sequence with the first (row index zero) new sequence
                    this_seq_init[ix_cluster_split] = (this_ml_sequence_split[0]).reshape(this_ml_sequence_split.shape[1])

                    #add the second new sequence (row index one) to the stack of sequences, 
                    #so that you now have N_S sequences instead of N_S-1
                    this_seq_init           = np.hstack((this_seq_init.T, this_ml_sequence_split[1])).T
                    
                    #initialize fraction of subjects in each subtype to be uniform
                    this_f_init             = np.array([1.] * N_S) / float(N_S)
                    
 
                    print(' + Finding ML solution from hierarchical initialisation')
                    
                    
                    if self.apoe_flag:
                        this_ml_sequence,       \
                        this_ml_f,              \
                        this_ml_likelihood,     \
                        this_ml_sequence_mat,   \
                        this_ml_f_mat,          \
                        this_ml_likelihood_mat, \
                        this_em_likelihood_histories,\
                        this_ml_genetic_weights,\
                        this_ml_genetic_weights_mat = self._find_ml_mixture(sustainData, this_seq_init, this_f_init)
                    
                    else:
                        this_ml_sequence,       \
                        this_ml_f,              \
                        this_ml_likelihood,     \
                        this_ml_sequence_mat,   \
                        this_ml_f_mat,          \
                        this_ml_likelihood_mat, \
                        this_em_likelihood_histories = self._find_ml_mixture(sustainData, this_seq_init, this_f_init)

                    # Choose the most probable SuStaIn model from the different
                    # possible SuStaIn models initialised by splitting each subtype
                    # in turn
                    # FIXME: these arrays have an unnecessary additional axis with size = N_startpoints - remove it further upstream
                    if this_ml_likelihood[0] > ml_likelihood:
                        ml_likelihood       = this_ml_likelihood[0]
                        ml_sequence         = this_ml_sequence[:, :, 0]
                        ml_f                = this_ml_f[:, 0]
                        ml_likelihood_mat   = this_ml_likelihood_mat[0]
                        ml_sequence_mat     = this_ml_sequence_mat[:, :, 0]
                        ml_f_mat            = this_ml_f_mat[:, 0]
                        # we're not removing the startingpoint dimension bc we want the EM lik from all startingpoints
                        em_likelihood_histories = this_em_likelihood_histories
                        #ml_likelihood_mat   = this_ml_likelihood_mat
                        
                        if self.apoe_flag:
                            ml_genetic_weights = this_ml_genetic_weights[:,:,0]
                            ml_genetic_weights_mat = this_ml_genetic_weights_mat[:,:,0]
                        
                    print('- ML likelihood is', this_ml_likelihood[0])
                else:
                    print(f'Cluster {ix_cluster_split + 1} of {N_S - 1} too small for subdivision')
            print(f'Overall ML likelihood is', ml_likelihood)

        if self.apoe_flag:
            return ml_sequence, ml_f, ml_likelihood, ml_sequence_mat, ml_f_mat, ml_likelihood_mat, em_likelihood_histories, ml_genetic_weights, ml_genetic_weights_mat
        else:
            return ml_sequence, ml_f, ml_likelihood, ml_sequence_mat, ml_f_mat, ml_likelihood_mat, em_likelihood_histories

    #********************************************

    def _find_ml(self, sustainData):
        # Fit the maximum likelihood model
        #
        # OUTPUTS:
        # ml_sequence   - the ordering of the stages for each subtype
        # ml_f          - the most probable proportion of individuals belonging to each subtype
        # ml_likelihood - the likelihood of the most probable SuStaIn model

        partial_iter                        = partial(self._find_ml_iteration, sustainData)
        seed_sequences = np.random.SeedSequence(self.global_rng.integers(1e10))
        pool_output_list                    = self.pool.map(partial_iter, seed_sequences.spawn(self.N_startpoints))

        if ~isinstance(pool_output_list, list):
            pool_output_list                = list(pool_output_list)

        ml_sequence_mat                     = np.zeros((1, sustainData.getNumStages(), self.N_startpoints)) #np.zeros((1, self.stage_zscore.shape[1], self.N_startpoints))
        ml_f_mat                            = np.zeros((1, self.N_startpoints))
        ml_likelihood_mat                   = np.zeros(self.N_startpoints)
        
        if self.apoe_flag:
            N_S = 1 # is find ml just for 1 subtype
            ml_genetic_weights_mat          = np.zeros((N_S, self.N_genetic_categories, self.N_startpoints))
        
        #em_likelihood_histories = []
        
        # creating an array to store all iterations of all startingpoints EM likelihoods
        MaxIter = 100
        em_likelihood_histories = np.nan * np.ones((MaxIter,self.N_startpoints ))
        
        for i in range(self.N_startpoints):
            ml_sequence_mat[:, :, i]        = pool_output_list[i][0]
            ml_f_mat[:, i]                  = pool_output_list[i][1]
            ml_likelihood_mat[i]            = pool_output_list[i][2]
            # append the em lik history 
            #em_likelihood_histories.append(pool_output_list[i][3])
            em_likelihood_histories[:,i]    = pool_output_list[i][3].ravel()
            
            if self.apoe_flag:
                ml_genetic_weights_mat[:,:,i]            = pool_output_list[i][4]
                
        # save the array of likelihoods  ml_likelihood_mat
    
        ix                                  = np.argmax(ml_likelihood_mat)
        ml_sequence                         = ml_sequence_mat[:, :, ix]
        ml_f                                = ml_f_mat[:, ix]
        ml_likelihood                       = ml_likelihood_mat[ix]
        if self.apoe_flag:
            ml_genetic_weights              = ml_genetic_weights_mat[:,:,ix]
            
            return (ml_sequence, ml_f, ml_genetic_weights, ml_likelihood, 
                    ml_sequence_mat, ml_f_mat, ml_genetic_weights_mat, ml_likelihood_mat, em_likelihood_histories)

        return ml_sequence, ml_f, ml_likelihood, ml_sequence_mat, ml_f_mat, ml_likelihood_mat, em_likelihood_histories

    def _find_ml_iteration(self, sustainData, seed_seq):
        #Convenience sub-function for above

        # Get process-appropriate Generator
        rng = np.random.default_rng(seed_seq)

        # randomly initialise the sequence of the linear z-score model
        seq_init                        = self._initialise_sequence(sustainData, rng)
        f_init                          = [1]
        
        N_S = seq_init.shape[0]
        #initialise genetic weights
        if self.apoe_flag:
            genetic_weights_init = self._initialise_genetic_weights(N_S,rng)
        
        if self.apoe_flag:
            
            this_ml_sequence,   \
            this_ml_f,          \
            this_ml_genetic_weights,\
            this_ml_likelihood, \
            _,                  \
            _,                  \
            _,                  \
            samples_likelihood              = self._perform_em(sustainData, seq_init, f_init, rng, genetic_weights_init)

            
        else:
            
            this_ml_sequence,   \
            this_ml_f,          \
            this_ml_likelihood, \
            _,                  \
            _,                  \
            samples_likelihood              = self._perform_em(sustainData, seq_init, f_init, rng)

        if self.apoe_flag:
            return this_ml_sequence, this_ml_f, this_ml_likelihood,  samples_likelihood, this_ml_genetic_weights
        else:
            return this_ml_sequence, this_ml_f, this_ml_likelihood, samples_likelihood
    
    # Please try both initialisation methods in experiments to decide which one works better
    
    def _initialise_genetic_weights(self, N_S,rng=None):
        """
        Initialises genetic weights using a flat, unconstrained Dirichlet distribution.
        
        EXPERIMENTAL BEHAVIOR TO TRACK:
        - Space Exploration: MAXIMUM. Each parallel startpoint gets wildly different 
          initial weights (e.g., one worker might guess a subtype is 90% APOE4 carriers, 
          while another guesses 5%). Great for breaking out of local minima.
        - Convergence Speed: SLOWER. Because the initial guesses are often highly 
          unrealistic compared to the actual cohort, the EM loop must spend its first 
          few iterations doing heavy lifting to pull these weights back toward reality.
        """
        # alpha=[1, 1, 1] creates a uniform distribution over a probability simplex.
        # It natively ensures that each row sums to exactly 1.0 while maximizing 
        # variance between different random draws across parallel workers.
        # Shape output: (N_S, 3) representing [Genotype_0, Genotype_1, Genotype_2] per subtype.
        if rng is None:
            rng = getattr(self, 'global_rng', np.random.default_rng())
            # get global
        alpha_vector = [1] * self.N_genetic_categories
        genetic_weights = rng.dirichlet(alpha=alpha_vector, size=N_S)
        
        return genetic_weights
    
    # def _initialise_genetic_weights(self,N_S, rng=None):
    #     """
    #     Initialises genetic weights by blending the true global population frequencies 
    #     with a random Dirichlet matrix using a tunable mixing weight (w).
        
    #     EXPERIMENTAL BEHAVIOR TO TRACK:
    #     - Space Exploration: BALANCED. Workers are structurally unique from each other,
    #       but they stay clustered within a reasonable neighborhood of the population baseline.
    #     - Convergence Speed: FASTER. Because the model begins near the true demographic 
    #       gravity center of your dataset, it eliminates the "burn-in" iterations 
    #       wasted on corrections, allowing the clinical sequence shuffler to get to work immediately.
    #     """
    #     # Fetch the pre-calculated global background rates of your non-null dataset
    #     # e.g., returns a stable vector like [0.542, 0.361, 0.096]
    #     global_frequencies = self._global_genetic_frequencies 
        
    #     # Replicate the global background row across all target subtypes.
    #     # This acts as your "Exploitation" matrix (fully informed by reality).
    #     # Shape: (N_S, 3)
    #     genetic_weights_baseline = np.tile(global_frequencies, (N_S, 1))
        
    #     # Generate a wide, unconstrained random Dirichlet matrix.
    #     # This acts as your "Exploration" channel, adding unique variance to each parallel worker.
    #     # Shape: (N_S, 3)
    #     alpha_vector = [1] * self.N_genetic_categories # how many vals there are: 3 for 0,1,2 allele or 2 for 0,1
        
    #     if rng is None:
    #         rng = getattr(self, 'global_rng', np.random.default_rng())
    #     random_exploration = rng.dirichlet(alpha=alpha_vector, size=N_S)
        
    #     # Convex Combination blending: 
    #     # w = 0.70 means the starting point is 70% anchored to the real dataset demographics
    #     # and 30% dedicated to pure random exploration. 
    #     # NOTE: If your experiments show too little exploration, drop w to 0.50.
    #     w = 0.70 
    #     #w = 0.30
    #     genetic_weights_init = (w * genetic_weights_baseline) + ((1.0 - w) * random_exploration)
        
    #     # Numerical Safeguards:
    #     # 1. Clip to ensure floating-point variances never create 0.0 or negative probabilities,
    #     #    which would crash the log-likelihood calculation downstream.
    #     genetic_weights_init = np.clip(genetic_weights_init, 1e-5, 1.0)
        
    #     # 2. Re-normalize row-wise to ensure that the mathematical integrity of the mixture prior 
    #     #    is intact (every subtype row must sum strictly to 1.0).
    #     genetic_weights_init /= np.sum(genetic_weights_init, axis=1, keepdims=True)
        
    #     return genetic_weights_init
    
    
    
    
    #********************************************

    def _find_ml_split(self, sustainData):
        # Fit a mixture of two models
        #
        #
        # OUTPUTS:
        # ml_sequence   - the ordering of the stages for each subtype
        # ml_f          - the most probable proportion of individuals belonging to each subtype
        # ml_likelihood - the likelihood of the most probable SuStaIn model

        N_S                                 = 2

        partial_iter                        = partial(self._find_ml_split_iteration, sustainData)
        seed_sequences = np.random.SeedSequence(self.global_rng.integers(1e10))
        pool_output_list                    = self.pool.map(partial_iter, seed_sequences.spawn(self.N_startpoints))

        if ~isinstance(pool_output_list, list):
            pool_output_list                = list(pool_output_list)

        ml_sequence_mat                     = np.zeros((N_S, sustainData.getNumStages(), self.N_startpoints))
        ml_f_mat                            = np.zeros((N_S, self.N_startpoints))
        ml_likelihood_mat                   = np.zeros((self.N_startpoints, 1))
        
        if self.apoe_flag:
            ml_genetic_weights_mat          = np.zeros((N_S,self.N_genetic_categories,self.N_startpoints))

        for i in range(self.N_startpoints):
            ml_sequence_mat[:, :, i]        = pool_output_list[i][0]
            ml_f_mat[:, i]                  = pool_output_list[i][1]
            ml_likelihood_mat[i]            = pool_output_list[i][2]
            
            if self.apoe_flag:
                
                ml_genetic_weights_mat[:,:,i] = pool_output_list[i][3]

        ix                                  = [np.where(ml_likelihood_mat == max(ml_likelihood_mat))[0][0]] #ugly bit of code to get first index where likelihood is maximum

        ml_sequence                         = ml_sequence_mat[:, :, ix]
        ml_f                                = ml_f_mat[:, ix]
        ml_likelihood                       = ml_likelihood_mat[ix]
        if self.apoe_flag:
            ml_genetic_weights              = ml_genetic_weights_mat[:,:,ix]
            
        return ml_sequence, ml_f, ml_likelihood, ml_sequence_mat, ml_f_mat, ml_likelihood_mat

    def _find_ml_split_iteration(self, sustainData, seed_seq):
        #Convenience sub-function for above

        # Get process-appropriate Generator
        rng = np.random.default_rng(seed_seq)

        N_S                                 = 2

        # randomly initialise individuals as belonging to one of the two subtypes (clusters)
        min_N_cluster                       = 0
        while min_N_cluster == 0:
            vals = rng.random(sustainData.getNumSamples())
            cluster_assignment = np.ceil(N_S * vals).astype(int)
            # Count cluster sizes
            # Guarantee 1s and 2s counts with minlength=3
            # Ignore 0s count with [1:]
            cluster_sizes = np.bincount(cluster_assignment, minlength=3)[1:]
            # Get the minimum cluster size
            min_N_cluster = cluster_sizes.min()

        # initialise the stages of the two models by fitting a single model to each of the two sets of individuals
        seq_init                            = np.zeros((N_S, sustainData.getNumStages()))
        for s in range(N_S):
            index_s                         = cluster_assignment.reshape(cluster_assignment.shape[0], ) == (s + 1)
            temp_sustainData                = sustainData.reindex(index_s)

            temp_seq_init                   = self._initialise_sequence(sustainData, rng)
            if self.apoe_flag:
                temp_genetic_weights_init = self._initialise_genetic_weights(N_S=1,rng=rng) 
                seq_init[s, :], _, _, _, _, _, _,_   = self._perform_em(temp_sustainData, temp_seq_init, [1], rng, temp_genetic_weights_init)
            else:  
                seq_init[s, :], _, _, _, _, _   = self._perform_em(temp_sustainData, temp_seq_init, [1], rng)

        f_init                              = np.array([1.] * N_S) / float(N_S)
        if self.apoe_flag:  
            genetic_weights_init = self._initialise_genetic_weights(N_S,rng)
            
            # optimise the mixture of two models from the initialisation
            this_ml_sequence, \
            this_ml_f, \
            this_ml_genetic_weights,\
            this_ml_likelihood, _, _, _,_         = self._perform_em(sustainData, seq_init, f_init, rng, genetic_weights_init)
            

            return this_ml_sequence, this_ml_f, this_ml_likelihood, this_ml_genetic_weights
        
        else:
            
            # optimise the mixture of two models from the initialisation
            this_ml_sequence, \
            this_ml_f, \
            this_ml_likelihood, _, _, _         = self._perform_em(sustainData, seq_init, f_init, rng)

            return this_ml_sequence, this_ml_f, this_ml_likelihood

    #********************************************
    def _find_ml_mixture(self, sustainData, seq_init, f_init):
        # Fit a mixture of models
        #
        #
        # OUTPUTS:
        # ml_sequence   - the ordering of the stages for each subtype for the next SuStaIn model in the hierarchy
        # ml_f          - the most probable proportion of individuals belonging to each subtype for the next SuStaIn model in the hierarchy
        # ml_likelihood - the likelihood of the most probable SuStaIn model for the next SuStaIn model in the hierarchy

        N_S                                 = seq_init.shape[0]

        partial_iter                        = partial(self._find_ml_mixture_iteration, sustainData, seq_init, f_init)
        seed_sequences = np.random.SeedSequence(self.global_rng.integers(1e10))
        pool_output_list                    = self.pool.map(partial_iter, seed_sequences.spawn(self.N_startpoints))

        if ~isinstance(pool_output_list, list):
            pool_output_list                = list(pool_output_list)

        ml_sequence_mat                     = np.zeros((N_S, sustainData.getNumStages(), self.N_startpoints))
        ml_f_mat                            = np.zeros((N_S, self.N_startpoints))
        ml_likelihood_mat                   = np.zeros((self.N_startpoints, 1))
        
        # creating an array to store all iterations of all startingpoints EM likelihoods
        MaxIter = 100
        em_likelihood_histories = np.nan * np.ones((MaxIter,self.N_startpoints ))
        
        if self.apoe_flag:
            ml_genetic_weights_mat          = np.zeros((N_S,self.N_genetic_categories,self.N_startpoints))


        for i in range(self.N_startpoints):
            ml_sequence_mat[:, :, i]        = pool_output_list[i][0]
            ml_f_mat[:, i]                  = pool_output_list[i][1]
            ml_likelihood_mat[i]            = pool_output_list[i][2]
            
            em_likelihood_histories[:,i]    = pool_output_list[i][5].ravel()
            
            if self.apoe_flag:
                ml_genetic_weights_mat[:,:,i] = pool_output_list[i][6]
                
            

        ix                                  = np.where(ml_likelihood_mat == max(ml_likelihood_mat))
        ix                                  = ix[0]

        ml_sequence                         = ml_sequence_mat[:, :, ix]
        ml_f                                = ml_f_mat[:, ix]
        ml_likelihood                       = ml_likelihood_mat[ix]
        
        print('EM lik shape after',em_likelihood_histories.shape)
        
        if self.apoe_flag:
            ml_genetic_weights              = ml_genetic_weights_mat[:,:,ix]

            return ml_sequence, ml_f, ml_likelihood, ml_sequence_mat, ml_f_mat, ml_likelihood_mat, em_likelihood_histories, ml_genetic_weights, ml_genetic_weights_mat
        else:
            return ml_sequence, ml_f, ml_likelihood, ml_sequence_mat, ml_f_mat, ml_likelihood_mat, em_likelihood_histories

    def _find_ml_mixture_iteration(self, sustainData, seq_init, f_init, seed_seq):
        #Convenience sub-function for above
        
        N_S = seq_init.shape[0]
        # Get process-appropriate Generator
        rng = np.random.default_rng(seed_seq)
        
        # initialise genetic weights for each startingpoint diff based on rng
        if self.apoe_flag:
            genetic_weights_init = self._initialise_genetic_weights(N_S,rng)
            
        
        if self.apoe_flag:
            
            ml_sequence,        \
            ml_f,               \
            ml_genetic_weights, \
            ml_likelihood,      \
            samples_sequence,   \
            samples_f,          \
            samples_genetic_weights,\
            samples_likelihood,                = self._perform_em(sustainData, seq_init, f_init, rng, genetic_weights_init)
        
            return ml_sequence, ml_f, ml_likelihood, samples_sequence, samples_f, samples_likelihood, ml_genetic_weights, samples_genetic_weights
        
        else:
            ml_sequence,        \
            ml_f,               \
            ml_likelihood,      \
            samples_sequence,   \
            samples_f,          \
            samples_likelihood                  = self._perform_em(sustainData, seq_init, f_init, rng)
            return ml_sequence, ml_f, ml_likelihood, samples_sequence, samples_f, samples_likelihood

        
    #********************************************
    def _run_anatomical_burnin(self, sustainData, current_sequence, current_f, rng, burn_in_iters=10):
        """
        Anatomical Burn-In Pre-Routine:
        Optimizes sequence and cohort fractions for a fixed number of iterations
        using a completely flat, non-informative genetic prior matrix.
        Returns the stabilized sequence layout and adjusted cohort fractions.
        """
        N_S = current_sequence.shape[0]
        
        # 1. Initialize temporary flat uniform weights to blind the likelihood engine
        uniform_weights = np.ones((N_S, self.N_genetic_categories)) / self.N_genetic_categories
        
        S_stable = current_sequence.copy()
        f_stable = current_f.copy()
        
        # 2. Run the isolated burn-in loop
        for _ in range(burn_in_iters):
            S_stable, f_stable, _ = self._optimise_parameters(
                sustainData, S_stable, f_stable, rng, uniform_weights
            )
            
        print(f"--- Upfront Anatomical Burn-In Complete ({burn_in_iters} iterations) ---")
        return S_stable, f_stable
    
    def _perform_em(self, sustainData, current_sequence, current_f, rng, current_genetic_weights=None):
        # allow method to be updated from input
        
        method = self.em_loop_type
        #method = 'alternating'
        #method = 'combined'
        
        use_burn_in_phase = False
        
        if self.apoe_flag:
            if use_burn_in_phase:
                # Re-assign the starting parameters to your stabilized spatial layouts
                current_sequence, current_f = self._run_anatomical_burnin(
                    sustainData, current_sequence, current_f, rng, burn_in_iters=0
                )
                
            if method == 'alternating':
                #print('Using',method,'method' )
                return self._perform_em_alternating(sustainData, current_sequence, current_f, rng, current_genetic_weights)
            else:
                #print('Using',method,'method')
                return self._perform_em_combined(sustainData, current_sequence, current_f, rng, current_genetic_weights)

        # Perform an E-M procedure to estimate parameters of SuStaIn model
        MaxIter                             = 100

        N                                   = sustainData.getNumStages()    #self.stage_zscore.shape[1]
        N_S                                 = current_sequence.shape[0]
        current_likelihood, _, _, _, _      = self._calculate_likelihood(sustainData, current_sequence, current_f)

        terminate                           = 0
        iteration                           = 0
        samples_sequence                    = np.nan * np.ones((MaxIter, N, N_S))
        samples_f                           = np.nan * np.ones((MaxIter, N_S))
        samples_likelihood                  = np.nan * np.ones((MaxIter, 1))

        samples_sequence[0, :, :]           = current_sequence.reshape(current_sequence.shape[1], current_sequence.shape[0])
        current_f                           = np.array(current_f).reshape(len(current_f))
        samples_f[0, :]                     = current_f
        samples_likelihood[0]               = current_likelihood
        
        while terminate == 0:
            
            # optimising sequence and f
            
            candidate_sequence,     \
            candidate_f,            \
            candidate_likelihood            = self._optimise_parameters(sustainData, current_sequence, current_f, rng)

            HAS_converged                   = np.fabs((candidate_likelihood - current_likelihood) / max(candidate_likelihood, current_likelihood)) < 1e-6
            if HAS_converged:
                print('EM converged in', iteration + 1, 'iterations')
                terminate                   = 1
            else:
                if candidate_likelihood > current_likelihood:
                    current_sequence        = candidate_sequence
                    current_f               = candidate_f
                    current_likelihood      = candidate_likelihood

            
            samples_sequence[iteration, :, :] = current_sequence.T.reshape(current_sequence.T.shape[0], N_S)
            samples_f[iteration, :]         = current_f
            samples_likelihood[iteration]   = current_likelihood

            if iteration == (MaxIter - 1):
                terminate                   = 1
            iteration                       = iteration + 1

        ml_sequence                         = current_sequence
        ml_f                                = current_f
        ml_likelihood                       = current_likelihood
        return ml_sequence, ml_f, ml_likelihood, samples_sequence, samples_f, samples_likelihood
 
    
    def _perform_em_alternating(self, sustainData, current_sequence, current_f, rng, current_genetic_weights): # do i need to pass current_genetic_weights as input var too?

        # Perform an E-M procedure to estimate parameters of SuStaIn model
        MaxIter                             = 100

        N                                   = sustainData.getNumStages()    #self.stage_zscore.shape[1]
        N_S                                 = current_sequence.shape[0]
        
        current_likelihood, _, _, _, _      = self._calculate_likelihood(sustainData, current_sequence, current_f, current_genetic_weights)

        terminate                           = 0
        iteration                           = 0
        samples_sequence                    = np.nan * np.ones((MaxIter, N, N_S))
        samples_f                           = np.nan * np.ones((MaxIter, N_S))
        samples_likelihood                  = np.nan * np.ones((MaxIter, 1))
        samples_genetics                    = np.nan * np.ones((MaxIter, N_S, self.N_genetic_categories))

        samples_sequence[0, :, :]           = current_sequence.reshape(current_sequence.shape[1], current_sequence.shape[0])
        current_f                           = np.array(current_f).reshape(len(current_f))
        samples_f[0, :]                     = current_f
        samples_likelihood[0]               = current_likelihood
        samples_genetics[0,:,:]             = current_genetic_weights
        
            
        HAS_converged_seq = False
        HAS_converged_gen = False
        while terminate == 0:
            
            # optimising sequence and f
            
            if not HAS_converged_seq:
                # print('Optimising biomarker sequence structure...')
                candidate_sequence,     \
                candidate_f,            \
                candidate_likelihood        = self._optimise_parameters(sustainData, current_sequence, current_f, rng,current_genetic_weights)
                
                if np.fabs((candidate_likelihood - current_likelihood) / max(candidate_likelihood, current_likelihood)) < 1e-6:
                    HAS_converged_seq = True
                    #print(f'Biomarker sequence layout converged at iteration {iteration}')
                else:
                    if candidate_likelihood > current_likelihood:
                        current_sequence        = candidate_sequence
                        current_f               = candidate_f
                        current_likelihood      = candidate_likelihood
                        #print('Accepting new clinical biomarker order')
            
            
            # optimise genetic parameters
            if not HAS_converged_gen:
                # print('Optimising genetic parameters...')
                candidate_genetic_weights, \
                candidate_genetic_likelihood        = self._optimise_genetic_parameters(sustainData, current_sequence, current_f, current_genetic_weights)
                
                #print('Candidate genetic weights', candidate_genetic_weights)
                #print('Cand gen lik',candidate_genetic_likelihood )
                #print('Current lik', current_likelihood)
                # Check localized genetic convergence criterion
                if np.fabs((candidate_genetic_likelihood - current_likelihood) / max(candidate_genetic_likelihood, current_likelihood)) < 1e-6:
                    HAS_converged_gen = True
                    print(f'Genetic weight estimation converged at iteration {iteration}')
                else:
                    if candidate_genetic_likelihood > current_likelihood:
                        current_likelihood = candidate_genetic_likelihood
                        current_genetic_weights = candidate_genetic_weights
                        #print('Accepting new genetic weights matrix')
                
                
            if HAS_converged_seq and HAS_converged_gen :
                print('EM converged in', iteration + 1, 'iterations')
                terminate                   = 1
            

            samples_sequence[iteration, :, :] = current_sequence.T.reshape(current_sequence.T.shape[0], N_S)
            samples_f[iteration, :]           = current_f
            samples_genetics[iteration, :, :] = current_genetic_weights  # <-- Log weights snapshot here
            samples_likelihood[iteration]     = current_likelihood

            if iteration == (MaxIter - 1):
                terminate                   = 1
            iteration                       = iteration + 1

        ml_sequence                         = current_sequence
        ml_f                                = current_f
        ml_genetic_weights                  = current_genetic_weights
        ml_likelihood                       = current_likelihood
        
        return ml_sequence, ml_f, ml_genetic_weights ,ml_likelihood, samples_sequence, samples_f, samples_genetics, samples_likelihood
    
    
    
    
    # 1 step perform em 
    # IF combined method works better, rename it as perform_em and use it everywhere!
    def _perform_em_combined(self, sustainData, current_sequence, current_f, rng, current_genetic_weights): # do i need to pass current_genetic_weights as input var too?
        ''' In this function I will allow both original params (sequence,f) and new params (genetic weights) to 
        optimise in the same EM sequence
        
        '''
      
    
        # Perform an E-M procedure to estimate parameters of SuStaIn model
        MaxIter                             = 100

        N                                   = sustainData.getNumStages()    #self.stage_zscore.shape[1]
        N_S                                 = current_sequence.shape[0]
        
        
        if self.apoe_flag:
            current_likelihood, _, _, _, _      = self._calculate_likelihood(sustainData, current_sequence, current_f, current_genetic_weights)
        else:
            current_likelihood, _, _, _, _      = self._calculate_likelihood(sustainData, current_sequence, current_f)

        terminate                           = 0
        iteration                           = 0
        samples_sequence                    = np.nan * np.ones((MaxIter, N, N_S))
        samples_f                           = np.nan * np.ones((MaxIter, N_S))
        samples_likelihood                  = np.nan * np.ones((MaxIter, 1))
        if self.apoe_flag:   
            samples_genetics                    = np.nan * np.ones((MaxIter, N_S, self.N_genetic_categories))

        samples_sequence[0, :, :]           = current_sequence.reshape(current_sequence.shape[1], current_sequence.shape[0])
        current_f                           = np.array(current_f).reshape(len(current_f))
        samples_f[0, :]                     = current_f
        samples_likelihood[0]               = current_likelihood
        if self.apoe_flag:
            samples_genetics[0,:,:]             = current_genetic_weights
        
            
        while terminate == 0:
            
            # optimising sequence and f
            if self.apoe_flag:
                candidate_sequence,     \
                candidate_f,            \
                candidate_likelihood,   \
                candidate_genetic_weights   = self._optimise_parameters_combined(sustainData, current_sequence, current_f, rng, current_genetic_weights)
            
            else:
                
                candidate_sequence,     \
                candidate_f,            \
                candidate_likelihood            = self._optimise_parameters(sustainData, current_sequence, current_f, rng)

            HAS_converged                   = np.fabs((candidate_likelihood - current_likelihood) / max(candidate_likelihood, current_likelihood)) < 1e-6
            if HAS_converged:
                print('EM converged in', iteration + 1, 'iterations')
                terminate                   = 1
            else:
                if candidate_likelihood > current_likelihood:
                    current_sequence        = candidate_sequence
                    current_f               = candidate_f
                    current_likelihood      = candidate_likelihood
                    if self.apoe_flag:
                        current_genetic_weights = candidate_genetic_weights

            samples_sequence[iteration, :, :] = current_sequence.T.reshape(current_sequence.T.shape[0], N_S)
            samples_f[iteration, :]           = current_f
            if self.apoe_flag:   
                samples_genetics[iteration, :, :] = current_genetic_weights  # <-- Log weights snapshot here
            samples_likelihood[iteration]     = current_likelihood

            if iteration == (MaxIter - 1):
                terminate                   = 1
            iteration                       = iteration + 1

        ml_sequence                         = current_sequence
        ml_f                                = current_f
        ml_likelihood                       = current_likelihood
        
        if self.apoe_flag:  
            ml_genetic_weights                  = current_genetic_weights
            return ml_sequence, ml_f, ml_genetic_weights ,ml_likelihood, samples_sequence, samples_f, samples_genetics, samples_likelihood
        else:
            return ml_sequence, ml_f,ml_likelihood, samples_sequence, samples_f, samples_likelihood
        

    def _calculate_likelihood(self, sustainData, S, f, genetic_weights=None):
        # Computes the likelihood of a mixture of models
        #
        #
        # OUTPUTS:
        # loglike               - the log-likelihood of the current model
        # total_prob_subj       - the total probability of the current SuStaIn model for each subject
        # total_prob_stage      - the total probability of each stage in the current SuStaIn model
        # total_prob_cluster    - the total probability of each subtype in the current SuStaIn model
        # p_perm_k              - the probability of each subjects data at each stage of each subtype in the current SuStaIn model

        M                                   = sustainData.getNumSamples()  #data_local.shape[0]
        N_S                                 = S.shape[0]
        N                                   = sustainData.getNumStages()    #self.stage_zscore.shape[1]

        f                                   = np.array(f).reshape(N_S, 1, 1)
        f_val_mat                           = np.tile(f, (1, N + 1, M))
        f_val_mat                           = np.transpose(f_val_mat, (2, 1, 0))

        p_perm_k                            = np.zeros((M, N + 1, N_S))

        for s in range(N_S):
            p_perm_k[:, :, s]               = self._calculate_likelihood_stage(sustainData, S[s])  #self.__calculate_likelihood_stage_linearzscoremodel_approx(data_local, S[s])

        
      
        if self.apoe_flag: # need to implement a flag as input to model
        
            # M= numb patients; N_S = no. subtypes; N=no. stages
            # apoe_dummy shape: (M, 3), genetic_weights shape: (N_S, 3)
            # genetic_prior shape: (M, N_S) 
            apoe_dummy = sustainData.apoe
            
            # # fake genetic prior to test lik
            # rng_mock = np.random.default_rng(42)
            # genetic_weights = rng_mock.dirichlet(alpha=[1, 1, 1], size=N_S)
            #print('Genetic weights',genetic_weights.shape, genetic_weights)
            
            # 1. Multiply matrices to get a patient-by-subtype prior: shape (M, N_S)
            #print('APoe dummy',apoe_dummy)
            #print('genetic weights',genetic_weights)
            genetic_prior = apoe_dummy @ genetic_weights.T
            
            # 2. Reshape to (M, 1, N_S) so it stretches perfectly across the (N + 1) stages
            genetic_prior = genetic_prior[:, np.newaxis, :]
            #print('Genetic prior 3D shape:', genetic_prior.shape)
            
            
            total_prob_cluster                  = np.squeeze(np.sum(p_perm_k * f_val_mat * genetic_prior, 1))
        else:
            total_prob_cluster                  = np.squeeze(np.sum(p_perm_k * f_val_mat, 1))
        
        
        # print(' total_prob_cluster   ',total_prob_cluster.shape, total_prob_cluster)
        if self.apoe_flag:
            total_prob_stage                    = np.sum(p_perm_k * f_val_mat * genetic_prior, 2)
        else:  
            total_prob_stage                    = np.sum(p_perm_k * f_val_mat, 2)
        total_prob_subj                     = np.sum(total_prob_stage, 1)

        loglike                             = np.sum(np.log(total_prob_subj + 1e-250))
      

        return loglike, total_prob_subj, total_prob_stage, total_prob_cluster, p_perm_k

    def _estimate_uncertainty_sustain_model(self, sustainData, seq_init, f_init, genetic_weights_init=None):
        # Estimate the uncertainty in the subtype progression patterns and
        # proportion of individuals belonging to the SuStaIn model
        #
        #
        # OUTPUTS:
        # ml_sequence       - the most probable ordering of the stages for each subtype found across MCMC samples
        # ml_f              - the most probable proportion of individuals belonging to each subtype found across MCMC samples
        # ml_likelihood     - the likelihood of the most probable SuStaIn model found across MCMC samples
        # samples_sequence  - samples of the ordering of the stages for each subtype obtained from MCMC sampling
        # samples_f         - samples of the proportion of individuals belonging to each subtype obtained from MCMC sampling
        # samples_likeilhood - samples of the likelihood of each SuStaIn model sampled by the MCMC sampling
        if self.apoe_flag:
            
            # Perform a few initial passes where the perturbation sizes of the MCMC uncertainty estimation are tuned
            seq_sigma_opt, f_sigma_opt, genetic_sigma_opt   = self._optimise_mcmc_settings(sustainData, seq_init, f_init,genetic_weights_init)
            
            #print('Genetics sigma (Step size) after dyanmic optimisation:',genetic_sigma_opt)
        
            # Run the full MCMC algorithm to estimate the uncertainty
            ml_sequence,        \
            ml_f,               \
            ml_likelihood,      \
            samples_sequence,   \
            samples_f,          \
            samples_likelihood, \
            ml_genetic_weights, \
            samples_genetic_weights          = self._perform_mcmc(sustainData, seq_init, f_init, self.N_iterations_MCMC, seq_sigma_opt, f_sigma_opt,genetic_weights_init,genetic_sigma_opt )

            return ml_sequence, ml_f, ml_likelihood, samples_sequence, samples_f, samples_likelihood, ml_genetic_weights, samples_genetic_weights


        
        # Perform a few initial passes where the perturbation sizes of the MCMC uncertainty estimation are tuned
        seq_sigma_opt, f_sigma_opt          = self._optimise_mcmc_settings(sustainData, seq_init, f_init)

        # Run the full MCMC algorithm to estimate the uncertainty
        ml_sequence,        \
        ml_f,               \
        ml_likelihood,      \
        samples_sequence,   \
        samples_f,          \
        samples_likelihood                  = self._perform_mcmc(sustainData, seq_init, f_init, self.N_iterations_MCMC, seq_sigma_opt, f_sigma_opt)

        return ml_sequence, ml_f, ml_likelihood, samples_sequence, samples_f, samples_likelihood
    
    def _optimise_mcmc_settings(self, sustainData, seq_init, f_init, genetic_weights_init=None):

        # Optimise the perturbation size for the MCMC algorithm
        n_iterations_MCMC_optimisation      = int(1e4)  # FIXME: set externally

        n_passes_optimisation               = 3

        seq_sigma_currentpass               = 1
        f_sigma_currentpass                 = 0.01  # magic number
        if self.apoe_flag:
            # step size for genetic_Weights initialisation
            genetics_sigma_currentpass         = 0.01

        N_S                                 = seq_init.shape[0]

        for i in range(n_passes_optimisation):
            if self.apoe_flag:
                _, _, _, samples_sequence_currentpass, samples_f_currentpass, _,_, samples_genetic_currentpass = self._perform_mcmc(   sustainData,
                                                                                                                                     seq_init,
                                                                                                                                     f_init,
                                                                                                                                     n_iterations_MCMC_optimisation,
                                                                                                                                     seq_sigma_currentpass,
                                                                                                                                     f_sigma_currentpass,
                                                                                                                                     genetic_weights_init,
                                                                                                                                     genetics_sigma_currentpass)
            else:
                
                _, _, _, samples_sequence_currentpass, samples_f_currentpass, _ = self._perform_mcmc(   sustainData,
                                                                                                         seq_init,
                                                                                                         f_init,
                                                                                                         n_iterations_MCMC_optimisation,
                                                                                                         seq_sigma_currentpass,
                                                                                                         f_sigma_currentpass)

            samples_position_currentpass    = np.zeros(samples_sequence_currentpass.shape)
            for s in range(N_S):
                for sample in range(n_iterations_MCMC_optimisation):
                    temp_seq                        = samples_sequence_currentpass[s, :, sample]
                    temp_inv                        = np.array([0] * samples_sequence_currentpass.shape[1])
                    temp_inv[temp_seq.astype(int)]  = np.arange(samples_sequence_currentpass.shape[1])
                    samples_position_currentpass[s, :, sample] = temp_inv

            seq_sigma_currentpass           = np.std(samples_position_currentpass, axis=2, ddof=1)  # np.std is different to Matlab std, which normalises to N-1 by default
            seq_sigma_currentpass[seq_sigma_currentpass < 0.01] = 0.01  # magic number

            f_sigma_currentpass             = np.std(samples_f_currentpass, axis=1, ddof=1)         # np.std is different to Matlab std, which normalises to N-1 by default
            
            if self.apoe_flag:
                genetics_sigma_currentpass = np.std(samples_genetic_currentpass, axis = 2, ddof=1)# # on what axis2 and ddof

        seq_sigma_opt                       = seq_sigma_currentpass
        f_sigma_opt                         = f_sigma_currentpass
        
        if self.apoe_flag:
            genetics_sigma_opt              = genetics_sigma_currentpass

            return seq_sigma_opt, f_sigma_opt, genetics_sigma_opt
        else:
            return seq_sigma_opt, f_sigma_opt

    def _evaluate_likelihood_setofsamples(self, sustainData, samples_sequence, samples_f, samples_genetic_weights=None):
    
        n_total                             = samples_sequence.shape[2]
    
        #reduce the number of samples to speed this function up
        if n_total >= 1e6:
            N_samples                       = int(np.round(n_total/1000))
        elif n_total >= 1e5:
            N_samples                       = int(np.round(n_total/100))
        else:
            N_samples                       = n_total        
        select_samples                      = np.round(np.linspace(0, n_total - 1, N_samples)).astype(int)               
    
        samples_sequence                    = samples_sequence[:, :, select_samples]
        samples_f                           = samples_f[:, select_samples]
        
        if self.apoe_flag:
            samples_genetic_weights         = samples_genetic_weights[:,:,select_samples]
    
        # Take MCMC samples of the uncertainty in the SuStaIn model parameters
        M                                   = sustainData.getNumSamples()   #data_local.shape[0]
        n_iterations                        = samples_sequence.shape[2]
        samples_likelihood_subj             = np.zeros((M, n_iterations))
        for i in range(n_iterations):
            S                               = samples_sequence[:, :, i]
            f                               = samples_f[:, i]
            if self.apoe_flag:
                genetic_weights             = samples_genetic_weights[:,:,i]
                _, likelihood_sample_subj, _, _, _  = self._calculate_likelihood(sustainData, S, f,genetic_weights)
            else:
                _, likelihood_sample_subj, _, _, _  = self._calculate_likelihood(sustainData, S, f)

            samples_likelihood_subj[:, i]   = likelihood_sample_subj

        return samples_likelihood_subj


    # ********************* ABSTRACT METHODS
    @abstractmethod
    def _initialise_sequence(self, sustainData, rng):
        pass

    @abstractmethod
    def _calculate_likelihood_stage(self, sustainData, S):
        pass

    @abstractmethod
    def _optimise_parameters(self, sustainData, S_init, f_init, rng):
        pass

    @abstractmethod
    def _perform_mcmc(self, sustainData, seq_init, f_init, n_iterations, seq_sigma, f_sigma, genetic_weights_init=None):
        pass

    @abstractmethod
    def _plot_sustain_model():
        pass

    @staticmethod
    @abstractmethod
    def plot_positional_var():
        pass

    @abstractmethod
    def subtype_and_stage_individuals_newData(self):    #up to the implementations to define exact number of params here
        pass

    # ********************* STATIC METHODS
    @staticmethod
    def calc_coeff(sig):
        return 1. / np.sqrt(np.pi * 2.0) * sig

    @staticmethod
    def calc_exp(x, mu, sig):
        x = (x - mu) / sig
        return np.exp(-.5 * x * x)

    @staticmethod
    def check_biomarker_colours(biomarker_colours, biomarker_labels):
        if isinstance(biomarker_colours, dict):
            # Check each label exists
            assert all(i in biomarker_labels for i in biomarker_colours.keys()), "A label doesn't match!"
            # Check each colour exists
            assert all(mcolors.is_color_like(i) for i in biomarker_colours.values()), "A proper colour wasn't given!"
            # Add in any colours that aren't defined, allowing for partial colouration
            for label in biomarker_labels:
                if label not in biomarker_colours:
                    biomarker_colours[label] = "black"
        elif isinstance(biomarker_colours, (list, tuple)):
            # Check each colour exists
            assert all(mcolors.is_color_like(i) for i in biomarker_colours), "A proper colour wasn't given!"
            # Check right number of colours given
            assert len(biomarker_colours) == len(biomarker_labels), "The number of colours and labels do not match!"
            # Turn list of colours into a label:colour mapping
            biomarker_colours = {k:v for k,v in zip(biomarker_labels, biomarker_colours)}
        else:
            raise TypeError("A dictionary mapping label:colour or list/tuple of colours must be given!")
        return biomarker_colours

    # ********************* TEST METHODS
    @staticmethod
    @abstractmethod
    def generate_random_model():
        pass

    @staticmethod
    @abstractmethod
    def generate_data():
        pass

    @classmethod
    @abstractmethod
    def test_sustain(cls):
        pass


from multiprocessing import Value
import warnings
from tqdm.auto import tqdm
import numpy as np
from matplotlib import pyplot as plt
from scipy.stats import norm
import pandas as pd

# from pySuStaIn.AbstractSustain import AbstractSustainData
# from pySuStaIn.AbstractSustain import AbstractSustain

#*******************************************
#The data structure class for ZscoreSustain_APOE4. It holds the z-scored data that gets passed around and re-indexed in places.
class ZScoreSustainData(AbstractSustainData):

    def __init__(self, data, numStages,apoe=None):
        self.data           = data
        self.__numStages    = numStages
    
        if isinstance(apoe, pd.DataFrame):
            self.apoe =  apoe.to_numpy()  # Convert DataFrame to NumPy array
        else:
            self.apoe = apoe

    def getNumSamples(self):
        return self.data.shape[0]

    def getNumBiomarkers(self):
        return self.data.shape[1]

    def getNumStages(self):
        return self.__numStages

    def reindex(self, index):
        # print('Reindexing works for apoe:',self.apoe[index,:])
        
        if self.apoe is None: 
            return ZScoreSustainData(self.data[index,], self.__numStages)
            
        else:
            return ZScoreSustainData(self.data[index,], self.__numStages, self.apoe[index,:])

#*******************************************
#An implementation of the AbstractSustain class with multiple events for each biomarker based on deviations from normality, measured in z-scores.
#There are a fixed number of thresholds for each biomarker, specified at initialization of the ZscoreSustain object.
class ZscoreSustain_APOE4(AbstractSustain):

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
                 seed=None,
                 apoe4_status=None, 
                 apoe_flag=False,
                 em_loop_type = "combined" # or "alternating"
                 ):
        # The initializer for the z-score based events implementation of AbstractSustain
        # Parameters:
        #   data                        - !important! needs to be (positive) z-scores!
        #                                 dim: number of subjects x number of biomarkers
        #   Z_vals                      - a matrix specifying the z-score thresholds for each biomarker
        #                                 for M biomarkers and 3 thresholds (1,2 and 3 for example) this would be a dim: M x 3 matrix
        #   Z_max                       - a vector specifying the maximum z-score for each biomarker
        #                                 when using z-score thresholds of 1,2,3 this would typically be 5.
        #                                 for M biomarkers this would be a dim: M x 1 vector
        #   biomarker_labels            - the names of the biomarkers as a list of strings
        #   N_startpoints               - number of startpoints to use in maximum likelihood step of SuStaIn, typically 25
        #   N_S_max                     - maximum number of subtypes, should be 1 or more
        #   N_iterations_MCMC           - number of MCMC iterations, typically 1e5 or 1e6 but can be lower for debugging
        #   output_folder               - where to save pickle files, etc.
        #   dataset_name                - for naming pickle files
        #   use_parallel_startpoints    - boolean for whether or not to parallelize the maximum likelihood loop
        #   seed                        - random number seed
        
        #######################
        #Modifications
        # we take apoe4 carrier status as an input variable
        # apoe4_status                  - apoe4 allele carrier status, 0 or 1
        self.apoe_flag = apoe_flag
        
        self.em_loop_type = em_loop_type  # 'combined' or 'alternating'
        
        if self.apoe_flag:
            if apoe4_status is None:
                raise ValueError(
                "CRITICAL ERROR: 'apoe_flag' is set to True, but 'apoe4_status' is None. "
                "You must pass a valid array of genotypes to execute the genetic model pipeline. "
                "Alternatively, set 'apoe_flag=False' to execute standard baseline SuStaIn."
            )
                
        
        # clean the data to ensure no apoe4 are missing
        if self.apoe_flag:
            # remove patients with nan apoe 
            valid = ~np.isnan(apoe4_status).flatten()
            print(f"{data.shape[0]} patients initially")
            clean_data = data[valid, :]
            data = clean_data
            clean_apoe = apoe4_status[valid].astype(int)
            
            # transform apoe4 carrier status as a dummy varibale with 3 cols
            N = len(clean_apoe)
            unique_categories = np.unique(clean_apoe)
            self.N_genetic_categories = len(unique_categories)
            print(f" -> Detected {self.N_genetic_categories} unique genetic categories: {unique_categories}")
            
            # Transform APOE carrier status as a dummy variable dynamically scaled
            N = len(clean_apoe)
            apoe_dummy = np.zeros((N, self.N_genetic_categories)) # 
            apoe_dummy[np.arange(N), clean_apoe] = 1.0
            
            print(f"{clean_data.shape[0]} patients with non null apoe4 carrier status")
            
        # determine global APOE frequencies to start initialisation from for later
        if self.apoe_flag:
            genotype_counts = np.sum(apoe_dummy, axis=0)
            M_total = apoe_dummy.shape[0]
            # Add a tiny epsilon (1e-12) to prevent any potential division-by-zero 
            # if a genotype is entirely missing from a small pilot dataset
            self._global_genetic_frequencies = genotype_counts / float(M_total + 1e-12)
            print(f" -> [Genetics Setup] Global cohort background frequencies calculated: {self._global_genetic_frequencies}")
        
        # # testing out initialisation
        # test_rng = getattr(self, 'global_rng', np.random.default_rng(42))
        # test_N_S = 2
        # print('genetic initialisation function 2',self._initialise_genetic_weights(test_N_S,test_rng))
        
        
        N                               = data.shape[1]  # number of biomarkers
        assert (len(biomarker_labels) == N), "number of labels should match number of biomarkers"

        stage_zscore            = Z_vals.T.flatten()    #np.array([y for x in Z_vals.T for y in x])
        stage_zscore            = stage_zscore.reshape(1,len(stage_zscore))
        IX_select               = stage_zscore>0
        stage_zscore            = stage_zscore[IX_select]
        stage_zscore            = stage_zscore.reshape(1,len(stage_zscore))

        num_zscores             = Z_vals.shape[1]
        IX_vals                 = np.array([[x for x in range(N)]] * num_zscores).T
        stage_biomarker_index   = IX_vals.T.flatten()   #np.array([y for x in IX_vals.T for y in x])
        stage_biomarker_index   = stage_biomarker_index.reshape(1,len(stage_biomarker_index))
        stage_biomarker_index   = stage_biomarker_index[IX_select]
        stage_biomarker_index   = stage_biomarker_index.reshape(1,len(stage_biomarker_index))

        self.Z_vals                     = Z_vals
        self.stage_zscore               = stage_zscore
        self.stage_biomarker_index      = stage_biomarker_index

        self.min_biomarker_zscore       = [0] * N
        self.max_biomarker_zscore       = Z_max
        self.std_biomarker_zscore       = [1] * N

        self.biomarker_labels           = biomarker_labels

        numStages                       = stage_zscore.shape[1]
        
        if apoe_flag:
            self.__sustainData              = ZScoreSustainData(data, numStages,apoe_dummy)
        else:
            self.__sustainData              = ZScoreSustainData(data, numStages)
        
        

        super().__init__(self.__sustainData,
                         N_startpoints,
                         N_S_max,
                         N_iterations_MCMC,
                         output_folder,
                         dataset_name,
                         use_parallel_startpoints,
                         seed)


    def _initialise_sequence(self, sustainData, rng):
        # Randomly initialises a linear z-score model ensuring that the biomarkers
        # are monotonically increasing
        #
        #
        # OUTPUTS:
        # S - a random linear z-score model under the condition that each biomarker
        # is monotonically increasing

        N                                   = np.array(self.stage_zscore).shape[1]
        S                                   = np.zeros(N)
        for i in range(N):

            IS_min_stage_zscore             = np.array([False] * N)
            possible_biomarkers             = np.unique(self.stage_biomarker_index)
            for j in range(len(possible_biomarkers)):
                IS_unselected               = [False] * N
                for k in set(range(N)) - set(S[:i]):
                    IS_unselected[k]        = True

                this_biomarkers             = np.array([(np.array(self.stage_biomarker_index)[0] == possible_biomarkers[j]).astype(int) +
                                                        (np.array(IS_unselected) == 1).astype(int)]) == 2
                if not np.any(this_biomarkers):
                    this_min_stage_zscore   = 0
                else:
                    this_min_stage_zscore   = min(self.stage_zscore[this_biomarkers])
                if (this_min_stage_zscore):
                    temp                    = ((this_biomarkers.astype(int) + (self.stage_zscore == this_min_stage_zscore).astype(int)) == 2).T
                    temp                    = temp.reshape(len(temp), )
                    IS_min_stage_zscore[temp] = True

            events                          = np.array(range(N))
            possible_events                 = np.array(events[IS_min_stage_zscore])
            this_index                      = np.ceil(rng.random() * ((len(possible_events)))) - 1
            S[i]                            = possible_events[int(this_index)]

        S                                   = S.reshape(1, len(S))
        return S

    def _calculate_likelihood_stage(self, sustainData, S):
        '''
         Computes the likelihood of a single linear z-score model using an
         approximation method (faster)
        Outputs:
        ========
         p_perm_k - the probability of each subjects data at each stage of a particular subtype
         in the SuStaIn model
        '''

        N                                   = self.stage_biomarker_index.shape[1]
        S_inv                               = np.array([0] * N)
        S_inv[S.astype(int)]                = np.arange(N)
        possible_biomarkers                 = np.unique(self.stage_biomarker_index)
        B                                   = len(possible_biomarkers)
        point_value                         = np.zeros((B, N + 2))

        # all the arange you'll need below
        arange_N                            = np.arange(N + 2)

        for i in range(B):
            b                               = possible_biomarkers[i]
            event_location                  = np.concatenate([[0], S_inv[(self.stage_biomarker_index == b)[0]], [N]])
            event_value                     = np.concatenate([[self.min_biomarker_zscore[i]], self.stage_zscore[self.stage_biomarker_index == b], [self.max_biomarker_zscore[i]]])
            for j in range(len(event_location) - 1):

                if j == 0:  # FIXME: nasty hack to get Matlab indexing to match up - necessary here because indices are used for linspace limits

                    # original
                    #temp                   = np.arange(event_location[j],event_location[j+1]+2)
                    #point_value[i,temp]    = np.linspace(event_value[j],event_value[j+1],event_location[j+1]-event_location[j]+2)

                    # fastest by a bit
                    temp                    = arange_N[event_location[j]:(event_location[j + 1] + 2)]
                    N_j                     = event_location[j + 1] - event_location[j] + 2
                    point_value[i, temp]    = ZscoreSustain_APOE4.linspace_local2(event_value[j], event_value[j + 1], N_j, arange_N[0:N_j])

                else:
                    # original
                    #temp                   = np.arange(event_location[j] + 1, event_location[j + 1] + 2)
                    #point_value[i, temp]   = np.linspace(event_value[j],event_value[j+1],event_location[j+1]-event_location[j]+1)

                    # fastest by a bit
                    temp                    = arange_N[(event_location[j] + 1):(event_location[j + 1] + 2)]
                    N_j                     = event_location[j + 1] - event_location[j] + 1
                    point_value[i, temp]    = ZscoreSustain_APOE4.linspace_local2(event_value[j], event_value[j + 1], N_j, arange_N[0:N_j])

        stage_value                         = 0.5 * point_value[:, :point_value.shape[1] - 1] + 0.5 * point_value[:, 1:]

        M                                   = sustainData.getNumSamples()   #data_local.shape[0]
        p_perm_k                            = np.zeros((M, N + 1))

        # optimised likelihood calc - take log and only call np.exp once after loop
        sigmat = np.array(self.std_biomarker_zscore)

        factor                              = np.log(1. / np.sqrt(np.pi * 2.0) * sigmat)
        coeff                               = np.log(1. / float(N + 1))

        # original
        """
        for j in range(N+1):
            x                   = (data-np.tile(stage_value[:,j],(M,1)))/sigmat
            p_perm_k[:,j]       = coeff+np.sum(factor-.5*x*x,1)
        """
        # faster - do the tiling once
        # stage_value_tiled                   = np.tile(stage_value, (M, 1))
        # N_biomarkers                        = stage_value.shape[0]
        # for j in range(N + 1):
        #     stage_value_tiled_j             = stage_value_tiled[:, j].reshape(M, N_biomarkers)
        #     x                               = (sustainData.data - stage_value_tiled_j) / sigmat  #(data_local - stage_value_tiled_j) / sigmat
        #     p_perm_k[:, j]                  = coeff + np.sum(factor - .5 * np.square(x), 1)
        # p_perm_k                            = np.exp(p_perm_k)

        # even faster - do in one go
        x = (sustainData.data[:, :, None] - stage_value) / sigmat[None, :, None]
        p_perm_k = coeff + np.sum(factor[None, :, None] - 0.5 * np.square(x), 1)
        p_perm_k = np.exp(p_perm_k)

        return p_perm_k
    

    def _optimise_parameters(self, sustainData, S_init, f_init, rng, genetic_weights= None):
        '''
        Core Behavior:
          - Optimizes clinical biomarker Sequences (S) and Cohort Fractions (f).
          - Keeps Categorical Genetic Weights (W) COMPLETELY FIXED.
        
        Used By:
          - Baseline SuStaIn (apoe_flag = False).
          - The upfront "Anatomical Burn-In" pre-routine phase.
          - Step 1 of the 'alternating' EM optimization method.
        
        Mathematical Framework:
          Standard piece-wise linear z-score likelihood maximization. If genetic 
          weights are passed in, they act as static background scaling constants.
          '''
          
        # Optimise the parameters of the SuStaIn model

        M                                   = sustainData.getNumSamples()   #data_local.shape[0]
        N_S                                 = S_init.shape[0]
        N                                   = self.stage_zscore.shape[1]

        S_opt                               = S_init.copy()  # have to copy or changes will be passed to S_init
        f_opt                               = np.array(f_init).reshape(N_S, 1, 1)
        f_val_mat                           = np.tile(f_opt, (1, N + 1, M))
        f_val_mat                           = np.transpose(f_val_mat, (2, 1, 0))
        p_perm_k                            = np.zeros((M, N + 1, N_S))
        
        # --- GENETIC PRIOR INITIALIZATION ---
       
        if self.apoe_flag:
            apoe_dummy                      = sustainData.apoe # shape: (M, 3)
            # Pull active weights matrix from the class instance (populated during the EM loop)
            # shape: (N_S, 3)
            genetic_prior                   = apoe_dummy  @ genetic_weights.T #self.genetic_weights when we make it global param;
            genetic_prior                   = genetic_prior[:, np.newaxis, :] # shape: (M, 1, N_S)
        else:
            genetic_prior                   = 1.0 # Multiplicative identity factor if flag is off
        
        
        
        for s in range(N_S):
            p_perm_k[:, :, s]               = self._calculate_likelihood_stage(sustainData, S_opt[s])
        
        if self.apoe_flag: 
            p_perm_k_weighted                   = p_perm_k * f_val_mat * genetic_prior
        else:
            p_perm_k_weighted                   = p_perm_k * f_val_mat
            
        p_perm_k_norm                       = p_perm_k_weighted / np.sum(p_perm_k_weighted + 1e-250, axis=(1, 2), keepdims=True)
        f_opt                               = (np.squeeze(sum(sum(p_perm_k_norm))) / sum(sum(sum(p_perm_k_norm)))).reshape(N_S, 1, 1)
        f_val_mat                           = np.tile(f_opt, (1, N + 1, M))
        f_val_mat                           = np.transpose(f_val_mat, (2, 1, 0))
        order_seq                           = rng.permutation(N_S)  # this will produce different random numbers to Matlab

        for s in order_seq:
            order_bio                       = rng.permutation(N)  # this will produce different random numbers to Matlab
            for i in order_bio:
                current_sequence            = S_opt[s]
                current_location            = np.array([0] * len(current_sequence))
                current_location[current_sequence.astype(int)] = np.arange(len(current_sequence))

                selected_event              = i

                move_event_from             = current_location[selected_event]

                this_stage_zscore           = self.stage_zscore[0, selected_event]
                selected_biomarker          = self.stage_biomarker_index[0, selected_event]
                possible_zscores_biomarker  = self.stage_zscore[self.stage_biomarker_index == selected_biomarker]

                # slightly different conditional check to matlab version to protect python from calling min,max on an empty array
                min_filter                  = possible_zscores_biomarker < this_stage_zscore
                max_filter                  = possible_zscores_biomarker > this_stage_zscore
                events                      = np.array(range(N))
                if np.any(min_filter):
                    min_zscore_bound        = max(possible_zscores_biomarker[min_filter])
                    min_zscore_bound_event  = events[((self.stage_zscore[0] == min_zscore_bound).astype(int) + (self.stage_biomarker_index[0] == selected_biomarker).astype(int)) == 2]
                    move_event_to_lower_bound = current_location[min_zscore_bound_event] + 1
                else:
                    move_event_to_lower_bound = 0
                if np.any(max_filter):
                    max_zscore_bound        = min(possible_zscores_biomarker[max_filter])
                    max_zscore_bound_event  = events[((self.stage_zscore[0] == max_zscore_bound).astype(int) + (self.stage_biomarker_index[0] == selected_biomarker).astype(int)) == 2]
                    move_event_to_upper_bound = current_location[max_zscore_bound_event]
                else:
                    move_event_to_upper_bound = N
                    # FIXME: hack because python won't produce an array in range (N,N), while matlab will produce an array (N)... urgh
                if move_event_to_lower_bound == move_event_to_upper_bound:
                    possible_positions      = np.array([0])
                else:
                    possible_positions      = np.arange(move_event_to_lower_bound, move_event_to_upper_bound)
                possible_sequences          = np.zeros((len(possible_positions), N))
                possible_likelihood         = np.zeros((len(possible_positions), 1))
                possible_p_perm_k           = np.zeros((M, N + 1, len(possible_positions)))
                for index in range(len(possible_positions)):
                    current_sequence        = S_opt[s]

                    #choose a position in the sequence to move an event to
                    move_event_to           = possible_positions[index]

                    # move this event in its new position
                    current_sequence        = np.delete(current_sequence, move_event_from, 0)  # this is different to the Matlab version, which call current_sequence(move_event_from) = []
                    new_sequence            = np.concatenate([current_sequence[np.arange(move_event_to)], [selected_event], current_sequence[np.arange(move_event_to, N - 1)]])
                    possible_sequences[index, :] = new_sequence

                    possible_p_perm_k[:, :, index] = self._calculate_likelihood_stage(sustainData, new_sequence)

                    p_perm_k[:, :, s]       = possible_p_perm_k[:, :, index]
                    total_prob_stage        = np.sum(p_perm_k * f_val_mat, 2)
                    total_prob_subj         = np.sum(total_prob_stage, 1)
                    possible_likelihood[index] = np.sum(np.log(total_prob_subj + 1e-250))

                possible_likelihood         = possible_likelihood.reshape(possible_likelihood.shape[0])
                max_likelihood              = max(possible_likelihood)
                this_S                      = possible_sequences[possible_likelihood == max_likelihood, :]
                this_S                      = this_S[0, :]
                S_opt[s]                    = this_S
                this_p_perm_k               = possible_p_perm_k[:, :, possible_likelihood == max_likelihood]
                p_perm_k[:, :, s]           = this_p_perm_k[:, :, 0]

            S_opt[s]                        = this_S

        if self.apoe_flag:
            p_perm_k_weighted                   = p_perm_k * f_val_mat * genetic_prior
        else:
            p_perm_k_weighted                   = p_perm_k * f_val_mat
        
        #adding 1e-250 fixes divide by zero problem that happens rarely
        #p_perm_k_norm                       = p_perm_k_weighted / np.tile(np.sum(np.sum(p_perm_k_weighted, 1), 1).reshape(M, 1, 1), (1, N + 1, N_S))  # the second summation axis is different to Matlab version
        p_perm_k_norm                       = p_perm_k_weighted / np.sum(p_perm_k_weighted + 1e-250, axis=(1, 2), keepdims=True)

        f_opt                               = (np.squeeze(sum(sum(p_perm_k_norm))) / sum(sum(sum(p_perm_k_norm)))).reshape(N_S, 1, 1)
        f_val_mat                           = np.tile(f_opt, (1, N + 1, M))
        f_val_mat                           = np.transpose(f_val_mat, (2, 1, 0))
        f_opt                               = f_opt.reshape(N_S)

        if self.apoe_flag:
            total_prob_stage                    = np.sum(p_perm_k * f_val_mat* genetic_prior, 2)
        else:
            total_prob_stage                    = np.sum(p_perm_k * f_val_mat, 2)
        
        total_prob_subj                     = np.sum(total_prob_stage, 1)
        likelihood_opt                      = np.sum(np.log(total_prob_subj + 1e-250))
        
        return S_opt, f_opt, likelihood_opt
    
    def _optimise_genetic_parameters(self, sustainData, S, f, genetic_weights_init):
        """
        Core Behavior:
          - Optimizes Categorical Genetic Weights (W) ONLY.
          - Keeps Biomarker Sequences (S) and Cohort Fractions (f) COMPLETELY FIXED.
        
        Used By:
          - Step 2 of the old 'alternating' EM optimization method.
        Mathematical Framework:
          Computes a localized E-step to get patient cluster responsibilities, 
          then executes a closed-form Lagrangian optimization step to maximize 
          the categorical matrix weights under the row-sum constraint (sum(W) = 1).
        
        """
        apoe_dummy = sustainData.apoe
        N_S = S.shape[0]

        # 1. Run the standard likelihood call internally to get p_perm_k
        _, _, _, _, p_perm_k = self._calculate_likelihood(sustainData, S, f,genetic_weights_init)
        N_stages = p_perm_k.shape[1]
        M = sustainData.getNumSamples()

        # 2. Build the 3D fraction matrix for normalization
        f_opt_mat = np.array(f).reshape(N_S, 1, 1)
        f_val_mat = np.tile(f_opt_mat, (1, N_stages, M))
        f_val_mat = np.transpose(f_val_mat, (2, 1, 0)) 
        
        # 3. Compute responsibilities using the CURRENT stable weights
        genetic_prior_3d = (apoe_dummy @ genetic_weights_init.T)[:, np.newaxis, :]
        p_perm_k_weighted = p_perm_k * f_val_mat * genetic_prior_3d
        p_perm_k_norm = p_perm_k_weighted / np.sum(p_perm_k_weighted + 1e-250, axis=(1, 2), keepdims=True)

        # 4. M-Step: Compute the candidate genetic weights
        gamma = np.sum(p_perm_k_norm, axis=1)                               
        weights_numerator = gamma.T @ apoe_dummy                            
        weights_denominator = np.sum(gamma, axis=0, keepdims=True).T        
        genetic_weights_opt = weights_numerator / (weights_denominator + 1e-12)
        
        # 5. Calculate the new likelihood using these candidate weights
        updated_prior_3d = (apoe_dummy @ genetic_weights_opt.T)[:, np.newaxis, :]
        total_prob_stage = np.sum(p_perm_k * f_val_mat * updated_prior_3d, axis=2)
        total_prob_subj  = np.sum(total_prob_stage, axis=1)
        genetic_likelihood_opt = np.sum(np.log(total_prob_subj + 1e-250))
        
        return genetic_weights_opt, genetic_likelihood_opt
    
    def _optimise_parameters_combined(self, sustainData, S_init, f_init, rng, genetic_weights_init):
        """
        Core Behavior:
          - Optimizes Sequences (S), Fractions (f), AND Genetic Weights (W)
            SIMULTANEOUSLY within a single execution block.
        
        Used By:
          - The true 1-Step Simultaneous EM method ('combined').
        
        Mathematical Framework:
          1. E-Step: Freezes a single, unified patient responsibility mass (gamma)
                     using incoming sequence and genetic inputs.
          2. M-Step: Instantly updates BOTH fractions (f) and genetic weights (W) 
                     analytically from that same frozen gamma snapshot.
          3. Shuffler: Runs the greedy biomarker event-swapping loops immediately 
                       after to align the sequence to the new parameters.
        
        """
        M     = sustainData.getNumSamples()
        N_S   = S_init.shape[0]
        N     = self.stage_zscore.shape[1]

        S_opt = S_init.copy()
        
        # ---------------------------------------------------------------------
        # 1. INITIAL EXPECTATION STEP (Calculate Joint Likelihood & Gamma)
        # ---------------------------------------------------------------------
        f_opt     = np.array(f_init).reshape(N_S, 1, 1)
        f_val_mat = np.tile(f_opt, (1, N + 1, M))
        f_val_mat = np.transpose(f_val_mat, (2, 1, 0))
        p_perm_k  = np.zeros((M, N + 1, N_S))

        for s in range(N_S):
            p_perm_k[:, :, s] = self._calculate_likelihood_stage(sustainData, S_opt[s])

        # Apply categorical genetic prior tensor
        apoe_dummy        = sustainData.apoe
        genetic_prior     = (apoe_dummy @ genetic_weights_init.T)[:, np.newaxis, :]
        p_perm_k_weighted = p_perm_k * f_val_mat * genetic_prior
        
        # This is your frozen E-Step responsibility matrix snapshot
        p_perm_k_norm     = p_perm_k_weighted / np.sum(p_perm_k_weighted + 1e-250, axis=(1, 2), keepdims=True)
        gamma             = np.sum(p_perm_k_norm, axis=1) # Shape: (M, N_S)

        # ---------------------------------------------------------------------
        # 2. THE SIMULTANEOUS M-STEP (Analytical Updates for f and W)
        # ---------------------------------------------------------------------
        # Update cohort fractions (f) cleanly from gamma mass
        f_opt     = (np.squeeze(sum(sum(p_perm_k_norm))) / sum(sum(sum(p_perm_k_norm)))).reshape(N_S, 1, 1)
        f_val_mat = np.tile(f_opt, (1, N + 1, M))
        f_val_mat = np.transpose(f_val_mat, (2, 1, 0))
        
        # Update genetic weights (W) analytical matrix using the EXACT same gamma mass snapshot
        weights_numerator   = gamma.T @ apoe_dummy                                    
        weights_denominator = np.sum(gamma, axis=0, keepdims=True).T                
        genetic_weights_opt = weights_numerator / (weights_denominator + 1e-12)
        
        # Numerical safeguard to prevent downstream log-likelihood crashes
        genetic_weights_opt = np.clip(genetic_weights_opt, 1e-5, 1.0)
        genetic_weights_opt /= np.sum(genetic_weights_opt, axis=1, keepdims=True)
        
        # Refresh the active 3D genetic prior using optimized weights
        genetic_prior = (apoe_dummy @ genetic_weights_opt.T)[:, np.newaxis, :]

        # ---------------------------------------------------------------------
        # 3. COMBINATORIAL SEQUENCE SHUFFLER STEP (Greedy Search)
        # ---------------------------------------------------------------------
        order_seq = rng.permutation(N_S)
        for s in order_seq:
            order_bio = rng.permutation(N)
            for i in order_bio:
                current_sequence                               = S_opt[s]
                current_location                               = np.array([0] * len(current_sequence))
                current_location[current_sequence.astype(int)] = np.arange(len(current_sequence))

                selected_event   = i
                move_event_from  = current_location[selected_event]

                this_stage_zscore           = self.stage_zscore[0, selected_event]
                selected_biomarker          = self.stage_biomarker_index[0, selected_event]
                possible_zscores_biomarker  = self.stage_zscore[self.stage_biomarker_index == selected_biomarker]

                min_filter = possible_zscores_biomarker < this_stage_zscore
                max_filter = possible_zscores_biomarker > this_stage_zscore
                events     = np.array(range(N))
                
                if np.any(min_filter):
                    min_zscore_bound          = max(possible_zscores_biomarker[min_filter])
                    min_zscore_bound_event    = events[((self.stage_zscore[0] == min_zscore_bound).astype(int) + (self.stage_biomarker_index[0] == selected_biomarker).astype(int)) == 2]
                    move_event_to_lower_bound = current_location[min_zscore_bound_event] + 1
                else:
                    move_event_to_lower_bound = 0
                    
                if np.any(max_filter):
                    max_zscore_bound          = min(possible_zscores_biomarker[max_filter])
                    max_zscore_bound_event    = events[((self.stage_zscore[0] == max_zscore_bound).astype(int) + (self.stage_biomarker_index[0] == selected_biomarker).astype(int)) == 2]
                    move_event_to_upper_bound = current_location[max_zscore_bound_event]
                else:
                    move_event_to_upper_bound = N
                    
                if move_event_to_lower_bound == move_event_to_upper_bound:
                    possible_positions = np.array([0])
                else:
                    possible_positions = np.arange(move_event_to_lower_bound, move_event_to_upper_bound)
                    
                possible_sequences  = np.zeros((len(possible_positions), N))
                possible_likelihood = np.zeros((len(possible_positions), 1))
                possible_p_perm_k   = np.zeros((M, N + 1, len(possible_positions)))
                
                for index in range(len(possible_positions)):
                    current_sequence = S_opt[s]
                    move_event_to    = possible_positions[index]

                    current_sequence = np.delete(current_sequence, move_event_from, 0)
                    new_sequence     = np.concatenate([current_sequence[np.arange(move_event_to)], [selected_event], current_sequence[np.arange(move_event_to, N - 1)]])
                    possible_sequences[index, :] = new_sequence

                    possible_p_perm_k[:, :, index] = self._calculate_likelihood_stage(sustainData, new_sequence)

                    p_perm_k[:, :, s] = possible_p_perm_k[:, :, index]
                    
                    # Compute inner loop likelihood with the updated f and W parameters
                    total_prob_stage           = np.sum(p_perm_k * f_val_mat * genetic_prior, 2)
                    total_prob_subj            = np.sum(total_prob_stage, 1)
                    possible_likelihood[index] = np.sum(np.log(total_prob_subj + 1e-250))

                possible_likelihood = possible_likelihood.reshape(possible_likelihood.shape[0])
                max_likelihood      = max(possible_likelihood)
                this_S              = possible_sequences[possible_likelihood == max_likelihood, :]
                this_S              = this_S[0, :]
                S_opt[s]            = this_S
                this_p_perm_k       = possible_p_perm_k[:, :, possible_likelihood == max_likelihood]
                p_perm_k[:, :, s]   = this_p_perm_k[:, :, 0]

            S_opt[s] = this_S

        # ---------------------------------------------------------------------
        # 4. FINAL ALIGNMENT M-STEP (Re-align f one last time to the shuffled S)
        # ---------------------------------------------------------------------
        p_perm_k_weighted = p_perm_k * f_val_mat * genetic_prior
        p_perm_k_norm     = p_perm_k_weighted / np.sum(p_perm_k_weighted + 1e-250, axis=(1, 2), keepdims=True)

        f_opt = (np.squeeze(sum(sum(p_perm_k_norm))) / sum(sum(sum(p_perm_k_norm)))).reshape(N_S, 1, 1)
        f_val_mat = np.tile(f_opt, (1, N + 1, M))
        f_val_mat = np.transpose(f_val_mat, (2, 1, 0))
        f_opt = f_opt.reshape(N_S)
        
        # I dont know if i should update genetic weights one last time here or not

        total_prob_stage = np.sum(p_perm_k * f_val_mat * genetic_prior, 2)
        total_prob_subj  = np.sum(total_prob_stage, 1)
        likelihood_opt   = np.sum(np.log(total_prob_subj + 1e-250))

        return S_opt, f_opt, likelihood_opt, genetic_weights_opt
        

    def _perform_mcmc(self, sustainData, seq_init, f_init, n_iterations, seq_sigma, f_sigma, genetic_weights_init=None, genetics_sigma=None):
        # Take MCMC samples of the uncertainty in the SuStaIn model parameters

        N                                   = self.stage_zscore.shape[1]
        N_S                                 = seq_init.shape[0]

        if isinstance(f_sigma, float):  # FIXME: hack to enable multiplication
            f_sigma                         = np.array([f_sigma])

        samples_sequence                    = np.zeros((N_S, N, n_iterations))
        samples_f                           = np.zeros((N_S, n_iterations))
        samples_likelihood                  = np.zeros((n_iterations, 1))
        
        if self.apoe_flag:    
            samples_genetic_weights             = np.zeros((N_S, self.N_genetic_categories,n_iterations))
            samples_genetic_weights[:,:,0]      = genetic_weights_init
        
        samples_sequence[:, :, 0]           = seq_init  # don't need to copy as we don't write to 0 index
        samples_f[:, 0]                     = f_init

        # Reduce frequency of tqdm update to 0.1% of total for larger iteration numbers
        tqdm_update_iters = int(n_iterations/1000) if n_iterations > 100000 else None 

        for i in tqdm(range(n_iterations), "MCMC Iteration", n_iterations, miniters=tqdm_update_iters):
            if i > 0:
                seq_order                   = self.global_rng.permutation(N_S)  # this function returns different random numbers to Matlab
                for s in seq_order:
                    move_event_from         = int(np.ceil(N * self.global_rng.random())) - 1
                    current_sequence        = samples_sequence[s, :, i - 1]

                    current_location        = np.array([0] * N)
                    current_location[current_sequence.astype(int)] = np.arange(N)

                    selected_event          = int(current_sequence[move_event_from])
                    this_stage_zscore       = self.stage_zscore[0, selected_event]
                    selected_biomarker      = self.stage_biomarker_index[0, selected_event]
                    possible_zscores_biomarker = self.stage_zscore[self.stage_biomarker_index == selected_biomarker]

                    # slightly different conditional check to matlab version to protect python from calling min,max on an empty array
                    min_filter              = possible_zscores_biomarker < this_stage_zscore
                    max_filter              = possible_zscores_biomarker > this_stage_zscore
                    events                  = np.array(range(N))
                    if np.any(min_filter):
                        min_zscore_bound            = max(possible_zscores_biomarker[min_filter])
                        min_zscore_bound_event      = events[((self.stage_zscore[0] == min_zscore_bound).astype(int) + (self.stage_biomarker_index[0] == selected_biomarker).astype(int)) == 2]
                        move_event_to_lower_bound   = current_location[min_zscore_bound_event] + 1
                    else:
                        move_event_to_lower_bound   = 0

                    if np.any(max_filter):
                        max_zscore_bound            = min(possible_zscores_biomarker[max_filter])
                        max_zscore_bound_event      = events[((self.stage_zscore[0] == max_zscore_bound).astype(int) + (self.stage_biomarker_index[0] == selected_biomarker).astype(int)) == 2]
                        move_event_to_upper_bound   = current_location[max_zscore_bound_event]
                    else:
                        move_event_to_upper_bound   = N

                    # FIXME: hack because python won't produce an array in range (N,N), while matlab will produce an array (N)... urgh
                    if move_event_to_lower_bound == move_event_to_upper_bound:
                        possible_positions          = np.array([0])
                    else:
                        possible_positions          = np.arange(move_event_to_lower_bound, move_event_to_upper_bound)

                    distance                = possible_positions - move_event_from

                    if isinstance(seq_sigma, int):  # FIXME: change to float
                        this_seq_sigma      = seq_sigma
                    else:
                        this_seq_sigma      = seq_sigma[s, selected_event]

                    # use own normal PDF because stats.norm is slow
                    weight                  = AbstractSustain.calc_coeff(this_seq_sigma) * AbstractSustain.calc_exp(distance, 0., this_seq_sigma)
                    weight                  /= np.sum(weight)
                    index                   = self.global_rng.choice(range(len(possible_positions)), 1, replace=True, p=weight)  # FIXME: difficult to check this because random.choice is different to Matlab randsample

                    move_event_to           = possible_positions[index]

                    current_sequence        = np.delete(current_sequence, move_event_from, 0)
                    new_sequence            = np.concatenate([current_sequence[np.arange(move_event_to)], [selected_event], current_sequence[np.arange(move_event_to, N - 1)]])
                    samples_sequence[s, :, i] = new_sequence

                new_f                       = samples_f[:, i - 1] + f_sigma * self.global_rng.standard_normal()
                new_f                       = (np.fabs(new_f) / np.sum(np.fabs(new_f)))
                samples_f[:, i]             = new_f
                
                # check if we need to change how the perturbations are made and are they still normalised
                if self.apoe_flag:
                    new_weights                           = samples_genetic_weights[:,:,i-1] + genetics_sigma * self.global_rng.standard_normal(size=(N_S, self.N_genetic_categories))
                    new_weights                           = (np.fabs(new_weights) / np.sum(np.fabs(new_weights), axis=1, keepdims=True)) # on what axis
                    samples_genetic_weights[:, :, i]       = new_weights 
                    
            S                               = samples_sequence[:, :, i]
            f                               = samples_f[:, i]
            if self.apoe_flag:
                genetic_weights             = samples_genetic_weights[:, :, i]
                #print('Genetic weights of index i', genetic_weights)
            
            if self.apoe_flag:
                likelihood_sample, _, _, _, _   = self._calculate_likelihood(sustainData, S, f,genetic_weights)
            else:
                likelihood_sample, _, _, _, _   = self._calculate_likelihood(sustainData, S, f)
            samples_likelihood[i]           = likelihood_sample
            
            # this is the block updates (metropolis hastings)
            # in case it doesnt show good acceptance rates, try splitting it: Gibbs withn Hastings
            if i > 0:
                ratio                           = np.exp(samples_likelihood[i] - samples_likelihood[i - 1])
                if ratio < self.global_rng.random():
                    samples_likelihood[i]       = samples_likelihood[i - 1]
                    samples_sequence[:, :, i]   = samples_sequence[:, :, i - 1]
                    samples_f[:, i]             = samples_f[:, i - 1]
                    
                    if self.apoe_flag:
                        samples_genetic_weights[:, :, i] = samples_genetic_weights[:, :, i-1]

        perm_index                          = np.where(samples_likelihood == max(samples_likelihood))
        perm_index                          = perm_index[0]
        ml_likelihood                       = max(samples_likelihood)
        ml_sequence                         = samples_sequence[:, :, perm_index]
        ml_f                                = samples_f[:, perm_index]
        if self.apoe_flag:
            ml_genetic_weights              = samples_genetic_weights[:,:, perm_index]
            return ml_sequence, ml_f, ml_likelihood, samples_sequence, samples_f, samples_likelihood, ml_genetic_weights, samples_genetic_weights

        return ml_sequence, ml_f, ml_likelihood, samples_sequence, samples_f, samples_likelihood

    def _plot_sustain_model(self, *args, **kwargs):
        return ZscoreSustain_APOE4.plot_positional_var(*args, Z_vals=self.Z_vals, **kwargs)

    def subtype_and_stage_individuals_newData(self, data_new, samples_sequence, samples_f, N_samples):

        numStages_new                   = self.__sustainData.getNumStages() #data_new.shape[1]
        sustainData_newData             = ZScoreSustainData(data_new, numStages_new)

        ml_subtype,         \
        prob_ml_subtype,    \
        ml_stage,           \
        prob_ml_stage,      \
        prob_subtype,       \
        prob_stage,         \
        prob_subtype_stage          = self.subtype_and_stage_individuals(sustainData_newData, samples_sequence, samples_f, N_samples)

        return ml_subtype, prob_ml_subtype, ml_stage, prob_ml_stage, prob_subtype, prob_stage, prob_subtype_stage

    # ********************* STATIC METHODS
    @staticmethod
    def linspace_local2(a, b, N, arange_N):
        return a + (b - a) / (N - 1.) * arange_N

    @staticmethod
    def plot_positional_var(samples_sequence, samples_f, n_samples, Z_vals, biomarker_labels=None, ml_f_EM=None, cval=False, subtype_order=None, biomarker_order=None, title_font_size=12, stage_font_size=10, stage_label='SuStaIn Stage', stage_rot=0, stage_interval=1, label_font_size=10, label_rot=0, cmap="original", biomarker_colours=None, figsize=None, subtype_titles=None, separate_subtypes=False, save_path=None, save_kwargs={}):
        # Get the number of subtypes
        N_S = samples_sequence.shape[0]
        # Get the number of features/biomarkers
        N_bio = Z_vals.shape[0]
        # Check that the number of labels given match
        if biomarker_labels is not None:
            assert len(biomarker_labels) == N_bio
        # Set subtype order if not given
        if subtype_order is None:
            # Determine order if info given
            if ml_f_EM is not None:
                subtype_order = np.argsort(ml_f_EM)[::-1]
            # Otherwise determine order from samples_f
            else:
                subtype_order = np.argsort(np.mean(samples_f, 1))[::-1]
        elif isinstance(subtype_order, tuple):
            subtype_order = list(subtype_order)
        # Unravel the stage zscores from Z_vals
        stage_zscore = Z_vals.T.flatten()
        IX_select = np.nonzero(stage_zscore)[0]
        stage_zscore = stage_zscore[IX_select][None, :]
        # Get the z-scores and their number
        zvalues = np.unique(stage_zscore)
        N_z = len(zvalues)
        # Extract which biomarkers have which zscores/stages
        stage_biomarker_index = np.tile(np.arange(N_bio), (N_z,))
        stage_biomarker_index = stage_biomarker_index[IX_select]
        # Warn user of reordering if labels and order given
        if biomarker_labels is not None and biomarker_order is not None:
            warnings.warn(
                "Both labels and an order have been given. The labels will be reordered according to the given order!"
            )
        if biomarker_order is not None:
            # self._plot_biomarker_order is not suited to zscore version
            # Ignore for compatability, for now
            # One option is to reshape, sum position, and lowest->highest determines order
            if len(biomarker_order) > N_bio:
                biomarker_order = np.arange(N_bio)
        # Otherwise use default order
        else:
            biomarker_order = np.arange(N_bio)
        # If no labels given, set dummy defaults
        if biomarker_labels is None:
            biomarker_labels = [f"Biomarker {i}" for i in range(N_bio)]
        # Otherwise reorder according to given order (or not if not given)
        else:
            biomarker_labels = [biomarker_labels[i] for i in biomarker_order]
        # Check number of subtype titles is correct if given
        if subtype_titles is not None:
            assert len(subtype_titles) == N_S
        # Z-score colour definition
        if cmap == "original":
            # Hard-coded colours: hooray!
            colour_mat = np.array([[1, 0, 0], [1, 0, 1], [0, 0, 1], [0.5, 0, 1], [0, 1, 1], [0, 1, 0.5]])[:N_z]
            # We only have up to 5 default colours, so double-check
            if colour_mat.shape[0] > N_z:
                raise ValueError(f"Colours are only defined for {len(colour_mat)} z-scores!")
        else:
            raise NotImplementedError
        '''
        Note for future self/others: The use of any arbitrary colourmap is problematic, as when the same stage can have the same biomarker with different z-scores of different certainties, the colours need to mix in a visually informative way and there can be issues with RGB mixing/interpolation, particulary if there are >2 z-scores for the same biomarker at the same stage. It may be possible, but the end result may no longer be useful to look at.
        '''

        # Check biomarker label colours
        # If custom biomarker text colours are given
        if biomarker_colours is not None:
            biomarker_colours = AbstractSustain.check_biomarker_colours(
            biomarker_colours, biomarker_labels
        )
        # Default case of all-black colours
        # Unnecessary, but skips a check later
        else:
            biomarker_colours = {i:"black" for i in biomarker_labels}

        # Flag to plot subtypes separately
        if separate_subtypes:
            nrows, ncols = 1, 1
        else:
            # Determine number of rows and columns (rounded up)
            if N_S == 1:
                nrows, ncols = 1, 1
            elif N_S < 3:
                nrows, ncols = 1, N_S
            elif N_S < 7:
                nrows, ncols = 2, int(np.ceil(N_S / 2))
            else:
                nrows, ncols = 3, int(np.ceil(N_S / 3))
        # Total axes used to loop over
        total_axes = nrows * ncols
        # Create list of single figure object if not separated
        if separate_subtypes:
            subtype_loops = N_S
        else:
            subtype_loops = 1
        # Container for all figure objects
        figs = []
        # Loop over figures (only makes a diff if separate_subtypes=True)
        for i in range(subtype_loops):
            # Create the figure and axis for this subtype loop
            fig, axs = plt.subplots(nrows, ncols, figsize=figsize)
            figs.append(fig)
            # Loop over each axis
            for j in range(total_axes):
                # Normal functionality (all subtypes on one plot)
                if not separate_subtypes:
                    i = j
                # Handle case of a single array
                if isinstance(axs, np.ndarray):
                    ax = axs.flat[i]
                else:
                    ax = axs
                # Check if i is superfluous
                if i not in range(N_S):
                    ax.set_axis_off()
                    continue

                this_samples_sequence = samples_sequence[subtype_order[i],:,:].T
                N = this_samples_sequence.shape[1]

                # Construct confusion matrix (vectorized)
                # We compare `this_samples_sequence` against each position
                # Sum each time it was observed at that point in the sequence
                # And normalize for number of samples/sequences
                confus_matrix = (this_samples_sequence==np.arange(N)[:, None, None]).sum(1) / this_samples_sequence.shape[0]

                # Define the confusion matrix to insert the colours
                # Use 1s to start with all white
                confus_matrix_c = np.ones((N_bio, N, 3))

                # Loop over each z-score event
                for j, z in enumerate(zvalues):
                    # Determine which colours to alter
                    # I.e. red (1,0,0) means removing green & blue channels
                    # according to the certainty of red (representing z-score 1)
                    alter_level = colour_mat[j] == 0
                    # Extract the uncertainties for this z-score
                    confus_matrix_zscore = confus_matrix[(stage_zscore==z)[0]]
                    # Subtract the certainty for this colour
                    confus_matrix_c[
                        np.ix_(
                            stage_biomarker_index[(stage_zscore==z)[0]], range(N),
                            alter_level
                        )
                    ] -= np.tile(
                        confus_matrix_zscore.reshape((stage_zscore==z).sum(), N, 1),
                        (1, 1, alter_level.sum())
                    )
                if subtype_titles is not None:
                    title_i = subtype_titles[i]
                else:
                    # Add axis title
                    if cval == False:
                        temp_mean_f = np.mean(samples_f, 1)
                        # Shuffle vals according to subtype_order
                        # This defaults to previous method if custom order not given
                        vals = temp_mean_f[subtype_order]

                        if n_samples != np.inf:
                            title_i = f"Subtype {i+1} (f={vals[i]:.2f}, n={np.round(vals[i] * n_samples):n})"
                        else:
                            title_i = f"Subtype {i+1} (f={vals[i]:.2f})"
                    else:
                        title_i = f"Subtype {i+1} cross-validated"
                # Plot the colourized matrix
                ax.imshow(
                    confus_matrix_c[biomarker_order, :, :],
                    interpolation='nearest'
                )
                # Add the xticks and labels
                stage_ticks = np.arange(0, N, stage_interval)
                ax.set_xticks(stage_ticks)
                ax.set_xticklabels(stage_ticks+1, fontsize=stage_font_size, rotation=stage_rot)
                # Add the yticks and labels
                ax.set_yticks(np.arange(N_bio))
                # Add biomarker labels to LHS of every row only
                if (i % ncols) == 0:
                    ax.set_yticklabels(biomarker_labels, ha='right', fontsize=label_font_size, rotation=label_rot)
                    # Set biomarker label colours
                    for tick_label in ax.get_yticklabels():
                        tick_label.set_color(biomarker_colours[tick_label.get_text()])
                else:
                    ax.set_yticklabels([])
                # Make the event label slightly bigger than the ticks
                ax.set_xlabel(stage_label, fontsize=stage_font_size+2)
                ax.set_title(title_i, fontsize=title_font_size)
            # Tighten up the figure
            fig.tight_layout()
            # Save if a path is given
            if save_path is not None:
                # Modify path for specific subtype if specified
                # Don't modify save_path!
                if separate_subtypes:
                    save_name = f"{save_path}_subtype{i}"
                else:
                    save_name = f"{save_path}_all-subtypes"
                # Handle file format, avoids issue with . in filenames
                if "format" in save_kwargs:
                    file_format = save_kwargs.pop("format")
                # Default to png
                else:
                    file_format = "png"
                # Save the figure, with additional kwargs
                fig.savefig(
                    f"{save_name}.{file_format}",
                    **save_kwargs
                )
        return figs, axs

    # ********************* TEST METHODS
    @classmethod
    def test_sustain(cls, n_biomarkers, n_samples, n_subtypes, 
    ground_truth_subtypes, sustain_kwargs, seed=42):
        # Set a global seed to propagate
        np.random.seed(seed)
        # Create Z values
        Z_vals = np.tile(np.arange(1, 4), (n_biomarkers, 1))
        Z_vals[0, 2] = 0

        Z_max = np.full((n_biomarkers,), 5)
        Z_max[2] = 2

        ground_truth_sequences = cls.generate_random_model(Z_vals, n_subtypes)
        N_stages = np.sum(Z_vals > 0) + 1

        ground_truth_stages_control = np.zeros((int(np.round(n_samples * 0.25)), 1))
        ground_truth_stages_other = np.random.randint(1, N_stages+1, (int(np.round(n_samples * 0.75)), 1))
        ground_truth_stages = np.vstack((ground_truth_stages_control, ground_truth_stages_other)).astype(int)

        data, data_denoised, stage_value = cls.generate_data(
            ground_truth_subtypes,
            ground_truth_stages,
            ground_truth_sequences,
            Z_vals,
            Z_max
        )

        return cls(
            data, Z_vals, Z_max,
            **sustain_kwargs
        )

    @staticmethod
    def generate_random_model(Z_vals, N_S, seed=None):
        num_biomarkers = Z_vals.shape[0]

        stage_zscore = Z_vals.T.flatten()#[np.newaxis, :]

        IX_select = np.nonzero(stage_zscore)[0]
        stage_zscore = stage_zscore[IX_select]#[np.newaxis, :]
        num_zscores = Z_vals.shape[0]

        stage_biomarker_index = np.tile(np.arange(num_biomarkers), (num_zscores,))
        stage_biomarker_index = stage_biomarker_index[IX_select]#[np.newaxis, :]

        N = stage_zscore.shape[0]
        S = np.zeros((N_S, N))
        # Moved outside loop, no need
        possible_biomarkers = np.unique(stage_biomarker_index)

        for s in range(N_S):
            for i in range(N):

                IS_min_stage_zscore = np.full(N, False)
    
                for j in possible_biomarkers:
                    IS_unselected = np.full(N, False)
                    # I have no idea what purpose this serves, so leaving for now
                    for k in set(range(N)) - set(S[s][:i]):
                        IS_unselected[k] = True

                    this_biomarkers = np.logical_and(
                        stage_biomarker_index == possible_biomarkers[j],
                        np.array(IS_unselected) == 1
                    )
                    if not np.any(this_biomarkers):
                        this_min_stage_zscore = 0
                    else:
                        this_min_stage_zscore = np.min(stage_zscore[this_biomarkers])
                    
                    if this_min_stage_zscore:
                        IS_min_stage_zscore[np.logical_and(
                            this_biomarkers,
                            stage_zscore == this_min_stage_zscore
                        )] = True

                events = np.arange(N)
                possible_events = events[IS_min_stage_zscore]
                this_index = np.ceil(np.random.rand() * len(possible_events)) - 1
                
                S[s][i] = possible_events[int(this_index)]
        return S

    # TODO: Refactor this as above
    @staticmethod
    def generate_data(subtypes, stages, gt_ordering, Z_vals, Z_max):
        B = Z_vals.shape[0]
        stage_zscore = np.array([y for x in Z_vals.T for y in x])
        stage_zscore = stage_zscore.reshape(1,len(stage_zscore))
        IX_select = stage_zscore>0
        stage_zscore = stage_zscore[IX_select]
        stage_zscore = stage_zscore.reshape(1,len(stage_zscore))

        num_zscores = Z_vals.shape[1]
        IX_vals = np.array([[x for x in range(B)]] * num_zscores).T
        stage_biomarker_index = np.array([y for x in IX_vals.T for y in x])
        stage_biomarker_index = stage_biomarker_index.reshape(1,len(stage_biomarker_index))
        stage_biomarker_index = stage_biomarker_index[IX_select]
        stage_biomarker_index = stage_biomarker_index.reshape(1,len(stage_biomarker_index))

        min_biomarker_zscore = [0]*B
        max_biomarker_zscore = Z_max
        std_biomarker_zscore = [1]*B

        N = stage_biomarker_index.shape[1]
        N_S = gt_ordering.shape[0]

        possible_biomarkers = np.unique(stage_biomarker_index)
        stage_value = np.zeros((B,N+2,N_S))

        for s in range(N_S):
            S = gt_ordering[s,:]
            S_inv = np.array([0]*N)
            S_inv[S.astype(int)] = np.arange(N)
            for i in range(B):
                b = possible_biomarkers[i]
                event_location = np.concatenate([[0], S_inv[(stage_biomarker_index == b)[0]], [N]])

                event_value = np.concatenate([[min_biomarker_zscore[i]], stage_zscore[stage_biomarker_index == b], [max_biomarker_zscore[i]]])

                for j in range(len(event_location)-1):

                    if j == 0: # FIXME: nasty hack to get Matlab indexing to match up - necessary here because indices are used for linspace limits
                        index = np.arange(event_location[j],event_location[j+1]+2)
                        stage_value[i,index,s] = np.linspace(event_value[j],event_value[j+1],event_location[j+1]-event_location[j]+2)
                    else:
                        index = np.arange(event_location[j] + 1, event_location[j + 1] + 2)
                        stage_value[i,index,s] = np.linspace(event_value[j],event_value[j+1],event_location[j+1]-event_location[j]+1)

        M = stages.shape[0]
        data_denoised = np.zeros((M,B))
        for m in range(M):
            data_denoised[m,:] = stage_value[:,int(stages[m]),subtypes[m]]
        data = data_denoised + norm.ppf(np.random.rand(B,M).T)*np.tile(std_biomarker_zscore,(M,1))

        return data, data_denoised, stage_value
    