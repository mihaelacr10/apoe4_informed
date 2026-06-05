#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed June 3rd 9:40 2026

Experiment 1: Algorithmic Diagnostics
Objective: Compare Combined vs. Alternating EM initialization methods 
           by tracking and plotting their log-likelihood paths.

@author: mihaelacroitor
"""

import numpy as np
import matplotlib.pyplot as plt
from genetics_simulation_utils import simulate_apoe_sustain_dataset
from pathlib import Path
import os 
import shutil
import pandas as pd

# TODO: Import your actual model class or module here
# from your_apoe_sustain_module import ApoeInformedSuStaIn 

from apoe4_sustain import ZscoreSustain_APOE4


def run_diagnostic_benchmark(num_seeds=5):
    print(f"🔬 Running EM Diagnostics across {num_seeds} random seeds...")
    
    # Dictionaries to store step-by-step optimization paths for plotting
    # Master list to collect tabular summary rows across all seeds
    all_seed_metrics = []
    
    N_biomarkers = 5
    N_S_max = 3 # 2
    #N_startpoints = 15
    N_startpoints = 15
    N_iterations_MCMC = int(1e3) #int(1e4)
    BiomarkerNames = [f'Biomarker {i}' for i in range(N_biomarkers)]
    
    for seed_idx in range(num_seeds):
        print("\n=========================================")
        print(f" PROCESSING DATASET SEED: {seed_idx}")
        print("=========================================")
        
        # Base directory isolated specifically for this experiment
        base_sim_path = Path("experiments") / "01_algorithmic_diagnostics" 
        
        # Unique folder per dataset fold inside the diagnostics directory
        dataset_name = f"diagnostic_seed_{seed_idx}"
        dataset_folder = base_sim_path / dataset_name
        
        # 1. Generate unique dataset for this specific fold
        df, Z_vals, Z_max, gt_sequence, gt_f, W_true = simulate_apoe_sustain_dataset(
            N_S_gt=N_S_max, 
            N=N_biomarkers,
            genetic_signal_strength='moderate', 
            seed=seed_idx,
            save=True,
            output_path=base_sim_path,   
            dataset_name=dataset_name,    
            base_filename="raw_data"      
        )
        
        X_data = df[BiomarkerNames].values
        y_genetics = df['apoe_status'].values
        
        # 2. Define distinct model output paths nested inside that specific seed fold
        combined_output_folder   = dataset_folder / "combined"
        alternating_output_folder = dataset_folder / "alternating"
        
        # -----------------------------------------------------------------
        # RUN METHOD A: COMBINED UNIFIED EM LOOP
        # -----------------------------------------------------------------
        print("    👉 Running Combined EM configurations...")
        if combined_output_folder.exists():
            shutil.rmtree(combined_output_folder)
        combined_output_folder.mkdir(parents=True, exist_ok=True)
        
        sustain_input_combined = ZscoreSustain_APOE4(
            X_data, Z_vals, Z_max, BiomarkerNames, N_startpoints, N_S_max, 
            N_iterations_MCMC, combined_output_folder, dataset_name, False,
            apoe4_status=y_genetics, apoe_flag=True, em_loop_type='combined'
        )
        sustain_input_combined.run_sustain_algorithm()
        
        # -----------------------------------------------------------------
        # RUN METHOD B: ALTERNATING ITERATIVE EM LOOPS
        # -----------------------------------------------------------------
        print("    👉 Running Alternating EM configurations...")
        if alternating_output_folder.exists():
            shutil.rmtree(alternating_output_folder)
        alternating_output_folder.mkdir(parents=True, exist_ok=True)
        
        sustain_input_alternating = ZscoreSustain_APOE4(
            X_data, Z_vals, Z_max, BiomarkerNames, N_startpoints, N_S_max, 
            N_iterations_MCMC, alternating_output_folder, dataset_name, False,
            apoe4_status=y_genetics, apoe_flag=True, em_loop_type='alternating'
        )
        sustain_input_alternating.run_sustain_algorithm()
        
        # -------------------------------------------------------------------------
        # CALCULATE GROUND-TRUTH MAXIMUM LIKELIHOOD
        # -------------------------------------------------------------------------
        # Extract the hidden data structure using the private class handle
        sustainData = sustain_input_combined._AbstractSustain__sustainData
        
        # Compute exact true ceiling value
        true_max_lik, _, _, _, _ = sustain_input_combined._calculate_likelihood(
            sustainData, gt_sequence, gt_f, W_true
        )
        print(f" Ground-Truth True Max Likelihood calculated: {true_max_lik:.2f}")
        
        # -------------------------------------------------------------------------
        # LOAD EM LIK HISTORIES FROM THE SAVED SUBTYPE PICKLE FILES
        # -------------------------------------------------------------------------
        s = N_S_max - 1  # Target final subtype output stage index
        
        combined_pickle_file = combined_output_folder / 'pickle_files' / f"{dataset_name}_subtype{s}.pickle"
        alternating_pickle_file = alternating_output_folder / 'pickle_files' / f"{dataset_name}_subtype{s}.pickle"
        
        pk_combined = pd.read_pickle(combined_pickle_file)
        combined_histories = pk_combined["em_likelihood_histories"]
        
        pk_alternating = pd.read_pickle(alternating_pickle_file)
        alternating_histories = pk_alternating["em_likelihood_histories"]
        
        
        # =========================================================================
        # 1. CALCULATE CONVERGENCE METRICS FOR COMBINED EM 
        # =========================================================================
        c_max_lh = np.nanmax(combined_histories, axis=0)
        c_wins = c_max_lh >= true_max_lik
        c_win_count = c_wins.sum()
        
        c_mask = ~np.isnan(combined_histories)
        c_iters_all = np.sum(c_mask, axis=0)
        c_avg_all = np.mean(c_iters_all)
        
        c_avg_succ = np.mean(c_iters_all[c_wins]) if np.any(c_wins) else np.nan
        
        # =========================================================================
        # 2. CALCULATE CONVERGENCE METRICS FOR ALTERNATING EM 
        # =========================================================================
        a_max_lh = np.nanmax(alternating_histories, axis=0)
        a_wins = a_max_lh >= true_max_lik
        a_win_count = a_wins.sum()
        
        a_mask = ~np.isnan(alternating_histories)
        a_iters_all = np.sum(a_mask, axis=0)
        a_avg_all = np.mean(a_iters_all)
        
        a_avg_succ = np.mean(a_iters_all[a_wins]) if np.any(a_wins) else np.nan

        # =========================================================================
        # 3. POPULATE THE MASTER DICTIONARY COLLECTOR FOR THE CSV
        # =========================================================================
        all_seed_metrics.append({
            'Seed': f"Seed {seed_idx}",
            'True_Max_Lik': true_max_lik,
            
            # Combined Columns
            'Combined_Success': f"{c_win_count}/{N_startpoints}",
            'Combined_Success_Num': c_win_count,
            'Combined_Avg_Iters': c_avg_all,
            'Combined_Avg_Iters_Succ': c_avg_succ,
            
            # Alternating Columns
            'Alternating_Success': f"{a_win_count}/{N_startpoints}",
            'Alternating_Success_Num': a_win_count,
            'Alternating_Avg_Iters': a_avg_all,
            'Alternating_Avg_Iters_Succ': a_avg_succ
        })
        
        
        # -------------------------------------------------------------------------
        # DIAGNOSTIC PLOTTING: Side-by-Side Surface Convergence Dashboard
        # -------------------------------------------------------------------------
        fig, axes = plt.subplots(1, 2, figsize=(15, 6), sharey=True)
        
        # Left Panel: Combined Loop Type
        axes[0].plot(combined_histories, alpha=0.7, linewidth=1.5)
        axes[0].axhline(y=true_max_lik, color="black", linestyle=":", linewidth=2, zorder=5,
                        label=f"True Max Lik ({true_max_lik:.2f})")
        #axes[0].set_xlim(-2, 100)
        axes[0].set_title(f"Combined EM Loop ('combined')\n(All {N_startpoints} Multi-Starts)", fontsize=11, fontweight='bold')
        axes[0].set_xlabel('EM Iteration Step', fontsize=10)
        axes[0].set_ylabel('Log-Likelihood', fontsize=10)
        axes[0].grid(True, linestyle=':', alpha=0.6)
        axes[0].legend(loc="lower right")
        
        # Right Panel: Alternating Loop Type
        axes[1].plot(alternating_histories, alpha=0.7, linewidth=1.5, linestyle='--')
        axes[1].axhline(y=true_max_lik, color="black", linestyle=":", linewidth=2, zorder=5,
                        label=f"True Max Lik ({true_max_lik:.2f})")
        #axes[1].set_xlim(-2, 100)
        axes[1].set_title(f"Alternating EM Loops ('alternating')\n(All {N_startpoints} Multi-Starts)", fontsize=11, fontweight='bold')
        axes[1].set_xlabel('EM Iteration Step', fontsize=10)
        axes[1].grid(True, linestyle=':', alpha=0.6)
        axes[1].legend(loc="lower right")
        
        plt.suptitle(f"EM Optimization Diagnostic Profile: Dataset Seed {seed_idx}", 
                     fontsize=13, fontweight='bold', y=0.98)
        plt.tight_layout()
        
        # -------------------------------------------------------------------------
        # EXPORT GRAPH: Automatically save plot image inside the dataset folder
        # -------------------------------------------------------------------------
        plot_save_path = dataset_folder / f"em_convergence_profile_seed_{seed_idx}.png"
        plt.savefig(plot_save_path, dpi=300, bbox_inches='tight')
        print(f" Diagnostic chart safely exported to: {plot_save_path}")
        
        # Spawn window frame without interrupting loop progression
        plt.show(block=False)
    
        
    # =========================================================================
    # 4. CROSS-SEED AGGREGATION & EXPORT SYSTEM (DATAFRAME ONLY)
    # =========================================================================
    summary_df = pd.DataFrame(all_seed_metrics)
    
    # Calculate across-seed averages cleanly
    avg_row = {
        'Seed': 'AVERAGE',
        'True_Max_Lik': summary_df['True_Max_Lik'].mean(),
        'Combined_Success': f"{summary_df['Combined_Success_Num'].mean():.1f}/{N_startpoints}",
        'Combined_Success_Num': summary_df['Combined_Success_Num'].mean(),
        'Combined_Avg_Iters': summary_df['Combined_Avg_Iters'].mean(),
        'Combined_Avg_Iters_Succ': summary_df['Combined_Avg_Iters_Succ'].mean(),
        'Alternating_Success': f"{summary_df['Alternating_Success_Num'].mean():.1f}/{N_startpoints}",
        'Alternating_Success_Num': summary_df['Alternating_Success_Num'].mean(),
        'Alternating_Avg_Iters': summary_df['Alternating_Avg_Iters'].mean(),
        'Alternating_Avg_Iters_Succ': summary_df['Alternating_Avg_Iters_Succ'].mean()
    }
    summary_df = pd.concat([summary_df, pd.DataFrame([avg_row])], ignore_index=True)
    
    # Drop intermediate numeric columns
    display_df = summary_df.drop(columns=['Combined_Success_Num', 'Alternating_Success_Num'])
    
    # Clean formatting/rounding
    display_df['True_Max_Lik'] = display_df['True_Max_Lik'].round(2)
    display_df['Combined_Avg_Iters'] = display_df['Combined_Avg_Iters'].round(1)
    display_df['Combined_Avg_Iters_Succ'] = display_df['Combined_Avg_Iters_Succ'].round(1)
    display_df['Alternating_Avg_Iters'] = display_df['Alternating_Avg_Iters'].round(1)
    display_df['Alternating_Avg_Iters_Succ'] = display_df['Alternating_Avg_Iters_Succ'].round(1)
    
    # Save to CSV
    csv_save_path = base_sim_path / "final_experiment_summary.csv"
    display_df.to_csv(csv_save_path, index=False)
    
    # Clear console and print just the clean DataFrame object
    print("\n" + "="*50)
    print(" FINAL AGGREGATED METRICS DATAFRAME")
    print("="*50)
    print(display_df)
    print(f"\n💾 Saved directly to: {csv_save_path}\n")

if __name__ == '__main__':
    # Start with 5 seeds to quickly verify your logic paths
    run_diagnostic_benchmark(num_seeds=1)