#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed June 3rd 9:40 2026

Experiment 1: Algorithmic Diagnostics
Objective: Full 2x2 Factorial Matrix (EM Loop x Weight Init) over 3 Subtypes.
           Correctly maps and saves distinct likelihood curves and PVD maps.

@author: mihaelacroitor
"""

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from genetics_simulation_utils import simulate_apoe_sustain_dataset
from pathlib import Path
import os 
import shutil

from apoe4_sustain import ZscoreSustain_APOE4


def run_diagnostic_benchmark(num_seeds=5):
    init_methods = ['random', 'cohort_perturb']
    em_loops = ['combined', 'alternating']
    
    master_tabular_rows = []
    
    N_biomarkers = 5#5
    N_S_max = 3 #3
    N_startpoints = 15 # 15
    N_iterations_MCMC = int(1e3) 
    BiomarkerNames = [f'Biomarker {i}' for i in range(N_biomarkers)]
    
    base_sim_path = Path("experiments") / "01_algorithmic_diagnostics" / "outputs"

    for seed_idx in range(num_seeds):
        print("\n==================================================")
        print(f" GENERATING UNIFIED COHORT FOR DATASET SEED: {seed_idx}")
        print("==================================================")
        
        dataset_name = f"diagnostic_seed_{seed_idx}"
        dataset_folder = base_sim_path / dataset_name
        
        # 1. Generate the shared stable dataset for this seed
        df, Z_vals, Z_max, gt_sequence, gt_f, W_true = simulate_apoe_sustain_dataset(
            N_S_gt=N_S_max, N=N_biomarkers, genetic_signal_strength='moderate', 
            seed=seed_idx, save=True, output_path=base_sim_path,   
            dataset_name=dataset_name, base_filename="raw_data"      
        )
        
        X_data = df[BiomarkerNames].values
        y_genetics = df['apoe_status'].values
        M_samples = len(X_data)

        # -------------------------------------------------------------------------
        # VISUAL GROUND TRUTH PROGRESSION PATTERN (Once per seed)
        # -------------------------------------------------------------------------
        temp_sustain = ZscoreSustain_APOE4(
            X_data, Z_vals, Z_max, BiomarkerNames, N_startpoints, N_S_max, 
            N_iterations_MCMC, dataset_folder, dataset_name, False,
            apoe4_status=y_genetics, apoe_flag=True, em_loop_type='combined'
        )
        sustainData = temp_sustain._AbstractSustain__sustainData
        true_max_lik, _, _, _, _ = temp_sustain._calculate_likelihood(sustainData, gt_sequence, gt_f, W_true)
        print(f" 🎯 Dataset Ground-Truth True Max Likelihood: {true_max_lik:.2f}")

        temp_gt_sequence = np.tile(np.reshape(gt_sequence, (gt_sequence.shape[0], gt_sequence.shape[1], 1)), 100)
        temp_gt_f = np.asarray(gt_f).reshape(len(gt_f), 1)
        dynamic_gt_order = np.arange(gt_sequence.shape[0])
        
        ZscoreSustain_APOE4._plot_sustain_model(temp_sustain, temp_gt_sequence, temp_gt_f, M_samples, subtype_order=dynamic_gt_order)
        plt.suptitle(f'Figure 3: Ground Truth Progression Pattern (Seed {seed_idx})', fontweight='bold')
        gt_plot_path = dataset_folder / f"ground_truth_pattern_seed_{seed_idx}.png"
        plt.savefig(gt_plot_path, dpi=300, bbox_inches='tight')
        plt.close('all')

        # ---------------------------------------------------------------------
        # FACTORIAL SWEEP MATRIX LOOPS
        # ---------------------------------------------------------------------
        for init in init_methods:
            print(f"\n   🎬 Running Branch: Init Method = [{init.upper()}]")
            
            # Temporary cache to store tracks for side-by-side comparison plotting
            histories_for_plot = {}
            
            for em in em_loops:
                print(f"    👉 Executing Model: {init} + {em}...")
                specific_run_folder = dataset_folder / f"run_{init}_{em}"
                if specific_run_folder.exists():
                    shutil.rmtree(specific_run_folder)
                specific_run_folder.mkdir(parents=True, exist_ok=True)
                
                model = ZscoreSustain_APOE4(
                    X_data, Z_vals, Z_max, BiomarkerNames, N_startpoints, N_S_max, 
                    N_iterations_MCMC, specific_run_folder, dataset_name, False,
                    apoe4_status=y_genetics, apoe_flag=True, em_loop_type=em,
                    genetic_init_method=init
                )
                model.run_sustain_algorithm()
                
                # Load structural outputs out of the targeted subtype file
                s_target = N_S_max - 1  
                pickle_path = specific_run_folder / 'pickle_files' / f"{dataset_name}_subtype{s_target}.pickle"
                pk_data = pd.read_pickle(pickle_path)
                histories = pk_data["em_likelihood_histories"]
                
                # Save reference array to our side-by-side plotting cache
                histories_for_plot[em] = histories  
                
                # Metrics Crunching
                max_lh = np.nanmax(histories, axis=0)
                wins = max_lh >= true_max_lik
                win_count = wins.sum()
                iters_all = np.sum(~np.isnan(histories), axis=0)
                
                master_tabular_rows.append({
                    'Seed': f"Seed {seed_idx}", 'Init_Method': init, 'EM_Strategy': em,
                    'True_Max_Lik': true_max_lik, 'Success_Rate': f"{win_count}/{N_startpoints}",
                    'Success_Num': win_count, 'Avg_Iters_Total': np.mean(iters_all),
                    'Avg_Iters_Success': np.mean(iters_all[wins]) if np.any(wins) else np.nan
                })
                
                # -------------------------------------------------------------------------
                # VISUAL ASSET 1: Positional Variance Diagram (Saved inside individual run workspace)
                # -------------------------------------------------------------------------
                samples_sequence = pk_data["samples_sequence"]
                samples_f = pk_data["samples_f"]
                ml_f_EM = pk_data["ml_f_EM"]
                subtype_order = np.argsort(ml_f_EM)
                
                ZscoreSustain_APOE4._plot_sustain_model(
                    model, samples_sequence, samples_f, M_samples, 
                    subtype_order=subtype_order, biomarker_labels=BiomarkerNames
                )
                plt.suptitle(f'Figure 4: APOE-Informed SuStaIn Output ({init} | {em})', fontweight='bold')
                
                pvd_save_path = specific_run_folder / f"positional_variance_diagram_seed_{seed_idx}.png"
                plt.savefig(pvd_save_path, dpi=300, bbox_inches='tight')
                plt.close('all') 

            # -------------------------------------------------------------------------
            # VISUAL ASSET 2: Side-by-Side EM Likelihood Curves Dashboard
            # FIX: Placed outside the inner loop so data from both runs exists cleanly!
            # -------------------------------------------------------------------------
            fig_lik, axes = plt.subplots(1, 2, figsize=(15, 6), sharey=True)
            
            max_combined_len = np.max(np.sum(~np.isnan(histories_for_plot['combined']), axis=0))
            max_alternating_len = np.max(np.sum(~np.isnan(histories_for_plot['alternating']), axis=0))
            dynamic_x_max = int(max(max_combined_len, max_alternating_len) + 2)
                
            # Left Panel: Combined Loop Type
            axes[0].plot(histories_for_plot['combined'], alpha=0.7, linewidth=1.5)
            axes[0].axhline(y=true_max_lik, color="black", linestyle=":", linewidth=2, label=f"True Max Lik ({true_max_lik:.2f})")
            axes[0].set_title(f"Combined EM Loop ('combined') [{init}]\n(All {N_startpoints} Multi-Starts)", fontsize=11, fontweight='bold')
            axes[0].set_xlabel('EM Iteration Step', fontsize=10)
            axes[0].set_ylabel('Log-Likelihood', fontsize=10)
            axes[0].set_xlim(-0.5, dynamic_x_max)  
            axes[0].grid(True, linestyle=':', alpha=0.6)
            axes[0].legend(loc="lower right")
            
            # Right Panel: Alternating Loop Type
            axes[1].plot(histories_for_plot['alternating'], alpha=0.7, linewidth=1.5, linestyle='--')
            axes[1].axhline(y=true_max_lik, color="black", linestyle=":", linewidth=2, label=f"True Max Lik ({true_max_lik:.2f})")
            axes[1].set_title(f"Alternating EM Loops ('alternating') [{init}]\n(All {N_startpoints} Multi-Starts)", fontsize=11, fontweight='bold')
            axes[1].set_xlabel('EM Iteration Step', fontsize=10)
            axes[1].set_xlim(-0.5, dynamic_x_max)  
            axes[1].grid(True, linestyle=':', alpha=0.6)
            axes[1].legend(loc="lower right")
            
            plt.suptitle(f"EM Likelihood Optimization Diagnostics ({init}): Seed {seed_idx}", fontsize=13, fontweight='bold', y=0.98)
            plt.tight_layout()
            
            # Save the dashboard directly inside the parent seed folder
            lik_save_path = dataset_folder / f"em_likelihood_curves_{init}_seed_{seed_idx}.png"
            plt.savefig(lik_save_path, dpi=300, bbox_inches='tight')
            plt.close(fig_lik)
            print(f" 📈 Likelihood curve chart successfully exported to: {lik_save_path}")

    # =========================================================================
    # UNIFIED REPORT GENERATION & COMPILATION
    # =========================================================================
    master_df = pd.DataFrame(master_tabular_rows)
    
    summary_blocks = []
    for init in init_methods:
        for em in em_loops:
            subset = master_df[(master_df['Init_Method'] == init) & (master_df['EM_Strategy'] == em)]
            summary_blocks.append({
                'Seed': '📊 ACROSS-SEED AVG', 'Init_Method': init, 'EM_Strategy': em,
                'True_Max_Lik': subset['True_Max_Lik'].mean(),
                'Success_Rate': f"{subset['Success_Num'].mean():.1f}/{N_startpoints}",
                'Success_Num': subset['Success_Num'].mean(),
                'Avg_Iters_Total': subset['Avg_Iters_Total'].mean(),
                'Avg_Iters_Success': subset['Avg_Iters_Success'].mean()
            })
            
    final_master_df = pd.concat([master_df, pd.DataFrame(summary_blocks)], ignore_index=True)
    final_master_df = final_master_df.drop(columns=['Success_Num'])
    
    final_master_df['True_Max_Lik'] = final_master_df['True_Max_Lik'].round(2)
    final_master_df['Avg_Iters_Total'] = final_master_df['Avg_Iters_Total'].round(1)
    final_master_df['Avg_Iters_Success'] = final_master_df['Avg_Iters_Success'].round(1)
    
    master_csv_path = base_sim_path / "comprehensive_factorial_volatility_matrix.csv"
    final_master_df.to_csv(master_csv_path, index=False)
    
    print("\n" + "█"*90)
    print(" 🏆 FINAL CONSOLIDATED 2x2 FACTORIAL SUMMARY PERFORMANCE SHEET")
    print("█"*90)
    print(final_master_df.to_string(index=False))
    print(f"\n💾 Absolute publication summary matrix saved cleanly to: {master_csv_path}\n")


if __name__ == '__main__':
    run_diagnostic_benchmark(num_seeds=10)