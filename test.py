import numpy as np
import os
import shutil
from apoe4_sustain import APOE4Sustain

def main():
    # 1. Setup paths
    output_folder = "test_output"
    dataset_name = "smoke_test"
    if os.path.exists(output_folder):
        shutil.rmtree(output_folder)
    os.makedirs(output_folder)

    # 2. Generate Synthetic Data
    N_subjects = 20
    N_biomarkers = 3
    data = np.random.randn(N_subjects, N_biomarkers)
    
    # 3. Create Required Inputs for ZscoreSustain
    # SuStaInLabels: Names for your biomarkers
    SuStaInLabels = ["Hippocampus", "Entorhinal", "Amygdala"]
    
    # Z_vals: Which Z-scores to trigger for each biomarker (e.g., 1, 2, 3)
    # Shape should be (N_biomarkers, N_z_scores)
    Z_vals = np.array([[1, 2, 3], 
                       [1, 2, 3], 
                       [1, 2, 3]])
    
    # Z_max: The maximum Z-score considered
    Z_max = 3
    
    # apoe4_status: Our custom input (0 or 1)
    apoe4_status = np.random.randint(0, 2, size=(N_subjects, 1))

    # 4. Standard SuStaIn control parameters
    N_startpoints = 2
    N_S_max = 2              # Max number of subtypes to try
    N_iterations_MCMC = 10   # Minimal iterations for smoke test
    
    print("--- Initializing APOE4Sustain with All Required Inputs ---")
    
    # Initialize using the exact signature you provided
    sustain_model = APOE4Sustain(
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
        apoe4_status=apoe4_status
    )

    print("--- Starting SuStaIn Run ---")
    # This calls the run() method which eventually calls our overridden likelihood
    sustain_model.run()

    print("\n--- Smoke Test Finished Successfully ---")

if __name__ == "__main__":
    main()