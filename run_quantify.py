"""
Main entrypoint script to execute the MRSI Quantification pipeline.
Supports FSL (Newton / MH) and custom MCMC sampling methods.
"""

import os
import sys
import argparse
from datetime import datetime
from pathlib import Path
import pandas as pd
import numpy as np

# Imports FSL
import fsl_mrs.utils.mrs_io as mrs_io
from fsl_mrs.utils.preproc import nifti_mrs_proc as proc

# Imports depuis notre package src structuré
from src.tools_mcmc import align_mrsi_spectra
from src.utils import get_directory_names, load_basis_lip, gen_ppm_axis, get_unique_filename
from src.pipelines import run_FSL_Newton, run_FSL_MH, run_MCMC, save_mrsi_proc


def Run_quantify(root_directory, run_fsl_newton=True, run_fsl_metropolis=True, run_mcmc_gibbs=True):
    results_folder = 'Results_test'
    
    for foldername, subfolders, filenames in os.walk(root_directory):
        if os.path.basename(foldername) == 'CSI':
            print(f"\nProcessing directory: {foldername}")
            CSI_dir = foldername
            CSI_data = []
            affine_matrix = []
            Patient_ID, Exam = "Unknown", "Unknown"
            
            for filename in filenames:
                if filename.endswith("_hsvd2.nii"):  
                    print(f'Loading NIfTI MRS file: {filename}')
                    data = mrs_io.read_FID(os.path.join(CSI_dir, filename))
                    CSI_data.append(data)
                    affine_matrix.append(data.voxToWorldMat)
                    Patient_ID, Exam = get_directory_names(CSI_dir)
            
            if len(CSI_data) > 1: 
                print('Warning: More than one CSI file found to process.')
                
            for acqu in range(len(CSI_data)):
                print(f'Start time: {datetime.now()}')
                CSI_data_zf = proc.truncate_or_pad(CSI_data[acqu], 512, 'last')
                print('Zero padding (512 points): Done')
                
                # Load basis
                basis = load_basis_lip() 
                basis_array_FID = basis.original_basis_array / 10000 
                basis_array_spec = np.fft.fftshift(np.fft.fft(basis_array_FID, axis=0), axes=0) 

                P, M = basis_array_spec.shape
                names = basis._names
                ppmaxis = gen_ppm_axis(CSI_data_zf)
                t = np.linspace(0, P, P) * 0.001
                
                mrsi_proc = CSI_data_zf.mrs(basis=basis)
                CSI_data_zf_fc = align_mrsi_spectra(mrsi_proc, names, basis_array_spec, ppmaxis)
                mrsi_fc = CSI_data_zf_fc.mrs(basis=basis)
                affine = affine_matrix[acqu]
                
                result_path = os.path.join(CSI_dir, results_folder)
                p = Path(result_path)
                
                if not p.exists():
                    save_mrsi_proc(mrsi_fc, CSI_dir, affine, results_folder)
                    h5_out_path = os.path.join(CSI_dir, results_folder, f'MRSI_Results_{Patient_ID}_{Exam}')
                    h5_out_path = get_unique_filename(h5_out_path)

                    # 1. Option FSL Newton
                    if run_fsl_newton:
                        FSL_Newton_results_df = run_FSL_Newton(mrsi_fc, names, CSI_dir, affine, results_folder)
                        print('run_FSL_Newton Done')
                        with pd.HDFStore(h5_out_path, mode='w') as store:
                            for i, df in enumerate(FSL_Newton_results_df):
                                store.put(f'df_newton_{i}', df)
                    else:
                        print('Skip FSL_Newton')
                    
                    # 2. Option FSL Metropolis Hastings
                    if run_fsl_metropolis:
                        FSL_MH_results_df = run_FSL_MH(mrsi_fc, names, CSI_dir, affine, results_folder)
                        print('run_FSL_MH Done')
                        with pd.HDFStore(h5_out_path, mode='a') as store:
                            for i, df in enumerate(FSL_MH_results_df):
                                store.put(f'df_mh_{i}', df)
                    else:
                        print('Skip FSL_MH')

                    # 3. Option Custom MCMC Gibbs within MH
                    if run_mcmc_gibbs:
                        MCMC_results_df = run_MCMC(mrsi_fc, names, CSI_dir, affine, ppmaxis, basis_array_FID, t, results_folder)
                        print('run_MCMC Done')
                        MCMC_results_df.to_hdf(h5_out_path, key='MCMC_results_df', mode='a')
                    else:
                        print('Skip MCMC')
                    
                    print(f'End time: {datetime.now()}')
                else:
                    print('Result path already exists. Skipping acquisition.')


def main():
    parser = argparse.ArgumentParser(description="Pipeline orchestration for MRSI Quantification.")
    
    # Argument positionnel obligatoire
    parser.add_argument("root_directory", type=str, 
                        help="Path to the root directory containing the 'CSI' study folder.")
    
    # Arguments optionnels nommés (--nom)
    parser.add_argument("--newton", action="store_true", default=True,
                        help="Run the FSL Newton optimization method (default: True).")
    parser.add_argument("--no-newton", action="store_false", dest="newton",
                        help="Disable the FSL Newton method.")
                        
    parser.add_argument("--mh", action="store_true", default=True,
                        help="Run the FSL Metropolis-Hastings method (default: True).")
    parser.add_argument("--no-mh", action="store_false", dest="mh",
                        help="Disable the FSL Metropolis-Hastings method.")
                        
    parser.add_argument("--mcmc", action="store_true", default=True,
                        help="Run the custom Bayesian MCMC Gibbs sampling method (default: True).")
    parser.add_argument("--no-mcmc", action="store_false", dest="mcmc",
                        help="Disable the custom MCMC Gibbs method.")

    args = parser.parse_args()

    # Lancement du traitement
    Run_quantify(
        root_directory=args.root_directory, 
        run_fsl_newton=args.newton, 
        run_fsl_metropolis=args.mh, 
        run_mcmc_gibbs=args.mcmc
    )


if __name__ == '__main__':
    main()
