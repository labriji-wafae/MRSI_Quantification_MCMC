"""
Pipelines for MRSI fitting and quantification using FSL-MRS and MCMC.
"""

import os
from datetime import datetime
from multiprocessing import Pool
import numpy as np
import pandas as pd
import nibabel as nib
from scipy.optimize import minimize

# Imports spécialisés neuroimagerie
from fsl_mrs.utils import fitting, misc
from fsl_mrs.utils.misc import SpecToFID
from nifti_mrs.create_nmrs import gen_nifti_mrs

# Imports locaux depuis les autres modules du dossier src
from src.mcmc import MCMC_ber_laplace_MH_within_Gibbs, compute_H, objective_function, grad
from src.tools_mcmc import polynomial_baseline_correction
from src.utils import get_directory_names, check_directory_exists, get_unique_filename


def run_FSL_MH(mrsi_fc, names, CSI_dir, affine_matrix, results_folder):
    x_size, y_size, z_size = mrsi_fc.spatial_shape
    A_est_fsl_MH = np.zeros([mrsi_fc.numBasis, x_size, y_size, z_size])
    CRLB_est_fsl_MH = np.zeros([mrsi_fc.numBasis, x_size, y_size, z_size])
    X_est_fsl_MH = np.zeros(np.shape(mrsi_fc.data), dtype=complex)
    Gamma_est_fsl_MH = np.zeros([x_size, y_size, z_size])

    names_crlb = [name + '_crlb' for name in names]
    names_sd = [name + '_sd' for name in names]
    M = len(names)
    FSL_MH_results_df = []

    for i in range(3, 13, 1):
        for j in range(3, 13, 1):
            for k in range(2, 6, 1):
                mrs = mrsi_fc.mrs_by_index([i, j, k])
                mrs.rescaleForFitting()
                if np.mean(mrs.FID) != 0:
                    mrs.processForFitting()
                    metab_groups = misc.parse_metab_groups(mrs, 'combine_all')

                    Fitargs = {
                        'ppmlim': [1, 4.2],
                        'method': 'MH', 
                        'baseline_order': 6,
                        'metab_groups': metab_groups,
                        'model': 'voigt',
                        'MHSamples': 5
                    }

                    res = fitting.fit_FSLModel(mrs, **Fitargs)
                    fitResults = np.array(res.fitResults)

                    df_res = res.fitResults.copy()
                    df_res.insert(0, "Voxel Position", [(i, j, k)] * len(df_res))
                    
                    A_est = fitResults[0, 0:M]
                    Gamma_est = fitResults[0, M]
                    A_est_fsl_MH[:, i, j, k] = A_est
                    Gamma_est_fsl_MH[i, j, k] = Gamma_est
                    X_est_fsl_MH[i, j, k, :] = res.pred_spec
                    CRLB_est_fsl_MH[:, i, j, k] = res.crlb[:M]
                    sd_dict = {name: [sd] for name, sd in zip(names_sd, res.getUncertainties(type='raw'))}
                    crlb_dict = {name: [sd] for name, sd in zip(names_crlb, res.crlb)}

                    frames = [df_res, pd.DataFrame(sd_dict), pd.DataFrame(crlb_dict)]
                    result = pd.concat(frames, axis=1)
                    FSL_MH_results_df.append(result)
    
    Patient_ID, Exam = get_directory_names(CSI_dir)
    check_directory_exists(os.path.join(CSI_dir, results_folder))

    filepath = os.path.join(CSI_dir, results_folder, f'A_FSLMH_{Patient_ID}_{Exam}.nii.gz')
    A_FSLMH_nii = np.transpose(A_est_fsl_MH, (1, 2, 3, 0))
    nifti_img = nib.Nifti1Image(A_FSLMH_nii, affine=affine_matrix)
    filepath = get_unique_filename(filepath)
    nib.save(nifti_img, filepath)
    
    fid_to_save = np.zeros([x_size, y_size, z_size, mrsi_fc.FID_points], dtype=complex)
    for x in range(x_size):
        for y in range(y_size):
            for z in range(z_size):
                fid_to_save[x, y, z] = SpecToFID(X_est_fsl_MH[x, y, z] / np.max(np.abs(X_est_fsl_MH[x, y, z])))

    nifti_out_path = os.path.join(CSI_dir, results_folder, f'MRSI_X_FSL_MH_{Patient_ID}_{Exam}.nii.gz')
    nifti_out_path = get_unique_filename(nifti_out_path)
    gen_nifti_mrs(fid_to_save, 1 / mrsi_fc.header['bandwidth'], mrsi_fc.header['centralFrequency'], '1H', affine_matrix).save(nifti_out_path)
    
    filepath = os.path.join(CSI_dir, results_folder, f'CNI_FSLMH_{Patient_ID}_{Exam}.nii.gz')
    CNI = np.squeeze(A_est_fsl_MH[names.index('PCh'), :, :, :]) / np.squeeze(A_est_fsl_MH[names.index('NAA'), :, :, :])
    CNI[~np.isfinite(CNI)] = 0
    nifti_img = nib.Nifti1Image(CNI, affine_matrix)
    filepath = get_unique_filename(filepath)
    nib.save(nifti_img, filepath)

    return FSL_MH_results_df


def run_FSL_Newton(mrsi_fc, names, CSI_dir, affine_matrix, results_folder):
    x_size, y_size, z_size = mrsi_fc.spatial_shape
    A_est_fsl_Newton = np.zeros([mrsi_fc.numBasis, x_size, y_size, z_size])
    CRLB_est_fsl_Newton = np.zeros([mrsi_fc.numBasis, x_size, y_size, z_size])
    X_est_fsl_Newton = np.zeros(np.shape(mrsi_fc.data), dtype=complex)
    Gamma_est_fsl_Newton = np.zeros([x_size, y_size, z_size])

    names_crlb = [name + '_crlb' for name in names]
    names_sd = [name + '_sd' for name in names]
    FSL_Newton_results_df = []
    M = len(names)
    
    for i in range(3, 13, 1):
        for j in range(3, 13, 1):
            for k in range(2, 6, 1):
                mrs = mrsi_fc.mrs_by_index([i, j, k])
                mrs.rescaleForFitting()
                if np.mean(mrs.FID) != 0:
                    mrs.processForFitting()
                    metab_groups = misc.parse_metab_groups(mrs, 'combine_all')

                    Fitargs = {
                        'ppmlim': [1, 4.2],
                        'method': 'Newton', 
                        'baseline_order': 6,
                        'metab_groups': metab_groups,
                        'model': 'voigt'
                    }

                    res = fitting.fit_FSLModel(mrs, **Fitargs)
                    fitResults = np.array(res.fitResults)

                    df_res = res.fitResults.copy()
                    df_res.insert(0, "Voxel Position", [(i, j, k)])
                    
                    A_est = fitResults[0, 0:M]
                    Gamma_est = fitResults[0, M]
                    A_est_fsl_Newton[:, i, j, k] = A_est
                    Gamma_est_fsl_Newton[i, j, k] = Gamma_est
                    X_est_fsl_Newton[i, j, k, :] = res.pred_spec
                    CRLB_est_fsl_Newton[:, i, j, k] = res.crlb[:M]
                    sd_dict = {name: [sd] for name, sd in zip(names_sd, res.getUncertainties(type='raw'))}
                    crlb_dict = {name: [sd] for name, sd in zip(names_crlb, res.crlb)}

                    frames = [df_res, pd.DataFrame(sd_dict), pd.DataFrame(crlb_dict)]
                    result = pd.concat(frames, axis=1)
                    FSL_Newton_results_df.append(result)
    
    Patient_ID, Exam = get_directory_names(CSI_dir)
    check_directory_exists(os.path.join(CSI_dir, results_folder))

    filepath = os.path.join(CSI_dir, results_folder, f'A_FSLNewton_{Patient_ID}_{Exam}.nii.gz')
    A_FSLNewton_nii = np.transpose(A_est_fsl_Newton, (1, 2, 3, 0))
    nifti_img = nib.Nifti1Image(A_FSLNewton_nii, affine=affine_matrix)
    filepath = get_unique_filename(filepath)
    nib.save(nifti_img, filepath)
    
    fid_to_save = np.zeros([x_size, y_size, z_size, mrsi_fc.FID_points], dtype=complex)
    for x in range(x_size):
        for y in range(y_size):
            for z in range(z_size):
                fid_to_save[x, y, z] = SpecToFID(X_est_fsl_Newton[x, y, z] / np.max(np.abs(X_est_fsl_Newton[x, y, z])))

    nifti_out_path = os.path.join(CSI_dir, results_folder, f'MRSI_X_FSL_Newton_{Patient_ID}_{Exam}.nii.gz')
    nifti_out_path = get_unique_filename(nifti_out_path)
    gen_nifti_mrs(fid_to_save, 1 / mrsi_fc.header['bandwidth'], mrsi_fc.header['centralFrequency'], '1H', affine_matrix).save(nifti_out_path)
    
    filepath = os.path.join(CSI_dir, results_folder, f'CNI_FSLNewton_{Patient_ID}_{Exam}.nii.gz')
    CNI = np.squeeze(A_est_fsl_Newton[names.index('PCh'), :, :, :]) / np.squeeze(A_est_fsl_Newton[names.index('NAA'), :, :, :])
    CNI[~np.isfinite(CNI)] = 0
    nifti_img = nib.Nifti1Image(CNI, affine_matrix)
    filepath = get_unique_filename(filepath)
    nib.save(nifti_img, filepath)

    return FSL_Newton_results_df


def run_MCMC(mrsi_fc, names, CSI_dir, affine_matrix, faxis, basis_array_FID, t, results_folder):
    x_size, y_size, z_size = mrsi_fc.spatial_shape
    P, M = basis_array_FID.shape
    Nmc, Nbi = 4, 2

    A_mcmc_H_List = np.zeros((M, x_size * y_size * z_size))
    X_mcmc_H_List = np.zeros((P, x_size * y_size * z_size), dtype=complex)
    Tsigma2_mcmc_H_List = np.zeros((Nmc - Nbi, x_size * y_size * z_size))
    Gamma_List = np.zeros(x_size * y_size * z_size)
    Tgamma_List = np.zeros((Nmc - Nbi, x_size * y_size * z_size))
    TA_mcmc_H_List = np.zeros((M, Nmc - Nbi, x_size * y_size * z_size))
    Tw_mcmc_H_List = np.zeros((Nmc - Nbi, x_size * y_size * z_size))
    Ta_mcmc_H_List = np.zeros((Nmc - Nbi, x_size * y_size * z_size))
    totalTime_H_List = np.zeros(x_size * y_size * z_size)

    lb = np.zeros(M + 1)
    ub = np.hstack([np.inf * np.ones(M), 15])
    dp_est_matrix = np.zeros(x_size * y_size * z_size)
 
    tuples_list = []
    for vox_x in range(x_size):
        for vox_y in range(y_size):
            for vox_z in range(z_size):
                tuples_list.append((vox_x, vox_y, vox_z))

    mrs_sample = mrsi_fc.mrs_by_index([0, 0, 0])
    first, last = mrs_sample.ppmlim_to_range([1, 4.2])
    print(f'MCMC Start time : {datetime.now()}')

    iterP_args = [
        (iterP, mrsi_fc, basis_array_FID, t, lb, ub, Nmc, Nbi, first, last, tuples_list, faxis) 
        for iterP in range(x_size * y_size * z_size)
    ]

    with Pool() as pool:
        results = pool.starmap(process_iterP, iterP_args)
        
    for result in results:
        if result is not None:
            iterP = result['iterP']
            dp_est_matrix[iterP] = result['dp_est_value']
            A_mcmc_H_List[:, iterP] = result['A_mcmc_H']
            X_mcmc_H_List[:, iterP] = result['X_mcmc_H']
            TA_mcmc_H_List[:, :, iterP] = result['TA_mcmc_H']
            Tsigma2_mcmc_H_List[:, iterP] = result['Tsigma2_mcmc_H']
            Tw_mcmc_H_List[:, iterP] = result['Tw_mcmc_H']
            Ta_mcmc_H_List[:, iterP] = result['Ta_mcmc_H']
            totalTime_H_List[iterP] = result['totalTime_H']
            Gamma_List[iterP] = result['gamma']
            Tgamma_List[:, iterP] = result['Tgamma']

    print(f'MCMC End time : {datetime.now()}')

    A_est_mcmc = np.reshape(A_mcmc_H_List, (M, x_size, y_size, z_size))
    TA_mcmc = np.reshape(TA_mcmc_H_List, (M, Nmc - Nbi, x_size, y_size, z_size))
    X_mcmc = np.reshape(X_mcmc_H_List, (P, x_size, y_size, z_size))
    Tsigma2_mcmc_H_Matrix = np.reshape(Tsigma2_mcmc_H_List, (Nmc - Nbi, x_size, y_size, z_size))
    Tw_mcmc_H_Matrix = np.reshape(Tw_mcmc_H_List, (Nmc - Nbi, x_size, y_size, z_size))
    a_mcmc_H_Matrix = np.reshape(Ta_mcmc_H_List, (Nmc - Nbi, x_size, y_size, z_size))
    Tgamma_matrix = np.reshape(Tgamma_List, (Nmc - Nbi, x_size, y_size, z_size))
    Gamma_matrix = np.reshape(Gamma_List, (x_size, y_size, z_size))

    Patient_ID, Exam = get_directory_names(CSI_dir)
    check_directory_exists(os.path.join(CSI_dir, results_folder))

    filepath = os.path.join(CSI_dir, results_folder, f'A_MCMC_{Patient_ID}_{Exam}.nii.gz')
    A_mcmc_nii = np.transpose(A_est_mcmc, (1, 2, 3, 0))
    nifti_img = nib.Nifti1Image(A_mcmc_nii, affine=affine_matrix)
    filepath = get_unique_filename(filepath)
    nib.save(nifti_img, filepath)
    
    fid_to_save = np.zeros([x_size, y_size, z_size, mrsi_fc.FID_points], dtype=complex)
    for x in range(x_size):
        for y in range(y_size):
            for z in range(z_size):
                fid_to_save[x, y, z] = SpecToFID(X_mcmc[:, x, y, z] / np.max(np.abs(X_mcmc[:, x, y, z])))

    nifti_out_path = os.path.join(CSI_dir, results_folder, f'MRSI_X_MCMC_{Patient_ID}_{Exam}')
    nifti_out_path = get_unique_filename(nifti_out_path)
    gen_nifti_mrs(fid_to_save, 1 / mrsi_fc.header['bandwidth'], mrsi_fc.header['centralFrequency'], '1H', affine_matrix).save(nifti_out_path)

    df_mcmc = pd.DataFrame({
        'Tw_mcmc': [Tw_mcmc_H_Matrix],
        'Tgamma': [Tgamma_matrix],
        'gamma': [Gamma_matrix],
        'Ta_mcmc': [a_mcmc_H_Matrix], 
        'Tsigma2': [Tsigma2_mcmc_H_Matrix],
        'A_mcmc': [A_est_mcmc],
        'TA_mcmc': [TA_mcmc],
        'X_mcmc': [X_mcmc]
    })
    
    filepath = os.path.join(CSI_dir, results_folder, f'CNI_MCMC_{Patient_ID}_{Exam}.nii.gz')
    CNI = np.squeeze(A_est_mcmc[names.index('PCh'), :, :, :]) / np.squeeze(A_est_mcmc[names.index('NAA'), :, :, :])
    CNI[~np.isfinite(CNI)] = 0
    nifti_img = nib.Nifti1Image(CNI, affine_matrix)
    filepath = get_unique_filename(filepath)
    nib.save(nifti_img, filepath)
    
    return df_mcmc


def process_iterP(iterP, mrsi_fc, Basis_fid, t, lb, ub, Nmc, Nbi, first, last, tuples_list, faxis):
    vox_x, vox_y, vox_z = tuples_list[iterP]
    grps_info = 0
    M = Basis_fid.shape[1]
    initial_guess = np.zeros(M + 1)
    
    if (2 < vox_x < 13) and (2 < vox_y < 13) and (1 < vox_z < 6):
        mrs = mrsi_fc.mrs_by_index([vox_x, vox_y, vox_z])
        mrs.rescaleForFitting()
        obs = mrs.get_spec()
        scaling_factor = np.max(np.abs(obs))
        obs = obs / scaling_factor 
        obs = polynomial_baseline_correction(faxis, obs)
        
        output = minimize(objective_function, initial_guess, args=(t, Basis_fid, M, obs, first, last), method='TNC', jac=grad, bounds=list(zip(lb, ub)))
        dp_est_List = output.x[M:M*2]
        totalTime_H = 0 
        gamma_init = dp_est_List[0] 
        
        if np.mean(obs) != 0:
            A_mcmc_H, gamma, Tsigma2_mcmc_H, Ta, Tw, TSignal, TSignal_MAP, a, TZ, w, Chain_gamma, Chain_TSignal, Chain_sigma2m, Chain_phi0, Chain_phi1, Chain_epsilon = MCMC_ber_laplace_MH_within_Gibbs(output.x[0:M], gamma_init, Basis_fid, obs, Nmc, Nbi, t, 0, first, last)
        else: 
            return None
            
        H_corr = compute_H(Basis_fid, gamma, grps_info, t, np.mean(Chain_sigma2m[Nbi+1:]), np.mean(Chain_epsilon[Nbi+1:]), np.mean(Chain_phi0[Nbi+1:]), np.mean(Chain_phi1[Nbi+1:]))
    else: 
        A_mcmc_H = np.zeros(M) * np.nan 
        gamma_init = np.nan
        H_corr = np.zeros([Basis_fid.shape[0], M]) 
        TSignal = np.zeros((M, Nmc - Nbi)) 
        Tsigma2_mcmc_H = np.zeros(Nmc - Nbi)
        Tw = np.zeros((Nmc - Nbi))
        Ta = np.zeros((Nmc - Nbi))
        totalTime_H = 0
        gamma = 0
        Chain_gamma = np.zeros((Nmc))
        
    return {
        'iterP': iterP,
        'dp_est_value': gamma_init,
        'A_mcmc_H': A_mcmc_H,
        'X_mcmc_H': np.dot(H_corr, A_mcmc_H),
        'TA_mcmc_H': TSignal,
        'Tsigma2_mcmc_H': Tsigma2_mcmc_H,
        'Tw_mcmc_H': Tw,
        'Ta_mcmc_H': Ta,
        'totalTime_H': totalTime_H,
        'gamma': gamma,
        'Tgamma': Chain_gamma[Nbi:]
    }


def save_mrsi_proc(mrsi_fc, CSI_dir, affine_matrix, results_folder):
    x_size, y_size, z_size = mrsi_fc.spatial_shape
    fid_to_save = np.zeros([x_size, y_size, z_size, mrsi_fc.FID_points], dtype=complex) 
    for x in range(x_size):
        for y in range(y_size):
            for z in range(z_size):
                mrs = mrsi_fc.mrs_by_index([x, y, z])
                mrs.rescaleForFitting()
                fid_to_save[x, y, z] = mrs.FID / np.max(np.abs(mrs.FID))

    Patient_ID, Exam = get_directory_names(CSI_dir)
    check_directory_exists(os.path.join(CSI_dir, results_folder))
    nifti_out_path = os.path.join(CSI_dir, results_folder, f'MRSI_proc_{Patient_ID}_{Exam}.nii.gz')
    nifti_out_path = get_unique_filename(nifti_out_path)
    print(f'For proc MRSI save: New : {nifti_out_path}')

    gen_nifti_mrs(fid_to_save, 1 / mrsi_fc.header['bandwidth'], mrsi_fc.header['centralFrequency'], '1H', affine_matrix).save(nifti_out_path)
