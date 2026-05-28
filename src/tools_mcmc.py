"""
Preprocessing tools for MRSI data.
Implements spectral alignment (frequency shift) and polynomial baseline correction.
"""

import numpy as np
from scipy.optimize import curve_fit
from fsl_mrs.core.nifti_mrs import NIFTI_MRS
from fsl_mrs.utils.misc import SpecToFID
from nifti_mrs import create_nmrs


def align_mrsi_spectra(mrsi_data, names, basis_array_spec: np.ndarray, ppm: np.ndarray) -> NIFTI_MRS:
    """
    Align frequency shifts across all 3D voxels in an MRSI dataset.
    Optimized to compute index positions outside the main voxel loop.
    """
    size_x, size_y, size_z = mrsi_data.spatial_shape
    mrsi_array_align = np.zeros(
        [size_x, size_y, size_z, mrsi_data.FID_points], 
        dtype=complex
    )
    
    names_list = list(names)
    idx_naa = names_list.index('NAA') if 'NAA' in names_list else None
    idx_pch = names_list.index('PCh') if 'PCh' in names_list else None
    idx_cr = names_list.index('Cr') if 'Cr' in names_list else None

    for x in range(size_x):
        for y in range(size_y):
            for z in range(size_z):
                obs = mrsi_data.mrs_by_index([x, y, z]).get_spec()
                I_max = np.argmax(np.real(obs))
                current_ppm = ppm[I_max]
                
                Metab_H = None

                # Alignement basé sur les pics dominants de résonance
                if 1.8 <= current_ppm <= 2.2 and idx_naa is not None:  # NAA (~2.01 ppm)
                    Metab_H = np.real(basis_array_spec[:, idx_naa])
                elif 3.1 <= current_ppm <= 3.6 and idx_pch is not None:  # Cho / PCh (~3.2 ppm)
                    Metab_H = np.real(basis_array_spec[:, idx_pch])
                elif 2.8 <= current_ppm <= 3.1 and idx_cr is not None:   # Cr (~3.02 ppm)
                    Metab_H = np.real(basis_array_spec[:, idx_cr])
                elif 1.3 <= current_ppm <= 1.4:
                    print(f'High Lipid/lactate contamination possible for voxel [{x},{y},{z}]')

                # Application du décalage (roll) si un métabolite de référence est trouvé
                if Metab_H is not None:
                    I_metab = np.argmax(Metab_H)
                    shift = I_metab - I_max
                    obs = np.roll(obs, shift)

                mrsi_array_align[x, y, z] = SpecToFID(obs)

    # Reconstruction propre de l'objet NIFTI_MRS
    nmrs_obj = create_nmrs.gen_nifti_mrs(
        mrsi_array_align, 
        1 / mrsi_data.header['bandwidth'], 
        mrsi_data.header['centralFrequency']
    )
    return NIFTI_MRS(nmrs_obj, header=mrsi_data.header)


def align_mrs_spectra(mrs_data: np.ndarray, names, basis_array_spec: np.ndarray, ppm: np.ndarray):
    """
    Align frequency shift for a single voxel spectrum vector.
    """
    obs = mrs_data.copy()
    I_max = np.argmax(np.real(obs))
    current_ppm = ppm[I_max]
    Metab_H = None

    names_list = list(names)
    
    if 1.8 <= current_ppm <= 2.2 and 'NAA' in names_list:
        Metab_H = np.real(basis_array_spec[:, names_list.index('NAA')])
        print('Realigned on NAA')
    elif 3.1 <= current_ppm <= 3.6 and 'PCh' in names_list:
        Metab_H = np.real(basis_array_spec[:, names_list.index('PCh')])
        print('Realigned on PCh')
    elif 2.8 <= current_ppm <= 3.1 and 'Cr' in names_list:
        Metab_H = np.real(basis_array_spec[:, names_list.index('Cr')])
        print('Realigned on Cr')

    shift = 0
    if Metab_H is not None:
        I_metab = np.argmax(Metab_H)
        shift = I_metab - I_max
        print(f'I_metab = {I_metab} | I_max = {I_max} | Shift applied: {shift}')
        obs = np.roll(obs, shift)

    return obs, shift


def excludedata(ppm: np.ndarray, indices: np.ndarray) -> np.ndarray:
    """Create a boolean mask to exclude specific signal indices from fitting."""
    mask = np.ones(len(ppm), dtype=bool)
    mask[indices] = False
    return mask


def polynomial_fit(ppm: np.ndarray, y: np.ndarray, degree: int, exclude_mask: np.ndarray):
    """
    Fit a polynomial vector on non-excluded data points using numpy.polyfit 
    for maximum numerical speed and reliability.
    """
    # Remplacement de curve_fit par np.polyfit 
    coefficients = np.polyfit(ppm[exclude_mask], y[exclude_mask], deg=degree)
    return lambda x: np.polyval(coefficients, x)


def polynomial_baseline_correction(ppm: np.ndarray, y: np.ndarray) -> np.ndarray:
    """
    Perform a 3rd-degree polynomial baseline subtraction on the real and imaginary 
    parts of the MRS spectrum, excluding main metabolite windows.
    """
    # Sélection des zones hors métabolites majeurs pour modéliser le bruit de fond
    x1 = np.where((ppm > 0.7) & (ppm < 1.1))[0]
    x2 = np.where((ppm > 1.2) & (ppm < 1.6))[0]
    x3 = np.where((ppm > 1.9) & (ppm < 2.1))[0]
    x4 = np.where((ppm > 2.9) & (ppm < 3.35))[0]
    x5 = np.where((ppm > 3.5) & (ppm < 4.0))[0]
    
    x = np.concatenate((x1, x2, x3, x4, x5))

    exclude_re = excludedata(ppm, x)
    exclude_im = excludedata(ppm, x)

    cfun_re = polynomial_fit(ppm, np.real(y), degree=3, exclude_mask=exclude_re)
    cfun_im = polynomial_fit(ppm, np.imag(y), degree=3, exclude_mask=exclude_im)

    baseline_re = cfun_re(ppm)
    baseline_im = cfun_im(ppm)

    corrected_signal = y - baseline_re - 1j * baseline_im
    return corrected_signal
