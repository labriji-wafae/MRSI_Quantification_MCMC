"""
MCMC Core Core Module for MRSI Quantification.
Implements Gibbs sampling, Metropolis-Hastings within Gibbs, and Bayesian 
Bernoulli-Laplace sparse models for spectral fitting.
"""

import numpy as np
from scipy.stats import beta, norm
from scipy.stats import gamma as gamma_dist
from scipy.fftpack import fft, fftshift
from rtnorm import rtnorm


def map_sparse(TSignal: np.ndarray, TZ: np.ndarray) -> np.ndarray:
    """
    Map sparse signal based on the support matrix TZ using vectorized operations.
    """
    L, C = TSignal.shape
    # Calcul vectorisé sur l'axe des colonnes (axis=1) pour éviter la boucle for
    tmp = np.sum(TZ, axis=1) - C / 2
    
    # Calcul des moyennes pour toutes les lignes
    means = np.mean(TSignal, axis=1)
    
    # Si la condition est respectée, on garde la moyenne, sinon 0
    signal_map = np.where(tmp > 0, means, 0.0)
    return signal_map


def sample_w(theta: np.ndarray, alpha0: float, alpha1: float) -> float:
    """
    Sample the sparse weight hyperparameter 'w' from a Beta distribution.
    """
    M = len(theta)
    n1 = np.sum(theta != 0)
    n0 = M - n1

    coeff1 = alpha0 + n1
    coeff0 = alpha1 + n0

    return beta.rvs(coeff1, coeff0)


def compute_H(H0: np.ndarray, gamma: float, grps_info: int, t: np.ndarray, 
              sigma2m: float, epsilon: float, phi0: float, phi1: float) -> np.ndarray:
    """
    Compute the transformed basis matrix H incorporating phase and damping factors.
    """
    M = H0.shape[1]
    H = H0.copy()
    
    for midx in range(M):
        if grps_info == 0:
            # Vectorized transformation per metabolite column
            damping_term = np.exp(-t * ((gamma + sigma2m * t) + 1j * epsilon))
            phase_term = np.exp(1j * phi0)
            H[:, midx] = phase_term * FID2Spec(H0[:, midx] * damping_term)
    return H


def sample_gamma_MH(Observation, VSignal, sigma2, H0, H, gamma, grps_info, t, 
                    acc, rej, sigma2_m, epsilon, phi0, phi1, first, last):
    """
    Metropolis-Hastings sampling for the lineshape damping parameter (gamma).
    """
    Hstar = H.copy()
    M = H.shape[1]
    pvar = 1.0  # Proposal variance
    mu_gamma = 5.0
    sigma_gamma = 2.5

    # Propose a new candidate
    gamma_star = gamma + np.random.randn() * pvar
    
    for midx in range(M):
        Hstar[:, midx] = fftshift(fft(H0[:, midx] * np.exp(-t * gamma_star)))

    # Compute probability densities safely
    diff_old = Observation[first:last] - np.dot(H[first:last, :], VSignal)
    diff_star = Observation[first:last] - np.dot(Hstar[first:last, :], VSignal)
    
    pdf_old = np.exp(-np.linalg.norm(diff_old)**2 / sigma2) * norm.pdf(gamma, mu_gamma, np.sqrt(sigma_gamma))
    pdf_star = np.exp(-np.linalg.norm(diff_star)**2 / sigma2) * norm.pdf(gamma_star, mu_gamma, np.sqrt(sigma_gamma))

    if gamma_star < 0 or gamma_star > 14 or not np.isfinite(gamma_star):
        accept = 0.0
    else:
        # Avoid division by zero warnings
        accept = min(pdf_star / (pdf_old + np.finfo(float).eps), 1.0)

    urn = np.random.rand()
    accept_result = urn <= accept
    
    if accept_result:
        acc += 1
        gamma = gamma_star
    else:
        rej += 1

    return gamma, accept_result, acc, rej, pvar


def sample_x_l1_Cplx_Operator(X, a, w, sigma2, y, Operator):
    """
    Sample metabolite concentrations (X) and support indicators (Z) 
    using the Bernoulli-Laplace prior distribution.
    """
    M = len(X)
    Z = np.zeros(M)

    for i in range(M):
        ui1 = np.zeros((M, 1))
        ui1[i, 0] = 1

        hi = np.dot(Operator, ui1)
        hi_norm = np.real(np.dot(hi.conj().T, hi))
        hi_norm = max(hi_norm, np.finfo(float).eps)

        X_1i = X.copy()
        X_1i[i] = 0
        
        ei = y - Operator @ X_1i
        hiTei = hi.conj().T @ ei
        eiThi = ei.conj().T @ hi
        
        vari = sigma2 / (2 * hi_norm)
        vari = max(vari, np.finfo(float).eps)
        mui_plus = vari * (np.real(eiThi + hiTei.T) / sigma2 - 1 / a)

        Kplus = (np.sqrt(vari * np.pi / 2)) * (np.sqrt(2 * vari * np.pi)) * (1 / a) * np.exp(mui_plus**2 / (2 * vari)) * (1 - norm.cdf(-mui_plus / np.sqrt(vari)))

        bet = 1 / max(Kplus, np.finfo(float).eps)
        wstar = 1 / (bet / w + 1 - bet)
        
        try:
            choix1 = np.random.binomial(1, wstar)
        except Exception:
            choix1 = 0
            
        X[i] = 0.0
        Z[i] = 0.0
        
        if choix1 == 1:
            X[i] = rtnorm(0, np.inf, mui_plus, np.sqrt(vari))
            Z[i] = 1.0

    return X, Z


def MCMC_ber_laplace_MH_within_Gibbs(Signal, gamma0, H0, Observation, Nit, burn, t, grps_info, first, last):
    """
    Main Gibbs Sampler orchestration loop for Bayesian MRSI quantification.
    """
    alphaw0, alphaw1 = 1.01, 1.01
    a_sigma2, b_sigma2 = 1e-4, 1e-4
    a_a, b_a = 0.001, 0.001

    # Safe handling of dimensions (Prevents index out of bounds error on integers)
    M = Signal.size
    P = Observation.size
    
    VSignal = Signal.flatten()
    VObservation = Observation.flatten()
    
    Chain_a = np.zeros(Nit)
    Chain_sigma2 = np.zeros(Nit)
    Chain_w = np.zeros(Nit)
    Chain_TSignal = np.zeros((M, Nit))
    
    # Structures de stockage après la période de burn-in
    keep_size = Nit - burn
    TSignal_out = np.zeros((M, keep_size))
    TSignal_MAP_out = np.zeros((M, keep_size))
    TH = np.zeros((P, M, keep_size), dtype=complex)
    TZ = np.zeros((M, keep_size))
    Tsigma2 = np.zeros(keep_size)
    Ta = np.zeros(keep_size)
    Tw = np.zeros(keep_size)
    
    Nx = 0
    Chain_Signal = np.zeros(M)

    acc_chain = np.zeros(keep_size) if grps_info == 0 else np.zeros((keep_size, M))
    Chain_gamma = np.zeros(Nit)

    # Initializations
    H = compute_H(H0, gamma0, grps_info, t, 0, 0, 0, 0)
    gamma = gamma0
    acc, rej = 0, 0
    sigma2m, epsilon, phi0, phi1 = 0.0, 0.0, 0.0, 0.0

    for ni in range(Nit):
        coeff1 = np.sum(VSignal != 0) + a_a
        coeff2 = b_a + np.linalg.norm(VSignal, 1)
        a = 1 / gamma_dist.rvs(coeff1, scale=1 / coeff2)
        Chain_a[ni] = a

        w = sample_w(VSignal, alphaw0, alphaw1)
        Chain_w[ni] = w

        c1 = a_sigma2 + P
        error = Observation - np.dot(H, VSignal)
        c2 = b_sigma2 + np.linalg.norm(error.flatten())**2
        sigma2 = 1 / (gamma_dist.rvs(c1, scale=1 / c2) + np.finfo(float).eps)
        Chain_sigma2[ni] = sigma2

        H = compute_H(H0, gamma, grps_info, t, sigma2m, epsilon, phi0, phi1)
        gamma, accept_result, acc, rej, _ = sample_gamma_MH(
            Observation, VSignal, sigma2, H0, H, gamma, grps_info, t, 
            acc, rej, sigma2m, epsilon, phi0, phi1, first, last
        )
        Chain_gamma[ni] = gamma
        H = compute_H(H0, gamma, grps_info, t, sigma2m, epsilon, phi0, phi1)

        VSignal, VZ = sample_x_l1_Cplx_Operator(VSignal, a, w, sigma2, VObservation[first:last], H[first:last, :])
        Chain_TSignal[:, ni] = VSignal

        if ni >= burn and Nx < keep_size:
            TSignal_out[:, Nx] = VSignal
            TH[:, :, Nx] = H
            Tsigma2[Nx] = sigma2
            TZ[:, Nx] = VZ
            Ta[Nx] = a
            Tw[Nx] = w
            
            Chain_Signal = ((Nx * Chain_Signal) + VSignal) / (Nx + 1)
            Signal_MAP = map_sparse(TSignal_out[:, :Nx+1], TZ[:, :Nx+1])
            TSignal_MAP_out[:, Nx] = Signal_MAP
            acc_chain[Nx] = accept_result
            Nx += 1

    # Compute final estimators
    final_signal = Signal_MAP if 'Signal_MAP' in locals() else VSignal
    return (final_signal, np.mean(Chain_gamma), Tsigma2, Ta, Tw, TSignal_out, 
            TSignal_MAP_out, np.mean(Ta), TZ, np.mean(Tw), Chain_gamma, Chain_TSignal, 
            np.zeros(Nit), np.zeros(Nit), np.zeros(Nit), np.zeros(Nit))


def FID2Spec(FID_t: np.ndarray) -> np.ndarray:
    """Convert Free Induction Decay (FID) time signal to Spectrum domain."""
    return np.fft.fftshift(np.fft.fft(FID_t, axis=0))


def forward(params, t, Basis_fid, M, data, first, last):
    """Forward physics model for MRSI signal."""
    t_expanded = t[:, np.newaxis] 
    return FID2Spec((Basis_fid * np.exp(-t_expanded * params[M:M*2])) @ params[:M])[:, np.newaxis]


def objective_function(params, t, Basis_fid, M, data, first, last):
    """Chi-squared objective function for optimization."""
    S = forward(params, t, Basis_fid, M, data, first, last)
    err = (data[first:last, None] - S[first:last])
    return np.real(np.sum(err * np.conj(err)))


def grad(params, t, Basis_fid, M, data, first, last):
    """Gradient calculation vector for Newton-TNC optimization."""
    m = Basis_fid
    n = m.shape[1]    
    con = params[:M]
    gamma = params[M:2*M]
    
    E = np.exp(-(gamma) * t[:, np.newaxis])
    e_term = m * E
    
    c = con[:, np.newaxis]
    Fmet = FID2Spec(e_term)
    Ftmet = FID2Spec(t[:, np.newaxis] * e_term)
    Ftmetc = Ftmet @ c
    Fmetcon = Fmet @ c

    Spec = data[first:last, None]
    S = Fmetcon[first:last]
    
    dSdc = Fmet[first:last, :]
    dSdgamma = -Ftmetc[first:last, :]
    
    dS = np.concatenate((dSdc, dSdgamma), axis=1)
    return np.real(np.sum(S * np.conj(dS) + np.conj(S) * dS - np.conj(Spec) * dS - Spec * np.conj(dS), axis=0))


def l0_norm(vector: np.ndarray) -> int:
    """Compute the L0 pseudo-norm (number of non-zero elements)."""
    return int(np.count_nonzero(vector))


def MSE(ground_truth: np.ndarray, estimations: np.ndarray) -> np.ndarray:
    """Compute Mean Squared Error."""
    return np.mean(np.square(ground_truth - estimations), axis=-1)
