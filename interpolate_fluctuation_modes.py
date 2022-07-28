import numpy as np
from numba import jit, prange

@jit(nopython=True, cache=True, parallel=True, fastmath=True, nogil=True)
def interpolate_fluctuation_modes(E01, mat_stiffness, mat_thermal_strain, mat_id, n_gauss, strain_dof, n_modes, n_gp):
    K = np.zeros((2 * n_modes, 2 * n_modes))
    F = np.zeros((2 * n_modes, n_modes))
    E_approx = np.zeros((n_gp, strain_dof, n_modes))

    I = np.eye(strain_dof)
    for gp_id in prange(n_gp):

        phase_id = mat_id[gp_id // n_gauss]

        P = np.hstack((-I, mat_thermal_strain[phase_id]))

        E01t_C = E01[gp_id].T @ mat_stiffness[phase_id]

        K += E01t_C @ E01[gp_id]
        F += E01t_C @ P

    phi = np.linalg.lstsq(K, F)[0]

    for gp_id in prange(n_gp):
        E_approx[gp_id] = E01[gp_id] @ phi

    return E_approx

# similar assembly procedure that incorporates interpolation parameters
# E01t_C0 = E01[gp_id].T @ (C0[phase_id] + Cd[phase_id] * alpha_C[phase_id])
# K += E01t_C0 @ E01[gp_id]
# F += E01t_C0 @ np.hstack((-I, eps0[phase_id] + epsd[phase_id] * alpha_eps[phase_id]))

def update_affine_decomposition(E01, sampling_C, sampling_eps, n_modes, n_phases, n_gp, strain_dof, mat_id, n_gauss, quick=False):
    I = np.eye(strain_dof)

    K0 = np.zeros((2 * n_modes, 2 * n_modes))
    K1 = np.zeros((n_phases, 2 * n_modes, 2 * n_modes))
    F0 = np.zeros((2 * n_modes, n_modes))
    F1 = np.zeros((n_phases, 2 * n_modes, n_modes))
    F2, F3 = [np.zeros((n_phases, 2 * n_modes, 1)) for _ in range(2)]

    dim0 = 1 if quick else n_gp
    S001 = np.zeros((dim0, strain_dof, 2 * n_modes))
    S002 = np.zeros((dim0, n_phases, strain_dof, 2 * n_modes))
    S101 = np.zeros((dim0, strain_dof, n_modes))
    S102 = np.zeros((dim0, n_phases, strain_dof, n_modes))
    S103 = np.zeros((dim0, n_phases, strain_dof, 1))
    S104 = np.zeros((dim0, n_phases, strain_dof, 1))

    # extract C0 and deltaC such that C_approx = C0 + alpha * deltaC
    C0 = sampling_C[:, 0]
    deltaC = sampling_C[:, 1] - sampling_C[:, 0]
    eps0 = sampling_eps[:, 0]
    delta_eps = sampling_eps[:, 1] - sampling_eps[:, 0]

    for gp_id in range(n_gp):
        phase_id = mat_id[gp_id // n_gauss]
        C0phase = C0[phase_id]
        eps0phase = eps0[phase_id]
        deltaCphase = deltaC[phase_id]
        deltaEPSphase = delta_eps[phase_id]
        P0_phase = np.hstack((-I, eps0phase))
        E01_transposed = E01[gp_id].T

        # essential terms that construct all other precomputed terms
        pre_k0 = C0phase @ E01[gp_id]
        pre_k1 = deltaCphase @ E01[gp_id]
        pre_f0 = C0phase @ P0_phase
        pre_f1 = deltaCphase @ P0_phase
        pre_f2 = C0phase @ deltaEPSphase
        pre_f3 = deltaCphase @ deltaEPSphase

        K0 += E01_transposed @ pre_k0
        K1[phase_id] += E01_transposed @ pre_k1

        F0 += E01_transposed @ pre_f0
        F1[phase_id] += E01_transposed @ pre_f1
        F2[phase_id] += E01_transposed @ pre_f2
        F3[phase_id] += E01_transposed @ pre_f3

        idx0 = 0 if quick else gp_id
        S001[idx0] += pre_k0
        S101[idx0] += pre_f0

        S002[idx0, phase_id] += pre_k1
        S102[idx0, phase_id] += pre_f1
        S103[idx0, phase_id] += pre_f2
        S104[idx0, phase_id] += pre_f3

    return K0, K1, F0, F1, F2, F3, S001, S101, S103, S002, S102, S104

# @jit(nopython=True, cache=True, parallel=True, fastmath=True, nogil=True)
def update_stress_localization(E01, K0, K1, F0, F1, F2, F3, S001, S101, S103, S002, S102, S104, alpha_C, alpha_eps, alpha_C_eps,
                               strain_dof, n_modes, n_gp):
    K = K0 + np.sum(K1 * alpha_C, 0)  # sum over the phases
    F = F0 + np.sum(F1 * alpha_C, 0)
    F[:, -1:] += np.sum(F2 * alpha_eps + F3 * alpha_C_eps, 0)

    phi = np.linalg.lstsq(K, F, rcond=None)[0]
    # phi = np.linalg.lstsq(K, F)[0]

    E_approx = np.zeros((n_gp, strain_dof, n_modes))
    for gp_id in prange(n_gp):
        E_approx[gp_id] = E01[gp_id] @ phi

    S_approx = 0
    # S00 = S001 + np.sum(S002 * alpha_C, 1)  # sum over the phases
    # S11 = S101 + np.sum(S102 * alpha_C, 1)
    # S11[..., -1:] += np.sum(S103 * alpha_eps + S104 * alpha_C_eps, 1)
    # S_approx = S00 @ phi - S11

    return E_approx, S_approx
