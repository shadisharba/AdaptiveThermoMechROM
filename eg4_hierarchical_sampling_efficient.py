import numpy as np
import numpy.linalg as la
import matplotlib.pyplot as plt
from operator import itemgetter

from interpolate_fluctuation_modes import update_affine_decomposition, update_stress_localization
from microstructures import *
from optimize_alpha import opt1, opt2, opt4, naive, opt4_alphas
from utilities import read_h5, construct_stress_localization, volume_average, plot_and_save, cm, cheap_err_indicator

np.random.seed(0)
file_name, data_path, temp1, temp2 = itemgetter('file_name', 'data_path', 'temp1', 'temp2')(microstructures[4])
print(file_name, '\t', data_path)

n_tests = 100
n_hierarchical_levels = 5
test_temperatures = np.linspace(temp1, temp2, num=n_tests)
test_alphas = np.linspace(0, 1, num=n_tests)

# read reference solutions
mesh, ref = read_h5(file_name, data_path, test_temperatures)
mat_id = mesh['mat_id']
n_gauss = mesh['n_gauss']
n_elements = mesh['n_elements']
strain_dof = mesh['strain_dof']
global_gradient = mesh['global_gradient']
n_gp = mesh['n_integration_points']
n_phases = len(np.unique(mat_id))
n_modes = ref[idx]['strain_localization'].shape[-1]

# extract temperature dependent data from the reference solutions
# such as: material stiffness and thermal strain at each temperature and for all phases
ref_Cs = np.zeros((n_tests, *ref[0]['localization_mat_stiffness'].shape))  # n_tests x n_phases x 6 x 6
ref_epss = np.zeros((n_tests, *ref[0]['localization_mat_thermal_strain'].shape))  # n_tests x n_phases x 6 x 1
effSref = np.zeros((n_tests, strain_dof, n_modes))
for idx, alpha in enumerate(test_alphas):
    Eref = ref[idx]['strain_localization']
    ref_Cs[idx] = ref[idx]['localization_mat_stiffness']
    ref_epss[idx] = ref[idx]['localization_mat_thermal_strain']
    Sref = construct_stress_localization(Eref, ref_Cs[idx], ref_epss[idx], mat_id, n_gauss, strain_dof)
    effSref[idx] = volume_average(Sref)

err_indicators, err_eff_S, err_eff_C, err_eff_eps = [np.zeros((n_hierarchical_levels, n_tests)) for _ in range(4)]
interpolate_temp = lambda x1, x2, alpha: x1 + alpha * (x2 - x1)
err = lambda x, y: la.norm(x - y) / la.norm(y) * 100 if la.norm(y) > 0 else la.norm(x - y)

# alpha_all_levels is initialized with the first level of two samples
alpha_all_levels = [np.linspace(0, 1, num=2)]
given_alpha_levels = False
# alpha_all_levels = np.asarray([
#     np.asarray([0., 1.]),
#     np.asarray([0., 0.80808081, 1.]),
#     np.asarray([0., 0.80808081, 0.92929293, 1.]),
#     np.asarray([0., 0.60606061, 0.80808081, 0.92929293, 1.]),
#     np.asarray([0., 0.60606061, 0.80808081, 0.92929293, 0.96969697, 1.])
# ], dtype=object)
for level in range(n_hierarchical_levels):
    print(f'\n --- {level = :.2f} --- \n')

    # read sampling data given current sampling points. note that samples are reread in the next hierarchical level
    # but as long as everything is stored is h5 & no solvers are called there's no need for optimizing performance here
    alphas = alpha_all_levels[level]
    temperatures = interpolate_temp(temp1, temp2, alphas)
    n_samples = len(alphas)
    _, samples = read_h5(file_name, data_path, temperatures, get_mesh=False)
    # lists that contain quantities from sampling pairs
    E01s, sampling_Cs, sampling_epss = [], [], []
    for id0 in range(n_samples - 1):
        id1 = id0 + 1
        E0 = samples[id0]['strain_localization']
        E1 = samples[id1]['strain_localization']
        E01s.append(np.ascontiguousarray(np.concatenate((E0, E1), axis=-1)))
        # n_samples of [n_phases x 2 x 6 x 6]
        sampling_Cs.append(
            np.stack((samples[id0]['localization_mat_stiffness'], \
                      samples[id1]['localization_mat_stiffness'])).transpose([1, 0, 2, 3]))
        # n_samples of [n_phases x 2 x 6 x 1]
        sampling_epss.append(
            np.stack((samples[id0]['localization_mat_thermal_strain'],
                      samples[id1]['localization_mat_thermal_strain'])).transpose([1, 0, 2, 3]))

    # alphas_indexing will contain the id of each pair of samples needed to solve the problem at a specific temperature
    # temperatures are determined by the values contained in tes_alphas
    alphas_indexing = np.searchsorted(alphas, test_alphas) - 1
    alphas_indexing[0] = 0

    current_sampling_id = None
    K0, K1, F0, F1, F2, F3, S001, S101, S103, S002, S102, S104 = [None for _ in range(12)]

    for idx, alpha in enumerate(test_alphas):
        print(f'{alpha = :.2f}')

        sampling_C = sampling_Cs[alphas_indexing[idx]]
        sampling_eps = sampling_epss[alphas_indexing[idx]]

        # interpolated quantities using an implicit interpolation scheme with four DOF
        alpha_C, alpha_eps = opt4_alphas(sampling_C, sampling_eps, ref_Cs[idx], ref_epss[idx])
        alpha_C_eps = alpha_C * alpha_eps

        # Assemble the linear system only when new samples are considered
        if alphas_indexing[idx] != current_sampling_id:
            current_sampling_id = alphas_indexing[idx]

            K0, K1, F0, F1, F2, F3, S001, S101, S103, S002, S102, S104 = update_affine_decomposition(
                E01s[current_sampling_id], sampling_C, sampling_eps, n_modes, n_phases, n_gp, strain_dof, mat_id, n_gauss,
                quick=True)

        Eopt4, Sopt4 = update_stress_localization(E01s[current_sampling_id], K0, K1, F0, F1, F2, F3, S001, S101, S103, S002, S102,
                                                  S104, alpha_C, alpha_eps, alpha_C_eps, strain_dof, n_modes, n_gp)
        # if not given_alpha_levels:
        #     effSopt = volume_average(Sopt4)
        # else:
        #     effSopt = np.squeeze(Sopt4 / n_gp)

        Sopt4 = construct_stress_localization(Eopt4, ref_Cs[idx], ref_epss[idx], mat_id, n_gauss, strain_dof)

        effSopt = volume_average(Sopt4)
        err_eff_S[level, idx] = err(effSopt, effSref[idx])

        Capprox = effSopt[:6, :6]
        Cref = effSref[idx][:6, :6]
        err_eff_C[level, idx] = err(Capprox, Cref)
        err_eff_eps[level, idx] = err(la.inv(Capprox) @ effSopt[:, -1], la.inv(Cref) @ effSref[idx][:, -1])

    if not given_alpha_levels:
        err_indicators[level, idx] = cheap_err_indicator(Sopt4, global_gradient)
        max_err_idx = np.argmax(err_indicators[level])
        alpha_all_levels.append(np.unique(np.sort(np.hstack((alphas, test_alphas[max_err_idx])))))

idx = [idx for idx, microstructure in enumerate(microstructures) if file_name == microstructure['file_name']][0]
np.savez_compressed(f'output/eg4_{idx}', n_hierarchical_levels=n_hierarchical_levels, test_temperatures=test_temperatures,
                    err_indicators=err_indicators, err_eff_S=err_eff_S, err_eff_C=err_eff_C, err_eff_eps=err_eff_eps,
                    alpha_all_levels=np.asarray(alpha_all_levels, dtype=object))

# %%
import numpy as np
import numpy.linalg as la
import matplotlib.pyplot as plt
from utilities import plot_and_save, cm

loaded_qoi = np.load(f'output/eg4_{idx}.npz', allow_pickle=True)

n_hierarchical_levels = loaded_qoi['n_hierarchical_levels']
test_temperatures = loaded_qoi['test_temperatures']
err_indicators = loaded_qoi['err_indicators']
err_eff_S = loaded_qoi['err_eff_S']
err_eff_C = loaded_qoi['err_eff_C']
err_eff_eps = loaded_qoi['err_eff_eps']
alpha_all_levels = loaded_qoi['alpha_all_levels']

temp1 = test_temperatures[0]
temp2 = test_temperatures[-1]
interpolate_temp = lambda x1, x2, alpha: x1 + alpha * (x2 - x1)

for level in range(n_hierarchical_levels):
    print(f'alphas of level {level}: {alpha_all_levels[level]}')
print('\n')
for level in range(n_hierarchical_levels):
    print(f'temperatures of level {level}: {interpolate_temp(temp1, temp2, alpha_all_levels[level])}')
print('\n')
for level in range(n_hierarchical_levels):
    print(f'level {level}')
    print(f'{np.max(err_indicators[level]) = }')

xlabel = 'Temperature [K]'
styles = ['-', '-', '--', '-.', ':', ':', ':', ':']
markers = ['s', 'd', '+', 'x', 'o']
colors = ['C0', 'C1', 'C2', 'C3', 'C4']
if not given_alpha_levels:
    err_indicators /= np.max(err_indicators)
    ylabel = 'Relative error $e_6$ [\%]'
    fig_name = f'eg4_{idx}_hierarchical_sampling_err_indicator'
    plt.figure(figsize=(6 * cm, 6 * cm), dpi=600)
    for level in range(n_hierarchical_levels):
        plt.plot(test_temperatures, err_indicators[level], label=f'{level + 2} samples', marker=markers[level],
                 color=colors[level], linestyle=styles[level], markevery=8)
    plot_and_save(xlabel, ylabel, fig_name, [temp1, temp2], [0, np.max(err_indicators)], loc='upper left')

ylabel = 'Relative error $e_4$ [\%]'
fig_name = f'eg4_{idx}_hierarchical_sampling_err_eff_S'
plt.figure(figsize=(6 * cm, 6 * cm), dpi=600)
for level in range(n_hierarchical_levels):
    plt.plot(test_temperatures, err_eff_S[level], label=f'{level + 2} samples', marker=markers[level], color=colors[level],
             linestyle=styles[level], markevery=8)
plot_and_save(xlabel, ylabel, fig_name, [temp1, temp2], [0, np.max(err_eff_S)], loc='upper left')

ylabel = 'Relative error $e_7$ [\%]'
fig_name = f'eg4_{idx}_hierarchical_sampling_err_eff_C'
plt.figure(figsize=(6 * cm, 6 * cm), dpi=600)
for level in range(n_hierarchical_levels):
    plt.plot(test_temperatures, err_eff_C[level], label=f'{level + 2} samples', marker=markers[level], color=colors[level],
             linestyle=styles[level], markevery=8)
plot_and_save(xlabel, ylabel, fig_name, [temp1, temp2], [0, np.max(err_eff_C)], loc='upper left')

ylabel = 'Relative error $e_8$ [\%]'
fig_name = f'eg4_{idx}_hierarchical_sampling_err_eff_eps'
plt.figure(figsize=(6 * cm, 6 * cm), dpi=600)
for level in range(n_hierarchical_levels):
    plt.plot(test_temperatures, err_eff_eps[level], label=f'{level + 2} samples', marker=markers[level], color=colors[level],
             linestyle=styles[level], markevery=8)
plot_and_save(xlabel, ylabel, fig_name, [temp1, temp2], [0, np.max(err_eff_eps)], loc='upper left')

# move to the new repo
# can provide a cleaned version of octahedron_rve with localization at the enriched points only
