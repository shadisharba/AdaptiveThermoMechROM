import numpy as np
import numpy.linalg as la
import matplotlib.pyplot as plt
from operator import itemgetter
from microstructures import *
from optimize_alpha import opt1, opt2, opt4, naive
from interpolate_fluctuation_modes import interpolate_fluctuation_modes
from utilities import read_h5, construct_stress_localization, volume_average, plot_and_save, cm, compute_err_indicator, \
    compute_residual, compute_residual_efficient, compute_err_indicator_efficient

np.random.seed(0)
file_name, data_path, temp1, temp2 = itemgetter('file_name', 'data_path', 'temp1', 'temp2')(microstructures[3])
print(file_name, '\t', data_path)

n_loading_directions = 10
n_tests = 10
n_hierarchical_levels = 5
test_temperatures = np.linspace(temp1, temp2, num=n_tests)
test_alphas = np.linspace(0, 1, num=n_tests)

mesh, ref = read_h5(file_name, data_path, test_temperatures)
mat_id = mesh['mat_id']
n_gauss = mesh['n_gauss']
n_elements = mesh['n_elements']

strains = np.random.normal(size=(n_loading_directions, mesh['strain_dof']))
strains /= la.norm(strains, axis=1)[:, None]

err_nodal_force = np.zeros((n_hierarchical_levels, n_tests, n_loading_directions))
err_indicators = np.zeros((n_hierarchical_levels, n_tests))
err_eff_stiffness = np.zeros((n_hierarchical_levels, n_tests))
alpha_levels = [np.linspace(0, 1, num=2)]

for level in range(n_hierarchical_levels):
    print(f'\n --- {level = :.2f} --- \n')
    alphas = alpha_levels[level]
    interpolate_temp = lambda x1, x2, alpha: x1 + alpha * (x2 - x1)
    temperatures = interpolate_temp(temp1, temp2, alphas)
    _, samples = read_h5(file_name, data_path, temperatures)
    for idx, alpha in enumerate(test_alphas):
        print(f'{alpha = :.2f}')
        temperature = test_temperatures[idx]

        upper_bound = np.where(alphas >= alpha)[0][0]
        id1 = upper_bound if upper_bound > 0 else 1
        id0 = id1 - 1

        E0 = samples[id0]['strain_localization']
        E1 = samples[id1]['strain_localization']
        E01 = np.ascontiguousarray(np.concatenate((E0, E1), axis=-1))

        sampling_C = np.stack(
            (samples[id0]['localization_mat_stiffness'], samples[id1]['localization_mat_stiffness'])).transpose([1, 0, 2, 3])
        sampling_eps = np.stack((samples[id0]['localization_mat_thermal_strain'],
                                 samples[id1]['localization_mat_thermal_strain'])).transpose([1, 0, 2, 3])

        # reference values
        Eref = ref[idx]['strain_localization']
        ref_C = ref[idx]['localization_mat_stiffness']
        ref_eps = ref[idx]['localization_mat_thermal_strain']
        normalization_factor_mech = ref[idx]['normalization_factor_mech']

        Sref = construct_stress_localization(Eref, ref_C, ref_eps, mesh['mat_id'], mesh['n_gauss'], mesh['strain_dof'])
        effCref = volume_average(Sref)

        # interpolated quantities using an implicit interpolation scheme with four DOF
        approx_C, approx_eps = opt4(sampling_C, sampling_eps, ref_C, ref_eps)
        Eopt4 = interpolate_fluctuation_modes(E01, approx_C, approx_eps, mesh['mat_id'], mesh['n_gauss'], mesh['strain_dof'])
        Sopt4 = construct_stress_localization(Eopt4, ref_C, ref_eps, mesh['mat_id'], mesh['n_gauss'], mesh['strain_dof'])
        effCopt4 = volume_average(Sopt4)

        err_indicators[level,
                       idx] = np.mean(np.max(np.abs(compute_err_indicator_efficient(Sopt4, mesh['global_gradient'])),
                                             axis=0)) / normalization_factor_mech * 100

        for strain_idx, strain in enumerate(strains):
            zeta = np.hstack((strain, 1))
            stress_opt4 = np.einsum('ijk,k', Sopt4, zeta, optimize='optimal')
            residual = compute_residual_efficient(stress_opt4, mesh['global_gradient'])

            err_nodal_force[level, idx, strain_idx] = la.norm(residual, np.inf) / normalization_factor_mech * 100

        err = lambda x, y: np.mean(la.norm(x - y) / la.norm(y)) * 100
        err_eff_stiffness[level, idx] = err(effCopt4, effCref)

    # max_err_idx = np.argmax(np.mean(err_nodal_force[level], axis=1))
    max_err_idx = np.argmax(err_indicators[level])
    alpha_levels.append(np.sort(np.hstack((alphas, test_alphas[max_err_idx]))))
    print(f'{np.max(np.mean(err_nodal_force[level], axis=1)) = }')
    print(f'{np.max(err_indicators[level]) = }')

np.savez_compressed('output/eg3', n_hierarchical_levels=n_hierarchical_levels, test_temperatures=test_temperatures,
                    err_nodal_force=err_nodal_force, err_indicators=err_indicators, err_eff_stiffness=err_eff_stiffness,
                    alpha_levels=np.asarray(alpha_levels, dtype=object))

# %%
import numpy as np
import numpy.linalg as la
import matplotlib.pyplot as plt
from utilities import plot_and_save, cm

loaded_qoi = np.load('output/eg3.npz', allow_pickle=True)
n_hierarchical_levels = loaded_qoi['n_hierarchical_levels']
test_temperatures = loaded_qoi['test_temperatures']
err_nodal_force = loaded_qoi['err_nodal_force']
err_indicators = loaded_qoi['err_indicators']
err_eff_stiffness = loaded_qoi['err_eff_stiffness']
alpha_levels = loaded_qoi['alpha_levels']

temp1 = test_temperatures[0]
temp2 = test_temperatures[-1]
interpolate_temp = lambda x1, x2, alpha: x1 + alpha * (x2 - x1)

for level in range(n_hierarchical_levels):
    print(f'alphas of level {level}: {alpha_levels[level]}')
print('\n')
for level in range(n_hierarchical_levels):
    print(f'temperatures of level {level}: {interpolate_temp(temp1, temp2, alpha_levels[level])}')
print('\n')
for level in range(n_hierarchical_levels):
    print(f'level {level}')
    print(f'{np.max(np.mean(err_nodal_force[level], axis=1)) = }')
    print(f'{np.max(err_indicators[level]) = }')

xlabel = 'Temperature [K]'
markers = ['s', 'd', '+', 'x', 'o']
colors = ['C0', 'C1', 'C2', 'C3', 'C4']

fig_name = 'eg3_hierarchical_sampling_err_nodal_force'
ylabel = 'Relative error $e_1$ [\%]'
plt.figure(figsize=(6 * cm, 6 * cm), dpi=600)
for level in range(n_hierarchical_levels):
    plt.plot(test_temperatures, np.mean(err_nodal_force[level], axis=1), label=f'{level + 2} samples', marker=markers[level],
             color=colors[level], markevery=8)
plot_and_save(xlabel, ylabel, fig_name, [temp1, temp2], [0, np.max(np.mean(err_nodal_force, axis=-1))], loc='upper left')

fig_name = 'eg3_hierarchical_sampling_err_indicator'
ylabel = 'Relative error $e_6$ [\%]'
plt.figure(figsize=(6 * cm, 6 * cm), dpi=600)
for level in range(n_hierarchical_levels):
    plt.plot(test_temperatures, err_indicators[level], label=f'{level + 2} samples', marker=markers[level], color=colors[level],
             markevery=8)
plot_and_save(xlabel, ylabel, fig_name, [temp1, temp2], [0, np.max(err_indicators)], loc='upper left')

# styles = ['-', '-', '--', '-.', ':', ':', ':', ':']
fig_name = 'eg3_hierarchical_sampling_err_nodal_vs_indicator'
ylabel = 'Normalized error [-]'
err_indicators /= np.max(err_indicators)
err_nodal_force_mat = np.mean(err_nodal_force, axis=-1)
err_nodal_force_mat /= np.max(err_nodal_force_mat)
plt.figure(figsize=(10 * cm, 6 * cm), dpi=600)
for level in range(n_hierarchical_levels):
    plt.plot(test_temperatures, err_nodal_force_mat[level], label=rf'$e_1$ {level + 2} samples', marker=markers[level],
             color=colors[level], linestyle='-', markevery=8)
    plt.plot(test_temperatures, err_indicators[level], label=rf'$e_6$ {level + 2} samples', marker=markers[level],
             color=colors[level], linestyle=':', markevery=8)
plot_and_save(xlabel, ylabel, fig_name, [temp1, temp2], [0, np.max(err_indicators)], loc='upper left')

fig_name = 'eg3_hierarchical_sampling_err_eff_stiffness'
ylabel = 'Relative error $e_4$ [\%]'
plt.figure(figsize=(6 * cm, 6 * cm), dpi=600)
for level in range(n_hierarchical_levels):
    plt.plot(test_temperatures, err_eff_stiffness[level], label=f'{level + 2} samples', marker=markers[level],
             color=colors[level], markevery=8)
plot_and_save(xlabel, ylabel, fig_name, [temp1, temp2], [0, np.max(err_eff_stiffness)], loc='upper left')
