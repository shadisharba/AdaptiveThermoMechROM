import numpy as np
import numpy.linalg as la
import matplotlib.pyplot as plt
from operator import itemgetter
from microstructures import *
from optimize_alpha import opt1, opt2, opt4, naive
from interpolate_fluctuation_modes import interpolate_fluctuation_modes
from utilities import read_h5, construct_stress_localization, volume_average, plot_and_save, cm, compute_residual, \
    compute_residual_efficient

np.random.seed(0)
file_name, data_path, temp1, temp2 = itemgetter('file_name', 'data_path', 'temp1', 'temp2')(microstructures[0])
print(file_name, '\t', data_path)

n_samples = 2
n_loading_directions = 10
n_tests = 10
temperatures = np.linspace(temp1, temp2, num=n_samples)
alphas = np.linspace(0, 1, num=n_samples)
test_temperatures = np.linspace(temp1, temp2, num=n_tests)
test_alphas = np.linspace(0, 1, num=n_tests)

mesh, ref = read_h5(file_name, data_path, test_temperatures)
_, samples = read_h5(file_name, data_path, temperatures)
mat_id = mesh['mat_id']
n_gauss = mesh['n_gauss']
n_elements = mesh['n_elements']

strains = np.random.normal(size=(n_loading_directions, mesh['strain_dof']))
strains /= la.norm(strains, axis=1)[:, None]

n_approaches = 5
err_strain_localization, err_stress_localization, err_eff_stiffness = [np.zeros((n_approaches, n_tests)) for _ in range(3)]
err_eff_stress, err_nodal_force = [np.zeros((n_approaches, n_tests * n_loading_directions)) for _ in range(2)]

for idx, alpha in enumerate(test_alphas):
    print(f'{alpha = :.2f}')
    temperature = test_temperatures[idx]

    interpolate_temp = lambda x1, x2: x1 + alpha * (x2 - x1)

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

    # interpolated quantities using an explicit interpolation scheme with one DOF
    approx_C, approx_eps = naive(alpha, sampling_C, sampling_eps, ref_C, ref_eps)
    Enaive = interpolate_temp(E0, E1)
    Snaive = construct_stress_localization(Enaive, ref_C, ref_eps, mesh['mat_id'], mesh['n_gauss'], mesh['strain_dof'])
    effCnaive = volume_average(Snaive)

    # interpolated quantities using an explicit interpolation scheme with one DOF
    Eopt0 = interpolate_fluctuation_modes(E01, approx_C, approx_eps, mesh['mat_id'], mesh['n_gauss'], mesh['strain_dof'])
    Sopt0 = construct_stress_localization(Eopt0, ref_C, ref_eps, mesh['mat_id'], mesh['n_gauss'], mesh['strain_dof'])
    effCopt0 = volume_average(Sopt0)

    # interpolated quantities using an implicit interpolation scheme with one DOF
    approx_C, approx_eps = opt1(sampling_C, sampling_eps, ref_C, ref_eps)
    Eopt1 = interpolate_fluctuation_modes(E01, approx_C, approx_eps, mesh['mat_id'], mesh['n_gauss'], mesh['strain_dof'])
    Sopt1 = construct_stress_localization(Eopt1, ref_C, ref_eps, mesh['mat_id'], mesh['n_gauss'], mesh['strain_dof'])
    effCopt1 = volume_average(Sopt1)

    # interpolated quantities using an implicit interpolation scheme with two DOF
    approx_C, approx_eps = opt2(sampling_C, sampling_eps, ref_C, ref_eps)
    Eopt2 = interpolate_fluctuation_modes(E01, approx_C, approx_eps, mesh['mat_id'], mesh['n_gauss'], mesh['strain_dof'])
    Sopt2 = construct_stress_localization(Eopt2, ref_C, ref_eps, mesh['mat_id'], mesh['n_gauss'], mesh['strain_dof'])
    effCopt2 = volume_average(Sopt2)

    # interpolated quantities using an implicit interpolation scheme with four DOF
    approx_C, approx_eps = opt4(sampling_C, sampling_eps, ref_C, ref_eps)
    Eopt4 = interpolate_fluctuation_modes(E01, approx_C, approx_eps, mesh['mat_id'], mesh['n_gauss'], mesh['strain_dof'])
    Sopt4 = construct_stress_localization(Eopt4, ref_C, ref_eps, mesh['mat_id'], mesh['n_gauss'], mesh['strain_dof'])
    effCopt4 = volume_average(Sopt4)

    err = lambda x, y: np.mean(la.norm(x - y, axis=(-1, -2)) / la.norm(y, axis=(-1, -2))) * 100
    err_vec = lambda x, y: np.mean(la.norm(x - y, axis=(-1)) / la.norm(y, axis=(-1))) * 100

    err_strain_localization[:, idx] = [err(Enaive, Eref), err(Eopt0, Eref), err(Eopt1, Eref), err(Eopt2, Eref), err(Eopt4, Eref)]
    err_stress_localization[:, idx] = [err(Snaive, Sref), err(Sopt0, Sref), err(Sopt1, Sref), err(Sopt2, Sref), err(Sopt4, Sref)]
    err_eff_stiffness[:, idx] = [
        err(effCnaive, effCref),
        err(effCopt0, effCref),
        err(effCopt1, effCref),
        err(effCopt2, effCref),
        err(effCopt4, effCref)
    ]

    for strain_idx, strain in enumerate(strains):
        zeta = np.hstack((strain, 1))

        eff_stress_ref = effCref @ zeta
        eff_stress_naive = effCnaive @ zeta
        eff_stress_opt0 = effCopt0 @ zeta
        eff_stress_opt1 = effCopt1 @ zeta
        eff_stress_opt2 = effCopt2 @ zeta
        eff_stress_opt4 = effCopt4 @ zeta
        err_eff_stress[:, idx * n_loading_directions + strain_idx] = [
            err_vec(eff_stress_naive, eff_stress_ref),
            err_vec(eff_stress_opt0, eff_stress_ref),
            err_vec(eff_stress_opt1, eff_stress_ref),
            err_vec(eff_stress_opt2, eff_stress_ref),
            err_vec(eff_stress_opt4, eff_stress_ref)
        ]

        stress_naive = np.einsum('ijk,k', Snaive, zeta, optimize='optimal')
        stress_opt0 = np.einsum('ijk,k', Sopt0, zeta, optimize='optimal')
        stress_opt1 = np.einsum('ijk,k', Sopt1, zeta, optimize='optimal')
        stress_opt2 = np.einsum('ijk,k', Sopt2, zeta, optimize='optimal')
        stress_opt4 = np.einsum('ijk,k', Sopt4, zeta, optimize='optimal')

        residuals = compute_residual_efficient([stress_naive, stress_opt0, stress_opt1, stress_opt2, stress_opt4],
                                               mesh['global_gradient'])

        err_nodal_force[:, idx * n_loading_directions +
                        strain_idx] = la.norm(residuals, np.inf, axis=0) / normalization_factor_mech * 100

np.savez_compressed('output/eg2', err_nodal_force=err_nodal_force, err_eff_stiffness=err_eff_stiffness,
                    err_eff_stress=err_eff_stress, err_strain_localization=err_strain_localization,
                    err_stress_localization=err_stress_localization, n_loading_directions=n_loading_directions,
                    n_approaches=n_approaches)
# %%
import numpy as np
import numpy.linalg as la
import matplotlib.pyplot as plt
from utilities import plot_and_save, cm, ecdf

loaded_qoi = np.load('output/eg2.npz')
err_nodal_force = loaded_qoi['err_nodal_force']
err_eff_stiffness = loaded_qoi['err_eff_stiffness']
err_eff_stress = loaded_qoi['err_eff_stress']
err_strain_localization = loaded_qoi['err_strain_localization']
err_stress_localization = loaded_qoi['err_stress_localization']
n_loading_directions = loaded_qoi['n_loading_directions']
n_approaches = loaded_qoi['n_approaches']
markevery = 8 * n_loading_directions

xlabel = '$x$ [\%]'
labels = ['N', r'O$_0$', 'O$_1$', 'O$_2$', 'O$_4$']
markers = ['s', 'd', '+', 'x', 'o']
colors = ['C0', 'C1', 'C2', 'C3', 'C4']

fig_name = 'eg2_err_strain_localization'
ylabel = '$P(e_2<x)$ [-]'
fig, ax = plt.subplots(figsize=(6 * cm, 6 * cm), dpi=600)
axins = ax.inset_axes([0.1, 0.65, 0.2, 0.3])
for idx in range(5):
    plt.step(*ecdf(err_strain_localization[idx]), label=labels[idx], marker=markers[idx], color=colors[idx], markevery=8)
    axins.step(*ecdf(err_strain_localization[idx]), label=labels[idx], marker=markers[idx], color=colors[idx], markevery=8)
axins.set_xlim(-0.05, 0.5)
axins.set_ylim(0.6, 1)
axins.get_yaxis().set_visible(False)
ax.indicate_inset_zoom(axins, facecolor=(0.8, 0.8, 0.8, 0.6), edgecolor=(0.3, 0.3, 0.3, 0.3), lw=0.1)
plot_and_save(xlabel, ylabel, fig_name, [-0.1, np.max(err_strain_localization) * 1.1], [0, 1], loc='lower right')

fig_name = 'eg2_err_stress_localization'
ylabel = '$P(e_3<x)$ [-]'
fig, ax = plt.subplots(figsize=(6 * cm, 6 * cm), dpi=600)
axins = ax.inset_axes([0.1, 0.65, 0.2, 0.3])
for idx in range(5):
    plt.step(*ecdf(err_stress_localization[idx]), label=labels[idx], marker=markers[idx], color=colors[idx], markevery=8)
    axins.step(*ecdf(err_stress_localization[idx]), label=labels[idx], marker=markers[idx], color=colors[idx], markevery=8)
axins.set_xlim(-0.05, 0.2)
axins.set_ylim(0.6, 1)
axins.get_yaxis().set_visible(False)
ax.indicate_inset_zoom(axins, facecolor=(0.8, 0.8, 0.8, 0.6), edgecolor=(0.3, 0.3, 0.3, 0.3), lw=0.1)
plot_and_save(xlabel, ylabel, fig_name, [-0.1, np.max(err_stress_localization) * 1.1], [0, 1], loc='lower right')

fig_name = 'eg2_err_eff_stiffness'
ylabel = '$P(e_4<x)$ [-]'
fig, ax = plt.subplots(figsize=(6 * cm, 6 * cm), dpi=600)
axins = ax.inset_axes([0.1, 0.65, 0.2, 0.3])
for idx in range(5):
    plt.step(*ecdf(err_eff_stiffness[idx]), label=labels[idx], marker=markers[idx], color=colors[idx], markevery=8)
    axins.step(*ecdf(err_eff_stiffness[idx]), label=labels[idx], marker=markers[idx], color=colors[idx], markevery=8)
axins.set_xlim(-0.05, 0.2)
axins.set_ylim(0.6, 1)
axins.get_yaxis().set_visible(False)
ax.indicate_inset_zoom(axins, facecolor=(0.8, 0.8, 0.8, 0.6), edgecolor=(0.3, 0.3, 0.3, 0.3), lw=0.1)
plot_and_save(xlabel, ylabel, fig_name, [-0.1, np.max(err_eff_stiffness) * 1.1], [0, 1], loc='lower right')

fig_name = 'eg2_err_eff_stress'
ylabel = '$P(e_5<x)$ [-]'
fig, ax = plt.subplots(figsize=(6 * cm, 6 * cm), dpi=600)
axins = ax.inset_axes([0.45, 0.15, 0.2, 0.3])
for idx in range(5):
    plt.step(*ecdf(err_eff_stress[idx]), label=labels[idx], marker=markers[idx], color=colors[idx], markevery=markevery)
    axins.step(*ecdf(err_eff_stress[idx]), label=labels[idx], marker=markers[idx], color=colors[idx], markevery=markevery)
axins.set_xlim(-0.05, 0.2)
axins.set_ylim(0.6, 1)
ax.indicate_inset_zoom(axins, facecolor=(0.8, 0.8, 0.8, 0.6), edgecolor=(0.3, 0.3, 0.3, 0.3), lw=0.1)
plot_and_save(xlabel, ylabel, fig_name, [-0.1, np.max(err_eff_stress) * 1.1], [0, 1], loc='lower right')

fig_name = 'eg2_err_nodal_force'
ylabel = '$P(e_1<x)$ [-]'
fig, ax = plt.subplots(figsize=(6 * cm, 6 * cm), dpi=600)
axins = ax.inset_axes([0.45, 0.15, 0.2, 0.3])
for idx in range(n_approaches):
    x, y = ecdf(err_nodal_force[idx])
    print(f'err_nodal_force {np.max(x[y<=0.99]) = :2.2e}')
    plt.step(x, y, label=labels[idx], marker=markers[idx], color=colors[idx], markevery=markevery)
    axins.step(x, y, label=labels[idx], marker=markers[idx], color=colors[idx], markevery=markevery)
axins.set_xlim(-0.05, 0.5)
axins.set_ylim(0.6, 1)
ax.indicate_inset_zoom(axins, facecolor=(0.8, 0.8, 0.8, 0.6), edgecolor=(0.3, 0.3, 0.3, 0.3), lw=0.1)
plot_and_save(xlabel, ylabel, fig_name, [-0.1, np.max(err_nodal_force) * 1.1], [0, 1], loc='lower right')

with np.printoptions(precision=4, suppress=True, formatter={'float': '{:>2.2e}'.format}, linewidth=100):
    print(f'{np.max(err_nodal_force,axis=1) = }')
    print(f'{np.max(err_strain_localization,axis=1) = }')
    print(f'{np.max(err_stress_localization,axis=1) = }')
    print(f'{np.max(err_eff_stiffness,axis=1) = }')
    print(f'{np.max(err_eff_stress,axis=1) = }')
