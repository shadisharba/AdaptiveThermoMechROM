import numpy as np
import numpy.linalg as la
import matplotlib.pyplot as plt
from material_parameters import *
from optimize_alpha import naive, opt1, opt2, opt4
from utilities import plot_and_save, cm

temp1 = 293
temp2 = 1300
n_tests = 25
test_temperatures = np.linspace(temp1, temp2, num=n_tests)
test_alphas = np.linspace(0, 1, num=n_tests)
n_approaches = 5

# TODO new error indicator
err_indic = np.zeros((n_tests))

for mat_id in range(2):
    sampling_C = [[stiffness_cu(temp1), stiffness_cu(temp2)], [stiffness_wsc(temp1), stiffness_wsc(temp2)]]
    sampling_eps = [[thermal_strain_cu(temp1), thermal_strain_cu(temp2)], [thermal_strain_wsc(temp1), thermal_strain_wsc(temp2)]]
    max_eig_value, trace_eps = [np.zeros((n_approaches, n_tests)) for _ in range(2)]

    for idx, alpha in enumerate(test_alphas):
        print(f'{alpha = :.2f}')
        temperature = test_temperatures[idx]

        interpolate_temp = lambda x1, x2: x1 + alpha * (x2 - x1)

        # reference values
        ref_C = [stiffness_cu(temperature), stiffness_wsc(temperature)]
        ref_eps = [thermal_strain_cu(temperature), thermal_strain_wsc(temperature)]
        ref_max_eig_value = np.max(la.eigvalsh(ref_C[mat_id]))
        ref_trace_eps = np.sum(ref_eps[mat_id][:3])

        # interpolated quantities using the explicit temperature interpolation scheme
        approx_C, approx_eps = naive(alpha, sampling_C, sampling_eps, ref_C, ref_eps)
        naive_max_eig_value = np.max(la.eigvalsh(approx_C[mat_id]))
        naive_trace_eps = np.sum(approx_eps[mat_id][:3])

        # interpolated quantities using an implicit interpolation scheme with one DOF
        approx_C, approx_eps = opt1(sampling_C, sampling_eps, ref_C, ref_eps)
        opt1_max_eig_value = np.max(la.eigvalsh(approx_C[mat_id]))
        opt1_trace_eps = np.sum(approx_eps[mat_id][:3])

        # interpolated quantities using an implicit interpolation scheme with two DOF
        approx_C, approx_eps = opt2(sampling_C, sampling_eps, ref_C, ref_eps)
        opt2_max_eig_value = np.max(la.eigvalsh(approx_C[mat_id]))
        opt2_trace_eps = np.sum(approx_eps[mat_id][:3])

        # interpolated quantities using an implicit interpolation scheme with four DOF
        approx_C, approx_eps = opt4(sampling_C, sampling_eps, ref_C, ref_eps)
        opt4_max_eig_value = np.max(la.eigvalsh(approx_C[mat_id]))
        opt4_trace_eps = np.sum(approx_eps[mat_id][:3])

        max_eig_value[0, idx] = ref_max_eig_value
        max_eig_value[1, idx] = naive_max_eig_value
        max_eig_value[2, idx] = opt1_max_eig_value
        max_eig_value[3, idx] = opt2_max_eig_value
        max_eig_value[4, idx] = opt4_max_eig_value

        trace_eps[0, idx] = ref_trace_eps
        trace_eps[1, idx] = naive_trace_eps
        trace_eps[2, idx] = opt1_trace_eps
        trace_eps[3, idx] = opt2_trace_eps
        trace_eps[4, idx] = opt4_trace_eps

        # TODO clean
        approx_C, approx_eps = naive(alpha, sampling_C, sampling_eps, ref_C, ref_eps)
        err = lambda x, y: la.norm(x - y) / (la.norm(y) if la.norm(y) > 0 else 1)
        # err_indic[idx] += err(la.eigvalsh(approx_C[mat_id]), la.eigvalsh(ref_C[mat_id])) + err(opt4_trace_eps, ref_trace_eps)
        # err_indic[idx] += err(opt4_max_eig_value, ref_max_eig_value) + err(opt4_trace_eps, ref_trace_eps)
        # err_indic[idx] += err(opt4_max_eig_value, ref_max_eig_value)
        err_indic[idx] += err(np.sort(la.eigvalsh(approx_C[mat_id])), np.sort(la.eigvalsh(ref_C[mat_id])))
        # err_indic[idx] += err(naive_trace_eps, ref_trace_eps)
        # err_indic[idx] += 0.7 * err(np.sort(la.eigvalsh(approx_C[mat_id])), np.sort(la.eigvalsh(ref_C[mat_id]))) + 0.3 * err(
        #     naive_trace_eps, ref_trace_eps)

    max_eig_value /= np.max(max_eig_value)
    trace_eps /= np.max(trace_eps)

    fig_name = f'eg1_max_eig{mat_id}'
    xlabel = 'Temperature [K]'
    ylabel = 'Normalized max($\lambda(\mathbb{C})$) [-]'
    labels = ['R', 'O$_0$', 'O$_1$', 'O$_2$', 'O$_4$']
    markers = ['s', 'd', '+', 'x', 'o']
    colors = ['C0', 'C1', 'C2', 'C3', 'C4']

    plt.figure(figsize=(6 * cm, 6 * cm), dpi=600)
    for idx in range(n_approaches):
        plt.plot(test_temperatures, max_eig_value[idx], label=labels[idx], marker=markers[idx], color=colors[idx])
    plot_and_save(xlabel, ylabel, fig_name, [temp1, temp2], [None if mat_id else 0, 1])

    fig_name = f'eg1_tr_thermal_strain{mat_id}'
    xlabel = 'Temperature [K]'
    ylabel = r'Normalized tr($\boldsymbol{\varepsilon}_\uptheta$) [-]'

    plt.figure(figsize=(6 * cm, 6 * cm), dpi=600)
    for idx in range(n_approaches):
        plt.plot(test_temperatures, trace_eps[idx], label=labels[idx], marker=markers[idx], color=colors[idx])
    plot_and_save(xlabel, ylabel, fig_name, [temp1, temp2], [0, 1])

fig_name = f'eg1_cumulative_error_indicator'
xlabel = 'Temperature [K]'
ylabel = r'Cumulative error indicator [-]'
plt.figure(figsize=(6 * cm, 6 * cm), dpi=600)
plt.plot(test_temperatures, err_indic, label=labels[0], marker=markers[0], color=colors[0])
plot_and_save(xlabel, ylabel, fig_name, [temp1, temp2], [0, None])
