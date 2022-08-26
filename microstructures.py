import numpy as np
from pathlib import Path as path

microstructures = [{
    'data_path': '/ms_1p/dset0_sim',
    'file_name': path("input/striped_normal_4x4x4.h5"),
    'temp1': 300,
    'temp2': 1300,
    'n_tests': 100,
    'sampling_alphas': None
}, {
    'data_path': '/ms_1p/dset0_sim',
    'file_name': path("input/sphere_normal_16x16x16_10samples.h5"),
    'temp1': 300,
    'temp2': 1300,
    'n_tests': 10,
    'sampling_alphas': None
}, {
    'data_path': '/ms_1p/dset0_sim',
    'file_name': path("input/sphere_normal_32x32x32_10samples.h5"),
    'temp1': 300,
    'temp2': 1300,
    'n_tests': 10,
    'sampling_alphas': None
}, {
    'data_path': '/ms_1p/dset0_sim',
    'file_name': path("input/sphere_combo_16x16x16_10samples.h5"),
    'temp1': 300,
    'temp2': 1300,
    'n_tests': 10,
    'sampling_alphas': None
}, {
    'data_path': '/ms_1p/dset0_sim',
    'file_name': path("input/octahedron_normal_16x16x16_10samples.h5"),
    'temp1': 300,
    'temp2': 1300,
    'n_tests': 10,
    'sampling_alphas': None
}, {
    'data_path': '/ms_1p/dset0_sim',
    'file_name': path("input/octahedron_combo_16x16x16_10samples.h5"),
    'temp1': 300,
    'temp2': 1300,
    'n_tests': 10,
    'sampling_alphas': None
}, {
    'data_path':
    '/ms_1p/dset0_sim',
    'file_name':
    path("input/octahedron_combo_32x32x32.h5"),
    'temp1':
    300,
    'temp2':
    1300,
    'n_tests':
    100,
    'sampling_alphas':
    np.asarray([
        np.asarray([0., 1.]),
        np.asarray([0., 0.82828283, 1.]),
        np.asarray([0., 0.82828283, 0.93939394, 1.]),
        np.asarray([0., 0.60606061, 0.82828283, 0.93939394, 1.]),
        np.asarray([0., 0.60606061, 0.82828283, 0.93939394, 0.97979798, 1.])
    ], dtype=object)
}, {
    'data_path':
    '/image_data/dset_0_sim',
    'file_name':
    path("input/random_rve_vol20.h5"),
    'temp1':
    300,
    'temp2':
    1300,
    'n_tests':
    100,
    'sampling_alphas':
    np.asarray([
        np.asarray([0., 1.]),
        np.asarray([0., 0.859, 1.]),
        np.asarray([0., 0.859, 0.949, 1.]),
        np.asarray([0., 0.667, 0.859, 0.949, 1.]),
        np.asarray([0., 0.667, 0.859, 0.949, 0.98, 1.]),
        np.asarray([0., 0.465, 0.667, 0.859, 0.949, 0.98, 1.]),
        np.asarray([0., 0.465, 0.667, 0.778, 0.859, 0.949, 0.98, 1.]),
        np.asarray([0., 0.465, 0.667, 0.778, 0.859, 0.909, 0.949, 0.98, 1.]),
        np.asarray([0., 0.465, 0.667, 0.778, 0.859, 0.909, 0.949, 0.98, 0.99, 1.]),
        np.asarray([0., 0.465, 0.667, 0.778, 0.859, 0.909, 0.949, 0.97, 0.98, 0.99, 1.])
    ], dtype=object)
}, {
    'data_path':
    '/image_data/dset_0_sim',
    'file_name':
    path("input/random_rve_vol40.h5"),
    'temp1':
    300,
    'temp2':
    1300,
    'n_tests':
    100,
    'sampling_alphas':
    np.asarray([
        np.asarray([0., 1.]),
        np.asarray([0., 0.899, 1.]),
        np.asarray([0., 0.899, 0.97, 1.]),
        np.asarray([0., 0.717, 0.899, 0.97, 1.]),
        np.asarray([0., 0.515, 0.717, 0.899, 0.97, 1.]),
        np.asarray([0., 0.515, 0.717, 0.899, 0.939, 0.97, 1.]),
        np.asarray([0., 0.515, 0.717, 0.828, 0.899, 0.939, 0.97, 1.]),
        np.asarray([0., 0.515, 0.717, 0.828, 0.899, 0.939, 0.97, 0.99, 1.]),
        np.asarray([0., 0.515, 0.717, 0.828, 0.899, 0.939, 0.97, 0.98, 0.99, 1.]),
        np.asarray([0., 0.333, 0.515, 0.717, 0.828, 0.899, 0.939, 0.97, 0.98, 0.99, 1.])
    ], dtype=object)
}, {
    'data_path':
    '/image_data/dset_0_sim',
    'file_name':
    path('input/random_rve_vol60.h5'),
    'temp1':
    300,
    'temp2':
    1300,
    'n_tests':
    100,
    'sampling_alphas':
    np.asarray([
        np.asarray([0., 1.]),
        np.asarray([0., 0.899, 1.]),
        np.asarray([0., 0.727, 0.899, 1.]),
        np.asarray([0., 0.727, 0.899, 0.97, 1.]),
        np.asarray([0., 0.525, 0.727, 0.899, 0.97, 1.]),
        np.asarray([0., 0.525, 0.727, 0.838, 0.899, 0.97, 1.]),
        np.asarray([0., 0.525, 0.727, 0.838, 0.899, 0.939, 0.97, 1.]),
        np.asarray([0., 0.525, 0.727, 0.838, 0.899, 0.939, 0.97, 0.99, 1.]),
        np.asarray([0., 0.343, 0.525, 0.727, 0.838, 0.899, 0.939, 0.97, 0.99, 1.]),
        np.asarray([0., 0.343, 0.525, 0.727, 0.788, 0.838, 0.899, 0.939, 0.97, 0.99, 1.])
    ], dtype=object)
}]
