import numpy as np
import json
import matplotlib.pyplot as plt
from pathlib import Path
import torch
from acm.data.io_tools import *
import acm.observables.emc as emc
from typing import List, Dict

plt.rc('text', usetex=True)
plt.rc('font', family='serif')


def get_data(
    statistic,
    return_model=True,
    select_coordinates={},
    slice_coordinates={},
    select_indices=[],
):
    stat = getattr(emc, statistic)
    observable = stat(
        select_mocks=select_mocks,
        select_coordinates=select_coordinates,
        slice_coordinates=slice_coordinates,
        select_indices=select_indices,
    )
    covariance_matrix = observable.get_covariance_matrix(divide_factor=64)
    data_x = observable.lhc_x
    data_y = observable.lhc_y
    error = np.sqrt(np.diag(covariance_matrix))
    sep = observable.separation
    if return_model:
        model = observable.get_model_prediction(data_x)
        return sep, data_y, error, model, observable.selected_bin_idx
    return sep, data_y, error, observable.selected_bin_idx.tolist()


def read_greedy_bins(target="lcdm"):
    bins_dir = "/pscratch/sd/c/cuesta/greedy_fisher/"
    if target == "lcdm":
        filename = "fixed_blcdm_marginalised_hod_v2.json"
    elif target == "blcdm":
        filename = "marginalised_hod_lcdm_v2.json"
    bins_fn = Path(bins_dir) / filename
    with open(bins_fn, "r") as f:
        selection_history = json.load(f)
    return selection_history["selection_history"]


def find_bins_in_greedy(selection_history, selected_bin_idx, statistic):
    bin_idx, color = [], []
    for i, selected_hist in enumerate(selection_history):
        if (
            selected_hist["statistic"] == statistic
            and selected_hist["bin_idx"] in selected_bin_idx
        ):
            bin_idx.append(selected_bin_idx.tolist().index(selected_hist["bin_idx"]))
            color.append(selected_hist["improvement"])
    return bin_idx, color


selection_history = read_greedy_bins()
selection_history_blcdm = read_greedy_bins("blcdm")

color_line_model = "#2c3e50"
color_all = "#6c757d"
color_lcdm = "#3498db"
color_blcdm = "#9b59b6"

fig, ax = plt.subplots(3, 4, figsize=(14, 10))

# projected correlation function
select_mocks = {"cosmo_idx": 0, "hod_idx": 30}

statistic = "GalaxyProjectedCorrelationFunction"
select_coordinates = {}

sep, data_y, error, model, selected_bin_idx = get_data(statistic)

greedy_bin_idx, _ = find_bins_in_greedy(
    selection_history,
    selected_bin_idx,
    "wp",
)
greedy_bin_idx_blcdm, _ = find_bins_in_greedy(
    selection_history_blcdm,
    selected_bin_idx,
    "wp",
)


ax[0, 0].errorbar(
    sep,
    data_y,
    error,
    markersize=4.0,
    elinewidth=1.0,
    marker="o",
    ls="",
    color=color_all,
    alpha=0.7,
)
ax[0, 0].errorbar(
    sep[greedy_bin_idx],
    data_y[greedy_bin_idx],
    error[greedy_bin_idx],
    markersize=6.0,
    elinewidth=1.0,
    marker="^",
    ls="",
    color=color_lcdm,
    alpha=0.8,
)
ax[0, 0].errorbar(
    sep[greedy_bin_idx_blcdm],
    data_y[greedy_bin_idx_blcdm],
    error[greedy_bin_idx_blcdm],
    markersize=6.0,
    elinewidth=1.0,
    marker="^",
    ls="",
    color=color_blcdm,
    alpha=0.8,
)
ax[0, 0].plot(sep, model, color=color_line_model)
ax[0, 0].set_xscale("log")
ax[0, 0].set_yscale("log")
ax[0][0].set_xlabel(r"$r\,[h^{-1}{\rm Mpc}]$", fontsize=15)
ax[0][0].set_ylabel(r"$w_p(r)$", fontsize=15)
ax[0][0].set_title(r'$\textrm{Projected 2PCF}$', fontsize=15)

# 2PCF multipoles
statistic = "GalaxyCorrelationFunctionMultipoles"
for ell in [0, 2]:
    select_coordinates = {
        "multipoles": [ell],
    }
    slice_coordinates = {"s": (10, 150)}
    sep, data_y, error, model, selected_bin_idx = get_data(
        statistic,
        select_coordinates=select_coordinates,
        slice_coordinates=slice_coordinates,
    )
    sep = sep[(sep >= 10) & (sep <= 150)]
    greedy_bin_idx, greedy_color = find_bins_in_greedy(
        selection_history,
        selected_bin_idx,
        "tpcf",
    )
    greedy_bin_idx_blcdm, _ = find_bins_in_greedy(
        selection_history_blcdm,
        selected_bin_idx,
        "tpcf",
    )

    ax[0, 1].errorbar(
        sep,
        sep**2 * data_y,
        sep**2 * error,
        markersize=4.0,
        elinewidth=1.0,
        marker="o",
        ls="",
        color=color_all,
        alpha=0.7,
    )
    ax[0, 1].errorbar(
        sep[greedy_bin_idx],
        sep[greedy_bin_idx] ** 2 * data_y[greedy_bin_idx],
        sep[greedy_bin_idx] ** 2 * error[greedy_bin_idx],
        markersize=6.0,
        elinewidth=1.0,
        marker="^",
        ls="",
        color=color_lcdm,
        alpha=0.8,
    )
    ax[0, 1].errorbar(
        sep[greedy_bin_idx_blcdm],
        sep[greedy_bin_idx_blcdm] ** 2 * data_y[greedy_bin_idx_blcdm],
        sep[greedy_bin_idx_blcdm] ** 2 * error[greedy_bin_idx_blcdm],
        markersize=6.0,
        elinewidth=1.0,
        marker="^",
        ls="",
        color=color_blcdm,
        alpha=0.8,
    )

    ax[0, 1].plot(sep, sep**2 * model, color=color_line_model)
ax[0][1].set_xlabel(r"$s\,[h^{-1}{\rm Mpc}]$", fontsize=15)
ax[0][1].set_ylabel(r"$s^2\xi_\ell(s)\,[h^{-2}{\rm Mpc}^2]$", fontsize=15)
# ax[0][1].set_title(r'$\textrm{2PCF multipoles}$', fontsize=15)


# # power spectrum
statistic = "GalaxyPowerSpectrumMultipoles"
for ell in [0, 2]:
    select_coordinates = {"multipoles": [ell], "cosmo_idx": 0, "hod_idx": 30}
    slice_coordinates = {"k": (0.0, 0.5)}
    sep, data_y, error, model, selected_bin_idx = get_data(
        statistic,
        select_coordinates=select_coordinates,
        slice_coordinates=slice_coordinates,
    )

    ax[0, 2].errorbar(
        sep,
        sep * data_y,
        sep * error,
        markersize=4.0,
        elinewidth=1.0,
        marker="o",
        ls="",
        color=color_all,
        alpha=0.7,
    )
    ax[0, 2].plot(sep, sep * model, color=color_line_model)
ax[0][2].set_xlabel(r"$k\,[h/{\rm Mpc}]$", fontsize=15)
ax[0][2].set_ylabel(r"$k\,P(k)\,[h^{-2}{\rm Mpc}^2]$", fontsize=15)
# ax[0][2].set_title(r'$\textrm{Power spectrum multipoles}$', fontsize=15)


# # density-split statistics
statistic = "DensitySplitPowerSpectrumMultipoles"
for q in [0, 1, 3, 4]:
    select_coordinates = {
        "multipoles": [0],
        "cosmo_idx": 0,
        "hod_idx": 30,
        "quantiles": [q],
        "statistics": ["quantile_data_power"],
    }
    slice_coordinates = {"k": (0.0, 0.5)}
    sep, data_y, error, model, selected_bin_idx = get_data(
        statistic,
        select_coordinates=select_coordinates,
        slice_coordinates=slice_coordinates,
    )

    greedy_bin_idx, greedy_color = find_bins_in_greedy(
        selection_history,
        selected_bin_idx,
        "density_split",
    )

    greedy_bin_idx_blcdm, _ = find_bins_in_greedy(
        selection_history_blcdm,
        selected_bin_idx,
        "density_split",
    )

    ax[1, 0].errorbar(
        sep,
        sep**2 * data_y,
        sep**2 * error,
        markersize=4.0,
        elinewidth=1.0,
        marker="o",
        ls="",
        color=color_all,
        alpha=0.7,
    )
    ax[1, 0].errorbar(
        sep[greedy_bin_idx],
        sep[greedy_bin_idx] ** 2 * data_y[greedy_bin_idx],
        sep[greedy_bin_idx] ** 2 * error[greedy_bin_idx],
        markersize=6.0,
        elinewidth=1.0,
        marker="^",
        ls="",
        color=color_lcdm,
        alpha=0.8,
    )
    ax[1, 0].errorbar(
        sep[greedy_bin_idx_blcdm],
        sep[greedy_bin_idx_blcdm] ** 2 * data_y[greedy_bin_idx_blcdm],
        sep[greedy_bin_idx_blcdm] ** 2 * error[greedy_bin_idx_blcdm],
        markersize=6.0,
        elinewidth=1.0,
        marker="^",
        ls="",
        color=color_blcdm,
        alpha=0.8,
    )

    ax[1, 0].plot(sep, sep**2 * model, color=color_line_model)
ax[1][0].set_xlabel(r"$k\,[h/{\rm Mpc}]$", fontsize=15)
ax[1][0].set_ylabel(r"$k\,P(k)\,[h^{-2}{\rm Mpc}^2]$", fontsize=15)
# ax[1][0].set_title(r'$\textrm{Density-split multipoles}$', fontsize=15)

# # Minkowski functionals
statistic = "MinkowskiFunctionals"
slice_coordinates = {}
sep, data_y, error, model, selected_bin_idx = get_data(
    statistic, slice_coordinates=slice_coordinates
)
greedy_bin_idx, greedy_color = find_bins_in_greedy(
    selection_history,
    selected_bin_idx,
    "minkowski",
)
greedy_bin_idx_blcdm, _ = find_bins_in_greedy(
    selection_history_blcdm,
    selected_bin_idx,
    "minkowski",
)
ax[2, 0].errorbar(
    sep[::3],
    data_y[::3],
    error[::3],
    markersize=3.0,
    elinewidth=1.0,
    marker="o",
    ls="",
    color=color_all,
    alpha=0.7,
)
ax[2, 0].errorbar(
    sep[greedy_bin_idx][::3],
    data_y[greedy_bin_idx][::3],
    error[greedy_bin_idx][::3],
    markersize=6.0,
    elinewidth=1.0,
    marker="^",
    ls="",
    color=color_lcdm,
    alpha=0.8,
)
ax[2, 0].errorbar(
    sep[greedy_bin_idx_blcdm][::3],
    data_y[greedy_bin_idx_blcdm][::3],
    error[greedy_bin_idx_blcdm][::3],
    markersize=6.0,
    elinewidth=1.0,
    marker="^",
    ls="",
    color=color_blcdm,
    alpha=0.8,
)


ax[2, 0].plot(sep[::3], model[::3], color=color_line_model)
# ax[2][0].set_xlabel(r'$\textrm{Overdensity } \Delta$', fontsize=15)
ax[2][0].set_ylabel(r"$W_i$", fontsize=15)
# ax[2][0].set_title(r'$\textrm{Minkowski functionals}$', fontsize=15)

# # Wavelet scattering transform
statistic = "WaveletScatteringTransform"
sep, data_y, error, model, selected_bin_idx = get_data(
    statistic, slice_coordinates=slice_coordinates
)
greedy_bin_idx, greedy_color = find_bins_in_greedy(
    selection_history,
    selected_bin_idx,
    "wst",
)
greedy_bin_idx_blcdm, _ = find_bins_in_greedy(
    selection_history_blcdm,
    selected_bin_idx,
    "wst",
)
ax[2, 1].errorbar(
    sep[::2],
    data_y[::2],
    error[::2],
    markersize=3.0,
    elinewidth=1.0,
    marker="o",
    ls="",
    color=color_all,
    alpha=0.7,
)
ax[2, 1].errorbar(
    sep[greedy_bin_idx][::2],
    data_y[greedy_bin_idx][::2],
    error[greedy_bin_idx][::2],
    markersize=6.0,
    elinewidth=1.0,
    marker="^",
    ls="",
    color=color_lcdm,
    alpha=0.8,
)
ax[2, 1].errorbar(
    sep[greedy_bin_idx_blcdm][::2],
    data_y[greedy_bin_idx_blcdm][::2],
    error[greedy_bin_idx_blcdm][::2],
    markersize=6.0,
    elinewidth=1.0,
    marker="^",
    ls="",
    color=color_blcdm,
    alpha=0.8,
)


ax[2, 1].plot(sep[::2], model[::2], color=color_line_model)
# ax[2][1].set_xlabel(r'$\textrm{Coefficient index}$', fontsize=15)
# ax[2][1].set_ylabel(r'$\textrm{WST coefficient}$', fontsize=15)
# ax[2][1].set_title(r'$\textrm{Wavelet scattering transform}$', fontsize=15)

# Density PDF
# statistic = f'GalaxyOverdensityPDF'
# select_filters = {'cosmo_idx': 0, 'hod_idx': 30}
# slice_filters = {}
# sep, data_y, error, model = get_data(statistic)
# ax[1, 2].errorbar(sep, data_y, error, markersize=3.0, elinewidth=1.0,
#                 marker='o', ls='', color='dimgrey')
# ax[1, 2].plot(sep, model, color='r')
# ax[1][2].set_xlim(-1.2, 2)
# ax[1][2].set_xlabel(r'$\textrm{Overdensity } \Delta$', fontsize=15)
# ax[1][2].set_ylabel(r'$\textrm{PDF}$', fontsize=15)
# ax[1][2].set_title(r'$\textrm{Overdensity PDF}$', fontsize=15)

# # Cumulant Generating Function
# statistic = f'CumulantGeneratingFunction'
# select_filters = {'cosmo_idx': 0, 'hod_idx': 30}
# slice_filters = {}
# sep, data_y, error = get_data(statistic, return_model=False)
# ax[1, 3].errorbar(sep[::2], data_y[::2], error[::2], markersize=3.0, elinewidth=1.0,
#                 marker='o', ls='', color='dimgrey')
# # ax[1, 3].plot(sep, model, color='r')
# # ax[1][3].set_xlim(-1.2, 2)
# ax[1][3].set_xlabel(r'$\lambda$', fontsize=15)
# ax[1][3].set_ylabel(r'$\log\langle e^{\lambda \delta}\rangle$', fontsize=15)
# ax[1][3].set_title(r'$\textrm{Cumulant Generating Function}$', fontsize=15)

# # Minimum Spanning Tree
# statistic = f'MinimumSpanningTree'
# select_filters = {'cosmo_idx': 0, 'hod_idx': 30}
# slice_filters = {}
# sep, data_y, error, model = get_data(statistic)
# ax[1, 1].errorbar(sep[::3], data_y[::3], error[::3], markersize=3.0, elinewidth=1.0,
#                 marker='o', ls='', color='dimgrey')
# ax[1, 1].plot(sep, model, color='r')
# # ax[1][2].set_xlim(-1.2, 2)
# ax[1][1].set_xlabel(r'$\textrm{Coefficient index}$', fontsize=15)
# # ax[1][2].set_ylabel(r'$\textrm{PDF}$', fontsize=15)
# ax[1][1].set_title(r'$\textrm{MST coefficient}$', fontsize=15)

# # Voxel Void-galaxy 2PCF
# statistic = 'VoxelVoidGalaxyCorrelationFunctionMultipoles'
# for ell in [0, 2]:
#     select_filters = {'multipoles': [ell], 'cosmo_idx': 0, 'hod_idx': 30}
#     slice_filters = {}
#     sep, data_y, error, model = get_data(statistic)
#     ax[2, 2].errorbar(sep[::2], data_y[::2], error[::2], markersize=3.0, elinewidth=1.0,
#                     marker='o', ls='', color='dimgrey')
#     ax[2, 2].plot(sep[::2], model[::2], color='r')
# ax[2][2].set_xlabel(r'$s\,[h^{-1}{\rm Mpc}]$', fontsize=15)
# ax[2][2].set_ylabel(r'$\xi_\ell(s)$', fontsize=15)
# ax[2][2].set_title(r'$\textrm{Voxel Void-galaxy CF}$', fontsize=15)

# DT Void-galaxy 2PCF
statistic = "DTVoidGalaxyCorrelationFunctionMultipoles"
for ell in [0, 2]:
    select_coordinates = {
        "multipoles": [ell],
    }
    sep, data_y, error, model, selected_bin_idx = get_data(
        statistic, select_coordinates=select_coordinates
    )

    greedy_bin_idx, greedy_color = find_bins_in_greedy(
        selection_history,
        selected_bin_idx,
        "dt_gv",
    )
    greedy_bin_idx_blcdm, _ = find_bins_in_greedy(
        selection_history_blcdm,
        selected_bin_idx,
        "dt_gv",
    )
    ax[2, 3].errorbar(
        sep[::2],
        data_y[::2],
        error[::2],
        markersize=3.0,
        elinewidth=1.0,
        marker="o",
        ls="",
        color=color_all,
        alpha=0.7,
    )
    ax[2, 3].errorbar(
        sep[greedy_bin_idx][::2],
        data_y[greedy_bin_idx][::2],
        error[greedy_bin_idx][::2],
        markersize=6.0,
        elinewidth=1.0,
        marker="^",
        ls="",
        color=color_lcdm,
        alpha=0.8,
    )
    ax[2, 3].errorbar(
        sep[greedy_bin_idx_blcdm][::2],
        data_y[greedy_bin_idx_blcdm][::2],
        error[greedy_bin_idx_blcdm][::2],
        markersize=6.0,
        elinewidth=1.0,
        marker="^",
        ls="",
        color=color_lcdm,
        alpha=0.8,
    )

    ax[2, 3].plot(sep[::2], model[::2], color=color_line_model)
# ax[2][3].set_xlabel(r'$s\,[h^{-1}{\rm Mpc}]$', fontsize=15)
# ax[2][3].set_ylabel(r'$\xi_\ell(s)$', fontsize=15)
# ax[2][3].set_title(r'$\textrm{DT Void-galaxy CF}$', fontsize=15)

# Density-field cumulants
# statistic = 'cgf_r10'
# select_filters = {'cosmo_idx': 0, 'hod_idx': 30}
# slice_filters = {}
# sep, data_y, error = get_data(statistic, return_model=False)
# ax[2, 2].errorbar(sep, data_y, error, markersize=3.0, elinewidth=1.0,
#                 marker='o', ls='', color='dimgrey')
# ax[2, 2].plot(sep[::2], model[::2], color='r')
# ax[2][2].set_xlabel(r'$\textrm{Overdensity } \Delta$', fontsize=15)
# ax[2][2].set_ylabel(r'$\textrm{PDF}$', fontsize=15)
# ax[2][2].set_title(r'$\textrm{Cumulant generating function}$', fontsize=15)


for ax in fig.axes:
    ax.xaxis.set_tick_params(labelsize=15)
    ax.yaxis.set_tick_params(labelsize=15)

plt.tight_layout()
plt.savefig("fig/summaries_multipanel.pdf", bbox_inches="tight")
