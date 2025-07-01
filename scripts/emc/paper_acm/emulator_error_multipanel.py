import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import torch
from acm.data.io_tools import *
import acm.observables.emc as emc

plt.rc('text', usetex=True)
plt.rc('font', family='serif')


def get_data(statistic, return_model=True):
    stat = getattr(emc, statistic)
    observable = stat(select_coordinates=select_coordinates, slice_coordinates=slice_coordinates)
    covariance_matrix = observable.get_covariance_matrix(divide_factor=64)
    emulator_error = observable.get_emulator_error()
    data_error = np.sqrt(np.diag(covariance_matrix))
    sep = observable.separation
    return sep, emulator_error, data_error


slice_coordinates = {}

fig, ax = plt.subplots(3, 4, figsize=(14, 10))

# projected correlation function
statistic = 'GalaxyProjectedCorrelationFunction'
select_coordinates = {}
sep, emulator_error, data_error = get_data(statistic)
ax[0, 0].plot(sep, emulator_error/data_error)
# ax[0, 0].set_xscale('log')
ax[0][0].set_xlabel(r'$r\,[h^{-1}{\rm Mpc}]$', fontsize=15)
ax[0][0].set_ylabel(r'$(X_{\rm model} - X_{\rm data})/\sigma_{\rm data}$', fontsize=15)
ax[0][0].set_title(r'$\textrm{Projected 2PCF}$', fontsize=15)

# 2PCF multipoles
statistic = 'GalaxyCorrelationFunctionMultipoles'
for ell in [0, 2]:
    select_coordinates = {'multipoles': [ell]}
    sep, emulator_error, data_error = get_data(statistic)
    ax[0, 1].plot(sep, emulator_error/data_error, label=f'$\ell={ell}$')
ax[0][1].set_xlabel(r'$s\,[h^{-1}{\rm Mpc}]$', fontsize=15)
ax[0][1].set_ylabel(r'$(X_{\rm model} - X_{\rm data})/\sigma_{\rm data}$', fontsize=15)
ax[0][1].set_title(r'$\textrm{2PCF multipoles}$', fontsize=15)
ax[0][1].legend(fontsize=13)

# power spectrum
statistic = 'GalaxyPowerSpectrumMultipoles'
for ell in [0, 2]:
    select_coordinates = {'multipoles': [ell]}
    sep, emulator_error, data_error = get_data(statistic)
    ax[0, 2].plot(sep, emulator_error/data_error, label=f'$\ell={ell}$')

ax[0][2].set_xlabel(r'$k\,[h/{\rm Mpc}]$', fontsize=15)
ax[0][2].set_ylabel(r'$(X_{\rm model} - X_{\rm data})/\sigma_{\rm data}$', fontsize=15)
ax[0][2].set_title(r'$\textrm{Power spectrum multipoles}$', fontsize=15)
ax[0][2].legend(fontsize=13)

# bispectrum
statistic = 'GalaxyBispectrumMultipoles'
for ell in [0, 2]:
    select_coordinates = {'multipoles': [ell]}
    sep, emulator_error, data_error = get_data(statistic)
    ax[0, 3].plot(emulator_error/data_error, label=f'$\ell={ell}$')
ax[0, 3].set_xlabel(r'$\textrm{bin index}$', fontsize=15)
ax[0][3].set_ylabel(r'$(X_{\rm model} - X_{\rm data})/\sigma_{\rm data}$', fontsize=15)
ax[0][3].set_title(r'$\textrm{Bispectrum multipoles}$', fontsize=15)
ax[0][3].legend(fontsize=13)

# density-split statistics
statistic = 'DensitySplitPowerSpectrumMultipoles'
for ell in [0, 2]:
    select_coordinates = {'multipoles': [ell],
                          'statistics': ['quantile_data_power'], 'quantiles': [4]}
    # slice_coordinates = {'k': (0.0, 0.5)}
    sep, emulator_error, data_error = get_data(statistic)
    ax[1, 0].plot(sep, emulator_error/data_error, label=f'$\ell={ell}$')

ax[1][0].set_xlabel(r'$k\,[h/{\rm Mpc}]$', fontsize=15)
ax[1][0].set_ylabel(r'$(X_{\rm model} - X_{\rm data})/\sigma_{\rm data}$', fontsize=15)
ax[1][0].set_title(r'$\textrm{Density-split multipoles}$', fontsize=15)
ax[1][0].legend(fontsize=13)

# Minkowski functionals
statistic = 'MinkowskiFunctionals'
sep, emulator_error, data_error = get_data(statistic)
ax[2, 0].plot(emulator_error/data_error)
ax[2][0].set_xlabel(r'$\textrm{bin index}$', fontsize=15)
ax[2][0].set_ylabel(r'$(X_{\rm model} - X_{\rm data})/\sigma_{\rm data}$', fontsize=15)
ax[2][0].set_title(r'$\textrm{Minkowski functionals}$', fontsize=15)

# Wavelet scattering transform
statistic = 'WaveletScatteringTransform'
sep, emulator_error, data_error = get_data(statistic)
ax[2, 1].plot(emulator_error/data_error)
ax[2][1].set_xlabel(r'$\textrm{bin index}$', fontsize=15)
ax[2][1].set_ylabel(r'$(X_{\rm model} - X_{\rm data})/\sigma_{\rm data}$', fontsize=15)
ax[2][1].set_title(r'$\textrm{Wavelet scattering transform}$', fontsize=15)

# Density PDF
statistic = f'GalaxyOverdensityPDF'
sep, emulator_error, data_error = get_data(statistic)
ax[1, 2].plot(emulator_error/data_error)
ax[1][2].set_xlabel(r'$\textrm{bin index}$', fontsize=15)
ax[1][2].set_ylabel(r'$(X_{\rm model} - X_{\rm data})/\sigma_{\rm data}$', fontsize=15)
ax[1][2].set_title(r'$\textrm{Overdensity PDF}$', fontsize=15)

# Cumulant Generating Function
statistic = f'CumulantGeneratingFunction'
sep, emulator_error, data_error = get_data(statistic)
ax[1, 3].plot(np.arange(len(emulator_error)), emulator_error/data_error)
ax[1][3].set_xlabel(r'$\textrm{bin index}$', fontsize=15)
ax[1][3].set_ylabel(r'$(X_{\rm model} - X_{\rm data})/\sigma_{\rm data}$', fontsize=15)
ax[1][3].set_title(r'$\textrm{Cumulant Generating Function}$', fontsize=15)

# Minimum Spanning Tree
statistic = f'MinimumSpanningTree'
sep, emulator_error, data_error = get_data(statistic)
ax[1, 1].plot(emulator_error/data_error)
ax[1][1].set_xlabel(r'$\textrm{Coefficient index}$', fontsize=15)
ax[1][1].set_ylabel(r'$(X_{\rm model} - X_{\rm data})/\sigma_{\rm data}$', fontsize=15)
ax[1][1].set_title(r'$\textrm{Minimum Spanning Tree}$', fontsize=15)

# Voxel Void-galaxy 2PCF
statistic = 'VoxelVoidGalaxyCorrelationFunctionMultipoles'
for ell in [0, 2]:
    select_coordinates = {'multipoles': [ell]}
    sep, emulator_error, data_error = get_data(statistic)
    ax[2, 2].plot(sep, emulator_error/data_error, label=f'$\ell={ell}$')
ax[2][2].set_xlabel(r'$s\,[h^{-1}{\rm Mpc}]$', fontsize=15)
ax[2][2].set_ylabel(r'$(X_{\rm model} - X_{\rm data})/\sigma_{\rm data}$', fontsize=15)
ax[2][2].set_title(r'$\textrm{Void-galaxy CCF}$', fontsize=15)
ax[2][2].legend(fontsize=13)

# # DT Void-galaxy 2PCF
# statistic = 'DTVoidGalaxyCorrelationFunctionMultipoles'
# for ell in [0, 2]:
#     select_coordinates = {'multipoles': [ell]}
#     sep, emulator_error, data_error = get_data(statistic)
#     ax[2, 3].plot(sep, emulator_error/data_error, label=f'$\ell={ell}$')
# ax[2][3].set_xlabel(r'$s\,[h^{-1}{\rm Mpc}]$', fontsize=15)
# ax[2][3].set_ylabel(r'$(X_{\rm model} - X_{\rm data})/\sigma_{\rm data}$', fontsize=15)
# ax[2][3].set_title(r'$\textrm{DT Void-galaxy CF}$', fontsize=15)

# VIDE void size function
statistic = 'VIDEVoidSizeFunction'
sep, emulator_error, data_error = get_data(statistic)
print(data_error)
ax[2, 3].plot(sep, emulator_error/data_error)
ax[2][3].set_xlabel(r'$R_{\rm void}\,[h^{-1}{\rm Mpc}]$', fontsize=15)
ax[2][3].set_ylabel(r'$(X_{\rm model} - X_{\rm data})/\sigma_{\rm data}$', fontsize=15)
ax[2][3].set_title(r'$\textrm{Void size function}$', fontsize=15)

for ax in fig.axes:
    ax.xaxis.set_tick_params(labelsize=15)
    ax.yaxis.set_tick_params(labelsize=15)

plt.tight_layout()
plt.savefig('emulator_error_multipanel.pdf', bbox_inches='tight')
plt.savefig('emulator_error_multipanel.png', bbox_inches='tight', dpi=300)