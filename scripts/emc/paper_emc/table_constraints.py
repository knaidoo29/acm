import matplotlib.pyplot as plt
from pathlib import Path
from getdist import plots, MCSamples, loadMCSamples
from sunbird.inference.samples import Chain
from cosmoprimo.fiducial import AbacusSummit
import numpy as np
from tabulate import tabulate, SEPARATING_LINE

plt.rc('text', usetex=True)
plt.rc('font', family='serif')

base_dir = Path('/global/cfs/cdirs/desicollab/users/epaillas/acm/fits_emc/diffsky/')

def get_samples(statistics, date='aug25'):
    if statistics == 'greedy':
        data_dir = base_dir / f'{date}/greedy/galsampled_67120_fixedAmp_mean_{diffsky_sampling}_v0.3/cosmo-base_hod-{hod_model}/'
        data_fn = Path(data_dir) / f"chain_number_density+minkowski+wp+tpcf+bk+dsc_pk+wst+vide_gv+pdf+cgf.npy"
    else:
        data_dir = base_dir / f'{date}/galsampled_67120_fixedAmp_mean_{diffsky_sampling}_v0.3/cosmo-base_hod-{hod_model}/'
        data_fn = Path(data_dir) / f"chain_number_density+{statistics}.npy"
    chain = Chain.load(data_fn)
    samples = Chain.to_getdist(chain, add_derived=True)
    Omega_m = samples.getParams().Omega_m
    sigma8 = samples.getParams().sigma8_m
    S8 = sigma8 * np.sqrt(Omega_m / 0.3)
    samples.addDerived(S8, name='S8', label=r'S_8')
    labels = chain.data['labels']
    markers = chain.markers
    markers.update({'S8': markers['sigma8_m'] * np.sqrt(markers['Omega_m'] / 0.3)})
    return samples, labels, markers


diffsky_sampling = 'mass'
hod_model = 'base-VB-AB-CB-s'

params_lcdm = ['omega_cdm', 'sigma8_m', 'n_s']

stats = [
    'wp',
    'minkowski',
    'tpcf',
    'pk',
    'wst',
    'bk',
    'dsc_pk',
    'dt_gv',
    'vide_gv',
    'vide_vsf_rsd',
    'mst',
    'cgf',
    'pdf',
    'greedy',
]
labels = [
    r'Projected 2PCF',
    r'Minkowski functionals',
    r'2PCF multipoles',
    r'Power spectrum multipoles',
    r'Wavelet scattering',
    r'Bispectrum multipoles',
    r'Density-split clustering',
    r'DT void-galaxy CCF',
    r'VIDE void-galaxy CCF',
    r'VIDE void size function',
    r'Minimum Spanning Tree'
    r'Cumulant generating function',
    r'Overdensity PDF',
    r'Greedy combination',
]

table = []
table.append([r'$\bm{\Lambda}$\textbf{CDM}'])
for stat, label in zip(stats, labels):
    samples, param_labels, markers = get_samples(stat, date='aug25')
    mean = np.array([samples[param].mean() for param in params_lcdm])
    std = np.array([samples[param].std() for param in params_lcdm])
    bias = [(markers[param] - mean[i])/std[i] for i, param in enumerate(params_lcdm)]
    
    constraints = [rf"${samples[param].mean():.5f} \pm {samples[param].std():.5f}$" for param in params_lcdm]
    constraints += [f"${bias[i]:.2f}\sigma$" for i in range(len(bias))]

    table.append([label] + constraints)

param_labels['Omega_m'] = r'$\Omega_{\rm m}$'
param_labels['S8'] = r'$S_8$'

header = ['Statistic'] + [param_labels[param] for param in params_lcdm] + [rf"$\Delta${param_labels[param]}" for param in params_lcdm]
print(tabulate(table, tablefmt='latex_raw', headers=header, floatfmt=".5f"))
