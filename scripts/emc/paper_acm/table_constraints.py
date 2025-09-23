import matplotlib.pyplot as plt
from pathlib import Path
from getdist import plots, MCSamples, loadMCSamples
from sunbird.inference.samples import Chain
from cosmoprimo.fiducial import AbacusSummit
import numpy as np
from tabulate import tabulate, SEPARATING_LINE

plt.rc('text', usetex=True)
plt.rc('font', family='serif')

base_dir = Path('/global/cfs/cdirs/desicollab/users/epaillas/acm/fits_emc/abacus/')


def get_samples(statistics, date='aug25', cosmo_model='base', hod_model='base-VB-AB-CB-s'):
    if statistics == 'greedy':
        data_dir = base_dir / f'{date}/greedy/c000_hod030/cosmo-{cosmo_model}_hod-{hod_model}/'
        data_fn = Path(data_dir) / f'chain_number_density+minkowski+wp+tpcf+bk+dsc_pk+wst+vide_gv+pdf+cgf.npy'
    else:
        data_dir = base_dir / f'{date}/c000_hod030/cosmo-{cosmo_model}_hod-{hod_model}/'
        data_fn = Path(data_dir) / f"chain_number_density+{statistics}.npy"
    chain = Chain.load(data_fn)
    samples = Chain.to_getdist(chain, add_derived=False)
    labels = chain.data['labels']
    return samples, labels

def print_cosmo_constraints():
    params_lcdm = ['omega_b', 'omega_cdm', 'sigma8_m', 'n_s']
    params_w0wa = ['w0_fld', 'wa_fld']
    params_Nur = ['N_ur']
    table = []
    table.append([r'$\bm{\Lambda}$\textbf{CDM}'])
    for stat, label in zip(stats, labels):
        samples, param_labels = get_samples(stat, date='aug25', cosmo_model='base')
        constraints = [rf"${samples[param].std():.5f}$" for param in params_lcdm]
        constraints += ['---' for _ in params_w0wa]
        constraints += ['---' for _ in params_Nur]
        table.append([label] + constraints)
    table.append([r'$\bm{w_0w_a}$\textbf{CDM}'])
    for stat, label in zip(stats, labels):
        samples, param_labels = get_samples(stat, cosmo_model='base-w0-wa')
        constraints = [rf"${samples[param].std():.5f}$" for param in params_lcdm]
        constraints += [rf"${samples[param].std():.5f}$" for param in params_w0wa]
        constraints += ['---' for _ in params_Nur]
        table.append([label] + constraints)
    table.append([r'$\bm{\Lambda}$\textbf{CDM}$+N_{\rm ur}$'])
    for stat, label in zip(stats, labels):
        samples, param_labels = get_samples(stat, date='aug25', cosmo_model='base-Nur')
        constraints = [rf"${samples[param].std():.5f}$" for param in params_lcdm]
        constraints += ['---' for _ in params_w0wa]
        constraints += [rf"${samples[param].std():.5f}$" for param in params_Nur]
        table.append([label] + constraints)
    header = ['Statistic'] + [f'$\Delta$' + param_labels[param] for param in params_lcdm] + \
        [f'$\Delta$' + param_labels[param] for param in params_w0wa] + \
        [f'$\Delta$' + param_labels[param] for param in params_Nur]
    print(tabulate(table, tablefmt='latex_raw', headers=header, floatfmt=".5f"))

def print_hod_constraints():
    params = ['logM_cut', 'logM_1', 'sigma', 'kappa', 'alpha', 's', 'alpha_c', 'alpha_s', 'B_cen', 'B_sat']
    table = []
    for stat, label in zip(stats, labels):
        samples, param_labels = get_samples(stat, date='aug25', model='LCDM')
        constraints = [rf"${samples[param].std():.5f}$" for param in params_lcdm]
        table.append([label] + constraints)
    header = ['Statistic'] + [f'$\Delta$' + param_labels[param] for param in params]
    print(tabulate(table, tablefmt='latex_raw', headers=header, floatfmt=".5f"))


if __name__ == '__main__':

    stats = [
        'wp',
        'tpcf',
        'pk',
        'bk',
        'dsc_pk',
        'wst',
        'mst',
        'minkowski',
        'dt_gv',
        'vide_gv',
        'voxel_voids',
        'vide_vsf_rsd',
        'pdf',
        'cgf',
        'greedy',
    ]
    labels = [
        r'Projected 2PCF',
        r'2PCF multipoles',
        r'Power spectrum multipoles',
        r'Bispectrum multipoles',
        r'Density-split clustering',
        r'Wavelet scattering',
        'Minimum spanning tree',
        r'Minkowski functionals',
        r'DT void-galaxy CCF',
        'VIDE void-galaxy CCF',
        'Voxel void-galaxy CCF',
        'VIDE void size function',
        'Overdensity PDF',
        'Cumulant generating function',
        r'Greedy combination',
    ]

    print_cosmo_constraints()
    # print_hod_constraints()