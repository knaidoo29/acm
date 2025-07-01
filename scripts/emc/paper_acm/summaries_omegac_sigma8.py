import matplotlib.pyplot as plt
from pathlib import Path
from getdist import plots, MCSamples, loadMCSamples
from sunbird.inference.samples import Chain
from cosmoprimo.fiducial import AbacusSummit
import numpy as np

plt.rc('text', usetex=True)
plt.rc('font', family='serif')


chains = []
legend_labels = []

# params = ['logM_cut', 'logM_1', 'sigma', 'kappa', 'alpha']
# params = ['omega_cdm', 'sigma8_m', 'n_s', 'logM_cut', 'logM_1', 'sigma', 'kappa', 'alpha']
params = ['omega_cdm', 'sigma8_m', 'n_s']
# params = ['A_cen', 'A_sat', 'B_cen', 'B_sat']

stats = [
    # 'wp',
    'minkowski',
    'tpcf',
    'bk',
    'wst'
]
# labels = [r'$w_p(r_p)$', r'Minkowski F.', r'$\xi_\ell$', r'$B_\ell$', 'Wavelet ST']
labels = [
    # r'$\textrm{Projected 2PCF}$',
    r'$\textrm{Minkowski Func.}$',
    r'$\textrm{2PCF multipoles}$',
    r'$\textrm{Bispectrum}$',
    r'$\textrm{Wavelet ST}$'
]

for stat, label in zip(stats, labels):
    data_dir = f'/global/cfs/cdirs/desicollab/users/epaillas/acm/fits_emc/abacus/jun28/c000_hod030/LCDM_baseHOD/'
    data_fn = Path(data_dir) / f"chain_number_density+{stat}.npy"
    chain = Chain.load(data_fn)
    samples = Chain.to_getdist(chain, add_derived=True)
    chains.append(samples)
    legend_labels.append(label)

data_dir = f'/global/cfs/cdirs/desicollab/users/epaillas/acm/fits_emc/abacus/optimal/apr11/c000_hod030/LCDM/'
data_fn = Path(data_dir) / 'chain_number_density+minkowski_apr11+wp+tpcf+bk+dsc_pk+wst_apr11+dt_gv.npy'
# data_fn = Path(data_dir) / 'chain_number_density+minkowski_apr11+wp+tpcf+pk+bk+dsc_pk+wst_apr11+dt_gv+voxel_voids+pdf_r10+cgf_r10.npy'
chain = Chain.load(data_fn)
samples = Chain.to_getdist(chain, add_derived=True)
chains.append(samples)
legend_labels.append('Greedy combo')

markers = chain.markers
cosmo = AbacusSummit(0)
markers.update({'Omega_m': cosmo['Omega_m'], 'h': cosmo['h']})
    
g = plots.get_subplot_plotter(width_inch=6)
g.settings.constrained_layout = False
g.settings.axis_marker_lw = 1.0
g.settings.axis_marker_ls = "--"
g.settings.title_limit_labels = False
g.settings.axis_marker_color = "k"
g.settings.legend_colored_text = True
g.settings.figure_legend_frame = True
g.settings.figure_legend_ncol = 1
g.settings.linewidth_contour = 1.0
g.settings.legend_fontsize = 18.5
g.settings.axes_fontsize = 18
g.settings.axes_labelsize = 25
g.settings.axis_tick_x_rotation = 45
# g.settings.axis_tick_max_labels = 6
g.settings.solid_colors = ['#4165c0', '#e770a2', '#5ac3be', '#696969', '#f79a1e', '#ba7dcd'][::-1]
g.settings.line_styles = g.settings.solid_colors

g.triangle_plot(
    roots=chains,
    legend_labels=legend_labels,
    markers=markers,
    params=params,
    filled=True,
    legend_loc='upper right',
    # filled=[True, False, False],
    # title_limit=1,
    # params=['logM_cut', 'logM_1', 'sigma', 'kappa', 'alpha']
    # legend_labels=stats
)

plt.savefig('summaries_omegam_sigma8.png', dpi=300, bbox_inches='tight')
plt.savefig('summaries_omegam_sigma8.pdf', bbox_inches='tight')