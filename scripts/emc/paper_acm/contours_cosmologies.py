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
params = ['omega_cdm', 'sigma8_m']
# params = ['A_cen', 'A_sat', 'B_cen', 'B_sat']

# cosmos = [1, 2, 3, 4]
# hods = [50, 13, 44, 0]

cosmos = [1, 2, 3, 4]
hods = [50, 13, 44, 0]


for cosmo_idx, hod_idx in zip(cosmos, hods):
    data_dir = f'/global/cfs/cdirs/desicollab/users/epaillas/acm/fits_emc/abacus/aug25/greedy/c{cosmo_idx:03}_hod{hod_idx:03}/cosmo-base_hod-base-VB-AB-CB-s/'
    # data_fn = Path(data_dir) / 'chain_number_density+minkowski_apr11+wp+tpcf+bk+dsc_pk+wst_apr11+dt_gv.npy'
    # data_fn = Path(data_dir) / 'chain_number_density+minkowski+wp+tpcf+bk+dsc_pk+wst+dt_gv.npy'
    data_fn = Path(data_dir) / 'chain_number_density+minkowski+wp+tpcf+bk+dsc_pk+wst+vide_gv+pdf+cgf.npy'
    # data_fn = Path(data_dir) / 'chain_number_density+minkowski_apr11+wp+tpcf+pk+bk+dsc_pk+wst_apr11+dt_gv+voxel_voids+pdf_r10+cgf_r10.npy'
    chain = Chain.load(data_fn)
    samples = Chain.to_getdist(chain, add_derived=True)
    chains.append(samples)
    legend_labels.append(rf'\textrm{{c{cosmo_idx:03d} cosmology}}')

markers = {
    'omega_cdm': [AbacusSummit(cosmo_idx)['omega_cdm'] for cosmo_idx in cosmos],
    'Omega_m': [AbacusSummit(cosmo_idx)['Omega_m'] for cosmo_idx in cosmos],
    'sigma8_m': [AbacusSummit(cosmo_idx).sigma8_m for cosmo_idx in cosmos],
}

# markers = chain.markers
# cosmo = AbacusSummit(0)
# markers.update({'Omega_m': cosmo['Omega_m'], 'h': cosmo['h']})
    
g = plots.get_subplot_plotter(width_inch=6)
g.settings.constrained_layout = True
g.settings.axis_marker_lw = 1.0
g.settings.axis_marker_ls = "--"
g.settings.title_limit_labels = False
g.settings.axis_marker_color = "k"
g.settings.legend_colored_text = True
g.settings.figure_legend_frame = False
g.settings.figure_legend_ncol = 1
g.settings.linewidth_contour = 1.0
g.settings.legend_fontsize = 23
g.settings.axes_fontsize = 20
g.settings.axes_labelsize = 28
g.settings.axis_tick_x_rotation = 45
g.settings.num_plot_contours = 3
# g.settings.axis_tick_max_labels = 6
g.settings.solid_colors = ['#4165c0', '#e770a2', '#5ac3be', '#696969', '#f79a1e'][::-1]
g.settings.line_styles = g.settings.solid_colors

g.triangle_plot(
    roots=chains,
    legend_labels=legend_labels,
    # markers=markers,
    params=params,
    filled=True,
    legend_loc='upper right',
    # filled=[True, False, False],
    # title_limit=1,
    # params=['logM_cut', 'logM_1', 'sigma', 'kappa', 'alpha']
    param_limits={'omega_cdm': [0.11, 0.135]},
    # legend_labels=stats,
)

colors = ['#e770a2', '#5ac3be', '#696969', '#f79a1e', '#ba7dcd']
symbols = ['o', 'X', '^', 'v']
for i, cosmo_idx in enumerate(cosmos):
    g.fig.axes[0].scatter(
        markers['omega_cdm'][i],
        markers['sigma8_m'][i],
        s=50,
        color=colors[i],
        marker=symbols[i],
        edgecolors='k',
        linewidths=1.0,
    )

plt.savefig('contours_cosmologies_v3.5.png', dpi=300, bbox_inches='tight')
plt.savefig('contours_cosmologies_v3.5.pdf', bbox_inches='tight')