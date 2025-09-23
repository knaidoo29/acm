import matplotlib.pyplot as plt
from pathlib import Path
from getdist import plots, MCSamples, loadMCSamples
from sunbird.inference.samples import Chain
import sys
from cosmoprimo.fiducial import AbacusSummit
sys.path.insert(1, '/global/cfs/cdirs/desicollab/users/epaillas/code/gqc-y3-bao/cosmo_paper/desi-y3-kp/scripts')
sys.path.insert(1, '/global/cfs/cdirs/desicollab/users/epaillas/code/gqc-y3-bao/cosmo_paper/desi-y3-kp/')
from y3_bao_cosmo_tools import load_cobaya_samples
import numpy as np

plt.rc('text', usetex=True)
plt.rc('font', family='serif')

def summit_cosmo_table(params):
    import pandas
    table_fn = "/global/cfs/cdirs/desicollab/users/epaillas/code/sunbird/sunbird/inference/priors/summit_cosmologies.txt"
    df = pandas.read_csv(table_fn, delimiter=',')
    df.columns = df.columns.str.strip()
    df.columns = list(df.columns.str.strip('# ').values)
    return np.c_[[df[param] for param in params]].T


abacus = summit_cosmo_table(params=['w0_fld', 'wa_fld'])



desi_d5_cmb = load_cobaya_samples(model='base_w_wa', dataset=['desi-bao-all', 'desy5sn', 'planck2018-lowl-TT-clik', 'planck2018-lowl-EE-clik', 'planck-NPIPE-highl-CamSpec-TTTEEE', 'planck-act-dr6-lensing'], convergence=True)
# desi_d5_cmb.addDerived(desi_d5_cmb.getParams().w, name='w0_fld', label='w_0')
# desi_d5_cmb.addDerived(desi_d5_cmb.getParams().wa, name='wa_fld', label='w_a')

chains = []
legend_labels = []

# params = ['logM_cut', 'logM_1', 'sigma', 'kappa', 'alpha']
# params = ['omega_cdm', 'sigma8_m', 'n_s', 'logM_cut', 'logM_1', 'sigma', 'kappa', 'alpha']
params = ['w0_fld', 'wa_fld']
# params = ['A_cen', 'A_sat', 'B_cen', 'B_sat']

# cosmos = [0, 2, 178]
# hods = [30, 13, 0]

cosmos = [0, 2]
hods = [30, 13]

for cosmo_idx, hod_idx in zip(cosmos, hods):
    # data_dir = f'/global/cfs/cdirs/desicollab/users/epaillas/acm/fits_emc/abacus/optimal/apr11/c{cosmo_idx:03}_hod{hod_idx:03}/w0wa_nrun_Nur/'
    # data_dir = f'/global/cfs/cdirs/desicollab/users/epaillas/acm/fits_emc/abacus/optimal/apr11/c{cosmo_idx:03}_hod{hod_idx:03}/w0wa_nrun_Nur/'
    # data_dir = f'/global/cfs/cdirs/desicollab/users/epaillas/acm/fits_emc/diffsky/aug11/greedy/galsampled_67120_fixedAmp_mean_mass_conc_v0.3/cosmo-base_hod-base-VB-AB-CB-s/'
    # data_fn = Path(data_dir) / 'chain_number_density+minkowski_apr11+wp+tpcf+bk+dsc_pk+wst_apr11+dt_gv.npy'

    data_dir = f'/global/cfs/cdirs/desicollab/users/epaillas/acm/fits_emc/abacus/aug25/greedy-bLCDM/c{cosmo_idx:03}_hod{hod_idx:03}/cosmo-base-w0-wa_hod-base-VB-AB-CB-s/'
    data_fn = Path(data_dir) / 'chain_number_density+minkowski+wp+tpcf+bk+dsc_pk+wst+vide_gv+vide_vsf_rsd+mst+pdf+cgf.npy'
    # data_fn = Path(data_dir) / 'chain_number_density+minkowski+wp+tpcf+bk+dsc_pk+wst+vide_gv+pdf.npy'

    # data_fn = Path(data_dir) / 'chain_number_density+minkowski_apr11+wp+tpcf+pk+bk+dsc_pk+wst_apr11+dt_gv+voxel_voids+pdf_r10+cgf_r10.npy'
    chain = Chain.load(data_fn)
    samples = Chain.to_getdist(chain, add_derived=False,
    # settings={'fine_bins_2D': 128, 'smooth_scale_1D': 0.22, 'smooth_scale_2D': 0.5},
    )
    chains.append(samples)
    legend_labels.append(rf'\textrm{{Greedy comb.~(c{cosmo_idx:03d})}}')

# data_fn = '/global/cfs/cdirs/desicollab/users/epaillas/acm/fits_emc/abacus/apr11/c000_hod030/w0wa/chain_prior.npy'
# chain = Chain.load(data_fn)
# samples = Chain.to_getdist(
#     chain,
#     add_derived=False,
#     settings={'fine_bins_2D': 128, 'smooth_scale_1D': 0.5, 'smooth_scale_2D': 0.5},
# )
# chains.append(samples)
# legend_labels.append(r'$\textrm{Prior}$')


chains.append(desi_d5_cmb)
legend_labels.append(r'$\textrm{BAO+CMB+SNe}$')

markers = {
    'w0_fld': [AbacusSummit(cosmo_idx)['w0_fld'] for cosmo_idx in cosmos],
    'wa_fld': [AbacusSummit(cosmo_idx)['wa_fld'] for cosmo_idx in cosmos],
}

# markers = chain.markers
# cosmo = AbacusSummit(0)
# markers.update({'Omega_m': cosmo['Omega_m'], 'h': cosmo['h']})
    
g = plots.get_subplot_plotter(width_inch=6)
g.settings.constrained_layout = True
g.settings.axis_marker_lw = 1.0
g.settings.axis_marker_ls = "--"
g.settings.title_limit_labels = False
g.settings.legend_colored_text = True
g.settings.figure_legend_frame = False
g.settings.figure_legend_ncol = 1
g.settings.linewidth_contour = 1.0
g.settings.legend_fontsize = 20
g.settings.axes_fontsize = 20
g.settings.axes_labelsize = 28
g.settings.axis_tick_x_rotation = 45
g.settings.num_plot_contours = 2
# g.settings.axis_tick_max_labels = 6
g.settings.solid_colors = ['#f79a1e', '#e770a2', 'dimgrey', '#5ac3be'][::-1]
g.settings.axis_marker_color = g.settings.solid_colors
# g.settings.line_styles = g.settings.solid_colors

g.triangle_plot(
    roots=chains,
    legend_labels=legend_labels,
    # markers=markers,
    params=params,
    legend_loc='upper right',
    filled=[True, True, True],
    # title_limit=1,
    # params=['logM_cut', 'logM_1', 'sigma', 'kappa', 'alpha']
    param_limits={'w0_fld': [-1.2, -0.6], 'wa_fld': [-1.5, 0.8]},
    # legend_labels=stats,
)

colors = ['#f79a1e', '#e770a2', 'dimgrey', '#5ac3be']
# symbols = ['X', '^']
# for i, cosmo_idx in enumerate(cosmos):
#     g.fig.axes[0].scatter(
#         markers['w0_fld'][i],
#         markers['wa_fld'][i],
#         s=50,
#         color=colors[i],
#         marker=symbols[i],
#         edgecolors='k',
#         linewidths=1.0,
#     )

g.fig.axes[0].scatter(
    abacus[:, 0],
    abacus[:, 1],
    s=20,
    color='crimson',
    marker='o',
    edgecolors='k',
    linewidths=1.0,
    label=r'$\textrm{AbacusSummit}$',
)

g.fig.axes[0].legend(loc='lower left', fontsize=12, handletextpad=0.5,
                     handlelength=0, ncol=2, columnspacing=0.5)

# w0 truth lines
g.fig.axes[1].axvline(-1.0, color=colors[1], linestyle='--', linewidth=1.0)
g.fig.axes[1].axvline(-0.7, color=colors[2], linestyle='--', linewidth=1.0)

# wa truth lines
g.fig.axes[2].axvline(0.0, color=colors[1], linestyle='--', linewidth=1.0)
g.fig.axes[2].axvline(-0.5, color=colors[2], linestyle='--', linewidth=1.0)

plt.savefig('contours_cosmologies_w0wa_v3.5.png', dpi=300, bbox_inches='tight')
plt.savefig('contours_cosmologies_w0wa_v3.5.pdf', bbox_inches='tight')