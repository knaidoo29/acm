from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
import json

plt.rc('text', usetex=True)
plt.rc('font', family='serif')

data_path = Path('/pscratch/sd/e/epaillas/emc/greedy_fisher')
# data_path = Path('/pscratch/sd/c/cuesta/greedy_fisher')


def get_fisher_per_statistic(fisher_dict, normalize=True,):
    statistics = list(fisher_dict['bins'].keys())

    fisher_per_statistic = {
        statistic: 0.0 for statistic in statistics
    }

    selection_history = fisher_dict['selection_history']
    for selected in selection_history:
        if np.isfinite(selected['improvement']):
            if selected['improvement'] > 0.:
                fisher_per_statistic[selected['statistic']] += selected['improvement']
        else:
            fisher_per_statistic[selected['statistic']]  += np.abs(selected['fisher_after'])
    if normalize:
        fisher_per_statistic = {stat: val / np.sum(list(fisher_per_statistic.values())) for stat, val in fisher_per_statistic.items()}
    return fisher_per_statistic

def plot_fisher_convergence(ax=None, show=False):
    def read_fisher(filename):
        with open(data_path / filename, 'r') as f:
            greedy = json.load(f)
        return greedy
    import json
    from matplotlib.patheffects import PathPatchEffect, SimpleLineShadow, Normal
    from pathlib import Path
    import numpy as np
    import matplotlib.pyplot as plt
    if ax is None:
        fig, ax = plt.subplots(figsize=(4, 3))
    filenames = [
        'fixed_blcdm_marginalised_hod_v2.json',
        'marginalised_hod_lcdm_v2.json',
        'hod_v2.json'
    ]
    labels = [r'Greedy-$\Lambda$CDM', r'Greedy-b$\Lambda$CDM', 'Greedy-HOD']
    linestyles = ['-', '--', '-.']
    # colors = ['#4165c0', '#e770a2', '#5ac3be']
    colors = ['#440154', '#31688e', '#fde725']
    for fn, label, ls, color in zip(filenames, labels, linestyles, colors):
        fisher = read_fisher(fn)['fisher']
        ax.plot(np.arange(len(fisher)), fisher/np.max(fisher), label=label, ls=ls, color=color, lw=2.3,
        path_effects=[SimpleLineShadow(shadow_color="black", linewidth=1.5, offset=(1, -1)),Normal()])
    ax.set_ylim(0, 1.1)
    ax.set_xlim(0, 200)
    ax.tick_params(axis='both', labelsize=8)
    # ax.legend(fontsize=12, loc='lower right', frameon=True)
    ax.set_xlabel(r'$\textrm{Iteration}$', fontsize=10)
    ax.set_ylabel(r'$\textrm{Information}$', fontsize=10)
    ax.hlines(1, 0, 200, color='k', linestyle='--', lw=1.0)
    return ax
    # plt.savefig('fisher_convergence.pdf', bbox_inches='tight')
    # if show:
    #     plt.show()
    # plt.close()



with open(data_path / 'fixed_blcdm_marginalised_hod_v3.5.json', 'r') as f:
    greedy_lcdm = json.load(f)

with open(data_path / 'marginalised_hod_lcdm_v3.5.json', 'r') as f:
    greedy_blcdm = json.load(f)

with open(data_path / 'hod_v3.5.json', 'r') as f:
    greedy_hod = json.load(f)

with open(data_path / 'all_v3.5.json', 'r') as f:
    greedy_all = json.load(f)


fisher_per_statistic_lcdm = get_fisher_per_statistic(greedy_lcdm)
fisher_per_statistic_blcdm = get_fisher_per_statistic(greedy_blcdm)
fisher_per_statistic_hod = get_fisher_per_statistic(greedy_hod)
fisher_per_statistic_all = get_fisher_per_statistic(greedy_all)


labels = { 
    'wp': 'Projected 2PCF',
    'tpcf': '2PCF',
    'bk':  'Bispectrum',
    'minkowski': 'Minkowski Func.',
    'wst': 'Wavelet ST',
    'dt_gv': 'Void-galaxy CCF',
    'vide_gv': 'Void-galaxy CCF',
    'density_split': 'Density Split',
    'vide_vsf': 'Void size function',
    'mst': 'M. Spanning Tree',
    'pdf': 'Overdensity PDF',
    'cgf': 'Cumulant GF',

}


stats = [labels[stat] for stat in list(fisher_per_statistic_lcdm.keys())]

x = np.arange(len(stats))
# width = 0.25
width = 0.18

fig, ax = plt.subplots(figsize=(8, 4))
bars1 = ax.bar(x - width, list(fisher_per_statistic_lcdm.values()), width, label=r'Greedy-$\Lambda$CDM', color='#440154',) 
bars2 = ax.bar(x, list(fisher_per_statistic_blcdm.values()), width, label=r'Greedy-b$\Lambda$CDM', color='#31688e',)
bars3 = ax.bar(x + width, list(fisher_per_statistic_hod.values()), width, label='Greedy-HOD', color='#fde725',)
bars4 = ax.bar(x + 2*width, list(fisher_per_statistic_all.values()), width, label='Greedy-All', color='#73d055', alpha=0.7,)

import matplotlib.patheffects as pe

# After creating bars1, bars2, bars3
# for bars in [bars1, bars2, bars3]:
for bars in [bars1, bars2, bars3, bars4]:
    for bar in bars:
        bar.set_path_effects([
            pe.withStroke(linewidth=2, foreground="black", alpha=0.7)  # shadow/outline
        ])

# for bars in [bars1, bars2, bars3]:
#     for bar in bars:
#         # add a grey rectangle behind each bar as shadow
#         ax.add_patch(plt.Rectangle(
#             (bar.get_x() + 0.02, 0),    # slightly shifted in x
#             bar.get_width(),
#             bar.get_height(),
#             color="black",
#             alpha=0.5,
#             zorder=bar.zorder-1  # put behind the bar
#         ))

left, bottom, width, height = [0.79, 0.69, 0.17, 0.25]
ax2 = fig.add_axes([left, bottom, width, height])
ax2 = plot_fisher_convergence(ax=ax2, show=False)

ax.set_xlabel('Statistics', fontsize=13)
ax.set_ylabel('Fisher Information Density', fontsize=13)
ax.set_xticks(x)
ax.tick_params(axis='y', labelsize=11)
ax.set_xticklabels(stats, rotation=45, fontsize=12)
ax.legend(loc='upper left')
# ax.tick_params(axis='x', length=15)
ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('greedy_bar_plot_v3.5_all.pdf', bbox_inches='tight')