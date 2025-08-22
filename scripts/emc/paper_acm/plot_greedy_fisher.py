from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
import json

data_path = Path('/pscratch/sd/c/cuesta/greedy_fisher')


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


with open(data_path / 'fixed_blcdm_marginalised_hod_v2.json', 'r') as f:
    greedy_lcdm = json.load(f)

with open(data_path / 'marginalised_hod_lcdm_v2.json', 'r') as f:
    greedy_blcdm = json.load(f)

with open(data_path / 'hod_v2.json', 'r') as f:
    greedy_hod = json.load(f)


fisher_per_statistic_lcdm = get_fisher_per_statistic(greedy_lcdm)
fisher_per_statistic_blcdm = get_fisher_per_statistic(greedy_blcdm)
fisher_per_statistic_hod = get_fisher_per_statistic(greedy_hod)


labels = { 
    'wp': 'Projected 2PCF',
    'tpcf': '2PCF',
    'bk':  'Bispectrum',
    'minkowski': 'Minkowski Func.',
    'wst': 'Wavelet ST',
    'dt_gv': 'DT Voids',
    'density_split': 'Density Split',
}


stats = [labels[stat] for stat in list(fisher_per_statistic_lcdm.keys())]

x = np.arange(len(stats))
width = 0.25

fig, ax = plt.subplots(figsize=(8, 4))
bars1 = ax.bar(x - width, list(fisher_per_statistic_lcdm.values()), width, label='Greedy-LCDM', color='#440154',) 
bars2 = ax.bar(x, list(fisher_per_statistic_blcdm.values()), width, label='Greedy-bLCDM', color='#31688e',)
bars3 = ax.bar(x + width, list(fisher_per_statistic_hod.values()), width, label='Greedy-HOD', color='#fde725',)

ax.set_xlabel('Statistics')
ax.set_ylabel('Fisher Information Density')
ax.set_xticks(x)
ax.set_xticklabels(stats, rotation=45)
ax.legend()
ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('greedy_bar_plot.pdf', bbox_inches='tight')