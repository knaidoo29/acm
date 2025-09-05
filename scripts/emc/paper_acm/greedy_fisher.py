import json
from pathlib import Path
import acm.observables.emc as emc
from acm.compression.greedy_fisher import run_greedy_fisher

def get_parameter_idx(lhc_x_names, parameters):
    return [
        lhc_x_names.index(param) for param in parameters
    ]

def run_marginalised_greedy_fisher(
    statistics, max_bins, fixed_parameters_idx, nuisance_parameters_idx, warmup=False,
):
    if warmup:
        bins_init, _, _ = run_greedy_fisher(
            statistics,
            max_bins=25, 
            add_emulator_error=True,
            fixed_parameters_idx=fixed_parameters_idx,
        )
    else:
        bins_init = None

    bins_marg, _, fisher_marg, selection_history = run_greedy_fisher(
        statistics,
        max_bins=max_bins, 
        add_emulator_error=True,
        fixed_parameters_idx=fixed_parameters_idx,
        nuisance_parameters_idx=nuisance_parameters_idx,
        initial_selection_bins=bins_init,
    )
    return bins_marg, fisher_marg, selection_history

if __name__ == '__main__':
    max_bins = 200
    data_path = Path('/pscratch/sd/e/epaillas/emc/greedy_fisher')
    fixed_parameters = ['omega_b']
    lcdm_parameters = ['omega_b', 'omega_cdm', 'sigma8_m', 'n_s']
    blcdm_parameters = ['nrun', 'N_ur', 'w0_fld', 'wa_fld']
    hod_parameters = [
        'logM_cut', 'logM_1', 'sigma', 'alpha', 'kappa', 'alpha_c',
        'alpha_s', 's', 'A_cen', 'A_sat', 'B_cen', 'B_sat'
    ]
    select_mocks={'cosmo_idx': [0], 'hod_idx': [30,],}
    statistics = {
        'wp': emc.GalaxyProjectedCorrelationFunction(
            select_mocks=select_mocks,
        ),
        'tpcf': emc.GalaxyCorrelationFunctionMultipoles(
            select_mocks=select_mocks,
        ),
        # 'pk': emc.GalaxyPowerSpectrumMultipoles(
        #     select_mocks=select_mocks,
        # ),
        'bk': emc.GalaxyBispectrumMultipoles(
            select_mocks=select_mocks,
        ),
        'density_split': emc.DensitySplitPowerSpectrumMultipoles(
            select_mocks=select_mocks,
        ),
        'minkowski': emc.MinkowskiFunctionals(
            select_mocks=select_mocks,
        ),
        # 'dt_gv': emc.DTVoidGalaxyCorrelationFunctionMultipoles(
        #     select_mocks=select_mocks,
        # ),
        'vide_gv': emc.VIDEVoidGalaxyCorrelationFunctionMultipoles(
            select_mocks=select_mocks,
        ),
        'vide_vsf': emc.VIDEVoidSizeFunction(
            select_mocks=select_mocks,
        ),
        'wst': emc.WaveletScatteringTransform(
            select_mocks=select_mocks,
        ),
        'mst': emc.MinimumSpanningTree(
            select_mocks=select_mocks,
        ),
        'pdf': emc.GalaxyOverdensityPDF(
            select_mocks=select_mocks,
        ),
        'cgf': emc.CumulantGeneratingFunction(
            select_mocks=select_mocks,
        ),
    }


    lhc_x_names = statistics['wp'].lhc_x_names
    blcdm_param_idx = get_parameter_idx(lhc_x_names, blcdm_parameters)
    lcdm_param_idx = get_parameter_idx(lhc_x_names, lcdm_parameters)
    hod_param_idx = get_parameter_idx(lhc_x_names, hod_parameters)

    # Greedy optimised all
    bins, _, fisher, selection_history = run_greedy_fisher(
       statistics,
       max_bins=max_bins, 
       add_emulator_error=True,
       fixed_parameters_idx=None,
    )
    all_fisher = {
       'fisher': fisher,
       'bins': bins,
       'selection_history': selection_history,
    }
    with open(data_path / 'all_v3.5.json', 'w') as f:
       json.dump(all_fisher, f)

    # # Greedy fixed blcdm
    # bins, _, fisher, selection_history_fixed = run_greedy_fisher(
    #    statistics,
    #    max_bins=max_bins, 
    #    add_emulator_error=True,
    #    fixed_parameters_idx=blcdm_param_idx,
    # )
    # all_fisher = {
    #    'fisher': fisher,
    #    'bins': bins,
    #    'selection_history': selection_history_fixed,
    # }
    # with open(data_path / 'fixed_blcdm_v2.json', 'w') as f:
    #    json.dump(all_fisher, f)

    # #  Fix blcdm, marginalised over hod
    # bins_fixed_blcdm, fisher_fixed_blcdm, selection_history_fixed_blcdm = run_marginalised_greedy_fisher(
    #     statistics,
    #     max_bins=max_bins, 
    #     fixed_parameters_idx=blcdm_param_idx,
    #     nuisance_parameters_idx=hod_param_idx,
    # )
    # all_fisher = {
    #     'fisher': fisher_fixed_blcdm,
    #     'bins': bins_fixed_blcdm,
    #     'selection_history': selection_history_fixed_blcdm,
    # }
    # with open(data_path / 'fixed_blcdm_marginalised_hod_v3.5.json', 'w') as f:
    #     json.dump(all_fisher, f)


    # #  Fix lcdm, marginalised over hod
    # # bins_fixed_lcdm, fisher_fixed_lcdm, selection_history_fixed_lcdm = run_marginalised_greedy_fisher(
    # #     statistics,
    # #     max_bins=max_bins, 
    # #     fixed_parameters_idx=lcdm_param_idx,
    # #     nuisance_parameters_idx=hod_param_idx,
    # # )
    # # all_fisher = {
    # #     'fisher': fisher_fixed_lcdm,
    # #     'bins': bins_fixed_lcdm,
    # #     'selection_history': selection_history_fixed_lcdm,
    # # }
    # # with open(data_path / 'fixed_lcdm_marginalised_hod_v2.json', 'w') as f:
    # #     json.dump(all_fisher, f)


    # bins_marg_lcdm, fisher_marg_lcdm, selection_history_marg_lcdm = run_marginalised_greedy_fisher(
    #     statistics,
    #     max_bins=max_bins, 
    #     fixed_parameters_idx=None,
    #     nuisance_parameters_idx=hod_param_idx + lcdm_param_idx,
    # )
    # all_fisher = {
    #     'fisher': fisher_marg_lcdm,
    #     'bins': bins_marg_lcdm,
    #     'selection_history': selection_history_marg_lcdm,
    # }
    # with open(data_path / 'marginalised_hod_lcdm_v3.5.json', 'w') as f:
    #     json.dump(all_fisher, f)


    # bins_hod, fisher_hod, selection_history_hod = run_marginalised_greedy_fisher(
    #     statistics,
    #     max_bins=max_bins, 
    #     fixed_parameters_idx=blcdm_param_idx,
    #     nuisance_parameters_idx=lcdm_param_idx,
    # )
    # all_fisher = {
    #     'fisher': fisher_hod,
    #     'bins': bins_hod,
    #     'selection_history': selection_history_hod,
    # }
    # with open(data_path / 'hod_v3.5.json', 'w') as f:
    #     json.dump(all_fisher, f)
