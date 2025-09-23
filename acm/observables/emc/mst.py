from .base import BaseObservable
import logging


class MinimumSpanningTree(BaseObservable):
    """
    Class for the Emulator's Mock Challenge minimum spanning tree.
    """

    def __init__(self, phase_correction=False, **kwargs):
        self.logger = logging.getLogger(self.__class__.__name__)
        self.stat_name = 'mst'
        self.sep_name = 'bin_idx'

        if phase_correction and hasattr(self, 'compute_phase_correction'):
            self.logger.info('Computing phase correction.')
            self.phase_correction = self.compute_phase_correction()

        super().__init__(**kwargs)

    @property
    def lhc_indices(self):
        """
        Indices of the Latin hypercube samples, including variations in cosmology and HOD parameters.
        """
        return {
            'cosmo_idx': list(range(0, 5)) + list(range(13, 14)) + list(range(100, 127)) + list(range(130, 182)),
            'hod_idx': list(range(350)),
        }

    @property
    def test_set_indices(self):
        """
        Indices of the test set samples, including variations in cosmology and HOD parameters.
        """
        return {
            'cosmo_idx': list(range(0, 5)) + list(range(13, 14)),
            'hod_idx': list(range(350)),
        }


    @property
    def coordinates(self):
        """
        Coordinates of the data and model vectors.
        """
        return{
            self.sep_name: self.separation,
        }
    
    @property
    def coordinates_indices(self):
        """
        Indices of the (flat) coordinates of the data and model vectors.
        """
        return{'bin_idx': list(range(len(self.separation)))}

    @property
    def model_fn(self):
        # return f'/pscratch/sd/e/epaillas/emc/trained_models/mst/cosmo+hod/optuna/last-v8.ckpt'
        return '/pscratch/sd/k/knaidoo/ACM/MockChallenge/emulators/emulate_3p0/last-v2.ckpt'

    def get_emulator_error(self, select_filters=None, slice_filters=None):
        from sunbird.data.data_utils import convert_to_summary
        from pathlib import Path
        import numpy as np
        error_dir = '/pscratch/sd/e/epaillas/emc/v1.1/emulator_error/'
        error_fn = Path(error_dir) / f'{self.stat_name}.npy'
        error = np.load(error_fn, allow_pickle=True).item()['emulator_error']
        coords = self.coordinates_indices if self.select_indices else self.coordinates
        coords_shape = tuple(len(v) for k, v in coords.items())
        dimensions = list(coords.keys())
        error = error.reshape(*coords_shape)
        select_filters = self.select_coordinates if self.select_coordinates else self.select_indices
        slice_filters = self.slice_coordinates
        return convert_to_summary(
            data=error, dimensions=dimensions, coords=coords,
            select_filters=select_filters, slice_filters=slice_filters
        ).values.reshape(-1)

    def create_diffsky_y(self):
        import numpy as np
        from pathlib import Path
        smoothing = '3p0'
        path = '/pscratch/sd/k/knaidoo/ACM/MockChallenge/data/'
        data = np.load(path + 'mst_diffsky_data_with_smoothing_%s_Npt_10.npz' % smoothing)
        vecs_diffsky = data['data'] # same as before but now for the diffsky data
        which_diffsky = data['which_diffskys'] # the diffsky names
        base_dir = Path('/pscratch/sd/e/epaillas/emc/v1.1/diffsky/data_vectors/')
        for phase in [1, 2]:
            for sample in ['mass', 'mass_conc']:
                idx = list(which_diffsky).index(f'67120_fixedAmp_{phase:03}_{sample}') 
                y = vecs_diffsky[idx]
                print(y)
                save_fn = base_dir / f'galsampled_67120_fixedAmp_{phase:03}_{sample}_v0.3/mst.npy'
                np.save(save_fn, {'bin_idx': np.arange(len(y)), 'diffsky_y': y})

    def create_lhc(self, cosmos=None, n_hod=350):
        """
        Create the Latin hypercube samples for the emulator (both input and output features).
        """
        import numpy as np
        x, x_names = self.create_lhc_x(cosmos=cosmos, n_hod=n_hod)
        bin_idx, y = self.create_lhc_y(cosmos=cosmos, n_hod=n_hod)
        cout = {'bin_idx': bin_idx, 'lhc_x': x, 'lhc_x_names': x_names, 'lhc_y': y}
        save_fn = '/pscratch/sd/e/epaillas/emc/v1.1/abacus/training_sets/cosmo+hod/mst.npy'
        np.save(save_fn, cout)
        return

    def create_lhc_x(self, cosmos=None, n_hod=350):
        import pandas
        import numpy as np
        if cosmos is None:
            cosmos = list(range(0, 5)) + list(range(13, 14)) + list(range(100, 127)) + list(range(130, 182))
        lhc_x = []
        for cosmo_idx in cosmos:
            data_dir = '/pscratch/sd/e/epaillas/emc/cosmo+hod_params/'
            data_fn = data_dir + f'AbacusSummit_c{cosmo_idx:03}.csv'
            lhc_x_i = pandas.read_csv(data_fn)
            lhc_x_names = list(lhc_x_i.columns)
            lhc_x_names = [name.replace(' ', '').replace('#', '') for name in lhc_x_names]
            lhc_x.append(lhc_x_i.values[:n_hod, :])
        lhc_x = np.concatenate(lhc_x)
        return lhc_x, lhc_x_names

    def create_lhc_y(self, cosmos=None, n_hod=350):
        import numpy as np
        if cosmos is None:
            cosmos = list(range(0, 5)) + list(range(13, 14)) + list(range(100, 127)) + list(range(130, 182))
        smoothing = '0p0'
        path = '/pscratch/sd/k/knaidoo/ACM/MockChallenge/data/'
        data = np.load(path + 'mst_emu_data_with_smoothing_%s_Npt_10.npz' % smoothing)
        aind = data['aind'] # abacus simulation index
        hind = data['hind'] # HOD index
        vecs_emu = data['data'] # same as before but now for the emulator data vectors.
        y = []
        for cosmo_idx in cosmos:
            for hod_idx in range(n_hod):
                if (cosmo_idx, hod_idx) in zip(aind, hind):
                    y.append(vecs_emu[list(zip(aind, hind)).index((cosmo_idx, hod_idx))])
                else:
                    y.append(np.zeros_like(vecs_emu[0]))
        bin_idx = np.arange(len(vecs_emu[0]))
        return bin_idx, np.array(y)

    def create_small_box_y(self):
        import numpy as np
        smoothing = '0p0' # or '3p0'
        path = '/pscratch/sd/k/knaidoo/ACM/MockChallenge/data/'
        data = np.load(path + 'mst_cov_data_with_smoothing_%s_Npt_10.npz' % smoothing)
        hind = data['hind'] # covariance HOD index
        vecs_cov = data['data'] # arrays of vectors, so vecs_cov[0] is the first array and so on.
        bin_idx = np.arange(len(vecs_cov[0]))
        cout = {'bin_idx': bin_idx, 'cov_y': vecs_cov}
        save_fn = '/pscratch/sd/e/epaillas/emc/v1.1/abacus/covariance_sets/small_box/mst.npy'
        np.save(save_fn, cout)
        return