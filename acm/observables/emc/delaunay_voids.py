from .base import BaseObservable
import logging


class DTVoidGalaxyCorrelationFunctionMultipoles(BaseObservable):
    """
    Class for the Emulator's Mock Challenge void-galaxy correlation
    function multipoles using the DT void finder.
    """
    def __init__(self, phase_correction=False, **kwargs):
        self.logger = logging.getLogger(self.__class__.__name__)
        self.stat_name = 'dt_gv'
        self.sep_name = 's'

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
            'hod_idx': list(range(270)),
        }

    @property
    def test_set_indices(self):
        """
        Indices of the test set samples, including variations in cosmology and HOD parameters.
        """
        return {
            'cosmo_idx': list(range(0, 5)) + list(range(13, 14)),
            'hod_idx': list(range(270)),
        }


    @property
    def coordinates(self):
        """
        Coordinates of the data and model vectors.
        """
        return{
            'multipoles': [0, 2],
            's': self.separation,
        }
    
    @property
    def coordinates_indices(self):
        """
        Indices of the (flat) coordinates of the data and model vectors.
        """
        return{'bin_idx': list(range(2 * len(self.separation)))}

    @property
    def model_fn(self):
        # return '/pscratch/sd/d/dforero/emc/trained_models/DTvoids_gv/cosmo+hod/optuna/last.ckpt'
        return '/pscratch/sd/e/epaillas/emc/v1.1/trained_models/best/DTVoidGalaxyCorrelationFunctionMultipoles/last.ckpt'

    def create_lhc(self, n_hod=20, cosmos=None, phase_idx=0, seed_idx=0):
        """
        Create the Latin hypercube samples for the emulator (both input and output features).
        """
        x, x_names = self.create_lhc_x(cosmos=cosmos, n_hod=n_hod)
        sep, y = self.create_lhc_y(n_hod=n_hod, cosmos=cosmos, phase_idx=phase_idx, seed_idx=seed_idx)
        return sep, x, x_names, y

    def create_lhc_y(self, n_hod=100, cosmos=None, phase_idx=0, seed_idx=0):
        """
        Create the output features for the emulator (the galaxy correlation function multipoles).
        """
        import numpy as np
        from pycorr import TwoPointCorrelationFunction
        base_dir = f'/pscratch/sd/d/dforero/projects/ac_emc/data/training_sets/dtfe6/tpcf/z0.5/yuan23_prior2/'
        if cosmos is None:
            cosmos = list(range(0, 5)) + list(range(13, 14)) + list(range(100, 127)) + list(range(130, 182))
        y = []
        for cosmo_idx in cosmos:
            print(cosmo_idx)
            data_dir = base_dir + f'c{cosmo_idx:03}_ph{phase_idx:03}/seed0/'
            for hod_idx in range(n_hod):
                data_fn = f"{data_dir}/gv_tpcf_hod{hod_idx:03d}.npy"
                data = TwoPointCorrelationFunction.load(data_fn)[::4].select((0, 150))
                s, multipoles = data(ells=(0, 2), return_sep=True)
                y.append(np.concatenate(multipoles))
        return s, np.array(y)

    def create_lhc_x(self, cosmos=None, n_hod=100):
        """
        Create the input features for the emulator (the cosmological and HOD parameters).
        """
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

    def create_small_box_y(self):
        """
        Create the output features for the emulator (the galaxy correlation function multipoles)
        from the small AbacusSummit box.
        """
        from pathlib import Path
        from pycorr import TwoPointCorrelationFunction
        import numpy as np
        import glob
        y = []
        data_dir = f'/pscratch/sd/d/dforero/projects/ac_emc/data/training_sets/dtfe6/tpcf/z0.5/full_prior2/small_onelos_hod466/'
        data_fns = glob.glob(f"{data_dir}/gv_tpcf_ph*.npy")
        for data_fn in data_fns:
            data = TwoPointCorrelationFunction.load(data_fn)[::4].select((0, 150))
            s, multipoles = data(ells=(0, 2), return_sep=True)
            y.append(np.concatenate(multipoles))
        return s, np.array(y)

    def create_diffsk_y(self):
        """
        Create the output features for the emulator (the galaxy correlation function multipoles).
        """
        import numpy as np
        from pathlib import Path
        from pycorr import TwoPointCorrelationFunction
        self.logger.info('Creating Diffsky y for the DT void-galaxy correlation function multipoles.')
        base_dir = f'/pscratch/sd/d/dforero/projects/ac_emc/data/data_vectors/diffsky/unit/dtfe6/tpcf/z0.5'
        for phase in [1, 2]:
            for sample in ['mass', 'mass_conc']:
                data_fn = f"{base_dir}/gv_tpcf_galsampled_diffsky_mock_67120_fixedAmp_{phase:03}_{sample}_v0.3.npy"
                data = TwoPointCorrelationFunction.load(data_fn)[::4].select((0, 150))
                s, multipoles = data(ells=(0, 2), return_sep=True)
                save_dir = Path('/pscratch/sd/e/epaillas/emc/v1.1/diffsky/data_vectors/')
                save_fn = save_dir / f'galsampled_67120_fixedAmp_{phase:03}_{sample}_v0.3/dt_gv.npy'
                cout = {'s': s, 'diffsky_y': np.concatenate(multipoles)}
                np.save(save_fn, cout)
        return


if __name__ == '__main__':
    import logging
    logging.basicConfig(level=logging.INFO)
    dt_gv = DTVoidGalaxyCorrelationFunctionMultipoles()
    dt_gv.create_diffsk_y()
    # Uncomment the following line to create the small box y
    # s, y = dt_gv.create_small_box_y()
    # print(s, y.shape)