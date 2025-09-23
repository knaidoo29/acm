from .base import BaseObservable
import logging


class VIDEVoidGalaxyCorrelationFunctionMultipoles(BaseObservable):
    """
    Class for the Emulator's Mock Challenge void-galaxy correlation
    function multipoles using the VIDE void finder.
    """
    def __init__(self, phase_correction=False, **kwargs):
        self.logger = logging.getLogger(self.__class__.__name__)
        self.stat_name = 'vide_gv'
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
            'hod_idx': list(range(100)),
        }

    @property
    def test_set_indices(self):
        """
        Indices of the test set samples, including variations in cosmology and HOD parameters.
        """
        return {
            'cosmo_idx': list(range(0, 5)) + list(range(13, 14)),
            'hod_idx': list(range(100)),
        }

    @property
    def small_box_indices(self):
        """
        Indices of the covariance samples, including variations in phase and HOD parameters.
        """
        return {
            'phase_idx': list(range(1786)),
        }

    @property
    def coordinates(self):
        """
        Coordinates of the data and model vectors.
        """
        return{
            'stack': [0, 1, 2, 3],
            'multipoles': [0, 2],
            's': self.separation,
        }
    
    @property
    def coordinates_indices(self):
        """
        Indices of the (flat) coordinates of the data and model vectors.
        """
        return{'bin_idx': list(range(4 * 2 * len(self.separation)))}

    @property
    def model_fn(self):
        return f'/pscratch/sd/e/epaillas/emc/v1.1/trained_models/best/VIDEVoidGalaxyCorrelationFunctionMultipoles/last.ckpt'

    def create_diffsky_y(self):
        import numpy as np
        from pathlib import Path
        s = np.load('/global/u1/e/epaillas/vide_data/diffsky_x_values_Mpc_units.npy')
        data_fn = '/global/u1/e/epaillas/vide_data/diffsky_stacked_multipoles_Mpc_units.npy'
        data = np.load(data_fn)
        base_dir = Path('/pscratch/sd/e/epaillas/emc/v1.1/diffsky/data_vectors/')
        idx = 0
        for phase in [1, 2]:
            for sample in ['mass', 'mass_conc']:
                y = data[idx]
                save_fn = base_dir / f'galsampled_67120_fixedAmp_{phase:03}_{sample}_v0.3/vide_gv.npy'
                np.save(save_fn, {'s': s, 'diffsky_y': y})
                idx += 1