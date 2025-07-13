import pytest
import numpy as np
import torch
from scipy.stats import multivariate_normal
from acm.compression.greedy_fisher import safe_inverse, run_greedy_fisher 
import acm.observables.emc as emc


STAT_NAMES = [
    'tpcf', 'bk', 'pk', 'dsc_pk', 'wp', 'vg_voids', 
    'dt_voids', 'cumulant', 'minkowski', 'wst', 'mst', 'pdf'
]

@pytest.fixture
def statistics():
    """Fixture that creates and returns the stat_map object for use in multiple tests"""
    select_mocks = {'cosmo_idx': [0], 'hod_idx': [30,]}
    return {
        'tpcf': emc.GalaxyCorrelationFunctionMultipoles(
            select_mocks=select_mocks,
        ),
        'bk': emc.GalaxyBispectrumMultipoles(
            select_mocks=select_mocks,
        ),
        'pk': emc.GalaxyPowerSpectrumMultipoles(
            select_mocks=select_mocks,
        ),
        'dsc_pk': emc.DensitySplitPowerSpectrumMultipoles(
            select_mocks=select_mocks,
        ),
        'wp': emc.GalaxyProjectedCorrelationFunction(
            select_mocks=select_mocks,
        ),
        'vg_voids': emc.VoxelVoidGalaxyCorrelationFunctionMultipoles(
            select_mocks=select_mocks,
        ),
        'dt_voids': emc.DTVoidGalaxyCorrelationFunctionMultipoles(
            select_mocks=select_mocks,
        ),
        'cumulant': emc.CumulantGeneratingFunction(
            select_mocks=select_mocks,
        ),
        'minkowski': emc.MinkowskiFunctionals(
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
    }



@pytest.mark.parametrize("stat_name", STAT_NAMES)
def test_well_behaved_covariance(statistics, stat_name, atol=1e-4, rtol=1e-4):
    """Test that covariance matrix and its inverse produce identity matrix."""
    stat = statistics[stat_name]
    small_box_y = stat.small_box_y
    covariance_matrix = np.cov(small_box_y.T)
    correction = stat.get_covariance_correction(
        n_s=len(small_box_y),
        n_d=len(covariance_matrix),
        n_theta=20,
        method='percival',
    )
    print(f"{stat_name}, correction = {correction}")
    precision_matrix = safe_inverse(covariance_matrix)
    identity_check = covariance_matrix @ precision_matrix 
    expected_identity = np.eye(covariance_matrix.shape[0])
    
    assert identity_check == pytest.approx(expected_identity, rel=rtol, abs=atol), \
        f"Covariance * inverse != identity for {stat_name}"

# @pytest.mark.parametrize("stat_name", STAT_NAMES)
# def test_well_behaved_fisher(statistics, stat_name):
#     """Test that Fisher information is positive."""
#     stat = statistics[stat_name]
#     fisher_information = get_individual_fisher_information(stat)
#     assert fisher_information > 0., f"Fisher < 0 for {stat_name}"



class MockStatistic:
    """Mock statistic class for testing"""
    def __init__(self, derivatives, covariance_data, emulator_error=None):
        self.n_bins, self.n_params = derivatives.shape
        self.derivatives = derivatives
        self.small_box_y = covariance_data
        self.lhc_x = np.zeros(self.n_params)  
        
        if emulator_error is None:
            self.emulator_error = np.eye(self.n_bins) * 1e-6
        else:
            self.emulator_error = emulator_error
            
        self.model = MockModel(self.derivatives)
    
    def get_emulator_error_matrix(self, method='std'):
        return self.emulator_error
    
    def get_covariance_correction(self, n_s, n_d, n_theta, method='percival-fisher'):
        return (n_s - n_d - 2) / (n_s - 1)

class MockModel:
    """Mock model that returns linear predictions"""
    def __init__(self, derivatives):
        self.derivatives = derivatives
        
    def get_prediction(self, x_batch, return_tensor=True, no_grad=False):
        if hasattr(x_batch, 'value'):
            x_tensor = x_batch.value
        else:
            x_tensor = x_batch
            
        if not isinstance(x_tensor, torch.Tensor):
            x_tensor = torch.tensor(x_tensor, dtype=torch.float32)
            
        if x_tensor.ndim == 1:
            x_tensor = x_tensor.unsqueeze(0)
            
        derivatives_tensor = torch.tensor(self.derivatives, dtype=torch.float32)
        
        predictions = x_tensor @ derivatives_tensor.T
        
        if not return_tensor:
            try:
                return predictions.detach().cpu().numpy()
            except RuntimeError:
                x_numpy = torch.zeros_like(x_tensor).detach().cpu().numpy()
                return x_numpy @ self.derivatives.T
        
        return predictions

@pytest.fixture
def known_solution_setup():
    """
    Create a test setup with known optimal solution.
    
    Returns a dictionary with test data where we know the optimal bin selection.
    """
    n_mocks = 2000
    np.random.seed(42)  
    
    derivatives = np.array([
        [2.0, 0.0],   # Bin 0: high sensitivity to theta1, none to theta2
        [0.0, 2.0],   # Bin 1: high sensitivity to theta2, none to theta1  
        [1.0, 1.0],   # Bin 2: moderate sensitivity to both
        [0.5, 0.5]    # Bin 3: low sensitivity to both
    ])
    
    # Design covariance structure
    # Make bins 0,1 have low noise, bin 2 medium noise, bin 3 high noise
    diagonal_vars = np.array([0.1, 0.1, 1.0, 10.0])  # Variances
    cov_matrix = np.diag(diagonal_vars)
    
    mock_data = multivariate_normal.rvs(
        mean=np.zeros(4), cov=cov_matrix, size=n_mocks, random_state=42
    )
    
    stat = MockStatistic(derivatives, mock_data)
    statistics = {'test_stat': stat}
    
    return {
        'statistics': statistics,
        'derivatives': derivatives,
        'diagonal_vars': diagonal_vars,
        'expected_order': [0, 1, 2, 3]  # Expected selection order by Fisher info
    }

class TestGreedyFisher:
    """Test class for greedy Fisher implementation"""
    
    def test_two_bin_selection(self, known_solution_setup):
        """Test that selecting 2 bins gives the optimal orthogonal pair"""
        statistics = known_solution_setup['statistics']
        
        selected_bins, final_fisher, history = run_greedy_fisher(
            statistics, max_bins=2, add_emulator_error=False, add_inverse_correction=False
        )
        
        selected_set = set(selected_bins['test_stat'])
        expected_set = {0, 1}  # Bins 0 and 1 should be selected (orthogonal, high SNR)
        
        assert selected_set == expected_set, (
            f"Expected bins {expected_set}, got {selected_set}. "
            f"Bins 0 and 1 are orthogonal with highest signal-to-noise ratio."
        )
    
    def test_three_bin_selection(self, known_solution_setup):
        """Test that selecting 3 bins adds the next best bin"""
        statistics = known_solution_setup['statistics']
        
        selected_bins, final_fisher, history = run_greedy_fisher(
            statistics, max_bins=3, add_emulator_error=False, add_inverse_correction=False
        )
        
        selected_set = set(selected_bins['test_stat'])
        expected_set = {0, 1, 2}  # Should add bin 2 next (better than bin 3)
        
        assert selected_set == expected_set, (
            f"Expected bins {expected_set}, got {selected_set}. "
            f"Bin 2 has higher Fisher information than bin 3."
        )
    
    def test_fisher_monotonicity(self, known_solution_setup):
        """Test that Fisher information increases monotonically"""
        statistics = known_solution_setup['statistics']
        
        selected_bins, final_fisher, history = run_greedy_fisher(
            statistics, max_bins=3, add_emulator_error=False, add_inverse_correction=False
        )
        
        # Check that each step increases (or at least doesn't decrease) Fisher info
        for i in range(1, len(history)):
            assert history[i] >= history[i-1], (
                f"Fisher information decreased from step {i-1} to {i}: "
                f"{history[i-1]:.6f} -> {history[i]:.6f}"
            )
    
    def test_single_bin_selection(self, known_solution_setup):
        """Test that single bin selection picks one of the best bins"""
        statistics = known_solution_setup['statistics']
        
        selected_bins, final_fisher, history = run_greedy_fisher(
            statistics, max_bins=1, add_emulator_error=False, add_inverse_correction=False
        )
        
        selected_bin = selected_bins['test_stat'][0]
        
        assert selected_bin in [0, 1], (
            f"Expected bin 0 or 1 (highest Fisher info), got bin {selected_bin}"
        )
    
    def test_all_bin_selection(self, known_solution_setup):
        """Test that all bins can be selected in the right order"""
        statistics = known_solution_setup['statistics']
        
        selected_bins, final_fisher, history = run_greedy_fisher(
            statistics, max_bins=4, add_emulator_error=False, add_inverse_correction=False
        )
        
        selected_list = selected_bins['test_stat']
        
        assert len(selected_list) == 4, f"Expected 4 bins selected, got {len(selected_list)}"
        assert set(selected_list) == {0, 1, 2, 3}, f"Not all bins selected: {selected_list}"
        
        # Check that bins 0 and 1 are selected first (in either order)
        first_two = set(selected_list[:2])
        assert first_two == {0, 1}, (
            f"Expected bins 0 and 1 to be selected first, got {first_two}"
        )
        
        # Check that bin 2 is selected before bin 3
        idx_2 = selected_list.index(2)
        idx_3 = selected_list.index(3)
        assert idx_2 < idx_3, f"Bin 2 should be selected before bin 3"
    
    def test_fisher_values_make_sense(self, known_solution_setup):
        """Test that Fisher information values are reasonable"""
        statistics = known_solution_setup['statistics']
        
        selected_bins, final_fisher, history = run_greedy_fisher(
            statistics, max_bins=4, add_emulator_error=False, add_inverse_correction=False
        )
        
        # Fisher information should be positive and finite
        for i, fisher_val in enumerate(history):
            assert np.isfinite(fisher_val), f"Fisher info at step {i} is not finite: {fisher_val}"
            assert fisher_val > -np.inf, f"Fisher info at step {i} is -inf: {fisher_val}"
        
        assert final_fisher > 0, f"Final Fisher info should be positive, got {final_fisher}"
        assert final_fisher < 1000, f"Final Fisher info seems too large: {final_fisher}"

