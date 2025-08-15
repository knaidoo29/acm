import numpy as np
import torch
from scipy.linalg import block_diag

def is_well_conditioned(matrix, threshold=1e10):
    condition_number = np.linalg.cond(matrix)
    return condition_number < threshold, condition_number

def safe_inverse(matrix,): 
    """Safely invert a potentially ill-conditioned matrix"""
    # if matrix is 1x1 return the inverse directly as a 1x1 matrix
    if matrix.shape == (1,1):
        return np.array([[1.0 / matrix[0,0]]])  # Return as 1x1 matrix, not scalar
    is_stable, cond_num = is_well_conditioned(matrix)
    if is_stable:
        try:
            return np.linalg.inv(matrix)
        except np.linalg.LinAlgError:
            print("Regular inversion failed despite good condition number")
            pass
    else:
        #print('Matrix is ill-conditioned, using pseudo-inverse')
        return np.linalg.pinv(matrix)

def get_pseudodeterminant(matrix, epsilon=1.e-10): 
    U, s, Vh = np.linalg.svd(matrix)
    threshold = np.max(s) * epsilon
    significant_s = s[s > threshold]
    return np.sum(np.log(significant_s))

def safe_log_determinant(matrix, epsilon=1e-10,):
    is_stable, cond_num = is_well_conditioned(matrix)
    if is_stable:
        try:
            sign, logdet = np.linalg.slogdet(matrix)
            if sign > 0:
                return logdet
        except np.linalg.LinAlgError:
            print("Regular log-determinant failed despite good condition number")
            pass
    #print('Matrix is ill-conditioned, using pseudo-log-determinant')
    return get_pseudodeterminant(matrix, epsilon)

class GreedyFisher:
    """
    Clean implementation of greedy bin selection with block-diagonal emulator errors.
    
    Key design principles:
    1. Consistent ordering: statistics appear in fixed order, bins in selection order
    2. Single source of truth for data combination
    3. Clear separation of concerns
    """
    
    def __init__(self, statistics, fixed_parameters_idx=None, nuisance_parameters_idx=None, volume_factor=64.0, initial_selection_bins=None):
        self.statistics = statistics
        self.statistics_order = list(statistics.keys())
        self.volume_factor = volume_factor
        self.fixed_parameters_idx = fixed_parameters_idx
        self.nuisance_parameters_idx = nuisance_parameters_idx
        self.initial_selection_bins = initial_selection_bins 
        self.precomputed = self._precompute_data()
        
    def get_gradient(self, statistic,): 
        fiducial_parameters = statistic.lhc_x
        fiducial_parameters = torch.tensor(fiducial_parameters.astype(np.float32), requires_grad=True,).unsqueeze(0)
        def model_fn(x_batch):
            return statistic.model.get_prediction(x_batch, return_tensor=True, no_grad=False,)
        gradients = torch.func.jacrev(model_fn)(fiducial_parameters).detach().squeeze().numpy()
        return gradients
        
    def _precompute_data(self):
        """Precompute all derivatives and covariance data once."""
        data = {
            'derivatives': {},
            'covariance_simulations': {},
            'emulator_error_matrices': {},
        }
        
        for stat_str, statistic in self.statistics.items():
            data['derivatives'][stat_str] = self.get_gradient(statistic)
            if self.fixed_parameters_idx is not None:
                # Remove fixed parameters from derivatives
                data['derivatives'][stat_str] = np.delete(
                    data['derivatives'][stat_str], 
                    self.fixed_parameters_idx, axis=1
                )

            data['covariance_simulations'][stat_str] = statistic.small_box_y
            data['emulator_error_matrices'][stat_str] = statistic.get_emulator_error_matrix(method='std_chi2_5sigma')
            
        return data
    
    def _combine_selected_data(self, selected_bins, new_bin_idx=None, new_bin_stat=None):
        """
        Single source of truth for combining data across statistics.
        
        Returns gradients, simulation_data, emulator_cov_matrix in consistent order.
        """
        gradient_parts = []
        data_parts = []
        emulator_blocks = []
        
        for stat_str in self.statistics_order:
            # Determine which bins to include for this statistic
            bins_to_use = list(selected_bins[stat_str])
            if stat_str == new_bin_stat and new_bin_idx is not None:
                bins_to_use.append(new_bin_idx)
                
            if len(bins_to_use) > 0:
                # Add data in consistent order
                gradient_parts.append(self.precomputed['derivatives'][stat_str][bins_to_use])
                data_parts.append(self.precomputed['covariance_simulations'][stat_str][:, bins_to_use])
                
                # Add emulator error block
                full_emulator_matrix = self.precomputed['emulator_error_matrices'][stat_str]
                emulator_block = full_emulator_matrix[np.ix_(bins_to_use, bins_to_use)]
                emulator_blocks.append(emulator_block)
        
        # Combine all parts
        if gradient_parts:
            combined_gradients = np.vstack(gradient_parts)
            combined_data = np.hstack(data_parts) 
            combined_emulator_cov = block_diag(*emulator_blocks)
        else:
            # Handle empty case
            n_params = list(self.precomputed['derivatives'].values())[0].shape[1]
            n_mocks = list(self.precomputed['covariance_simulations'].values())[0].shape[0]
            combined_gradients = np.zeros((0, n_params))
            combined_data = np.zeros((n_mocks, 0))
            combined_emulator_cov = np.zeros((0, 0))
            
        return combined_gradients, combined_data, combined_emulator_cov
    
    def _compute_precision_matrix(self, gradients, simulation_data, emulator_cov, 
                                add_emulator_error=True, add_inverse_correction=True):
        """Compute precision matrix for given data configuration."""
        # Handle edge case of no data (shouldn't happen in normal operation)
        if simulation_data.shape[1] == 0 or gradients.shape[0] == 0:
            return np.array([])
            
        # Build covariance from simulations
        prefactor = 1.0 / self.volume_factor
        covariance_matrix = prefactor * np.cov(simulation_data.T)
        covariance_matrix = np.atleast_2d(covariance_matrix)
        
        # Add emulator error if requested
        if add_emulator_error and emulator_cov.size > 0:
            covariance_matrix += emulator_cov
            
        # Apply correction if requested  
        if add_inverse_correction:
            correction = list(self.statistics.values())[0].get_covariance_correction(
                n_s=simulation_data.shape[0],
                n_d=simulation_data.shape[1], 
                n_theta=gradients.shape[1],
                method='percival-fisher',
            )
            covariance_matrix *= correction
            
        return safe_inverse(covariance_matrix)
    
    def _compute_fisher_information(self, gradients, precision_matrix,):
        """Compute Fisher information matrix and its log determinant."""
        # Handle edge cases of no bins selected
        if (gradients.shape[0] == 0 or 
            precision_matrix.size == 0 or 
            precision_matrix.ndim == 0 or
            precision_matrix.shape == ()):
            return float('-inf')
            
        fisher_matrix = gradients.T @ precision_matrix @ gradients
        if self.nuisance_parameters_idx is not None: 
            fisher_matrix = self._marginalize_fisher_matrix(fisher_matrix, self.nuisance_parameters_idx)
        return safe_log_determinant(fisher_matrix)

    def _marginalize_fisher_matrix(self, fisher_matrix, nuisance_parameters_idx):
        n_params = fisher_matrix.shape[0]
        target_mask = np.ones(n_params, dtype=bool)
        target_mask[nuisance_parameters_idx] = False
        target_parameters_idx = np.where(target_mask)[0]
        F_AA = fisher_matrix[np.ix_(target_parameters_idx, target_parameters_idx)]
        F_BB = fisher_matrix[np.ix_(nuisance_parameters_idx, nuisance_parameters_idx)]
        F_AB = fisher_matrix[np.ix_(target_parameters_idx, nuisance_parameters_idx)]
        F_BA = fisher_matrix[np.ix_(nuisance_parameters_idx, target_parameters_idx)]
        F_BB_inv = safe_inverse(F_BB)
        marginalized_fisher = F_AA - F_AB @ F_BB_inv @ F_BA
        return marginalized_fisher


    
    def evaluate_bin_addition(self, selected_bins, candidate_bins, candidate_stat,
                            add_emulator_error=True, add_inverse_correction=True):
        """
        Evaluate Fisher information for adding each candidate bin.
        
        Returns:
            best_idx: index in candidate_bins of best bin to add
            best_fisher: Fisher information achieved by adding that bin
        """
        fisher_values = []
        
        for bin_idx in candidate_bins:
            # Get combined data with this bin added
            gradients, sim_data, emulator_cov = self._combine_selected_data(
                selected_bins, new_bin_idx=bin_idx, new_bin_stat=candidate_stat
            )
            
            # Compute precision matrix and Fisher information
            precision_matrix = self._compute_precision_matrix(
                gradients, sim_data, emulator_cov, 
                add_emulator_error, add_inverse_correction
            )
            fisher_info = self._compute_fisher_information(gradients, precision_matrix)
            fisher_values.append(fisher_info)
        
        best_idx = np.argmax(fisher_values)
        return best_idx, fisher_values[best_idx]
    
    def greedy_selection(self, max_bins=10, add_emulator_error=True, 
                        add_inverse_correction=True, patience=10, min_improvement=0.001):
        """
        Perform greedy bin selection to maximize Fisher information.
        """
        # Initialize
        available_bins = {stat: list(range(self.precomputed['derivatives'][stat].shape[0])) 
                         for stat in self.statistics}
        if self.initial_selection_bins:
            selected_bins = self.initial_selection_bins.copy()
            # Remove selected bins from available bins
            for stat, bins in selected_bins.items():
                for bin_idx in bins:
                    if bin_idx in available_bins[stat]:
                        available_bins[stat].remove(bin_idx)
                        selection_history.append({
                            'statistic': stat,
                            'bin_idx': bin_idx,
                            'step': 'initial',
                            'fisher_before': None,
                            'fisher_after': None,
                            'improvement': None
                        })
            print(f"Starting with {sum(len(bins) for bins in selected_bins.values())} pre-selected bins")
            total_selected = sum(len(bins) for bins in selected_bins.values())
        else:
            selected_bins = {stat: [] for stat in self.statistics}
            total_selected = 0
        
        print(f'Total available bins: {sum(len(bins) for bins in available_bins.values())}')
        
        # Track progress
        current_fisher = float('-inf')
        fisher_history = []
        selection_history = []
        
        # Early stopping
        best_fisher = float('-inf')
        best_config = None
        no_improvement_count = 0
        
        while total_selected < max_bins:
            best_stat = None
            best_local_idx = None
            best_fisher_candidate = float('-inf')
            
            # Try adding a bin from each statistic
            for stat_str in self.statistics:
                if not available_bins[stat_str]:
                    continue
                    
                local_idx, fisher_value = self.evaluate_bin_addition(
                    selected_bins, available_bins[stat_str], stat_str,
                    add_emulator_error, add_inverse_correction
                )
                
                if fisher_value > best_fisher_candidate:
                    best_fisher_candidate = fisher_value
                    best_stat = stat_str
                    best_local_idx = local_idx
            
            if best_stat is None:
                print("No more bins can be added")
                break
                
            # Add the best bin
            actual_bin_idx = available_bins[best_stat][best_local_idx]
            selected_bins[best_stat].append(actual_bin_idx)
            available_bins[best_stat].remove(actual_bin_idx)
            
            improvement = best_fisher_candidate - current_fisher
            previous_fisher = current_fisher
            current_fisher = best_fisher_candidate
            fisher_history.append(current_fisher)
            selection_history.append({
                'statistic': best_stat,
                'bin_idx': actual_bin_idx,
                'step': total_selected + 1,
                'fisher_before': previous_fisher,
                'fisher_after': current_fisher,
                'improvement': improvement
            })
            total_selected += 1
            
            # Track best configuration for early stopping
            if current_fisher > best_fisher:
                best_fisher = current_fisher
                best_config = {stat: list(bins) for stat, bins in selected_bins.items()}
                no_improvement_count = 0
            else:
                rel_improvement = abs(improvement) / abs(best_fisher) if best_fisher != 0 else float('inf')
                if rel_improvement < min_improvement:
                    no_improvement_count += 1
                else:
                    no_improvement_count = 0
                    
            # Early stopping check
            if no_improvement_count >= patience:
                print(f"No significant improvement for {patience} iterations, stopping early")
                break
                
            # Progress logging
            if total_selected % 5 == 0 or total_selected == 1:
                print(f"Selected {total_selected}/{max_bins} bins, Fisher: {current_fisher:.4f}")
                distribution = ", ".join([f"{stat}: {len(bins)}" for stat, bins in selected_bins.items()])
                print(f"Distribution: {distribution}")
                print(f"Added {best_stat}:{actual_bin_idx} with improvement {improvement:.4f}")
        
        return selected_bins, current_fisher, fisher_history, selection_history

def map_indices_after_deletion_numpy(original_length, delete_indices, select_indices):
    """Same as above but using numpy for efficiency."""
    keep_mask = np.ones(original_length, dtype=bool)
    keep_mask[delete_indices] = False
    cumsum = np.cumsum(keep_mask) - 1
    
    select_indices = np.array(select_indices)
    valid_select = select_indices[keep_mask[select_indices]]
    
    new_select_indices = cumsum[valid_select]
    return new_select_indices

def run_greedy_fisher(statistics, max_bins=100, add_emulator_error=True, 
                     add_inverse_correction=True, fixed_parameters_idx=None,
                     nuisance_parameters_idx=None, initial_selection_bins=None,**kwargs):
    """
    Clean interface for running greedy Fisher information maximization.
    """
    if fixed_parameters_idx is not None and nuisance_parameters_idx is not None:
        nuisance_parameters_idx = map_indices_after_deletion_numpy(
            len(list(statistics.values())[0].lhc_x_names), fixed_parameters_idx, nuisance_parameters_idx
        )
    selector = GreedyFisher(
        statistics, 
        fixed_parameters_idx=fixed_parameters_idx,
        nuisance_parameters_idx=nuisance_parameters_idx,
        initial_selection_bins=initial_selection_bins,
    )
    
    selected_bins, final_fisher, fisher_history, selection_history = selector.greedy_selection(
        max_bins=max_bins,
        add_emulator_error=add_emulator_error, 
        add_inverse_correction=add_inverse_correction,
        **kwargs
    )
    
    print(f"\nFinal selection:")
    for stat, bins in selected_bins.items():
        print(f"{stat}: {len(bins)} bins selected")
    print(f"Total: {sum(len(bins) for bins in selected_bins.values())} bins")
    print(f"Final Fisher log-determinant: {final_fisher:.4f}")
    
    return selected_bins, final_fisher, fisher_history, selection_history