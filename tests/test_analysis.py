"""
Tests for XRD analysis functions
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import pytest
from tools.preprocessing import (
    normalize_xrd, trim_2theta, remove_xrd_nan,
    negative_to_zero, sort_dict_by_key, preprocess_pipeline
)
from tools.analysis import (
    run_nmf, norm_array, calculate_cosine_matrix,
    dim_reduction, run_dbscan, merge_cluster_ratio,
    get_argmax_prob, analyze_xrd_pipeline
)
from tools.data_loader import load_xrd_file, validate_xrd_data


class TestPreprocessing:
    """Test preprocessing functions"""

    def setup_method(self):
        """Set up test data"""
        self.test_data = {
            1: [[10, 20, 30, 40, 50], [100, 200, 300, 200, 100]],
            2: [[10, 20, 30, 40, 50], [150, 250, 350, 250, 150]]
        }

    def test_normalize_xrd(self):
        """Test intensity normalization"""
        result = normalize_xrd(self.test_data)

        # Check max intensity is 100
        for key, value in result.items():
            assert max(value[1]) == 100.0, f"Max intensity should be 100, got {max(value[1])}"

    def test_trim_2theta(self):
        """Test 2theta range trimming"""
        result = trim_2theta(self.test_data, ranges=[[15, 45]])

        # Check only values within range are kept
        for key, value in result.items():
            assert all(15 <= x <= 45 for x in value[0]), "Values outside range found"

    def test_remove_xrd_nan(self):
        """Test NaN removal"""
        data_with_nan = {
            1: [[10, 20, float('nan'), 40], [100, 200, float('nan'), 400]]
        }
        result = remove_xrd_nan(data_with_nan)

        # Check no NaN values
        for key, value in result.items():
            assert not any(np.isnan(x) for x in value[0]), "NaN in 2theta"
            assert not any(np.isnan(x) for x in value[1]), "NaN in intensity"

    def test_negative_to_zero(self):
        """Test negative value conversion"""
        data_with_negative = {
            1: [[10, 20, 30], [100, -50, 200]]
        }
        result = negative_to_zero(data_with_negative)

        # Check no negative values
        for key, value in result.items():
            assert all(x >= 0 for x in value[1]), "Negative intensity found"

    def test_sort_dict_by_key(self):
        """Test dictionary sorting"""
        unsorted_data = {3: [[], []], 1: [[], []], 2: [[], []]}
        result = sort_dict_by_key(unsorted_data)

        keys = list(result.keys())
        assert keys == sorted(keys), "Keys are not sorted"


class TestAnalysis:
    """Test analysis functions"""

    def setup_method(self):
        """Set up test data"""
        np.random.seed(42)
        n_samples = 10
        n_points = 100
        two_theta = np.linspace(10, 60, n_points).tolist()

        self.test_data = {}
        for i in range(n_samples):
            intensity = np.abs(np.random.randn(n_points) * 50 + 100).tolist()
            self.test_data[i + 1] = [two_theta, intensity]

    def test_run_nmf(self):
        """Test NMF decomposition"""
        basis, coeff, error = run_nmf(self.test_data, n_components=3)

        # Check output shapes
        assert basis.shape[0] == 3, "Wrong number of basis vectors"
        assert coeff.shape[0] == len(self.test_data), "Wrong number of coefficient rows"
        assert coeff.shape[1] == 3, "Wrong number of coefficient columns"
        assert 0 <= error <= 100, "Error should be percentage"

    def test_norm_array(self):
        """Test array normalization"""
        arrays = np.array([[10, 20, 30], [5, 10, 15]])
        result = norm_array(arrays, norm=100)

        for row in result:
            assert np.max(row) == 100.0, "Max value should be 100"

    def test_calculate_cosine_matrix(self):
        """Test cosine distance matrix"""
        arrays = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
        result = calculate_cosine_matrix(arrays, show_progress=False)

        # Check symmetry
        assert np.allclose(result, result.T), "Distance matrix should be symmetric"
        # Check diagonal is zero
        assert np.allclose(np.diag(result), 0), "Diagonal should be zero"

    def test_dim_reduction(self):
        """Test dimension reduction"""
        dist_matrix = np.array([[0, 1, 2], [1, 0, 1.5], [2, 1.5, 0]])
        result = dim_reduction(dist_matrix, method="MDS")

        assert result.shape == (3, 2), "Wrong output shape"

    def test_run_dbscan(self):
        """Test DBSCAN clustering"""
        embedding = np.array([[0, 0], [0.1, 0.1], [10, 10], [10.1, 10.1]])
        labels = run_dbscan(embedding, eps=1, min_samples=1)

        assert len(labels) == 4, "Wrong number of labels"
        # First two points should be in same cluster
        assert labels[0] == labels[1], "Nearby points should cluster together"

    def test_merge_cluster_ratio(self):
        """Test cluster ratio merging"""
        coeff = np.array([[1, 2, 3], [4, 5, 6]])
        labels = np.array([0, 0, 1])
        result = merge_cluster_ratio(coeff, labels)

        assert result.shape == (2, 2), "Wrong output shape"
        # Check cluster 0 sums components 0 and 1
        assert result[0, 0] == 3, "Wrong sum for cluster 0"
        assert result[0, 1] == 3, "Wrong sum for cluster 1"

    def test_get_argmax_prob(self):
        """Test argmax probability calculation"""
        cluster_ratio = np.array([[3, 1], [1, 4]])
        argmax, probs = get_argmax_prob(cluster_ratio)

        assert argmax == [0, 1], "Wrong argmax"
        assert len(probs) == 2, "Wrong number of probability rows"
        for prob in probs:
            assert abs(sum(prob) - 1.0) < 1e-6, "Probabilities should sum to 1"

    def test_analyze_xrd_pipeline(self):
        """Test full analysis pipeline"""
        result = analyze_xrd_pipeline(
            self.test_data,
            n_components=3,
            distance_method='cosine',
            dim_reduction_method='MDS',
            dbscan_eps=1.0,
            dbscan_min_samples=1,
            show_progress=False
        )

        # Check all expected keys
        expected_keys = ['basis_vector', 'coefficient', 'reconstruction_error',
                         'embedding', 'labels', 'cluster_ratio', 'argmax',
                         'probabilities', 'n_clusters']
        for key in expected_keys:
            assert key in result, f"Missing key: {key}"


class TestDataLoader:
    """Test data loading functions"""

    def test_validate_xrd_data_valid(self):
        """Test validation with valid data"""
        data = {
            'two_theta': np.array([10, 20, 30, 40, 50]),
            'intensity': np.array([100, 200, 300, 200, 100])
        }
        is_valid, error = validate_xrd_data(data)
        assert is_valid, f"Valid data rejected: {error}"

    def test_validate_xrd_data_empty(self):
        """Test validation with empty data"""
        data = {'two_theta': np.array([]), 'intensity': np.array([])}
        is_valid, error = validate_xrd_data(data)
        assert not is_valid, "Empty data should be invalid"

    def test_validate_xrd_data_length_mismatch(self):
        """Test validation with mismatched lengths"""
        data = {
            'two_theta': np.array([10, 20, 30]),
            'intensity': np.array([100, 200])
        }
        is_valid, error = validate_xrd_data(data)
        assert not is_valid, "Mismatched lengths should be invalid"


def run_tests():
    """Run all tests"""
    pytest.main([__file__, '-v'])


if __name__ == "__main__":
    run_tests()
