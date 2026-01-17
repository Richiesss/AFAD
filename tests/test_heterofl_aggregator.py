"""
Tests for HeteroFLAggregator

Verifies that the implementation follows the reference HeteroFL paper:
1. Sub-model distribution based on model_rate
2. Aggregation using update count (not sample count)
3. Proper index tracking for heterogeneous widths
"""

import pytest
import numpy as np
from flwr.common import parameters_to_ndarrays

from src.server.strategy.heterofl_aggregator import HeteroFLAggregator


class TestHeteroFLAggregator:
    """Test suite for HeteroFLAggregator"""

    def test_compute_param_idx_full_rate(self):
        """Test index computation for full model (rate=1.0)"""
        aggregator = HeteroFLAggregator()

        # Simulate Conv layer [out_channels=64, in_channels=3, H=3, W=3]
        global_params = [
            np.random.randn(64, 3, 3, 3),  # Conv weight
            np.random.randn(64),            # Conv bias
        ]

        param_idx = aggregator.compute_param_idx(global_params, model_rate=1.0)

        # Full rate should include all indices
        assert len(param_idx) == 2
        assert len(param_idx[0]) == 2  # (output_idx, input_idx)
        assert len(param_idx[0][0]) == 64  # All output channels
        assert len(param_idx[0][1]) == 3   # All input channels
        assert len(param_idx[1]) == 1  # Bias has 1D index
        assert len(param_idx[1][0]) == 64  # All bias elements

    def test_compute_param_idx_half_rate(self):
        """Test index computation for half model (rate=0.5)"""
        aggregator = HeteroFLAggregator()

        global_params = [
            np.random.randn(64, 3, 3, 3),  # Conv weight
            np.random.randn(64),            # Conv bias
        ]

        param_idx = aggregator.compute_param_idx(global_params, model_rate=0.5)

        # Half rate should include half the output channels
        assert len(param_idx[0][0]) == 32  # Half of 64
        # Input channels depend on previous layer, but first layer uses scaler
        assert len(param_idx[0][1]) == 2   # ceil(3 * 0.5) = 2
        assert len(param_idx[1][0]) == 32  # Bias follows output size

    def test_distribute_creates_submodel(self):
        """Test that distribute correctly extracts sub-model"""
        aggregator = HeteroFLAggregator()

        # Global model with known values
        global_params = [
            np.arange(64 * 3 * 3 * 3).reshape(64, 3, 3, 3).astype(float),
            np.arange(64).astype(float),
        ]

        sub_params = aggregator.distribute(global_params, "client_1", model_rate=0.5)

        # Sub-model should have smaller shapes
        assert sub_params[0].shape[0] == 32  # Half output channels
        assert sub_params[1].shape[0] == 32  # Half bias

        # Check that values are from the beginning (top-left)
        np.testing.assert_array_equal(sub_params[1], np.arange(32))

    def test_distribute_stores_param_idx(self):
        """Test that distribute stores param_idx for aggregation"""
        aggregator = HeteroFLAggregator()

        global_params = [np.random.randn(64, 3, 3, 3)]
        aggregator.distribute(global_params, "client_1", model_rate=0.5)

        assert "client_1" in aggregator.client_param_idx
        assert len(aggregator.client_param_idx["client_1"]) == 1

    def test_aggregate_update_count_not_sample_count(self):
        """
        Critical test: Verify aggregation uses update COUNT, not sample count

        Reference implementation:
        - count[k][idx] += 1  (not += num_examples)
        - average = sum / count
        """
        aggregator = HeteroFLAggregator()

        # Simple 1D parameter for easy verification
        global_params = [np.zeros(10)]

        # Distribute to 3 clients with same rate (all update same positions)
        for cid in ["c1", "c2", "c3"]:
            aggregator.distribute(global_params, cid, model_rate=1.0)

        # Client updates with different sample counts
        results = [
            ("c1", [np.ones(10) * 3.0], 100),   # value=3, samples=100
            ("c2", [np.ones(10) * 6.0], 200),   # value=6, samples=200
            ("c3", [np.ones(10) * 9.0], 300),   # value=9, samples=300
        ]

        updated = parameters_to_ndarrays(
            aggregator.aggregate("test", results, global_params)
        )

        # With update count: average = (3 + 6 + 9) / 3 = 6.0
        # With sample count (FedAvg): (3*100 + 6*200 + 9*300) / 600 = 6.5
        expected_heterofl = 6.0

        np.testing.assert_allclose(updated[0], np.ones(10) * expected_heterofl)

    def test_aggregate_preserves_non_updated_positions(self):
        """Test that positions not updated by any client retain global values"""
        aggregator = HeteroFLAggregator()

        # Global with non-zero values
        global_params = [np.ones(10) * 100.0]

        # Only client updates first 5 positions
        aggregator.distribute(global_params, "c1", model_rate=0.5)

        results = [
            ("c1", [np.ones(5) * 50.0], 100),
        ]

        updated = parameters_to_ndarrays(
            aggregator.aggregate("test", results, global_params)
        )

        # First 5 positions should be updated
        np.testing.assert_allclose(updated[0][:5], np.ones(5) * 50.0)
        # Last 5 positions should retain global value
        np.testing.assert_allclose(updated[0][5:], np.ones(5) * 100.0)

    def test_aggregate_heterogeneous_rates(self):
        """Test aggregation with clients having different model rates"""
        aggregator = HeteroFLAggregator()

        global_params = [np.zeros(10)]

        # Client 1: full model (rate=1.0)
        aggregator.distribute(global_params, "c1", model_rate=1.0)
        # Client 2: half model (rate=0.5)
        aggregator.distribute(global_params, "c2", model_rate=0.5)

        results = [
            ("c1", [np.ones(10) * 10.0], 100),  # Updates all 10 positions
            ("c2", [np.ones(5) * 20.0], 100),   # Updates first 5 positions
        ]

        updated = parameters_to_ndarrays(
            aggregator.aggregate("test", results, global_params)
        )

        # First 5 positions: average of 10 and 20 = 15
        np.testing.assert_allclose(updated[0][:5], np.ones(5) * 15.0)
        # Last 5 positions: only c1 updated = 10
        np.testing.assert_allclose(updated[0][5:], np.ones(5) * 10.0)

    def test_aggregate_simple_backward_compatible(self):
        """Test aggregate_simple for backward compatibility"""
        aggregator = HeteroFLAggregator()

        global_params = [np.zeros(10)]

        # Without client_id (old format) - params must be a list
        results = [
            ([np.ones(10) * 5.0], 100),
            ([np.ones(10) * 10.0], 200),
        ]

        updated = parameters_to_ndarrays(
            aggregator.aggregate_simple("test", results, global_params)
        )

        # Average = (5 + 10) / 2 = 7.5 (update count, not sample weighted)
        np.testing.assert_allclose(updated[0], np.ones(10) * 7.5)

    def test_2d_layer_aggregation(self):
        """Test aggregation for 2D layers (Linear)"""
        aggregator = HeteroFLAggregator()

        # Linear layer [out_features=8, in_features=4]
        global_params = [np.zeros((8, 4))]

        aggregator.distribute(global_params, "c1", model_rate=1.0)
        aggregator.distribute(global_params, "c2", model_rate=0.5)

        results = [
            ("c1", [np.ones((8, 4)) * 10.0], 100),
            ("c2", [np.ones((4, 2)) * 20.0], 100),  # Half size
        ]

        updated = parameters_to_ndarrays(
            aggregator.aggregate("test", results, global_params)
        )

        # Top-left quadrant (4x2): average of 10 and 20 = 15
        np.testing.assert_allclose(updated[0][:4, :2], np.ones((4, 2)) * 15.0)
        # Rest: only c1 updated = 10
        np.testing.assert_allclose(updated[0][4:, :], np.ones((4, 4)) * 10.0)
        np.testing.assert_allclose(updated[0][:4, 2:], np.ones((4, 2)) * 10.0)


class TestHeteroFLEdgeCases:
    """Edge case tests"""

    def test_empty_results(self):
        """Test aggregation with empty results"""
        aggregator = HeteroFLAggregator()
        global_params = [np.ones(10)]

        updated = parameters_to_ndarrays(
            aggregator.aggregate("test", [], global_params)
        )

        # Should return unchanged global params
        np.testing.assert_array_equal(updated[0], global_params[0])

    def test_single_client(self):
        """Test aggregation with single client"""
        aggregator = HeteroFLAggregator()
        global_params = [np.zeros(10)]

        aggregator.distribute(global_params, "c1", model_rate=1.0)

        results = [("c1", [np.ones(10) * 5.0], 100)]

        updated = parameters_to_ndarrays(
            aggregator.aggregate("test", results, global_params)
        )

        np.testing.assert_allclose(updated[0], np.ones(10) * 5.0)

    def test_infer_param_idx_fallback(self):
        """Test fallback index inference when param_idx not stored"""
        aggregator = HeteroFLAggregator()
        global_params = [np.zeros(10)]

        # Don't call distribute - test fallback
        results = [("unknown_client", [np.ones(5) * 5.0], 100)]

        # Should not crash, uses inferred indices
        updated = parameters_to_ndarrays(
            aggregator.aggregate("test", results, global_params)
        )

        # First 5 should be updated
        np.testing.assert_allclose(updated[0][:5], np.ones(5) * 5.0)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
