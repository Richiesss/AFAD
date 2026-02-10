"""
Tests for HeteroFLAggregator

Verifies that the implementation follows the reference HeteroFL paper:
1. Sub-model distribution based on model_rate
2. Aggregation using update count (not sample count)
3. Proper index tracking for heterogeneous widths
4. Output layer preservation (classification layer never width-scaled)
5. Label-split aggregation for output layer (Non-IID support)

Reference: Diao et al., "HeteroFL: Computation and Communication Efficient
Federated Learning for Heterogeneous Clients" (ICLR 2021)
"""

import numpy as np
import pytest
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
            np.random.randn(64),  # Conv bias
        ]

        param_idx = aggregator.compute_param_idx(global_params, model_rate=1.0)

        # Full rate should include all indices
        assert len(param_idx) == 2
        assert len(param_idx[0]) == 2  # (output_idx, input_idx)
        assert len(param_idx[0][0]) == 64  # All output channels
        assert len(param_idx[0][1]) == 3  # All input channels
        assert len(param_idx[1]) == 1  # Bias has 1D index
        assert len(param_idx[1][0]) == 64  # All bias elements

    def test_compute_param_idx_half_rate(self):
        """Test index computation for half model (rate=0.5)"""
        aggregator = HeteroFLAggregator()

        global_params = [
            np.random.randn(64, 3, 3, 3),  # Conv weight
            np.random.randn(64),  # Conv bias
        ]

        param_idx = aggregator.compute_param_idx(global_params, model_rate=0.5)

        # Half rate should include half the output channels
        assert len(param_idx[0][0]) == 32  # Half of 64
        # Input channels depend on previous layer, but first layer uses scaler
        assert len(param_idx[0][1]) == 2  # ceil(3 * 0.5) = 2
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
            ("c1", [np.ones(10) * 3.0], 100),  # value=3, samples=100
            ("c2", [np.ones(10) * 6.0], 200),  # value=6, samples=200
            ("c3", [np.ones(10) * 9.0], 300),  # value=9, samples=300
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
            ("c2", [np.ones(5) * 20.0], 100),  # Updates first 5 positions
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

    def test_2d_output_layer_aggregation(self):
        """
        Test aggregation for 2D output layer (Linear).

        When a single 2D layer is the only linear layer, it is detected as
        the output (classification) layer. Its output dimension is preserved
        at full size even when model_rate < 1.0.
        """
        aggregator = HeteroFLAggregator()

        # Output linear layer [num_classes=8, in_features=4]
        global_params = [np.zeros((8, 4))]

        aggregator.distribute(global_params, "c1", model_rate=1.0)
        aggregator.distribute(global_params, "c2", model_rate=0.5)

        # c1 gets full (8, 4); c2 gets (8, 2) because output preserved
        results = [
            ("c1", [np.ones((8, 4)) * 10.0], 100),
            ("c2", [np.ones((8, 2)) * 20.0], 100),
        ]

        updated = parameters_to_ndarrays(
            aggregator.aggregate("test", results, global_params)
        )

        # Left columns (:, :2): average of 10 and 20 = 15
        np.testing.assert_allclose(updated[0][:, :2], np.ones((8, 2)) * 15.0)
        # Right columns (:, 2:): only c1 = 10
        np.testing.assert_allclose(updated[0][:, 2:], np.ones((8, 2)) * 10.0)

    def test_hidden_linear_scaled_at_half_rate(self):
        """
        Test that hidden (non-output) linear layers ARE width-scaled.

        Model: Conv(16,3,3,3) -> Linear(8,16) -> Output Linear(10,8)
        At rate=0.5: Conv(8,2,3,3) -> Linear(4,8) -> Output(10,4)
        The output layer preserves its full output dim (10), but hidden
        linear layer output is scaled (8 -> 4).
        """
        aggregator = HeteroFLAggregator()

        global_params = [
            np.zeros((16, 3, 3, 3)),  # Conv weight
            np.zeros(16),  # Conv bias
            np.zeros((8, 16)),  # Hidden linear weight
            np.zeros(8),  # Hidden linear bias
            np.zeros((10, 8)),  # Output linear weight (last 2D)
            np.zeros(10),  # Output linear bias
        ]

        param_idx = aggregator.compute_param_idx(global_params, model_rate=0.5)

        # Conv: output scaled 16->8, input scaled 3->2
        assert len(param_idx[0][0]) == 8
        assert len(param_idx[0][1]) == 2
        # Conv bias: follows conv output = 8
        assert len(param_idx[1][0]) == 8

        # Hidden linear: output scaled 8->4, input follows conv output = 8
        assert len(param_idx[2][0]) == 4
        assert len(param_idx[2][1]) == 8
        # Hidden linear bias: follows hidden output = 4
        assert len(param_idx[3][0]) == 4

        # Output linear: output preserved at 10, input follows hidden output = 4
        assert len(param_idx[4][0]) == 10  # NOT scaled!
        assert len(param_idx[4][1]) == 4
        # Output bias: preserved at 10
        assert len(param_idx[5][0]) == 10  # NOT scaled!


class TestHeteroFLOutputLayerDetection:
    """Tests for output layer detection and preservation"""

    def test_find_output_layer_indices_with_linear(self):
        """Test detection of last 2D parameter as output layer"""
        global_params = [
            np.zeros((64, 3, 3, 3)),  # Conv weight
            np.zeros(64),  # Conv bias
            np.zeros((10, 64)),  # Output linear weight
            np.zeros(10),  # Output bias
        ]

        indices = HeteroFLAggregator._find_output_layer_indices(global_params)
        assert indices == {2, 3}  # Output weight + bias

    def test_find_output_layer_indices_no_linear(self):
        """Test that no output is detected when there are no 2D params"""
        global_params = [
            np.zeros((64, 3, 3, 3)),  # Conv weight only
            np.zeros(64),  # Conv bias
        ]

        indices = HeteroFLAggregator._find_output_layer_indices(global_params)
        assert indices == set()

    def test_find_output_layer_indices_multiple_linear(self):
        """Test that only the LAST 2D layer is detected as output"""
        global_params = [
            np.zeros((128, 64)),  # Hidden linear 1
            np.zeros(128),  # Hidden bias 1
            np.zeros((64, 128)),  # Hidden linear 2
            np.zeros(64),  # Hidden bias 2
            np.zeros((10, 64)),  # Output linear
            np.zeros(10),  # Output bias
        ]

        indices = HeteroFLAggregator._find_output_layer_indices(global_params)
        assert indices == {4, 5}  # Only the last one

    def test_output_layer_preserved_in_distribute(self):
        """
        Test that distribute preserves the output layer output dimension.

        Official HeteroFL: output layer always gets arange(output_size)
        (full num_classes), never scaled by model_rate.
        """
        aggregator = HeteroFLAggregator()

        # Conv(32,1,3,3) -> Linear(10,32)  (10 = num_classes)
        global_params = [
            np.ones((32, 1, 3, 3)),  # Conv weight
            np.ones(32),  # Conv bias
            np.ones((10, 32)),  # Output weight
            np.ones(10),  # Output bias
        ]

        sub_params = aggregator.distribute(global_params, "c1", model_rate=0.5)

        # Conv: scaled to (16, 1, 3, 3)
        assert sub_params[0].shape == (16, 1, 3, 3)
        assert sub_params[1].shape == (16,)

        # Output linear: output preserved at 10, input follows conv = 16
        assert sub_params[2].shape == (10, 16)
        assert sub_params[3].shape == (10,)


class TestHeteroFLLabelSplit:
    """Tests for label-split aggregation on output layer (Non-IID support)"""

    def test_aggregate_with_label_split(self):
        """
        Test that output layer only aggregates for labels each client has.

        Official HeteroFL combine():
        - For output weight/bias, only aggregate rows for client's label_split
        - Prevents client from overwriting weights for classes it hasn't seen
        """
        aggregator = HeteroFLAggregator()

        # Output layer: [num_classes=4, features=2]
        global_params = [
            np.ones((4, 2)) * 100.0,  # Output weight
            np.ones(4) * 100.0,  # Output bias
        ]

        aggregator.distribute(global_params, "c1", model_rate=1.0)
        aggregator.distribute(global_params, "c2", model_rate=1.0)

        results = [
            ("c1", [np.ones((4, 2)) * 10.0, np.ones(4) * 10.0], 100),
            ("c2", [np.ones((4, 2)) * 20.0, np.ones(4) * 20.0], 100),
        ]

        # Client 1 has labels [0, 1]; Client 2 has labels [2, 3]
        label_splits = {"c1": [0, 1], "c2": [2, 3]}

        updated = parameters_to_ndarrays(
            aggregator.aggregate(
                "test",
                results,
                global_params,
                client_label_splits=label_splits,
            )
        )

        # Output weight: rows 0,1 from c1 only = 10; rows 2,3 from c2 only = 20
        np.testing.assert_allclose(updated[0][0, :], [10.0, 10.0])
        np.testing.assert_allclose(updated[0][1, :], [10.0, 10.0])
        np.testing.assert_allclose(updated[0][2, :], [20.0, 20.0])
        np.testing.assert_allclose(updated[0][3, :], [20.0, 20.0])

        # Output bias: same pattern
        np.testing.assert_allclose(updated[1][:2], [10.0, 10.0])
        np.testing.assert_allclose(updated[1][2:], [20.0, 20.0])

    def test_aggregate_without_label_split_aggregates_all(self):
        """Without label_split, all clients contribute to all output positions"""
        aggregator = HeteroFLAggregator()

        global_params = [
            np.zeros((4, 2)),  # Output weight
            np.zeros(4),  # Output bias
        ]

        aggregator.distribute(global_params, "c1", model_rate=1.0)
        aggregator.distribute(global_params, "c2", model_rate=1.0)

        results = [
            ("c1", [np.ones((4, 2)) * 10.0, np.ones(4) * 10.0], 100),
            ("c2", [np.ones((4, 2)) * 20.0, np.ones(4) * 20.0], 100),
        ]

        # No label_split - standard aggregation
        updated = parameters_to_ndarrays(
            aggregator.aggregate("test", results, global_params)
        )

        # All positions: average of 10 and 20 = 15
        np.testing.assert_allclose(updated[0], np.ones((4, 2)) * 15.0)
        np.testing.assert_allclose(updated[1], np.ones(4) * 15.0)

    def test_label_split_with_overlapping_labels(self):
        """Test label split when clients share some labels"""
        aggregator = HeteroFLAggregator()

        global_params = [
            np.zeros((4, 2)),  # Output weight
            np.zeros(4),  # Output bias
        ]

        aggregator.distribute(global_params, "c1", model_rate=1.0)
        aggregator.distribute(global_params, "c2", model_rate=1.0)

        results = [
            ("c1", [np.ones((4, 2)) * 10.0, np.ones(4) * 10.0], 100),
            ("c2", [np.ones((4, 2)) * 20.0, np.ones(4) * 20.0], 100),
        ]

        # Overlapping: both have label 1
        label_splits = {"c1": [0, 1], "c2": [1, 2, 3]}

        updated = parameters_to_ndarrays(
            aggregator.aggregate(
                "test",
                results,
                global_params,
                client_label_splits=label_splits,
            )
        )

        # Label 0: only c1 = 10
        np.testing.assert_allclose(updated[0][0, :], [10.0, 10.0])
        # Label 1: both contribute, average = 15
        np.testing.assert_allclose(updated[0][1, :], [15.0, 15.0])
        # Label 2: only c2 = 20
        np.testing.assert_allclose(updated[0][2, :], [20.0, 20.0])
        # Label 3: only c2 = 20
        np.testing.assert_allclose(updated[0][3, :], [20.0, 20.0])


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

    def test_scalar_params_handled(self):
        """Test that scalar parameters (e.g. BN num_batches_tracked) are handled"""
        aggregator = HeteroFLAggregator()

        global_params = [
            np.zeros(16),  # BN weight
            np.zeros(16),  # BN bias
            np.zeros(16),  # BN running_mean
            np.zeros(16),  # BN running_var
            np.array(0),  # BN num_batches_tracked (scalar)
        ]

        param_idx = aggregator.compute_param_idx(global_params, model_rate=0.5)

        # Scalar should have empty index tuple
        assert param_idx[4] == ()

    def test_preserve_output_layer_disabled(self):
        """Test that output layer scaling works when preservation is disabled"""
        aggregator = HeteroFLAggregator()

        global_params = [
            np.zeros((10, 8)),  # 2D layer (would be detected as output)
        ]

        # With preservation enabled (default)
        idx_preserved = aggregator.compute_param_idx(global_params, model_rate=0.5)
        assert len(idx_preserved[0][0]) == 10  # Output preserved

        # With preservation disabled
        idx_scaled = aggregator.compute_param_idx(
            global_params, model_rate=0.5, preserve_output_layer=False
        )
        assert len(idx_scaled[0][0]) == 5  # Output scaled


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
