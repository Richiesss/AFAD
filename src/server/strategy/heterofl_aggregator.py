import numpy as np
from flwr.common import Parameters, ndarrays_to_parameters


class HeteroFLAggregator:
    """
    HeteroFL aggregation following the official implementation.

    Reference: Diao et al., "HeteroFL: Computation and Communication Efficient
    Federated Learning for Heterogeneous Clients" (ICLR 2021)
    https://github.com/diaoenmao/HeteroFL-Computation-and-Communication-Efficient-Federated-Learning-for-Heterogeneous-Clients

    Key mechanisms from the official implementation:
    1. split_model: Compute sub-model indices based on model_rate
    2. distribute: Extract sub-model parameters from global model
    3. aggregate: Count-based averaging (not sample-weighted)
    4. Output layer preservation: Classification layer is never width-scaled
    5. Label-split aggregation: Output layer only aggregated for labels client has
    """

    def __init__(self, global_model_rate: float = 1.0):
        self.global_model_rate = global_model_rate
        self.client_param_idx: dict[str, list[tuple]] = {}

    @staticmethod
    def _find_output_layer_indices(
        global_params: list[np.ndarray],
        num_preserved_tail_layers: int = 1,
    ) -> set[int]:
        """
        Detect preserved (non-width-scaled) layers by finding the last N
        2D weights and their following 1D biases.

        For vanilla HeteroFL (num_preserved_tail_layers=1), this finds only
        the classifier layer. For AFAD with FedGenModelWrapper
        (num_preserved_tail_layers=2), this also preserves the bottleneck
        layer whose output dim (latent_dim=32) must stay fixed.

        In the official HeteroFL, the output layer is identified by name
        ('linear' for resnet, last 'weight'/'bias' for conv). Since Flower
        uses positional arrays without names, we detect by position:
        the last N 2D parameters are preserved weights, and any immediately
        following 1D parameter with matching size is the corresponding bias.
        """
        output_indices: set[int] = set()

        # Find last N 2D parameters (scanning from end)
        found_2d_indices: list[int] = []
        for i in range(len(global_params) - 1, -1, -1):
            if len(global_params[i].shape) == 2:
                found_2d_indices.append(i)
                if len(found_2d_indices) >= num_preserved_tail_layers:
                    break

        for idx_2d in found_2d_indices:
            output_indices.add(idx_2d)
            output_dim = global_params[idx_2d].shape[0]

            # Check if next parameter is the matching bias
            if idx_2d + 1 < len(global_params):
                next_param = global_params[idx_2d + 1]
                if len(next_param.shape) == 1 and next_param.shape[0] == output_dim:
                    output_indices.add(idx_2d + 1)

        return output_indices

    def compute_param_idx(
        self,
        global_params: list[np.ndarray],
        model_rate: float,
        preserve_output_layer: bool = True,
        num_preserved_tail_layers: int = 1,
    ) -> list[tuple]:
        """
        Compute sub-model parameter indices based on model_rate.

        In HeteroFL, smaller sub-models use the leading portion of each layer.
        The output (classification) layer is never width-scaled -- all clients
        get the full num_classes output, matching the official implementation.

        Args:
            global_params: Global model parameters as numpy arrays
            model_rate: Client model rate (0.0-1.0), e.g. 0.5 = half width
            preserve_output_layer: If True, output layer keeps full output size
            num_preserved_tail_layers: Number of tail 2D layers to preserve.
                1 = classifier only (vanilla HeteroFL).
                2 = bottleneck + classifier (AFAD with FedGenModelWrapper).

        Returns:
            List of index tuples for each parameter
        """
        param_idx = []
        scaler = model_rate / self.global_model_rate

        # Detect output layer indices
        output_layer_indices = (
            self._find_output_layer_indices(global_params, num_preserved_tail_layers)
            if preserve_output_layer
            else set()
        )

        prev_output_idx = None

        for i, param in enumerate(global_params):
            shape = param.shape
            is_output_layer = i in output_layer_indices

            if len(shape) == 0:
                param_idx.append(())
                continue

            elif len(shape) == 1:
                if is_output_layer:
                    # Output bias: keep full size
                    idx = np.arange(shape[0])
                elif prev_output_idx is not None:
                    idx = prev_output_idx
                else:
                    output_size = int(np.ceil(shape[0] * scaler))
                    idx = np.arange(output_size)
                param_idx.append((idx,))

            elif len(shape) == 2:
                expected_in = int(np.ceil(shape[1] * scaler))
                if is_output_layer:
                    # Preserved tail layer (bottleneck or classifier):
                    # output dim is kept full; input follows the previous layer
                    # exactly (including preserved bottleneck output).
                    output_idx = np.arange(shape[0])
                    if prev_output_idx is not None:
                        input_idx = prev_output_idx
                    else:
                        input_idx = np.arange(expected_in)
                else:
                    output_size = int(np.ceil(shape[0] * scaler))
                    output_idx = np.arange(output_size)
                    if (
                        prev_output_idx is not None
                        and len(prev_output_idx) == expected_in
                    ):
                        input_idx = prev_output_idx
                    else:
                        # Residual/shortcut: input comes from a different branch,
                        # so prev_output_idx size may not match. Recompute directly.
                        input_idx = np.arange(expected_in)

                param_idx.append((output_idx, input_idx))
                prev_output_idx = output_idx

            elif len(shape) == 3:
                # ViT 3D params: [batch, seq, hidden] or [1, 1, hidden]
                # Only scale the last dim (hidden); batch and seq are preserved
                idx_0 = np.arange(shape[0])
                idx_1 = np.arange(shape[1])
                hidden_size = int(np.ceil(shape[2] * scaler))
                idx_2 = np.arange(hidden_size)
                param_idx.append((idx_0, idx_1, idx_2))
                prev_output_idx = idx_2

            elif len(shape) == 4:
                output_size = int(np.ceil(shape[0] * scaler))
                output_idx = np.arange(output_size)
                expected_in = int(np.ceil(shape[1] * scaler))

                if prev_output_idx is not None and len(prev_output_idx) == expected_in:
                    input_idx = prev_output_idx
                else:
                    # Residual/shortcut: input comes from a different branch,
                    # so prev_output_idx size may not match. Recompute directly.
                    input_idx = np.arange(expected_in)

                param_idx.append((output_idx, input_idx))
                prev_output_idx = output_idx

            else:
                scaled_shape = tuple(int(np.ceil(s * scaler)) for s in shape)
                idx_tuple = tuple(np.arange(s) for s in scaled_shape)
                param_idx.append(idx_tuple)
                if len(shape) > 0:
                    prev_output_idx = np.arange(scaled_shape[0])

        return param_idx

    def distribute(
        self,
        global_params: list[np.ndarray],
        client_id: str,
        model_rate: float,
        num_preserved_tail_layers: int = 1,
    ) -> list[np.ndarray]:
        """
        Extract sub-model from global model for a client.

        Uses torch.meshgrid-style indexing (matching the official implementation)
        to extract the sub-model parameters.
        """
        param_idx = self.compute_param_idx(
            global_params,
            model_rate,
            num_preserved_tail_layers=num_preserved_tail_layers,
        )
        self.client_param_idx[client_id] = param_idx

        sub_params = []
        for param, idx in zip(global_params, param_idx):
            if len(idx) == 0:
                sub_params.append(param.copy())
            elif len(idx) == 1:
                size = len(idx[0])
                sub_params.append(param[:size].copy())
            elif len(idx) == 2:
                out_size = len(idx[0])
                in_size = len(idx[1])
                if len(param.shape) == 2:
                    sub_params.append(param[:out_size, :in_size].copy())
                elif len(param.shape) == 4:
                    sub_params.append(param[:out_size, :in_size, :, :].copy())
                else:
                    sub_params.append(param.copy())
            elif len(idx) == 3:
                # 3D params (ViT pos_embedding, class_token)
                s0, s1, s2 = len(idx[0]), len(idx[1]), len(idx[2])
                sub_params.append(param[:s0, :s1, :s2].copy())
            else:
                sub_params.append(param.copy())

        return sub_params

    def aggregate(
        self,
        family: str,
        results: list[tuple[str, list[np.ndarray], int]],
        global_params: list[np.ndarray],
        client_label_splits: dict[str, list[int]] | None = None,
        num_preserved_tail_layers: int = 1,
    ) -> Parameters:
        """
        HeteroFL aggregation with count-based averaging.

        Following the official combine() method:
        - Count-based averaging (not sample-weighted)
        - Output layer: only aggregate for labels the client has data for
        - Non-updated positions retain their global values

        Args:
            family: Family name
            results: [(client_id, parameters, num_examples), ...]
            global_params: Current global parameters
            client_label_splits: Optional {client_id: [label_indices]} for
                output layer label-split aggregation (Non-IID support)
            num_preserved_tail_layers: Number of tail 2D layers preserved
                during distribute. Must match the value used in distribute().
        """
        if not results:
            return ndarrays_to_parameters(global_params)

        output_layer_indices = self._find_output_layer_indices(
            global_params, num_preserved_tail_layers
        )

        accumulated = [np.zeros_like(p) for p in global_params]
        count = [np.zeros_like(p) for p in global_params]

        for client_id, client_params, num_examples in results:
            param_idx = self.client_param_idx.get(client_id)

            if param_idx is None:
                param_idx = self._infer_param_idx(client_params, global_params)

            # Get label split for this client (if provided)
            label_split = None
            if client_label_splits and client_id in client_label_splits:
                label_split = client_label_splits[client_id]

            for i, (local_param, idx) in enumerate(zip(client_params, param_idx)):
                if i >= len(accumulated):
                    break

                is_output = i in output_layer_indices

                if len(idx) == 0:
                    accumulated[i] += local_param
                    count[i] += 1
                elif len(idx) == 1:
                    size = len(idx[0])
                    if is_output and label_split is not None:
                        # Output bias: only aggregate for client's labels
                        for lbl in label_split:
                            if lbl < size and lbl < len(local_param):
                                accumulated[i][lbl] += local_param[lbl]
                                count[i][lbl] += 1
                    else:
                        accumulated[i][:size] += local_param
                        count[i][:size] += 1
                elif len(idx) == 2:
                    out_size = len(idx[0])
                    in_size = len(idx[1])
                    if len(global_params[i].shape) == 2:
                        if is_output and label_split is not None:
                            # Output weight: only aggregate rows for client's labels
                            for lbl in label_split:
                                if lbl < out_size and lbl < local_param.shape[0]:
                                    accumulated[i][lbl, :in_size] += local_param[
                                        lbl, :in_size
                                    ]
                                    count[i][lbl, :in_size] += 1
                        else:
                            accumulated[i][:out_size, :in_size] += local_param
                            count[i][:out_size, :in_size] += 1
                    elif len(global_params[i].shape) == 4:
                        accumulated[i][:out_size, :in_size, :, :] += local_param
                        count[i][:out_size, :in_size, :, :] += 1
                    else:
                        slices = tuple(slice(0, len(ix)) for ix in idx)
                        accumulated[i][slices] += local_param
                        count[i][slices] += 1
                elif len(idx) == 3:
                    # 3D params (ViT pos_embedding, class_token)
                    s0, s1, s2 = len(idx[0]), len(idx[1]), len(idx[2])
                    accumulated[i][:s0, :s1, :s2] += local_param
                    count[i][:s0, :s1, :s2] += 1

        new_params = []
        for acc, cnt, orig in zip(accumulated, count, global_params):
            updated = orig.copy()
            mask = cnt > 0
            if np.any(mask):
                updated[mask] = acc[mask] / cnt[mask]
            new_params.append(updated)

        return ndarrays_to_parameters(new_params)

    def _infer_param_idx(
        self, client_params: list[np.ndarray], global_params: list[np.ndarray]
    ) -> list[tuple]:
        """Infer parameter indices from shapes (fallback for unknown clients)."""
        param_idx = []
        for client_p, global_p in zip(client_params, global_params):
            idx = tuple(np.arange(s) for s in client_p.shape)
            param_idx.append(idx)
        return param_idx

    def aggregate_simple(
        self,
        family: str,
        results: list[tuple[list[np.ndarray], int]],
        global_params: list[np.ndarray],
    ) -> Parameters:
        """
        Simple aggregation without client_id tracking (backward compatibility).

        Uses shape-based slicing for count-based averaging.
        """
        if not results:
            return ndarrays_to_parameters(global_params)

        accumulated = [np.zeros_like(p) for p in global_params]
        count = [np.zeros_like(p) for p in global_params]

        for client_params, num_examples in results:
            for i, p in enumerate(client_params):
                if i >= len(accumulated):
                    break

                target_shape = accumulated[i].shape
                src_shape = p.shape

                if len(target_shape) != len(src_shape):
                    continue

                valid = all(s <= t for s, t in zip(src_shape, target_shape))
                if not valid:
                    continue

                slices = tuple(slice(0, d) for d in src_shape)

                accumulated[i][slices] += p
                count[i][slices] += 1

        new_params = []
        for acc, cnt, orig in zip(accumulated, count, global_params):
            updated = orig.copy()
            mask = cnt > 0
            if np.any(mask):
                updated[mask] = acc[mask] / cnt[mask]
            new_params.append(updated)

        return ndarrays_to_parameters(new_params)
