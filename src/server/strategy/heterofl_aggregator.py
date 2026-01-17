from typing import List, Tuple, Dict
import numpy as np
from flwr.common import Parameters, ndarrays_to_parameters, parameters_to_ndarrays

class HeteroFLAggregator:
    """
    HeteroFL方式の同族内集約を行うクラス
    """
    def __init__(self):
        pass

    def aggregate(
        self, 
        family: str, 
        results: List[Tuple[np.ndarray, int]], 
        global_params: List[np.ndarray]
    ) -> Parameters:
        """
        Aggregate parameters within a family using HeteroFL logic (finding max shape).
        
        Args:
            family: Family name
            results: List of (parameters_ndarray, num_examples)
            global_params: Current global parameters for this family
            
        Returns:
            Parameters: Updated global parameters
        """
        # 1. Determine maximum shape for each layer across global and all clients
        # Initialize max_shapes with global_params shapes
        max_shapes = [p.shape for p in global_params]
        
        # Check against all client results
        for client_params, _ in results:
            # Extend max_shapes list if client has more layers (though unusual for HeteroFL)
            if len(client_params) > len(max_shapes):
                for _ in range(len(client_params) - len(max_shapes)):
                    max_shapes.append(client_params[len(max_shapes)].shape)
            
            for i, p in enumerate(client_params):
                current_max = max_shapes[i]
                # Compare dimensions
                if len(p.shape) != len(current_max):
                    # Different rank (e.g. Conv2d vs Linear flattened?), rare if arch is consistent.
                    # If this happens, likely incompatible architectures mixed (ResNet vs MLP).
                    # We keep the one with more dimensions or just current global?
                    # For safety, we trust the one with LARGER volume or just keep current max if rank matches.
                    pass 
                else:
                    new_shape = tuple(max(d1, d2) for d1, d2 in zip(current_max, p.shape))
                    max_shapes[i] = new_shape

        # 2. Create accumulators
        weighted_sum = [np.zeros(shape) for shape in max_shapes]
        weights_count = [np.zeros(shape) for shape in max_shapes]
        
        # 3. Aggregate
        for client_params, num_examples in results:
            for i, p in enumerate(client_params):
                if i >= len(weighted_sum):
                   break
                
                # Determine slice for this client's params
                # Assuming top-left alignment (standard HeteroFL)
                # Handle potential rank mismatch by skipping or reshaping? 
                # AFAD assumes same architecture (e.g. ResNet) but different widths.
                # Rank should match.
                
                target_shape = weighted_sum[i].shape
                src_shape = p.shape
                
                if len(target_shape) != len(src_shape):
                    # Skip incompatible layers (e.g. FC mismatch caused by Flatten size diff)
                    continue

                slices = tuple(slice(0, d) for d in src_shape)
                
                # Ensure we fit in target (max_shapes guarantees this, but safety check)
                valid = True
                for s, t in zip(src_shape, target_shape):
                    if s > t: valid = False
                
                if valid:
                    weighted_sum[i][slices] += p * num_examples
                    weights_count[i][slices] += num_examples

        # 4. Average
        new_params = []
        for w_sum, count, original_global in zip(weighted_sum, weights_count, global_params + [None]*(len(weighted_sum)-len(global_params))):
            # Avoid divide by zero
            # Where count > 0, we update. Where count == 0, we can keep original global or 0.
            # HeteroFL: Un-updated parameters should conceptually track global state.
            # If we extended global_params, original is None/smaller.
            
            updated = np.zeros_like(w_sum)
            mask = count > 0
            
            # Update where we have data
            updated[mask] = w_sum[mask] / count[mask]
            
            # Where no data from clients:
            # If we have original global value, keep it.
            if original_global is not None:
                # Need to copy original global into the new larger shape if needed
                # (Slicing logic)
                src_shape = original_global.shape
                target_shape = updated.shape
                
                if len(src_shape) == len(target_shape):
                     slices = tuple(slice(0, d) for d in src_shape)
                     # Only fill if mask is False (no update)? 
                     # Or does Global decay? Standard FL keeps old value.
                     # We blend: updated[~mask] = original[~mask] (roughly)
                     
                     # Extract original values for non-updated regions
                     # But 'updated' shape might be larger than 'original'.
                     # We need to map original into 'updated' buffer.
                     
                     # Strategy: Initialize 'final' with expanded original, then overwrite with updates?
                     # No, HeteroFL says global parameter is the union.
                     # We should preserve previous global values if valid.
                     
                     # Let's simple copy original to a temp buffer
                     temp_global = np.zeros_like(updated)
                     temp_global[slices] = original_global
                     
                     # Fill gaps
                     updated[~mask] = temp_global[~mask]
            
            new_params.append(updated)

        return ndarrays_to_parameters(new_params)
