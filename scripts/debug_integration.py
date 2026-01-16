import torch
import sys
import os
import numpy as np

# Add path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.models.registry import ModelRegistry
from src.routing.family_router import FamilyRouter
from src.server.strategy.afad_strategy import AFADStrategy
from src.server.generator.synthetic_generator import SyntheticGenerator
from src.client.afad_client import AFADClient
from src.data.mnist_loader import load_mnist_data
import flwr as fl
from flwr.common import ndarrays_to_parameters, parameters_to_ndarrays, FitRes, Status, Code
from flwr.server.client_proxy import ClientProxy

# Register models
import src.models.cnn.resnet
import src.models.vit.vit

class MockClientProxy(ClientProxy):
    def __init__(self, cid):
        super().__init__(cid)
    
    def get_properties(self, ins, timeout):
        return None
        
    def get_parameters(self, ins, timeout):
        return None
        
    def fit(self, ins, timeout):
        return None
        
    def evaluate(self, ins, timeout):
        return None
        
    def reconnect(self, ins, timeout):
        return None

def main():
    print("Starting manual integration test...")
    
    # 1. Setup Strategy
    router = FamilyRouter()
    generator = SyntheticGenerator(output_shape=(1, 28, 28))
    # Dummy initial params
    m = ModelRegistry.create_model("resnet18")
    initial_params = ndarrays_to_parameters([val.cpu().numpy() for val in m.state_dict().values()])
    
    strategy = AFADStrategy(
        initial_parameters=initial_params,
        initial_generator=generator,
        family_router=router
    )
    
    # 2. Setup Clients (1 CNN, 1 ViT)
    train_loaders, _ = load_mnist_data(num_clients=2, batch_size=4)
    
    client_cnn = AFADClient("0", ModelRegistry.create_model("resnet18"), train_loaders[0], epochs=1, device="cpu")
    client_vit = AFADClient("1", ModelRegistry.create_model("vit_tiny"), train_loaders[1], epochs=1, device="cpu")
    
    # 3. Simulate Round 1
    server_round = 1
    print(f"--- Round {server_round} ---")
    
    # Configure Fit
    # We need to mock ClientManager? Strategy.configure_fit takes client_manager.
    # But usually we can just pass a list of clients if we mock it.
    # Or purely manually create FitIns.
    
    # Let's manually create FitIns as if configure_fit returned them (skipping checking configure_fit logic for sampling)
    # But we want to test configure_fit's custom logic if any (adding config).
    
    # Mock Manager
    class MockClientManager:
        def sample(self, num_clients, min_num_clients=None):
            return [MockClientProxy("0"), MockClientProxy("1")]
            
        def num_available(self):
            return 2
            
    client_manager = MockClientManager()
    
    # Call strategy.configure_fit
    # Note: Our AFADStrategy calls super().configure_fit which calls client_manager.sample.
    fit_ins_list = strategy.configure_fit(server_round, initial_params, client_manager)
    print(f"Strategy generated {len(fit_ins_list)} FitIns instructions.")
    
    # 4. Clients Run Fit
    results = []
    
    for client_proxy, fit_ins in fit_ins_list:
        cid = client_proxy.cid
        target_client = client_cnn if cid == "0" else client_vit
        
        print(f"Client {cid} ({target_client.model.__class__.__name__}) training...")
        
        # Determine family for creating FitRes metrics (normally client does this)
        family, _ = router.detect_family(target_client.model)
        
        # Client fit
        updated_params, num_examples, metrics = target_client.fit(
            parameters_to_ndarrays(fit_ins.parameters), 
            fit_ins.config
        )
        
        # Add family to metrics (as per generic client implementation TODO)
        metrics["family"] = family
        
        # Create FitRes
        fit_res = FitRes(
            status=Status(code=Code.OK, message="Success"),
            parameters=ndarrays_to_parameters(updated_params),
            num_examples=num_examples,
            metrics=metrics
        )
        results.append((client_proxy, fit_res))
        
    print("Clients finished training.")
    
    # 5. Aggregate Fit
    print("Aggregating results...")
    agg_params, agg_metrics = strategy.aggregate_fit(server_round, results, failures=[])
    
    print("Aggregation complete.")
    print("Metrics:", agg_metrics)
    
    # Check if global models are updated in strategy
    print("Global Models keys:", strategy.global_models.keys())
    
    if "resnet" in strategy.global_models and "vit" in strategy.global_models:
        print("SUCCESS: Both families have updated global models.")
    else:
        print("FAILURE: Missing family updates.")
        sys.exit(1)
        
if __name__ == "__main__":
    main()
