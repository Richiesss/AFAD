import flwr as fl
import torch
import torch
import os
import sys

# Add project root to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.models.registry import ModelRegistry
from src.routing.family_router import FamilyRouter
from src.server.strategy.afad_strategy import AFADStrategy
from src.server.generator.synthetic_generator import SyntheticGenerator
from src.client.afad_client import AFADClient
from src.data.mnist_loader import load_mnist_data
from src.utils.logger import setup_logger

# Import models to register them
import src.models.cnn.resnet
import src.models.cnn.mobilenet
import src.models.vit.vit
import src.models.vit.deit

logger = setup_logger("AFADExperiment")

CLIENT_RESOURCES = {"num_cpus": 1.0, "num_gpus": 0.0} # Set GPU to 0 for safety in this env or use if available

def get_device():
    return "cuda" if torch.cuda.is_available() else "cpu"

def client_fn_builder(train_loaders, config):
    """
    Closure to create client_fn with access to data loaders
    """
    
    # Map CID to Model Name (Fixed for Phase 1 as per spec)
    # 0: resnet50, 1: mobilenetv3_large, 2: resnet18, 3: vit_tiny, 4: deit_small
    cid_to_model = {
        "0": "resnet50",
        "1": "mobilenetv3_large",
        "2": "resnet18",
        "3": "vit_tiny",
        "4": "deit_small"
    }

    def client_fn(cid: str) -> fl.client.Client:
        model_name = cid_to_model.get(cid, "resnet18")
        device = get_device()
        
        # Create Model
        model = ModelRegistry.create_model(model_name, num_classes=10)
        
        # Get Loader
        client_id_int = int(cid)
        train_loader = train_loaders[client_id_int % len(train_loaders)]
        
        # Create Client
        return AFADClient(
            cid=cid,
            model=model,
            train_loader=train_loader,
            epochs=config['training']['local_epochs'],
            device=device
        ).to_client()

    return client_fn

def main():
    # Load Config (Manually for now since Hydra setup is optional or I use simple loader)
    from src.utils.config_loader import load_config
    config_dict = load_config("config/afad_config.yaml")
    
    # Data
    train_loaders, test_loader = load_mnist_data(
        num_clients=config_dict['server']['min_clients'],
        batch_size=config_dict['data']['batch_size']
    )
    
    # Strategy Components
    family_router = FamilyRouter()
    generator = SyntheticGenerator(
        latent_dim=config_dict['strategy']['generator']['latent_dim'],
        num_classes=10
    )
    
    # Initial Parameters (Initialize a ResNet18 as 'global' placeholder)
    # In HeteroFL/AFAD, we might not have a single global model.
    # But Flower requires initial parameters usually.
    # We create a dummy model to serialize.
    initial_model = ModelRegistry.create_model("resnet18", num_classes=10)
    initial_params = fl.common.ndarrays_to_parameters(
        [val.cpu().numpy() for val in initial_model.state_dict().values()]
    )
    
    strategy = AFADStrategy(
        initial_parameters=initial_params,
        initial_generator=generator,
        family_router=family_router,
        min_fit_clients=config_dict['server']['min_fit_clients'],
        min_available_clients=config_dict['server']['min_clients'],
        fraction_fit=1.0,
        fraction_evaluate=0.0, # Disable evaluation for now to focus on fit
    )

    # Simulation
    logger.info("Starting simulation...")
    fl.simulation.start_simulation(
        client_fn=client_fn_builder(train_loaders, config_dict),
        num_clients=config_dict['server']['min_clients'],
        config=fl.server.ServerConfig(num_rounds=config_dict['experiment']['num_rounds']),
        strategy=strategy,
        client_resources=CLIENT_RESOURCES,
    )

if __name__ == "__main__":
    main()
