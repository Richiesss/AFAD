"""Test AFADClient initialization with Non-IID OrganAMNIST data."""
import sys
import time

sys.path.insert(0, "/home/AFAD")

print("Importing modules...", flush=True)
import torch
from src.data.medmnist_loader import load_organamnist_data
from src.models.registry import ModelRegistry
from src.models.fedgen_wrapper import FedGenModelWrapper
from src.server.generator.fedgen_generator import FedGenGenerator
from src.client.afad_client import AFADClient
import src.models.cnn.heterofl_resnet  # noqa

print("Imports OK", flush=True)

print("Loading Non-IID data...", flush=True)
t0 = time.time()
train_loaders, test_loader = load_organamnist_data(
    num_clients=10, batch_size=32, alpha=0.5, distribution="non_iid", seed=42
)
print(f"Data loaded in {time.time()-t0:.1f}s", flush=True)
print(f"  client 0: {len(train_loaders[0].dataset)} samples", flush=True)

for model_rate in [1.0, 0.5, 0.25]:
    print(f"\nTesting model_rate={model_rate}...", flush=True)

    print(f"  Creating base_model (rate={model_rate})...", flush=True)
    base_model = ModelRegistry.create_model(
        "heterofl_resnet18", num_classes=11, model_rate=model_rate
    )
    print("  base_model OK", flush=True)

    print("  Wrapping model...", flush=True)
    model = FedGenModelWrapper(base_model, latent_dim=32, num_classes=11)
    print("  wrap OK", flush=True)

    print("  Creating generator...", flush=True)
    generator = FedGenGenerator(noise_dim=32, num_classes=11, latent_dim=32)
    print("  generator OK", flush=True)

    print("  Creating AFADClient (proj_gamma=0.01)...", flush=True)
    t = time.time()
    client = AFADClient(
        cid="0",
        model=model,
        generator=generator,
        train_loader=train_loaders[0],
        epochs=1,
        device="cpu",
        family="cnn",
        model_rate=model_rate,
        model_name="heterofl_resnet18",
        num_classes=11,
        proto_gamma=0.01,
    )
    elapsed = time.time() - t
    print(f"  Client created in {elapsed:.2f}s", flush=True)
    print(f"  label_counts={client.label_counts}", flush=True)

print("\nAll tests passed!", flush=True)
