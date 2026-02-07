"""Integration test: verifies the full AFAD pipeline end-to-end."""

import numpy as np
from flwr.common import (
    Code,
    FitRes,
    Status,
    ndarrays_to_parameters,
)
from flwr.server.client_proxy import ClientProxy

import src.models.cnn.resnet  # noqa: F401 - register models
import src.models.vit.vit  # noqa: F401
from src.client.afad_client import AFADClient
from src.data.mnist_loader import load_mnist_data
from src.models.registry import ModelRegistry
from src.routing.family_router import FamilyRouter
from src.server.generator.synthetic_generator import SyntheticGenerator
from src.server.strategy.afad_strategy import AFADStrategy


class _MockClientProxy(ClientProxy):
    """Minimal ClientProxy for testing."""

    def __init__(self, cid: str):
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


def _build_strategy() -> AFADStrategy:
    """Create an AFADStrategy with model factories for ResNet18 and ViT-Tiny."""
    generator = SyntheticGenerator(latent_dim=32, num_classes=10)
    router = FamilyRouter()
    model_factories = {
        "resnet18": lambda num_classes=10: ModelRegistry.create_model(
            "resnet18", num_classes=num_classes
        ),
        "vit_tiny": lambda num_classes=10: ModelRegistry.create_model(
            "vit_tiny", num_classes=num_classes
        ),
    }
    initial_model = ModelRegistry.create_model("resnet18", num_classes=10)
    initial_params = ndarrays_to_parameters(
        [val.cpu().numpy() for val in initial_model.state_dict().values()]
    )
    return AFADStrategy(
        initial_parameters=initial_params,
        initial_generator=generator,
        family_router=router,
        model_factories=model_factories,
        fedgen_config={
            "temperature": 4.0,
            "gen_steps": 2,
            "distill_steps": 2,
            "batch_size": 8,
        },
        training_config={"lr": 0.01, "momentum": 0.9, "local_epochs": 1},
        min_fit_clients=2,
        min_available_clients=2,
        fraction_fit=1.0,
        fraction_evaluate=0.0,
    )


def _simulate_client_fit(
    client: AFADClient, config: dict
) -> tuple[list[np.ndarray], int, dict]:
    """Run a client fit with empty parameters (first round)."""
    return client.fit(parameters=[], config=config)


class TestIntegration:
    """Smoke test: 2 rounds with 2 clients (ResNet18 + ViT-Tiny)."""

    def setup_method(self):
        self.strategy = _build_strategy()
        self.train_loaders, self.test_loader = load_mnist_data(
            num_clients=2, batch_size=8, max_samples=200
        )

    def test_two_round_pipeline(self):
        """Run 2 rounds: verify HeteroFL aggregation + FedGen distillation."""
        client_cnn = AFADClient(
            cid="0",
            model=ModelRegistry.create_model("resnet18"),
            train_loader=self.train_loaders[0],
            epochs=1,
            device="cpu",
            family="cnn",
            model_name="resnet18",
        )
        client_vit = AFADClient(
            cid="1",
            model=ModelRegistry.create_model("vit_tiny"),
            train_loader=self.train_loaders[1],
            epochs=1,
            device="cpu",
            family="vit",
            model_name="vit_tiny",
        )
        clients = {"0": client_cnn, "1": client_vit}

        for round_num in range(1, 3):
            # Simulate client fit
            results = []
            for cid, client in clients.items():
                config = {"use_local_init": (round_num == 1), "round": round_num}
                updated_params, num_examples, metrics = client.fit([], config)
                proxy = _MockClientProxy(cid)
                fit_res = FitRes(
                    status=Status(code=Code.OK, message="ok"),
                    parameters=ndarrays_to_parameters(updated_params),
                    num_examples=num_examples,
                    metrics=metrics,
                )
                results.append((proxy, fit_res))

            # Aggregate
            agg_params, agg_metrics = self.strategy.aggregate_fit(
                round_num, results, failures=[]
            )

            assert agg_params is not None
            assert agg_metrics["total_clients"] == 2

        # After 2 rounds, strategy should have 2 model groups
        assert len(self.strategy.global_models) == 2

        # FedGen should have been triggered in round 2
        assert len(self.strategy.sig_to_model_name) == 2

    def test_global_models_have_correct_signatures(self):
        """Verify each client's model creates a unique signature group."""
        client_cnn = AFADClient(
            cid="0",
            model=ModelRegistry.create_model("resnet18"),
            train_loader=self.train_loaders[0],
            epochs=1,
            device="cpu",
            model_name="resnet18",
        )
        client_vit = AFADClient(
            cid="1",
            model=ModelRegistry.create_model("vit_tiny"),
            train_loader=self.train_loaders[1],
            epochs=1,
            device="cpu",
            model_name="vit_tiny",
        )

        # Round 1
        results = []
        for cid, client in [("0", client_cnn), ("1", client_vit)]:
            params, num_ex, metrics = client.fit([], {"use_local_init": True})
            proxy = _MockClientProxy(cid)
            results.append(
                (
                    proxy,
                    FitRes(
                        status=Status(code=Code.OK, message="ok"),
                        parameters=ndarrays_to_parameters(params),
                        num_examples=num_ex,
                        metrics=metrics,
                    ),
                )
            )

        self.strategy.aggregate_fit(1, results, failures=[])

        # Should have 2 unique signatures (ResNet18 != ViT-Tiny)
        assert len(self.strategy.global_models) == 2

        # Model names should be tracked
        model_names = set(self.strategy.sig_to_model_name.values())
        assert model_names == {"resnet18", "vit_tiny"}
