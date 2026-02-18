"""Integration test: verifies the full AFAD pipeline end-to-end."""

from flwr.common import (
    Code,
    FitRes,
    Status,
    ndarrays_to_parameters,
)
from flwr.server.client_proxy import ClientProxy

import src.models.cnn.resnet  # noqa: F401 - register models
import src.models.vit.vit  # noqa: F401
from src.client.heterofl_client import HeteroFLClient
from src.data.mnist_loader import load_mnist_data
from src.models.fedgen_wrapper import FedGenModelWrapper
from src.models.registry import ModelRegistry
from src.server.generator.fedgen_generator import FedGenGenerator
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


def _build_strategy(
    enable_fedgen: bool = True,
    enable_heterofl: bool = True,
    num_classes: int = 10,
) -> AFADStrategy:
    """Create an AFADStrategy with model factories for ResNet18 and ViT-Tiny."""
    latent_dim = 32
    generator = (
        FedGenGenerator(
            noise_dim=32, num_classes=num_classes, hidden_dim=64, latent_dim=latent_dim
        )
        if enable_fedgen
        else None
    )

    if enable_fedgen:
        # FedGen/AFAD modes: factories must create FedGenModelWrapper
        model_factories = {
            "resnet18": lambda num_classes=num_classes: FedGenModelWrapper(
                ModelRegistry.create_model("resnet18", num_classes=num_classes),
                latent_dim=latent_dim,
                num_classes=num_classes,
            ),
            "vit_tiny": lambda num_classes=num_classes: FedGenModelWrapper(
                ModelRegistry.create_model("vit_tiny", num_classes=num_classes),
                latent_dim=latent_dim,
                num_classes=num_classes,
            ),
        }
    else:
        # HeteroFL Only: plain models
        model_factories = {
            "resnet18": lambda num_classes=num_classes: ModelRegistry.create_model(
                "resnet18", num_classes=num_classes
            ),
            "vit_tiny": lambda num_classes=num_classes: ModelRegistry.create_model(
                "vit_tiny", num_classes=num_classes
            ),
        }
    initial_model = ModelRegistry.create_model("resnet18", num_classes=num_classes)
    initial_params = ndarrays_to_parameters(
        [val.cpu().numpy() for val in initial_model.state_dict().values()]
    )

    # Client model rates for HeteroFL (all full-rate for simplicity)
    client_model_rates = {"0": 1.0, "1": 1.0}

    return AFADStrategy(
        initial_parameters=initial_params,
        generator=generator,
        model_factories=model_factories,
        client_model_rates=client_model_rates,
        family_model_names={"cnn": "resnet18", "vit": "vit_tiny"},
        enable_fedgen=enable_fedgen,
        enable_heterofl=enable_heterofl,
        num_classes=num_classes,
        fedgen_config={
            "gen_lr": 1e-3,
            "batch_size": 8,
            "gen_epochs": 1,
            "teacher_iters": 2,
            "device": "cpu",
        },
        training_config={"lr": 0.01, "momentum": 0.9, "local_epochs": 1},
        min_fit_clients=2,
        min_available_clients=2,
        fraction_fit=1.0,
        fraction_evaluate=0.0,
    )


class TestIntegration:
    """Smoke test: 2 rounds with 2 clients (ResNet18 + ViT-Tiny)."""

    def setup_method(self):
        self.strategy = _build_strategy()
        self.train_loaders, self.test_loader = load_mnist_data(
            num_clients=2, batch_size=8, max_samples=200
        )

    def _make_clients(self, strategy=None):
        """Create CNN + ViT HeteroFL clients and register their families."""
        s = strategy or self.strategy
        s.set_client_family("0", "cnn")
        s.set_client_family("1", "vit")
        s._initialize_family_models()

        client_cnn = HeteroFLClient(
            cid="0",
            model=ModelRegistry.create_model("resnet18"),
            train_loader=self.train_loaders[0],
            epochs=1,
            device="cpu",
            family="cnn",
            model_name="resnet18",
        )
        client_vit = HeteroFLClient(
            cid="1",
            model=ModelRegistry.create_model("vit_tiny"),
            train_loader=self.train_loaders[1],
            epochs=1,
            device="cpu",
            family="vit",
            model_name="vit_tiny",
        )
        return {"0": client_cnn, "1": client_vit}

    def _run_round(self, clients, strategy, round_num):
        """Simulate one FL round and return aggregation results."""
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

        return strategy.aggregate_fit(round_num, results, failures=[])

    def test_two_round_pipeline(self):
        """Run 2 rounds: verify family-based aggregation + generator training."""
        clients = self._make_clients()

        for round_num in range(1, 3):
            agg_params, agg_metrics = self._run_round(clients, self.strategy, round_num)
            assert agg_params is not None
            assert agg_metrics["total_clients"] == 2

        # After 2 rounds, strategy should have 2 family global models
        assert len(self.strategy.family_global_models) == 2
        assert "cnn" in self.strategy.family_global_models
        assert "vit" in self.strategy.family_global_models

        # Generator should have been trained (2 families available)
        assert self.strategy._generator_trained

    def test_family_global_models_created(self):
        """Verify each family creates a unique global model entry."""
        clients = self._make_clients()

        # Round 1
        self._run_round(clients, self.strategy, round_num=1)

        # Should have 2 families
        assert len(self.strategy.family_global_models) == 2
        assert set(self.strategy.family_global_models.keys()) == {"cnn", "vit"}

    def test_label_counts_collected(self):
        """Verify label counts are collected from client fit metrics."""
        client = HeteroFLClient(
            cid="0",
            model=ModelRegistry.create_model("resnet18"),
            train_loader=self.train_loaders[0],
            epochs=1,
            device="cpu",
            model_name="resnet18",
        )

        params, num_ex, metrics = client.fit([], {"use_local_init": True})

        # label_counts should be in metrics as comma-separated string
        assert "label_counts" in metrics
        counts_str = metrics["label_counts"]
        counts = [int(x) for x in counts_str.split(",")]
        assert len(counts) == 10
        assert sum(counts) > 0

    def test_fedgen_disabled_no_generator_training(self):
        """With enable_fedgen=False, generator should not be trained."""
        strategy = _build_strategy(enable_fedgen=False)
        clients = self._make_clients(strategy)

        self._run_round(clients, strategy, round_num=1)
        assert not strategy._generator_trained

    def test_fedgen_only_mode_fedavg_aggregation(self):
        """enable_heterofl=False: FedAvg aggregation within each family."""
        strategy = _build_strategy(enable_heterofl=False, enable_fedgen=True)
        train_loaders, _test_loader = load_mnist_data(
            num_clients=2, batch_size=8, max_samples=200
        )

        strategy.set_client_family("0", "cnn")
        strategy.set_client_family("1", "vit")
        strategy._initialize_family_models()

        # FedGen mode: clients must use FedGenModelWrapper to match strategy
        latent_dim = 32
        client_cnn = HeteroFLClient(
            cid="0",
            model=FedGenModelWrapper(
                ModelRegistry.create_model("resnet18"),
                latent_dim=latent_dim,
            ),
            train_loader=train_loaders[0],
            epochs=1,
            device="cpu",
            family="cnn",
            model_name="resnet18",
        )
        client_vit = HeteroFLClient(
            cid="1",
            model=FedGenModelWrapper(
                ModelRegistry.create_model("vit_tiny"),
                latent_dim=latent_dim,
            ),
            train_loader=train_loaders[1],
            epochs=1,
            device="cpu",
            family="vit",
            model_name="vit_tiny",
        )
        clients = {"0": client_cnn, "1": client_vit}

        for round_num in range(1, 3):
            agg_params, agg_metrics = self._run_round(clients, strategy, round_num)
            assert agg_params is not None
            assert agg_metrics["total_clients"] == 2

        # Family global models should exist (FedAvg aggregation)
        assert len(strategy.family_global_models) == 2

        # Generator should have been trained (FedGen enabled + 2 families)
        assert strategy._generator_trained


class TestIntegration11Classes:
    """Test with num_classes=11 (OrganAMNIST-like)."""

    def setup_method(self):
        self.strategy = _build_strategy(num_classes=11)
        self.train_loaders, self.test_loader = load_mnist_data(
            num_clients=2, batch_size=8, max_samples=200
        )

    def test_11_class_pipeline(self):
        """Run 1 round with 11-class models to verify num_classes propagation."""
        self.strategy.set_client_family("0", "cnn")
        self.strategy.set_client_family("1", "vit")
        self.strategy._initialize_family_models()

        client_cnn = HeteroFLClient(
            cid="0",
            model=ModelRegistry.create_model("resnet18", num_classes=11),
            train_loader=self.train_loaders[0],
            epochs=1,
            device="cpu",
            family="cnn",
            model_name="resnet18",
            num_classes=11,
        )
        client_vit = HeteroFLClient(
            cid="1",
            model=ModelRegistry.create_model("vit_tiny", num_classes=11),
            train_loader=self.train_loaders[1],
            epochs=1,
            device="cpu",
            family="vit",
            model_name="vit_tiny",
            num_classes=11,
        )

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

        agg_params, agg_metrics = self.strategy.aggregate_fit(1, results, failures=[])
        assert agg_params is not None
        assert agg_metrics["total_clients"] == 2

    def test_11_class_label_counts(self):
        """Client with num_classes=11 should produce 11-element label_counts."""
        client = HeteroFLClient(
            cid="0",
            model=ModelRegistry.create_model("resnet18", num_classes=11),
            train_loader=self.train_loaders[0],
            epochs=1,
            device="cpu",
            model_name="resnet18",
            num_classes=11,
        )
        params, num_ex, metrics = client.fit([], {"use_local_init": True})
        counts = [int(x) for x in metrics["label_counts"].split(",")]
        assert len(counts) == 11
