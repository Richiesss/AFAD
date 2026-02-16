"""FedGen model wrapper (Zhu et al., ICML 2021).

Adds latent-layer support to any nn.Module by:
1. Replacing the classifier with a two-layer structure:
   bottleneck(feature_dim → latent_dim) + classifier(latent_dim → num_classes)
2. Providing forward_from_latent(z) that feeds generator output
   directly into the classifier, bypassing the feature extractor.

This matches the original FedGen design where models share a common
latent_dim and the generator targets this layer.
"""

import torch.nn as nn


def _find_classifier_info(model: nn.Module) -> tuple[str, int, nn.Linear]:
    """Detect the classifier layer of a model.

    Searches for common classifier attribute names and returns
    the attribute path, input features, and the layer itself.

    Returns:
        (attr_path, in_features, layer)
    """
    # Direct attributes
    for attr in ("fc", "classifier", "head"):
        if hasattr(model, attr):
            layer = getattr(model, attr)
            if isinstance(layer, nn.Linear):
                return attr, layer.in_features, layer

    # Nested: model.vit.heads (HeteroFLViT), model.heads (torchvision ViT)
    for attr in ("heads",):
        obj = model
        path = attr
        # Check nested .vit.heads
        if hasattr(model, "vit") and hasattr(model.vit, attr):
            obj = model.vit
            path = f"vit.{attr}"
        elif hasattr(model, attr):
            obj = model
            path = attr
        else:
            continue

        container = getattr(obj, attr)
        if isinstance(container, nn.Linear):
            return path, container.in_features, container
        if isinstance(container, nn.Sequential):
            for sublayer in container:
                if isinstance(sublayer, nn.Linear):
                    return path, sublayer.in_features, sublayer

    raise ValueError(
        f"Cannot find classifier layer in {type(model).__name__}. "
        "Expected one of: fc, classifier, head, heads."
    )


def _set_nested_attr(model: nn.Module, path: str, value: nn.Module) -> None:
    """Set a nested attribute like 'vit.heads'."""
    parts = path.split(".")
    obj = model
    for part in parts[:-1]:
        obj = getattr(obj, part)
    setattr(obj, parts[-1], value)


class FedGenModelWrapper(nn.Module):
    """Wraps a model with FedGen latent-layer support.

    The original classifier is replaced with:
      bottleneck: Linear(feature_dim → latent_dim)
      classifier: Linear(latent_dim → num_classes)

    For real data:
      forward(x) → backbone → bottleneck → classifier

    For generator output:
      forward_from_latent(z) → classifier

    Args:
        model: Base model (e.g., ResNet18, ViT).
        latent_dim: Latent representation dimension (must match generator).
        num_classes: Number of output classes.
    """

    def __init__(
        self,
        model: nn.Module,
        latent_dim: int = 32,
        num_classes: int = 10,
    ):
        super().__init__()
        self.latent_dim = latent_dim

        # Detect and remove the original classifier
        attr_path, feature_dim, _original = _find_classifier_info(model)
        self.feature_dim = feature_dim
        self._classifier_path = attr_path

        # Replace original classifier with Identity (backbone now outputs features)
        _set_nested_attr(model, attr_path, nn.Identity())
        self.backbone = model

        # New two-layer classifier matching FedGen architecture
        self.bottleneck = nn.Linear(feature_dim, latent_dim)
        self.classifier = nn.Linear(latent_dim, num_classes)

    def forward(self, x):
        """Normal forward: backbone → bottleneck → classifier."""
        features = self.backbone(x)
        latent = self.bottleneck(features)
        return self.classifier(latent)

    def forward_from_latent(self, z):
        """Forward from generator latent output: z → classifier.

        This is equivalent to the original FedGen's
        model(z, start_layer_idx=-1) which runs only the last layer.
        """
        return self.classifier(z)
