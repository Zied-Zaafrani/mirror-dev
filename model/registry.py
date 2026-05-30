"""
Registry utility for MIRROR modular component system.

Provides a generic Registry class for registering and building swappable
components (graph encoders, lab encoders, fusion modules, loss functions)
via decorator-based registration.

Usage:
    GRAPH_ENCODERS = Registry("graph_encoders")

    @GRAPH_ENCODERS.register("hgt_layer")
    class HGTLayer(nn.Module):
        ...

    layer_cls = GRAPH_ENCODERS.get("hgt_layer")
    layer = GRAPH_ENCODERS.build("hgt_layer", hidden_dim=64, num_heads=4)
"""


class Registry:
    """A named registry that maps string keys to classes.

    Supports decorator-based registration, class lookup, and instantiation.
    """

    def __init__(self, name: str):
        self.name = name
        self._registry: dict[str, type] = {}

    def register(self, key: str):
        """Decorator that registers a class under the given key.

        Usage:
            @registry.register("my_key")
            class MyClass:
                ...
        """
        def decorator(cls):
            if key in self._registry:
                raise ValueError(
                    f"Registry '{self.name}': key '{key}' already registered "
                    f"to {self._registry[key].__name__}. Cannot re-register "
                    f"with {cls.__name__}."
                )
            self._registry[key] = cls
            return cls
        return decorator

    def get(self, key: str) -> type:
        """Return the registered class for the given key (without instantiating).

        Raises KeyError with a helpful message listing available keys.
        """
        if key not in self._registry:
            available = ", ".join(sorted(self._registry.keys())) or "(none)"
            raise KeyError(
                f"Registry '{self.name}': unknown key '{key}'. "
                f"Available: [{available}]"
            )
        return self._registry[key]

    def build(self, key: str, **kwargs):
        """Instantiate and return the registered class for the given key."""
        cls = self.get(key)
        import logging
        logger = logging.getLogger(__name__)
        logger.info(f"  [Registry] Building {self.name}: '{key}' ({cls.__name__})")
        return cls(**kwargs)

    def list(self) -> list[str]:
        """Return sorted list of registered keys."""
        return sorted(self._registry.keys())

    def __contains__(self, key: str) -> bool:
        return key in self._registry

    def __repr__(self) -> str:
        return f"Registry(name='{self.name}', keys={self.list()})"


# ─────────────────────────────────────────────────────────────────────────────
# Global registry instances for MIRROR's swappable components
# ─────────────────────────────────────────────────────────────────────────────

GRAPH_ENCODERS = Registry("graph_encoders")
GRAPH_LAYERS = Registry("graph_layers")
LAB_ENCODERS = Registry("lab_encoders")
FUSION_MODULES = Registry("fusion_modules")
LOSS_FUNCTIONS = Registry("loss_functions")
TEMPORAL_ENCODERS = Registry("temporal_encoders")
AGGREGATORS = Registry("aggregators")
SCORERS = Registry("scorers")
PRETRAIN_TASKS = Registry("pretrain_tasks")
