"""
Backward-compatibility shim.

New code should import build_model from src.base_model (or base_model inside
Docker containers).  LiverTumourClassifier is retained for existing code that
still imports from this module.
"""
try:
    from src.base_model import build_model, ClassificationModelBase, _ARCH_REGISTRY  # noqa: F401
except ModuleNotFoundError:
    from base_model import build_model, ClassificationModelBase, _ARCH_REGISTRY  # noqa: F401


class LiverTumourClassifier:
    """Legacy factory class. Use build_model() for new code."""

    @staticmethod
    def get_model(config):
        """Create a classification model from a config namespace or object."""
        model_type  = getattr(config, 'model_type',  'resnet18')
        num_classes = getattr(config, 'num_classes', 5)
        return build_model(
            config={'model': {'model_type': model_type}},
            num_classes=num_classes,
            model_type=model_type,
        )
