from __future__ import annotations

__all__ = ["DatasetManifest", "ExperimentManifest", "ModelManifest"]


def __getattr__(name: str):
    if name in __all__:
        from jepa_glm.experiments.manifest import DatasetManifest, ExperimentManifest, ModelManifest

        return {
            "DatasetManifest": DatasetManifest,
            "ExperimentManifest": ExperimentManifest,
            "ModelManifest": ModelManifest,
        }[name]
    raise AttributeError(name)
