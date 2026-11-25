"""Optional trajectory capture and offline replay."""

from importlib import import_module

_EXPORTS = {
    "TrajectoryRecord": "src.harness.trajectory_store",
    "TrajectoryStore": "src.harness.trajectory_store",
    "ReplayRunner": "src.harness.replay_runner",
}

__all__ = sorted(_EXPORTS)


def __getattr__(name: str):
    if name not in _EXPORTS:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    value = getattr(import_module(_EXPORTS[name]), name)
    globals()[name] = value
    return value
