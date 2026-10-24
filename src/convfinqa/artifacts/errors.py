"""Errors raised by the artifact system."""


class ArtifactError(Exception):
    """Base class for artifact-system errors."""


class StaleArtifactError(ArtifactError):
    """A heavy artifact is stale or missing and heavy builds are not enabled."""


class BuildError(ArtifactError):
    """A builder failed."""
