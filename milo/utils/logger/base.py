from abc import ABC, abstractmethod


class BaseLogger(ABC):
    @abstractmethod
    def log(self, **kwargs) -> None:
        """Log data."""

    @abstractmethod
    def download(self) -> None:
        """Download data from remote logger."""

    @abstractmethod
    def close(self) -> None:
        """Close the logger."""
