from abc import ABC, abstractmethod


class BaseLogger(ABC):
    @abstractmethod
    def log(self, **kwargs) -> None:
        """Log data."""

    def download(self) -> None:
        pass

    @abstractmethod
    def close(self) -> None:
        """Close the logger."""
