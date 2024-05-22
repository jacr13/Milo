from abc import ABC, abstractmethod

from milo.data.buffer.base import ReplayBuffer
from milo.data.collector import Collector
from milo.policy.base import BasePolicy


class Trainer(ABC):
    def __init__(
        self,
        policy: BasePolicy,
        train_collector: Collector | None = None,
        test_collector: Collector | None = None,
        buffer: ReplayBuffer | None = None,
        max_epoch: int = 1000,
        batch_size: int = 64,
        step_per_epoch: int | None = None,
        repeat_per_collect: int | None = None,
        update_per_step: float = 1.0,
        step_per_collect: int | None = None,
        episode_per_collect: int | None = None,
        episode_per_test: int | None = None,
        eval_frequency: int = 10,
        save_frequency: int | None = None,
        save_best: bool = True,
        logger=None,
    ):
        self.policy = policy
        self.train_collector = train_collector
        self.test_collector = test_collector
        self.buffer = buffer
        self.max_epoch = max_epoch
        self.batch_size = batch_size

        self.step_per_epoch = step_per_epoch
        self.repeat_per_collect = repeat_per_collect
        self.update_per_step = update_per_step

        self.step_per_collect = step_per_collect
        self.episode_per_collect = episode_per_collect

        self.episode_per_test = episode_per_test
        self.eval_frequency = eval_frequency

        self.save_frequency = save_frequency
        self.save_best = save_best

        self.logger = logger

        self._interactions_count = 0

    @abstractmethod
    def policy_update_fn(self) -> dict:
        """Policy update logic (should be implemented by subclasses)."""

    def training_step(self) -> tuple[dict, dict]:

        if self.train_collector is not None:
            self.train_collector.reset()
            collect_stats = self.train_collector.collect(
                n_step=self.step_per_collect,
                n_episode=self.episode_per_collect,
            )
            self._interactions_count += collect_stats["collect_step"]
        else:
            assert self.buffer is not None, "Either train_collector or buffer must be provided."
            collect_stats = {}

        training_stats = self.policy_update_fn()
        return training_stats, collect_stats

    def test_step(self, collector: Collector | None = None, n_episode: int | None = None) -> dict:
        collector = collector or self.test_collector
        n_episodes = n_episode or self.episode_per_test
        assert n_episodes is not None, "Either provide `n_episode` or `trainer.episode_per_test`"
        assert collector is not None, "Provide `collector` or `trainer.test_collector`"

        collector.reset()
        return collector.collect(n_episode=n_episodes)

    def run(self) -> dict:
        best_eval_return = -float("inf")
        current_eval_return = -float("inf")
        training_stats = None
        test_stats = {}
        collect_stats = None
        stats = {}

        # TODO: Add pretraining
        try:
            for epoch in range(self.max_epoch):
                print(f"Epoch: {epoch}/{self.max_epoch}, Step: {self._interactions_count}")

                if self.save_frequency is not None and epoch % self.save_frequency == 0:
                    self.save()

                training_stats, collect_stats = self.training_step()

                if self.test_collector is not None and epoch % self.eval_frequency == 0:
                    test_stats = self.test_step()

                    current_eval_return = test_stats["return"]

                    # Save best
                    if current_eval_return > best_eval_return and self.save_best:
                        best_eval_return = current_eval_return
                        self.save()

                if self.logger is not None:
                    self.logger.log(
                        train=training_stats,
                        test=test_stats,
                        collection=collect_stats,
                        step=self._interactions_count,
                    )

        finally:
            self.close()
        return stats

    def save(self) -> None:
        return

    def load(self) -> None:
        return

    def close(self) -> None:
        return
