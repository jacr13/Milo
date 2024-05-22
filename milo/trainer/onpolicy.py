from trainer.base import Trainer


class OnpolicyTrainer(Trainer):
    def policy_update_fn(
        self,
    ) -> dict:
        """Perform one on-policy update by passing the entire buffer to the policy's update method."""
        assert self.train_collector is not None
        training_stat = self.policy.update(
            sample_size=None,
            buffer=self.train_collector.buffer,
            batch_size=self.batch_size,
            repeat=self.repeat_per_collect,
        )

        # Note: this is the main difference to the off-policy trainer!
        # The second difference is that batches of data are sampled without replacement
        # during training, whereas in off-policy or offline training, the batches are
        # sampled with replacement (and potentially custom prioritization).
        self.train_collector.reset_buffer(keep_statistics=True)

        return training_stat
