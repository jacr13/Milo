import os

import wandb


class WandbLogger:
    def __init__(
        self,
        experiment_name: str,
        project: str,
        group: str,
        config: dict | None = None,
        log_dir: str | None = None,
        entity: str | None = None,
        mode: str | None = None,
    ) -> None:
        self.experiment_name = experiment_name
        self.project = project
        self.group = group
        self.config = config
        self.log_dir = log_dir
        self.entity = entity
        self.mode = mode

        self._init()

    def _init(self) -> None:
        if self.log_dir is not None:
            os.makedirs(self.log_dir, exist_ok=True)

        wandb.init(
            project=self.project,
            name=self.experiment_name,
            id=self.experiment_name,
            group=self.group,
            config=self.config,
            dir=self.log_dir,
            entity=self.entity,
            mode=self.mode,
            resume="allow",
        )

    def log(self, **kwargs) -> None:
        step = kwargs.pop("step", None)
        data = {}
        for k, v in kwargs.items():
            if isinstance(v, dict):
                for k2, v2 in v.items():
                    data[f"{k}/{k2}"] = v2
            else:
                data[k] = v

        wandb.log(data, step=step)

    def download(self) -> None:
        pass

    def close(self) -> None:
        wandb.finish()
