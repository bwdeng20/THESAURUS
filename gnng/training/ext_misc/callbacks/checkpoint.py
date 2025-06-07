import warnings
from typing import Optional, Dict, Any
from typing_extensions import override
import logging
import os
import re
import time
import shutil
import yaml
import torch
import copy
from weakref import proxy
from datetime import timedelta
from pathlib import Path
from torch import Tensor
from lightning_utilities.core.rank_zero import rank_zero_info, rank_zero_warn
from .base import Callback
from gnng.typing import PathStr
from gnng.utils import get_filesystem, _is_local_file_protocol
from lightning.fabric.utilities.cloud_io import _is_dir

log = logging.getLogger(__name__)


class Checkpoint(Callback):
    r"""This is the base class for model checkpointing.

    Expert users may want to subclass it in case of writing custom :class:`~lightning.pytorch.callbacks.Checkpoint`
    callback, so that the trainer recognizes the custom class as a checkpointing callback.

    """


class ModelCheckpoint(Checkpoint):
    """Callback to save the model or model weights
    at some frequency.

    """

    CHECKPOINT_JOIN_CHAR = "-"
    CHECKPOINT_EQUALS_CHAR = "="
    CHECKPOINT_NAME_LAST = "last"
    FILE_EXTENSION = ".ckpt"
    STARTING_VERSION = 1

    def __init__(
        self,
        dirpath: Optional[PathStr] = None,
        filename: Optional[str] = None,
        monitor: Optional[str] = None,
        verbose: bool = False,
        save_last: Optional[bool] = None,
        save_top_k: int = 1,
        save_weights_only: bool = False,
        mode: str = "min",
        auto_insert_metric_name: bool = True,
        every_n_train_steps: Optional[int] = None,
        train_time_interval: Optional[timedelta] = None,
        every_n_epochs: Optional[int] = None,
        save_on_train_epoch_end: Optional[bool] = None,
        enable_version_counter: bool = True,
    ):
        super().__init__()
        self.monitor = monitor
        self.verbose = verbose
        self.save_last = save_last
        self.save_top_k = save_top_k
        self.save_weights_only = save_weights_only
        self.auto_insert_metric_name = auto_insert_metric_name
        self._save_on_train_epoch_end = save_on_train_epoch_end
        self._enable_version_counter = enable_version_counter

        self._last_global_step_saved = 0  # no need to save when no losses were taken
        self._last_time_checked: Optional[float] = None
        self.current_score: Optional[Tensor] = None
        self.best_k_models: Dict[str, Tensor] = {}
        self.kth_best_model_path = ""
        self.best_model_score: Optional[Tensor] = None
        self.last_model_score: Optional[Tensor] = None
        self.best_model_path = ""
        self.last_model_path = ""
        self._last_checkpoint_saved = ""

        self.kth_value: Tensor
        self.dirpath: Optional[PathStr]
        self.__init_monitor_mode(mode)
        self.__init_ckpt_dir(dirpath, filename)
        self.__init_triggers(every_n_train_steps, every_n_epochs, train_time_interval)
        self.__validate_init_configuration()

    def setup(self, trainer: Any, model: Any, stage: str) -> None:
        fabric = trainer.fabric
        dirpath = self.__resolve_ckpt_dir(trainer)
        dirpath = fabric.strategy.broadcast(dirpath)
        self.dirpath = dirpath
        self._fs = get_filesystem(self.dirpath or "")
        if fabric.is_global_zero and stage == "fit":
            self.__warn_if_dir_not_empty(self.dirpath)

    def on_train_start(self, trainer: Any, model: "torch.nn.Module") -> None:
        self._last_time_checked = time.monotonic()

    def on_train_batch_end(
        self,
        trainer: Any,
        model: "torch.nn.Module",
        outputs: Dict[str, Any],
        batch: Any,
        batch_idx: int,
    ) -> None:
        """Save checkpoint on train batch end if we meet the criteria for `every_n_train_steps`"""
        if self._should_skip_saving_checkpoint(trainer):
            return
        skip_batch = self._every_n_train_steps < 1 or (trainer.global_step % self._every_n_train_steps != 0)

        train_time_interval = self._train_time_interval
        skip_time = True
        now = time.monotonic()
        if train_time_interval:
            prev_time_check = self._last_time_checked
            skip_time = prev_time_check is None or (now - prev_time_check) < train_time_interval.total_seconds()
            # in case we have time differences across ranks
            # broadcast the decision on whether to checkpoint from rank 0 to avoid possible hangs
            skip_time = trainer.fabric.strategy.broadcast(skip_time)

        if skip_batch and skip_time:
            return
        if not skip_time:
            self._last_time_checked = now

        monitor_candidates = self._monitor_candidates(trainer)
        self._save_topk_checkpoint(trainer, monitor_candidates)
        self._save_last_checkpoint(trainer, monitor_candidates)

    def on_train_epoch_end(self, trainer: Any, model: "torch.nn.Module") -> None:
        """Save a checkpoint at the end of the training epoch."""
        if not self._should_skip_saving_checkpoint(trainer) and self._should_save_on_train_epoch_end(trainer):
            monitor_candidates = self._monitor_candidates(trainer)
            if self._every_n_epochs >= 1 and (trainer.current_epoch + 1) % self._every_n_epochs == 0:
                self._save_topk_checkpoint(trainer, monitor_candidates)
            self._save_last_checkpoint(trainer, monitor_candidates)

    def on_validation_end(self, trainer: Any, model: "torch.nn.Module") -> None:
        """Save a checkpoint at the end of the validation stage."""
        if not self._should_skip_saving_checkpoint(trainer) and not self._should_save_on_train_epoch_end(trainer):
            monitor_candidates = self._monitor_candidates(trainer)
            if self._every_n_epochs >= 1 and (trainer.current_epoch + 1) % self._every_n_epochs == 0:
                self._save_topk_checkpoint(trainer, monitor_candidates)
            self._save_last_checkpoint(trainer, monitor_candidates)

    @override
    def load_state_dict(self, state_dict: Dict[str, Any]) -> None:
        dirpath_from_ckpt = state_dict.get("dirpath", self.dirpath)

        if self.dirpath == dirpath_from_ckpt:
            self.best_model_score = state_dict["best_model_score"]
            self.last_model_score = state_dict["last_model_score"]
            self.kth_best_model_path = state_dict.get("kth_best_model_path", self.kth_best_model_path)
            self.kth_value = state_dict.get("kth_value", self.kth_value)
            self.best_k_models = state_dict.get("best_k_models", self.best_k_models)
            self.last_model_path = state_dict.get("last_model_path", self.last_model_path)
        else:
            warnings.warn(
                f"The dirpath has changed from {dirpath_from_ckpt!r} to {self.dirpath!r},"
                " therefore `best_model_score`, `kth_best_model_path`, `kth_value`, `last_model_path` and"
                " `best_k_models` won't be reloaded. Only `best_model_path` will be reloaded."
            )

        self.best_model_path = state_dict["best_model_path"]

    def _save_topk_checkpoint(self, trainer: Any, monitor_candidates: Dict[str, Tensor]) -> None:
        if self.save_top_k == 0:
            return
        # validate metric
        if self.monitor is not None:
            if self.monitor not in monitor_candidates:
                m = (
                    f"`ModelCheckpoint(monitor={self.monitor!r})` could not find the monitored key in the returned"
                    f" metrics: {list(monitor_candidates)}."
                    f" HINT: Did you call `log({self.monitor!r}, value)` in the `FabricTrainer`?"
                )
                raise RuntimeError(m)
            self._save_monitor_checkpoint(trainer, monitor_candidates)
        else:
            self._save_none_monitor_checkpoint(trainer, monitor_candidates)

    def _save_checkpoint(self, trainer: Any, filepath: str) -> None:
        trainer.save_checkpoint(filepath, state=None, only_weights=self.save_weights_only)
        self._last_global_step_saved = trainer.global_step
        self._last_checkpoint_saved = filepath

        # notify loggers
        if trainer.fabric.is_global_zero:
            for logger in trainer.fabric.loggers:
                logger.after_save_checkpoint(proxy(self))

    def _save_monitor_checkpoint(self, trainer: Any, monitor_candidates: Dict[str, Tensor]) -> None:
        assert self.monitor
        current = monitor_candidates.get(self.monitor)
        if self.check_monitor_top_k(trainer, current):
            assert current is not None
            self._update_best_and_save(current, trainer, monitor_candidates)
        elif self.verbose:
            epoch = monitor_candidates["epoch"]
            step = monitor_candidates["step"]
            rank_zero_info(f"Epoch {epoch:d}, global step {step:d}: {self.monitor!r} was not in top {self.save_top_k}")

    def _save_last_checkpoint(self, trainer: Any, monitor_candidates: Dict[str, Tensor]) -> None:
        if not self.save_last:
            return
        current = monitor_candidates.get(self.monitor)
        self.last_model_score = current
        filepath = self.format_checkpoint_name(monitor_candidates, self.CHECKPOINT_NAME_LAST)

        if self._enable_version_counter:
            version_cnt = self.STARTING_VERSION
            while self.file_exists(filepath, trainer) and filepath != self.last_model_path:
                filepath = self.format_checkpoint_name(monitor_candidates, self.CHECKPOINT_NAME_LAST, ver=version_cnt)
                version_cnt += 1

        # set the last model path before saving because it will be part of the state.
        previous, self.last_model_path = self.last_model_path, filepath
        if self._fs.protocol == "file" and self._last_checkpoint_saved and self.save_top_k != 0:
            self._link_checkpoint(trainer, self._last_checkpoint_saved, filepath)
        else:
            self._save_checkpoint(trainer, filepath)
        if previous and self._should_remove_checkpoint(trainer, previous, filepath):
            self._remove_checkpoint(trainer, previous)

    def _save_none_monitor_checkpoint(self, trainer: Any, monitor_candidates: Dict[str, Tensor]) -> None:
        filepath = self._get_metric_interpolated_filepath_name(monitor_candidates, trainer, self.best_model_path)
        # set the best model path before saving because it will be part of the state.
        previous, self.best_model_path = self.best_model_path, filepath
        self._save_checkpoint(trainer, filepath)

        if self.save_top_k == 1 and previous and self._should_remove_checkpoint(trainer, previous, filepath):
            self._remove_checkpoint(trainer, previous)

    def _update_best_and_save(self, current: Tensor, trainer: Any, monitor_candidates: Dict[str, Tensor]) -> None:
        k = len(self.best_k_models) + 1 if self.save_top_k == -1 else self.save_top_k

        del_filepath = None
        if len(self.best_k_models) == k and k > 0:
            del_filepath = self.kth_best_model_path
            self.best_k_models.pop(del_filepath)

        # do not save nan, replace with +/- inf
        if isinstance(current, Tensor) and torch.isnan(current):
            current = torch.tensor(float("inf" if self.mode == "min" else "-inf"), device=current.device)

        filepath = self._get_metric_interpolated_filepath_name(monitor_candidates, trainer, del_filepath)

        # save the current score
        self.current_score = current
        self.best_k_models[filepath] = current

        if len(self.best_k_models) == k:
            # monitor dict has reached k elements
            _op = max if self.mode == "min" else min
            self.kth_best_model_path = _op(self.best_k_models, key=self.best_k_models.get)  # type: ignore[arg-type]
            self.kth_value = self.best_k_models[self.kth_best_model_path]

        _op = min if self.mode == "min" else max
        self.best_model_path = _op(self.best_k_models, key=self.best_k_models.get)  # type: ignore[arg-type]
        self.best_model_score = self.best_k_models[self.best_model_path]

        if self.verbose:
            epoch = monitor_candidates["epoch"]
            step = monitor_candidates["step"]
            rank_zero_info(
                f"Epoch {epoch:d}, global step {step:d}: {self.monitor!r} reached {current:0.5f}"
                f" (best {self.best_model_score:0.5f}), saving model to {filepath!r} as top {k}"
            )
        self._save_checkpoint(trainer, filepath)

        if del_filepath and self._should_remove_checkpoint(trainer, del_filepath, filepath):
            self._remove_checkpoint(trainer, del_filepath)

    def check_monitor_top_k(self, trainer: Any, current: Optional[Tensor] = None) -> bool:
        fabric = trainer.fabric
        if current is None:
            return False

        if self.save_top_k == -1:
            return True

        less_than_k_models = len(self.best_k_models) < self.save_top_k
        if less_than_k_models:
            return True

        monitor_op = {"min": torch.lt, "max": torch.gt}[self.mode]
        should_update_best_and_save = monitor_op(current, self.best_k_models[self.kth_best_model_path])

        # If using multiple devices, make sure all processes are unanimous on the decision.
        should_update_best_and_save = fabric.strategy.reduce_boolean_decision(bool(should_update_best_and_save))

        return should_update_best_and_save

    def _get_metric_interpolated_filepath_name(
        self, monitor_candidates: Dict[str, Tensor], trainer: Any, del_filepath: Optional[str] = None
    ) -> str:
        filepath = self.format_checkpoint_name(monitor_candidates)

        if self._enable_version_counter:
            version_cnt = self.STARTING_VERSION
            while self.file_exists(filepath, trainer) and filepath != del_filepath:
                filepath = self.format_checkpoint_name(monitor_candidates, ver=version_cnt)
                version_cnt += 1

        return filepath

    def __warn_if_dir_not_empty(self, dirpath: PathStr) -> None:
        if self.save_top_k != 0 and _is_dir(self._fs, dirpath, strict=True) and len(self._fs.ls(dirpath)) > 0:
            rank_zero_warn(f"Checkpoint directory {dirpath} exists and is not empty.")

    def file_exists(self, filepath: PathStr, trainer: Any) -> bool:
        """Checks if a file exists on rank 0 and broadcasts the result to all other ranks, preventing the internal
        state to diverge between ranks."""
        exists = self._fs.exists(filepath)
        return trainer.fabric.strategy.broadcast(exists)

    def _monitor_candidates(self, trainer: Any) -> Dict[str, Tensor]:
        ces = trainer.get_current_epoch_summary()
        candidates = ces.get_summary("scalars")
        monitor_candidates = copy.deepcopy(candidates)
        # cast to int if necessary because `self.log("epoch", 123)` will convert it to float. if it's not a tensor
        # or does not exist we overwrite it as it's likely an error
        epoch = monitor_candidates.get("epoch")
        monitor_candidates["epoch"] = epoch.int() if isinstance(epoch, Tensor) else torch.tensor(trainer.current_epoch)
        step = monitor_candidates.get("step")
        monitor_candidates["step"] = step.int() if isinstance(step, Tensor) else torch.tensor(trainer.global_step)
        return monitor_candidates

    @property
    def state_key(self) -> str:
        return self._generate_state_key(
            monitor=self.monitor,
            mode=self.mode,
            every_n_train_steps=self._every_n_train_steps,
            every_n_epochs=self._every_n_epochs,
            train_time_interval=self._train_time_interval,
        )

    @override
    def state_dict(self) -> Dict[str, Any]:
        return {
            "monitor": self.monitor,
            "best_model_score": self.best_model_score,
            "best_model_path": self.best_model_path,
            "current_score": self.current_score,
            "dirpath": self.dirpath,
            "best_k_models": self.best_k_models,
            "kth_best_model_path": self.kth_best_model_path,
            "kth_value": self.kth_value,
            "last_model_path": self.last_model_path,
            "last_model_score": self.last_model_score
        }

    def __init_triggers(
        self,
        every_n_train_steps: Optional[int],
        every_n_epochs: Optional[int],
        train_time_interval: Optional[timedelta],
    ) -> None:
        # Default to running once after each validation epoch if neither
        # every_n_train_steps nor every_n_epochs is set
        if every_n_train_steps is None and every_n_epochs is None and train_time_interval is None:
            every_n_epochs = 1
            every_n_train_steps = 0
            log.debug("Both every_n_train_steps and every_n_epochs are not set. Setting every_n_epochs=1")
        else:
            every_n_epochs = every_n_epochs or 0
            every_n_train_steps = every_n_train_steps or 0

        self._train_time_interval: Optional[timedelta] = train_time_interval
        self._every_n_epochs: int = every_n_epochs
        self._every_n_train_steps: int = every_n_train_steps

    def __init_monitor_mode(self, mode: str) -> None:
        torch_inf = torch.tensor(torch.inf)
        mode_dict = {"min": (torch_inf, "min"), "max": (-torch_inf, "max")}

        if mode not in mode_dict:
            raise ValueError(f"`mode` can be {', '.join(mode_dict.keys())} but got {mode}")

        self.kth_value, self.mode = mode_dict[mode]

    def __init_ckpt_dir(self, dirpath: Optional[PathStr], filename: Optional[str]) -> None:
        self._fs = get_filesystem(dirpath if dirpath else "")

        if dirpath and _is_local_file_protocol(dirpath if dirpath else ""):
            dirpath = os.path.realpath(os.path.expanduser(dirpath))

        self.dirpath = dirpath
        self.filename = filename

    def _get_file_path(self, epoch, logs):
        """Returns the file path for checkpoint."""
        try:
            # `filepath` may contain placeholders such as `{epoch:02d}` and
            # `{mape:.2f}`. A mismatch between logged metrics and the path's
            # placeholders can cause formatting to fail.
            file_path = self.filepath.format(epoch=epoch + 1, **logs)
        except KeyError as e:
            raise KeyError('Failed to format this callback filepath: "{}". ' "Reason: {}".format(self.filepath, e))
        return file_path

    def __validate_init_configuration(self) -> None:
        if self.save_top_k < -1:
            raise ValueError(f"Invalid value for save_top_k={self.save_top_k}. Must be >= -1")

        if self._every_n_train_steps < 0:
            raise ValueError(f"Invalid value for every_n_train_steps={self._every_n_train_steps}. Must be >= 0")
        if self._every_n_epochs < 0:
            raise ValueError(f"Invalid value for every_n_epochs={self._every_n_epochs}. Must be >= 0")

        every_n_train_steps_triggered = self._every_n_train_steps >= 1
        every_n_epochs_triggered = self._every_n_epochs >= 1
        train_time_interval_triggered = self._train_time_interval is not None
        if every_n_train_steps_triggered + every_n_epochs_triggered + train_time_interval_triggered > 1:
            raise ValueError(
                f"Combination of parameters every_n_train_steps={self._every_n_train_steps}, "
                f"every_n_epochs={self._every_n_epochs} and train_time_interval={self._train_time_interval} "
                "should be mutually exclusive."
            )

        if self.monitor is None and self.save_top_k not in (-1, 0, 1):
            # -1: save all epochs, 0: nothing is saved, 1: save last epoch
            raise ValueError(
                f"ModelCheckpoint(save_top_k={self.save_top_k}, monitor=None) is not a valid"
                " configuration. No quantity for top_k to track."
            )

    def _should_skip_saving_checkpoint(self, trainer) -> bool:
        return self._last_global_step_saved == trainer.global_step  # already saved at the last step

    def _should_save_on_train_epoch_end(self, trainer: Any) -> bool:
        if self._save_on_train_epoch_end is not None:
            return self._save_on_train_epoch_end

        # if `check_val_every_n_epoch != 1`, we can't say when the validation dataloader will be loaded
        # so let's not enforce saving at every training epoch end
        if trainer.check_val_every_n_epoch != 1:
            return False

        # no validation means save on train epoch end
        if trainer._val_loader is None:
            return True

        # by default save after val epoch according to val loss or metrics
        return False

    def format_checkpoint_name(
        self, metrics: Dict[str, Tensor], filename: Optional[str] = None, ver: Optional[int] = None
    ) -> str:
        """Generate a filename according to the defined template.

        Example::

            >>> tmpdir = os.path.dirname(__file__)
            >>> ckpt = ModelCheckpoint(dirpath=tmpdir, filename='{epoch}')
            >>> os.path.basename(ckpt.format_checkpoint_name(dict(epoch=0)))
            'epoch=0.ckpt'
            >>> ckpt = ModelCheckpoint(dirpath=tmpdir, filename='{epoch:03d}')
            >>> os.path.basename(ckpt.format_checkpoint_name(dict(epoch=5)))
            'epoch=005.ckpt'
            >>> ckpt = ModelCheckpoint(dirpath=tmpdir, filename='{epoch}-{val_loss:.2f}')
            >>> os.path.basename(ckpt.format_checkpoint_name(dict(epoch=2, val_loss=0.123456)))
            'epoch=2-val_loss=0.12.ckpt'
            >>> os.path.basename(ckpt.format_checkpoint_name(dict(epoch=2, val_loss=0.12), filename='{epoch:d}'))
            'epoch=2.ckpt'
            >>> ckpt = ModelCheckpoint(dirpath=tmpdir,
            ... filename='epoch={epoch}-validation_loss={val_loss:.2f}',
            ... auto_insert_metric_name=False)
            >>> os.path.basename(ckpt.format_checkpoint_name(dict(epoch=2, val_loss=0.123456)))
            'epoch=2-validation_loss=0.12.ckpt'
            >>> ckpt = ModelCheckpoint(dirpath=tmpdir, filename='{missing:d}')
            >>> os.path.basename(ckpt.format_checkpoint_name({}))
            'missing=0.ckpt'
            >>> ckpt = ModelCheckpoint(filename='{step}')
            >>> os.path.basename(ckpt.format_checkpoint_name(dict(step=0)))
            'step=0.ckpt'

        """
        filename = filename or self.filename
        filename = self._format_checkpoint_name(filename, metrics, auto_insert_metric_name=self.auto_insert_metric_name)

        if ver is not None:
            filename = self.CHECKPOINT_JOIN_CHAR.join((filename, f"v{ver}"))

        ckpt_name = f"{filename}{self.FILE_EXTENSION}"
        return os.path.join(self.dirpath, ckpt_name) if self.dirpath else ckpt_name

    def _format_checkpoint_name(
        self,
        filename: Optional[str],
        metrics: Dict[str, Tensor],
        prefix: str = "",
        auto_insert_metric_name: bool = True,
    ) -> str:
        if not filename:
            # filename is not set, use default name
            filename = "{epoch}" + self.CHECKPOINT_JOIN_CHAR + "{step}"

        # check and parse user passed keys in the string
        groups = re.findall(r"(\{.*?)[:\}]", filename)

        # sort keys from longest to shortest to avoid replacing substring
        # eg: if keys are "epoch" and "epoch_test", the latter must be replaced first
        groups = sorted(groups, key=lambda x: len(x), reverse=True)

        for group in groups:
            name = group[1:]

            if auto_insert_metric_name:
                filename = filename.replace(group, name + self.CHECKPOINT_EQUALS_CHAR + "{" + name)

            # support for dots: https://stackoverflow.com/a/7934969
            filename = filename.replace(group, f"{{0[{name}]")

            if name not in metrics:
                metrics[name] = torch.tensor(0)
        filename = filename.format(metrics)

        if prefix:
            filename = self.CHECKPOINT_JOIN_CHAR.join([prefix, filename])

        return filename

    def _should_remove_checkpoint(self, trainer: Any, previous: str, current: str) -> bool:
        """Checks if the previous checkpoint should be deleted.

        A checkpoint won't be deleted if any of the cases apply:
        - The previous checkpoint is the same as the current checkpoint (means the old was already overwritten by new)
        - The previous checkpoint is not in the current checkpoint directory and the filesystem is local
        - The previous checkpoint is the checkpoint the Trainer resumed from and the filesystem is local

        """
        if previous == current:
            return False
        if not _is_local_file_protocol(previous):
            return True
        previous = Path(previous).absolute()
        resume_path = Path(trainer.ckpt_path).absolute() if trainer.ckpt_path is not None else None
        if resume_path is not None and previous == resume_path:
            return False
        assert self.dirpath is not None
        dirpath = Path(self.dirpath).absolute()
        return dirpath in previous.parents

    def _remove_checkpoint(self, trainer: Any, filepath: str) -> None:
        """Calls the strategy to remove the checkpoint file."""
        trainer.remove_checkpoint(filepath)

    @staticmethod
    def _link_checkpoint(trainer: Any, filepath: str, linkpath: str) -> None:
        fabric = trainer.fabric
        if fabric.is_global_zero:
            if os.path.islink(linkpath) or os.path.isfile(linkpath):
                os.remove(linkpath)
            elif os.path.isdir(linkpath):
                shutil.rmtree(linkpath)
            try:
                os.symlink(filepath, linkpath)
            except OSError:
                # on Windows, special permissions are required to create symbolic links as a regular user
                # fall back to copying the file
                shutil.copy(filepath, linkpath)
        fabric.strategy.barrier()

    def __resolve_ckpt_dir(self, trainer: Any) -> PathStr:
        """Determines model checkpoint save directory at runtime. Reference attributes from the trainer's logger to
        determine where to save checkpoints. The path for saving weights is set in this priority:

        1.  The ``ModelCheckpoint``'s ``dirpath`` if passed in
        2.  The ``Logger``'s ``log_dir`` if the trainer has loggers
        3.  The ``Trainer``'s ``default_root_dir`` if the trainer has no loggers

        The path gets extended with subdirectory "checkpoints".

        """
        if self.dirpath is not None:
            # short circuit if dirpath was passed to ModelCheckpoint
            return self.dirpath

        fabric = trainer.fabric

        if len(fabric.loggers) > 0:
            if fabric.loggers[0].save_dir is not None:
                save_dir = fabric.loggers[0].save_dir
            else:
                save_dir = trainer.default_root_dir
            name = fabric.loggers[0].name
            version = fabric.loggers[0].version
            version = version if isinstance(version, str) else f"version_{version}"
            ckpt_path = os.path.join(save_dir, str(name), version, "checkpoints")
        else:
            # if no loggers, use default_root_dir
            ckpt_path = os.path.join(trainer.default_root_dir, "checkpoints")

        return ckpt_path

    def to_yaml(self, filepath: Optional[PathStr] = None) -> None:
        """Saves the `best_k_models` dict containing the checkpoint paths with the corresponding scores to a YAML
        file."""
        best_k = {k: v.item() for k, v in self.best_k_models.items()}
        if filepath is None:
            assert self.dirpath
            filepath = os.path.join(self.dirpath, "best_k_models.yaml")
        with self._fs.open(filepath, "w") as fp:
            yaml.dump(best_k, fp)

    def __repr__(self):
        desc = f"{self.__class__.__name__}(monitor={self.monitor})"
        return desc
