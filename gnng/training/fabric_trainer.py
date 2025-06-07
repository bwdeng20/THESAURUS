import os
import re
import time
from dataclasses import dataclass
from collections.abc import Mapping, MutableMapping
from pathlib import Path
from typing import Any, Iterable, List, Literal, Optional, Union, cast, Callable, Sequence, Dict

from termcolor import colored
import lightning as L
import torch
from lightning.fabric.accelerators import Accelerator
from lightning.fabric.loggers import Logger
from lightning.fabric.strategies import Strategy
from lightning.fabric.utilities.rank_zero import rank_zero_warn
from lightning_utilities import apply_to_collection
from tqdm import tqdm
import warnings
from torchmetrics import MetricCollection, Metric
from gnng.nn.resolver_initializer import (
    optimizer_initializer,
    scheduler_initializer,
    metric_initializer,
    loss_initializer,
)
from gnng.typing import ParsableMetric, PathStr, ParsableOptimizer, ParsableScheduler, ParsableLoss
from gnng.utils import update_dict_no_collision
from gnng.utils import get_filesystem, normalize_string
from gnng.training.steps import configure_step
from gnng.training.metric_fns import naive_epoch_metric_fn
from gnng.training.loss_recorder import LossRecorder
from gnng.training.bo_recorder import BatchOutputRecorder
from gnng.training.summary_manager import SummaryManager
from lightning_utilities.core.enums import StrEnum


class TrainerStatus(StrEnum):
    INITIALIZING = "initializing"  # trainer creation
    TRAINING = "training"
    VALIDATING = "validating"
    TESTING = "testing"
    PREDICTING = "predicting"
    FINISHED = "finished"


@dataclass
class FabricTrainerConfig:
    accelerator: Union[str, Accelerator] = "auto"
    strategy: Union[str, Strategy] = "auto"
    devices: Union[List[int], str, int] = "auto"
    precision: Union[str, int] = "32-true"
    plugins: Optional[Union[str, Any]] = None
    callbacks: Optional[Union[List[Any], Any]] = None
    loggers: Optional[Union[Logger, List[Logger]]] = None
    max_epochs: Optional[int] = 500
    max_steps: Optional[int] = None
    grad_accum_steps: int = 1
    limit_train_batches: Union[int, float] = float("inf")
    limit_val_batches: Union[int, float] = float("inf")
    limit_test_batches: Union[int, float] = float("inf")
    check_val_every_n_epoch: int = 1
    use_distributed_sampler: bool = False
    default_root_dir: Optional[PathStr] = None
    checkpoint_frequency: int = 1
    ckpt_verbose: bool = False
    batch_out_on_cpu: bool = False


class FabricTrainer:
    RootSubDirStem = "version"

    def __init__(
            self,
            *,
            accelerator: Union[str, Accelerator] = "auto",
            strategy: Union[str, Strategy] = "auto",
            devices: Union[List[int], str, int] = "auto",
            precision: Union[str, int] = "32-true",
            plugins: Optional[Union[str, Any]] = None,
            callbacks: Optional[Union[List[Any], Any]] = None,
            loggers: Optional[Union[Logger, List[Logger]]] = None,
            max_epochs: Optional[int] = 500,
            max_steps: Optional[int] = None,
            grad_accum_steps: int = 1,
            limit_train_batches: Union[int, float] = float("inf"),
            limit_val_batches: Union[int, float] = float("inf"),
            limit_test_batches: Union[int, float] = float("inf"),
            check_val_every_n_epoch: int = 1,
            use_distributed_sampler: bool = False,
            default_root_dir: Optional[PathStr] = None,
            checkpoint_frequency: int = 1,
            ckpt_verbose: bool = False,
            batch_out_on_cpu: bool = False,
    ) -> None:
        """Exemplary Trainer with Fabric. This is a very simple trainer focused on readablity but with reduced
        featureset. As a trainer with more included features, we recommend using the
        :class:`lightning.pytorch.Trainer`.

        Args:
            accelerator: The hardware to run on. Possible choices are:
                ``"cpu"``, ``"cuda"``, ``"mps"``, ``"plain.yaml"``, ``"tpu"``, ``"auto"``.
            strategy: Strategy for how to run across multiple devices. Possible choices are:
                ``"dp"``, ``"ddp"``, ``"ddp_spawn"``, ``"deepspeed"``, ``"fsdp"``.
            devices: Number of devices to train on (``int``),
                which GPUs to train on (``list`` or ``str``), or ``"auto"``.
                The value applies per node.
            precision: Double precision (``"64"``), full precision (``"32"``), half precision AMP (``"16-mixed"``),
                or bfloat16 precision AMP (``"bf16-mixed"``).
            plugins: One or several custom plugins
            callbacks: A single callback or a list of callbacks. The following hooks are supported:
                - setup
                - on_train_start
                - on_train_epoch_start
                - on train_epoch_end
                - on_train_batch_start
                - on_train_batch_end
                - on_before_backward
                - on_after_backward
                - on_before_zero_grad
                - on_before_optimizer_step
                - on_train_end
                - on_validation_model_eval
                - on_validation_model_train
                - on_validation_epoch_start
                - on_validation_epoch_end
                - on_validation_batch_start
                - on_validation_batch_end
                - on_test_batch_start: (batch: Any, batch_idx: int)
                - on_test_batch_end: (outputs: STEP_OUTPUT, batch: Any, batch_idx)
                - on_test_model_train:Sets the model to train during the test  loop.
                - on_test_model_eval: Sets the model to eval during the test loop
                - on_test_epoch_start
                - on_test_epoch_end


            loggers: A single logger or a list of loggers. See :meth:`~lightning.fabric.fabric.Fabric.log` for more
                information.

            max_epochs: The maximum number of epochs to train
            max_steps: The maximum number of (optimizer) losses to train
            grad_accum_steps: How many batches to process before each optimizer step
            limit_train_batches: Limits the number of train batches per epoch
                If greater than number of batches in the dataloader, this has no effect.
            limit_val_batches: Limits the number of validation batches per epoch.
                If greater than number of batches in the dataloader, this has no effect.
            check_val_every_n_epoch: How many epochs to run before each validation epoch.
            use_distributed_sampler: Wraps the sampler of each dataloader with a respective distributed-aware sampler
                in case of distributed training.
            default_root_dir: Directory to store logs and checkpoints to.
            checkpoint_frequency: How many epochs to run before each checkpoint is written.

        Warning:
            callbacks written for the lightning trainer (especially making assumptions on the trainer), won't work!

        """
        self.state = TrainerStatus.INITIALIZING

        self.fabric = L.Fabric(
            accelerator=accelerator,
            strategy=strategy,
            devices=devices,
            precision=precision,
            plugins=plugins,
            callbacks=callbacks,
            loggers=loggers,
        )

        self.global_step = 0
        self.global_valid_step = 0
        self.global_test_step = 0
        self.grad_accum_steps: int = grad_accum_steps
        self.current_epoch = 0
        self.num_train_epochs = 0
        self.num_valid_epochs = 0
        self.num_test_epochs = 0

        self.max_epochs = max_epochs
        self.max_steps = max_steps
        self.should_stop = False

        # ensures limit_X_batches is either int or inf
        if not isinstance(limit_train_batches, int):
            assert limit_train_batches == float("inf")

        if not isinstance(limit_val_batches, int):
            assert limit_val_batches == float("inf")

        self.ckpt_verbose = ckpt_verbose
        self.limit_train_batches = limit_train_batches
        self.limit_val_batches = limit_val_batches
        self.limit_test_batches = limit_test_batches
        self.check_val_every_n_epoch = check_val_every_n_epoch
        self.use_distributed_sampler = use_distributed_sampler
        self.default_root_dir = self.parse_default_root_dir(default_root_dir)
        self.checkpoint_dir = self.default_root_dir / "checkpoints"
        self.checkpoint_frequency = checkpoint_frequency  # naive built-in checkpointing trigger if not None

        self.setup_loggers()
        self.setup_cbs()

        self._model = None
        self._batch_return_dict = {"train": {}, "valid": {}, "test": {}}  # batch-level output cache
        self.current_fit_epoch_summary = None  # train and val, epoch-level summary, including train/val metrics
        self.current_test_epoch_summary = None  # test,          epoch-level summary, including test metrics

        # Register when invoking .fit().
        # Unregister when .fit() is finished.
        # The concrete step implementation for specified model or data formats
        self._train_step_fn = None
        self._eval_step_fn = None
        self._test_step_fn = None  # may test with different logic from `_eval_step_fn`
        self._train_epoch_metric_fn = None
        self._eval_epoch_metric_fn = None
        self._test_epoch_metric_fn = None  # may test with different logic from `_eval_epoch_metric_fn`
        self._loss_fn = None
        self._metric_manager_dict: Optional[Dict[str, Metric]] = None
        self._metric_name: Optional[List[str]] = None
        self._train_loader = None
        self._val_loader = None
        self._test_loader = None
        self._current_state = None

        self._loss_manager_dict = {  # to manage all batch loss within one epoch
            "train_ls": LossRecorder(prefix="train/", fabric_instance=self.fabric),
            "valid_ls": LossRecorder(prefix="valid/", fabric_instance=self.fabric),
            "test_ls": LossRecorder(prefix="test/", fabric_instance=self.fabric),
        }

        self._bo_manager_dict = {  # to manage all batch output within one epoch
            "train_bo": BatchOutputRecorder(
                prefix="train/", fabric_instance=self.fabric, store_on_cpu=batch_out_on_cpu
            ),
            "valid_bo": BatchOutputRecorder(
                prefix="valid/", fabric_instance=self.fabric, store_on_cpu=batch_out_on_cpu
            ),
            "test_bo": BatchOutputRecorder(prefix="test/", fabric_instance=self.fabric, store_on_cpu=batch_out_on_cpu),
        }

        # Fixed Dummy properties defined by `pl.Trainer` and used by `pl.Callback` but not configurable for our Trainer
        self.fast_dev_run = False
        self.val_check_interval = 1.0
        self.sanity_checking = False

    def setup_cbs(self):
        queried_cb = self.get_callback_named(self.fabric, "ModelCheckpoint")
        if "modelcheckpoint" in queried_cb:
            self.checkpoint_frequency = None  # disable naive per epoch checkpointing
            # self.dirpath = self.checkpoint_dir # Force the callback use

    def setup_loggers(self):
        pass

    def parse_default_root_dir(self, default_root_dir=None):
        root_dir = default_root_dir or (Path(os.getcwd()) / "fabric_logs")
        root_dir = Path(root_dir)
        if root_dir.exists():
            assert root_dir.is_dir(), "root_dir exists but is not a directory."
            last_run_dir = self.get_latest_child(root_dir)
            last_run_info = re.findall(pattern=r"(\D+)_(\d+)", string=str(last_run_dir))  # {stem}_{V}
            if last_run_info:  # the last run_dir is successfully parsed
                run_dir_stem, run_dir_v = last_run_info[0]
            else:
                raise FileExistsError(
                    f"The detected last run subdirectory `{last_run_dir}` does not follow the "
                    f"run dir naming rule like `{self.RootSubDirStem}_#`. You can rename or delete"
                    f"if unimportant this file manually to get a consistent logging file system."
                )
            cur_run_v = int(run_dir_v) + 1
            cur_run_dir = root_dir / f"{self.RootSubDirStem}_{cur_run_v}"
        else:
            cur_run_dir = root_dir / f"{self.RootSubDirStem}_0"
        return cur_run_dir

    def get_metric_manager(self, stage):
        return self._metric_manager_dict[f"{stage}_mc"]

    def get_loss_manager(self, stage):
        return self._loss_manager_dict[f"{stage}_ls"]

    def get_bo_manager(self, stage):
        return self._bo_manager_dict[f"{stage}_bo"]

    @property
    def metric_name(self) -> List[str]:
        return self._metric_name

    def test(
            self,
            model: Optional[torch.nn.Module] = None,
            dataloader: Optional["torch.utils.data.DataLoader"] = None,
            ckpt_path: Optional[str] = None,
            metric: ParsableMetric = None,
            metric_kwargs: Optional[Dict[str, Any]] = None,
            test_step_fn: Callable = None,
            test_epoch_metric_fn: Callable = None,
            tqdm_frequency: Optional[int] = 1,  # per 1 epoch show tqdm batch-info bar
            test_step_fn_kwargs: Optional[Dict[str, Any]] = None,
            test_epoch_metric_fn_kwargs: Optional[Dict[str, Any]] = None,
            force_log_epoch: Optional[int] = None,
            auto_log: bool = False,
            force_no_dl_wrapper: bool = False,
            record_speed: bool = True
    ):
        assert dataloader is not None, "You must specify a dataloader to test the model."
        test_step_fn = test_step_fn or self._test_step_fn
        test_step_fn = test_step_fn or self._eval_step_fn
        test_step_fn_kwargs = test_step_fn_kwargs or {}

        test_epoch_metric_fn = test_epoch_metric_fn or self._test_epoch_metric_fn
        test_epoch_metric_fn = test_epoch_metric_fn or self._eval_epoch_metric_fn  # use that for valid
        test_epoch_metric_fn = test_epoch_metric_fn or naive_epoch_metric_fn

        self.current_test_epoch_summary = SummaryManager({"ckpt_path": ckpt_path})
        model = self.fabric.setup(model) if model is not None else self._model
        if not force_no_dl_wrapper:
            dataloader = self.fabric.setup_dataloaders(
                dataloader,
                use_distributed_sampler=self.use_distributed_sampler,
            )
        else:
            dataloader = dataloader

        if ckpt_path is not None:
            state = {"model": model}  # only model weights are needed for testing
            self.load(state, ckpt_path, strict=False, weights_only=True, load_warning=False)

        if metric is not None:  # if different metrics are considered for test stage
            metric = metric_initializer(metric, metric_kwargs)
            if not isinstance(metric, MetricCollection):  # wrapped with Collection
                metric = MetricCollection(metric)

            if  self._metric_manager_dict is None:
                self._metric_name = list(metric.keys())
                if self.fabric.is_global_zero:
                    self._metric_manager_dict = torch.nn.ModuleDict(
                        {
                            "test_mc": metric.clone(prefix="test/"),
                        }
                    )
                self._metric_manager_dict = self._metric_manager_dict.to(self.fabric.device)
            else:
                self._metric_manager_dict["test_mc"] = metric  # override the test metric manager on-the-fly

        tic = time.time()
        test_out = self.test_loop(
            model,
            dataloader,
            test_step_fn,
            test_epoch_metric_fn,
            self.limit_test_batches,
            tqdm_frequency,
            test_step_fn_kwargs,
            force_log_epoch,
            test_epoch_metric_fn_kwargs=test_epoch_metric_fn_kwargs,
            auto_log=auto_log,
        )
        if record_speed:
            torch.cuda.synchronize()
            toc= time.time()
            test_time = toc - tic
            test_out["test/time"] = test_time

        if self.fabric.is_global_zero:
            info_str = self.info_dict2fmt_str(test_out.get_summary("scalars"))
            info = f"[Epoch {self.current_epoch:4d}/{self.max_epochs - 1:4d}]{info_str}"
            self.fabric.print(colored(info, "green", attrs=["underline"]))
        self.state = TrainerStatus.FINISHED
        return test_out

    def fit(
            self,
            model: "torch.nn.Module",
            train_loader: "torch.utils.data.DataLoader",
            val_loader: Optional["torch.utils.data.DataLoader"] = None,
            optimizer: Optional[ParsableOptimizer] = None,
            optimizer_kwargs: Optional[Dict[str, Any]] = None,
            scheduler: Optional[ParsableScheduler] = None,
            scheduler_kwargs: Optional[Dict[str, Any]] = None,
            pl_scheduler_conf_kwargs: Optional[Dict[str, Any]] = None,
            criterion: Optional[ParsableLoss] = "CrossEntropy",
            criterion_kwargs: Optional[Dict[str, Any]] = None,
            metric: ParsableMetric = "Accuracy",
            metric_kwargs: Optional[Dict[str, Any]] = None,
            train_step_fn: Optional[Callable] = None,
            eval_step_fn: Optional[Callable] = None,  # by default, val/test use the same fn, so no `test_step_fn`
            train_epoch_metric_fn: Optional[Callable] = None,
            eval_epoch_metric_fn: Optional[Callable] = None,  # by default, val/test use the same fn, so no `test_epoch_metric_fn`
            ckpt_path: Optional[str] = None,
            verbose_frequency: Optional[int] = None,
            tqdm_frequency: Optional[int] = 1,
            train_step_fn_kwargs: Optional[Dict[str, Any]] = None,
            val_step_fn_kwargs: Optional[Dict[str, Any]] = None,
            train_epoch_metric_fn_kwargs: Optional[Dict[str, Any]] = None,
            val_epoch_metric_fn_kwargs: Optional[Dict[str, Any]] = None,
            force_no_dl_wrapper: bool = False,
            auto_log: bool = True,
    ):
        """
        Trains the model using the provided data loader and configuration.

        Parameters
        ----------
        model : torch.nn.Module
            The neural network model to be trained.
        train_loader : torch.utils.data.DataLoader
            DataLoader for training data.
        val_loader : Optional[torch.utils.data.DataLoader], optional
            DataLoader for validation data. Default is None.
        optimizer : Optional[ParsableOptimizer], optional
            Optimizer instance or configuration. Default is None.
        optimizer_kwargs : Optional[Dict[str, Any]], optional
            Additional arguments for the optimizer. Default is None.
        scheduler : Optional[ParsableScheduler], optional
            Learning rate scheduler instance or configuration. Default is None.
        scheduler_kwargs : Optional[Dict[str, Any]], optional
            Additional arguments for the scheduler. Default is None.
        pl_scheduler_conf_kwargs : Optional[Dict[str, Any]], optional
            Configuration for the PyTorch Lightning scheduler. Default is None.
        criterion : Optional[ParsableLoss], optional
            Loss function or configuration. Default is "CrossEntropy".
        criterion_kwargs : Optional[Dict[str, Any]], optional
            Additional arguments for the loss function. Default is None.
        metric : ParsableMetric, optional
            Metric for evaluating model performance. Default is "Accuracy".
        metric_kwargs : Optional[Dict[str, Any]], optional
            Additional arguments for the metric. Default is None.
        train_step_fn : Callable, optional
            Custom function for processing each training step. Default is None. YOU DO IT.
        eval_step_fn : Callable, optional
            Custom function for processing each evaluation step. Default is None. YOU DO IT.
        train_epoch_metric_fn : Callable, optional
            Custom function for processing metrics at the end of each training epoch. Default is None.
        eval_epoch_metric_fn : Callable, optional
            Custom function for processing metrics at the end of each evaluation epoch. Default is None.
        ckpt_path : Optional[str], optional
            Path to save model checkpoints. Default is None.
        verbose_frequency : Optional[int], optional
            Frequency (in epochs) to print verbose information. Default is None.
        tqdm_frequency : Optional[int], optional
            Frequency (in steps) to update the progress bar. Default is 1.
        train_step_fn_kwargs : Optional[Dict[str, Any]], optional
            Additional arguments for the training step function. Default is None.
        val_step_fn_kwargs : Optional[Dict[str, Any]], optional
            Additional arguments for the validation step function. Default is None.
        train_epoch_metric_fn_kwargs : Optional[Dict[str, Any]], optional
            Additional arguments for the training epoch metric function. Default is None.
        val_epoch_metric_fn_kwargs : Optional[Dict[str, Any]], optional
            Additional arguments for the validation epoch metric function. Default is None.
        force_no_dl_wrapper : bool, optional
            If True, disables the dataloader wrapper provided by the fabric. Default is False.
        auto_log : bool, optional
            If True, automatically logs information during training. Default is True.

        Returns
        -------
        None
            This function does not return anything, but it triggers the training process.
        """
        """The main entrypoint of the trainer, triggering the actual training. *args and **kwargs are for step_fn"""
        # =============================  Fabric Setup ====================================================
        # register for every fit()
        self._train_epoch_metric_fn = train_epoch_metric_fn or naive_epoch_metric_fn
        self._eval_epoch_metric_fn = eval_epoch_metric_fn or self._train_epoch_metric_fn
        self._train_step_fn = train_step_fn or configure_step(model)
        self._eval_step_fn = eval_step_fn or self._train_step_fn  # in general, the step logic is the same
        self._loss_fn = loss_initializer(criterion, criterion_kwargs)
        if isinstance(metric, str) and metric == "Accuracy":
            metric_kwargs = metric_kwargs or {"task": "multiclass", "num_classes": model.out_channels}
        metric = metric_initializer(metric, metric_kwargs)
        if not isinstance(metric, MetricCollection):  # wrapped with Collection
            metric = MetricCollection(metric)
        self._metric_name = list(metric.keys())
        if self.fabric.is_global_zero:
            self._metric_manager_dict = torch.nn.ModuleDict(
                {
                    "train_mc": metric.clone(prefix="train/"),
                    "valid_mc": metric.clone(prefix="valid/"),
                    "test_mc": metric.clone(prefix="test/"),
                }
            )

        if self.use_distributed_sampler:
            self.fabric.launch()
        self._metric_manager_dict = self._metric_manager_dict.to(self.fabric.device)

        # # setup dataloaders
        if not force_no_dl_wrapper:
            train_loader = self.fabric.setup_dataloaders(
                train_loader,
                use_distributed_sampler=self.use_distributed_sampler,
            )
        self._train_loader = train_loader

        if val_loader is not None:
            if not force_no_dl_wrapper:
                val_loader = self.fabric.setup_dataloaders(
                    val_loader,
                    use_distributed_sampler=self.use_distributed_sampler,
                )

        self._val_loader = val_loader

        # configure optimizer, model, and scheduler
        optimizer = optimizer_initializer(optimizer, optimizer_kwargs, model)
        pl_scheduler_conf_kwargs = pl_scheduler_conf_kwargs or {"interval": "epoch", "frequency": 1, "monitor": None}
        scheduler_cfg = scheduler_initializer(scheduler, scheduler_kwargs, optimizer, pl_scheduler_conf_kwargs)
        assert optimizer is not None
        model, optimizer = self.fabric.setup(model, optimizer)
        self._model = model

        # assemble state (current epoch and global step will be added in save)
        self._current_state = {"model": model, "optim": optimizer, "scheduler": scheduler_cfg}

        # load last checkpoint if available
        if ckpt_path is not None and os.path.isdir(ckpt_path):
            latest_checkpoint_path = self.get_latest_child(self.checkpoint_dir)
            if latest_checkpoint_path is not None:
                self.load(self._current_state, latest_checkpoint_path)

                # check if we even need to train here
                if self.max_epochs is not None and self.current_epoch >= self.max_epochs:
                    self.should_stop = True

        self.fabric.call("setup", self, model, stage="fit")
        # ============================== Fit = Many * (Train Epoch + Val Epoch) =======================================
        self.fabric.call("on_train_start", self.fabric, model)

        while not self.should_stop:  # train (and val) for many epochs
            self.current_fit_epoch_summary = SummaryManager({"epoch": self.current_epoch})  # reinit per epoch

            train_info = self.train_loop(
                model,
                optimizer,
                train_loader,
                limit_batches=self.limit_train_batches,
                scheduler_cfg=scheduler_cfg,
                tqdm_frequency=tqdm_frequency,
                train_step_fn_kwargs=train_step_fn_kwargs,
                train_epoch_metric_fn_kwargs=train_epoch_metric_fn_kwargs,
                auto_log=auto_log,
            )

            if self.should_validate:
                val_info = self.validation_loop(
                    model,
                    val_loader,
                    limit_batches=self.limit_val_batches,
                    tqdm_frequency=tqdm_frequency,
                    val_step_fn_kwargs=val_step_fn_kwargs,
                    val_epoch_metric_fn_kwargs=val_epoch_metric_fn_kwargs,
                    auto_log=auto_log,
                )
            else:
                val_info = {}  # epoch-level metric and loss

            self.fabric.call("on_validation_end", self, model)  # maybe Checkpoint cb hook call

            self.step_scheduler(model, scheduler_cfg, level="epoch", current_value=self.current_epoch)

            # raw checkpoint method
            if self.checkpoint_frequency is not None and self.current_epoch % self.checkpoint_frequency == 0:
                self.save_checkpoint(self.ckpt_path, self._current_state, only_weights=False)
            else:  # callback 'ModelCheckpoint' handles checkpointing in defined process hooks, so do nothing here
                pass

            if verbose_frequency is not None and self.current_epoch % verbose_frequency == 0:
                if self.fabric.is_global_zero:
                    train_info_str = self.info_dict2fmt_str(train_info.get_summary("scalars"))
                    val_info_str = self.info_dict2fmt_str(val_info.get_summary("scalars"))
                    info = (
                        f"[Epoch {self.current_epoch:4d}/{self.max_epochs - 1:4d}] Train and Val Summary\n"
                        f"{train_info_str}\n"
                        f"{val_info_str}"
                    )
                    self.fabric.print(colored(info, "green"))

            self.current_epoch += 1  # increase after all things done and before starting the next epoch
            # stopping condition on epoch level
            if self.max_epochs is not None and self.current_epoch >= self.max_epochs:
                self.should_stop = True

        # reset for next fit call
        self.should_stop = False
        # Should we unregister them automatically post one fit?
        # self._train_step_fn = None
        # self._eval_step_fn = None
        # self._loss_fn = None
        # self._metric_manager_dict= None
        # self._metric_name = None
        self.state = TrainerStatus.FINISHED

    def train_loop(
            self,
            model: Any,
            optimizer: Any,
            train_loader: Any,
            train_step_fn: Optional[Callable] = None,
            train_epoch_metric_fn: Optional[Callable] = None,
            limit_batches: Union[int, float] = float("inf"),
            scheduler_cfg: Optional[Mapping[str, Union[L.fabric.utilities.types.LRScheduler, bool, str, int]]] = None,
            tqdm_frequency: Optional[int] = None,
            train_step_fn_kwargs: Optional[Dict[str, Any]] = None,
            train_epoch_metric_fn_kwargs: Optional[Dict[str, Any]] = None,
            auto_log: bool = False,
    ):
        """The training loop running a single training epoch.

        Args:
            model: the LightningModule to train
            optimizer: the optimizer, optimizing the LightningModule.
            train_loader: The dataloader yielding the training batches.
            limit_batches: Limits the batches during this training epoch.
                If greater than the number of batches in the ``train_loader``, this has no effect.
            scheduler_cfg: The learning rate scheduler configuration.
                Have a look at :meth:`~lightning.pytorch.core.LightningModule.configure_optimizers`
                for supported values.

        """
        self.state = TrainerStatus.TRAINING

        train_step_fn = train_step_fn or self._train_step_fn
        train_step_fn_kwargs = train_step_fn_kwargs or {}
        train_epoch_metric_fn_kwargs = train_epoch_metric_fn_kwargs or {}
        train_epoch_metric_fn = train_epoch_metric_fn or self._train_epoch_metric_fn

        model.train()  # default train mode
        # you can override default train mode in this hook
        self.fabric.call("on_train_epoch_start", self, model, train_loader)

        if tqdm_frequency is not None and self.current_epoch % tqdm_frequency == 0:
            iterable = self.progbar_wrapper(
                train_loader,
                total=min(len(train_loader), limit_batches),
                desc=f"[Epoch {self.current_epoch:4d}|   {'Train':5s}]",
            )
        else:  # tqdm progress
            iterable = enumerate(train_loader)

        metric_manager = self._metric_manager_dict["train_mc"]
        loss_manager = self._loss_manager_dict["train_ls"]
        bo_manager = self._bo_manager_dict["train_bo"]
        metric_manager.reset()
        loss_manager.reset()
        bo_manager.reset()

        for batch_idx, batch_dt in iterable:
            # end epoch if stopping training completely or max batches for this epoch reached
            if self.should_stop or batch_idx >= limit_batches:
                break
            # check if optimizer should step in gradient accumulation
            should_optim_step = self.global_step % self.grad_accum_steps == 0

            self.fabric.call("on_train_batch_start", self, model, batch_dt, batch_idx)

            # Use this "fabric.no_backward_sync" when performing gradient accumulation and using a distributed
            # strategy (e.g., DDP). It will speed up your training loop by cutting redundant communication
            # between processes during the accumulation phase.
            # For single-device strategies, it is a no-op. Some strategies don’t support this.
            with self.fabric.no_backward_sync(model, enabled=not should_optim_step):
                # forward and backward
                out_dict: Mapping[str, Any] = train_step_fn(
                    model,
                    batch_dt,
                    loss_fn=self._loss_fn,
                    metric_manager=metric_manager,
                    loss_manager=loss_manager,
                    batch_out_manager=bo_manager,
                    current_epoch=self.current_epoch,
                    stage="train",
                    fabric=self.fabric,
                    batch_idx=batch_idx,
                    **train_step_fn_kwargs,
                )
                loss = out_dict["loss"]
                self.fabric.call("on_before_backward", self, model, loss)
                self.fabric.backward(loss)
                self.fabric.call("on_after_backward", self, model)

            if should_optim_step:
                # optimizer step
                self.fabric.call("on_before_optimizer_step", self, model, optimizer)
                optimizer.step()
                self.fabric.call("on_before_zero_grad", self, model, optimizer)
                optimizer.zero_grad()

            # avoid gradients in stored/accumulated values -> prevents potential OOM
            batch_return = self._batch_return_dict["train"] = apply_to_collection(
                out_dict, dtype=torch.Tensor, function=lambda x: x.detach()
            )

            self.fabric.call("on_train_batch_end", self, model, batch_return, batch_dt, batch_idx)

            # this guard ensures, we only step the scheduler (at Step/Batch Level) once per global step
            if should_optim_step:
                self.step_scheduler(model, scheduler_cfg, level="step", current_value=self.global_step)

            # add output values to progress bar
            self._format_iterable(iterable, batch_return, "train")

            # only increase global step if optimizer stepped
            self.global_step += int(should_optim_step)

            # stopping criterion on step level
            if self.max_steps is not None and self.global_step >= self.max_steps:
                self.should_stop = True
                break

        # LOGGING loss and metrics at Epoch Level； handle epoch-level intermediate results and artifacts.
        esd = train_epoch_metric_fn(self, model=self._model, stage="train", **train_epoch_metric_fn_kwargs)
        epoch_summary = SummaryManager(esd)
        loss_dict, num_sample_dict = loss_manager.compute()
        epoch_summary.update(loss_dict)
        if auto_log:
            self.log_metrics(epoch_summary, epoch=self.current_epoch)
        update_dict_no_collision(self.get_current_epoch_summary(), epoch_summary)

        self.fabric.call("on_train_epoch_end", self, model)
        self.num_train_epochs += 1
        return epoch_summary

    def log_metrics(
            self,
            metrics: Union[SummaryManager, MutableMapping[str, Any]],
            step: Optional[int] = None,
            epoch: Optional[int] = None,
            strict: bool = False,
    ) -> None:
        """Log multiple scalars, wandb medias, and wandb.Tables at once to all loggers that were added to Fabric.

        Args:
            metrics: A dictionary where the key is the name of the metric and the value to be logged.
                Any :class:`torch.Tensor` in the dictionary get detached from the graph automatically.
            step: Optional step number. Most Logger implementations auto-increment this value by one with every
                log call. You can specify your own value here.
            strict: If True, will raise error if any element of :obj:`data` collection is not a scalar; skip the
                non-scalar element otherwise.

        """
        all_loggers = self.fabric._loggers  # noqa
        if len(all_loggers) == 0:
            return

        if isinstance(metrics, SummaryManager):
            metrics2log = metrics.get_summary("scalars", "objects")
        else:
            metrics2log = {}  # tensor -> scalar
            for mt_name, mt_val in metrics.items():
                if isinstance(mt_val, torch.Tensor):
                    if mt_val.numel() == 1:
                        metrics2log[mt_name] = mt_val.item()
                    else:  # remove arrays
                        msg = f"Only scalars are to record, but got {mt_name}({mt_val.shape} {mt_val.dtype}) Tensor"
                        if strict:
                            raise ValueError(msg)
                        else:
                            warnings.warn(msg)
                else:
                    metrics2log[mt_name] = mt_val

        for logger in all_loggers:
            logger.log_metrics(metrics=metrics2log, step=step, epoch=epoch)

    def validation_loop(
            self,
            model: Any,
            val_loader: Any,
            val_step_fn: Optional[Callable] = None,
            val_epoch_metric_fn: Optional[Callable] = None,
            limit_batches: Union[int, float] = float("inf"),
            tqdm_frequency: Optional[int] = None,
            val_step_fn_kwargs: Optional[Dict[str, Any]] = None,
            val_epoch_metric_fn_kwargs: Optional[Dict[str, Any]] = None,
            auto_log: bool = False,
    ):
        self.state = TrainerStatus.VALIDATING
        res = self.eval_loop(
            model,
            val_loader,
            "valid",
            val_step_fn,
            val_epoch_metric_fn,
            limit_batches,
            tqdm_frequency,
            eval_step_fn_kwargs=val_step_fn_kwargs,
            eval_epoch_metric_fn_kwargs=val_epoch_metric_fn_kwargs,
            auto_log=auto_log,
        )
        self.num_valid_epochs += 1
        return res

    def test_loop(
            self,
            model: Any,
            test_loader: Any,
            test_step_fn: Optional[Callable] = None,
            test_epoch_metric_fn: Optional[Callable] = None,
            limit_batches: Union[int, float] = float("inf"),
            tqdm_frequency: Optional[int] = None,
            test_step_fn_kwargs: Optional[Dict[str, Any]] = None,
            force_log_epoch: Optional[int] = None,
            test_epoch_metric_fn_kwargs: Optional[Dict[str, Any]] = None,
            auto_log: bool = False,
    ):
        self.state = TrainerStatus.TESTING
        res = self.eval_loop(
            model,
            test_loader,
            "test",
            test_step_fn,
            test_epoch_metric_fn,
            limit_batches,
            tqdm_frequency,
            eval_step_fn_kwargs=test_step_fn_kwargs,
            force_log_epoch=force_log_epoch,
            eval_epoch_metric_fn_kwargs=test_epoch_metric_fn_kwargs,
            auto_log=auto_log,
        )
        self.num_test_epochs += 1
        return res

    def eval_loop(
            self,
            model: Any,
            eval_loader: Any,
            stage=None,
            eval_step_fn: Optional[Callable] = None,
            eval_epoch_metric_fn: Optional[Callable] = None,
            limit_batches: Union[int, float] = float("inf"),
            tqdm_frequency: Optional[int] = None,
            eval_step_fn_kwargs: Optional[Dict[str, Any]] = None,
            force_log_epoch: Optional[int] = None,
            eval_epoch_metric_fn_kwargs: Optional[Dict[str, Any]] = None,
            auto_log: bool = False,
    ):
        """The validation loop ruunning a single validation epoch.

        Args:
            model: the LightningModule to evaluate
            eval_loader: The dataloader yielding the validation/test batches.
            limit_batches: Limits the batches during this validation epoch.
                If greater than the number of batches in the ``eval_loader``, this has no effect.

        """
        assert stage is not None
        hook_stage_name = "validation" if stage in ("valid", "val", "validation") else stage
        # no validation if eval_loader wasn't passed
        if eval_loader is None:
            return {}

        eval_step_fn = eval_step_fn or self._eval_step_fn
        eval_step_fn_kwargs = eval_step_fn_kwargs or {}
        eval_epoch_metric_fn = eval_epoch_metric_fn or self._eval_epoch_metric_fn
        eval_epoch_metric_fn_kwargs = eval_epoch_metric_fn_kwargs or {}

        model.eval()  # Can call `model.eval()` or .train() in the hook of following line to change default eval state
        self.fabric.call(f"on_{hook_stage_name}_model_eval")

        torch.set_grad_enabled(False)
        # you can enable gradient in the hook here to override default no gradient during test stage
        self.fabric.call(f"on_{hook_stage_name}_epoch_start")

        if tqdm_frequency is not None and self.current_epoch % tqdm_frequency == 0:
            iterable = self.progbar_wrapper(
                eval_loader,
                total=min(len(eval_loader), limit_batches),
                desc=f"[Epoch {self.current_epoch:4d}|   {stage.capitalize():5s}]",
            )
        else:
            iterable = enumerate(eval_loader)

        metric_manager: "Metric" = self._metric_manager_dict[f"{stage}_mc"]
        loss_manager: "LossRecorder" = self._loss_manager_dict[f"{stage}_ls"]
        bo_manager = self._bo_manager_dict[f"{stage}_bo"]
        metric_manager.reset()
        loss_manager.reset()
        bo_manager.reset()

        for batch_idx, batch_dt in iterable:
            # end epoch if stopping training completely or max batches for this epoch reached
            if self.should_stop or batch_idx >= limit_batches:
                break

            self.fabric.call(f"on_{hook_stage_name}_batch_start", self, model, batch_dt, batch_idx)

            out_dict: Mapping[str, Any] = eval_step_fn(
                model,
                batch_dt,
                loss_fn=self._loss_fn,
                metric_manager=metric_manager,
                loss_manager=loss_manager,
                batch_out_manager=bo_manager,
                current_epoch=self.current_epoch,
                stage=stage,
                fabric=self.fabric,
                batch_idx=batch_idx,
                **eval_step_fn_kwargs,
            )

            # avoid gradients in stored/accumulated values -> prevents potential OOM
            # CAUTION! You may want to collect batch output and then do some differentiable operations on them
            batch_return = self._batch_return_dict[stage] = apply_to_collection(
                out_dict, torch.Tensor, lambda x: x.detach()
            )
            self.fabric.call(f"on_{hook_stage_name}_batch_end", batch_return, batch_dt, batch_idx)

            self._format_iterable(iterable, batch_return, stage)
            self.fabric.call(f"on_{hook_stage_name}_batch_end", self, model, batch_return, batch_dt, batch_idx)
            if hook_stage_name == "validation":
                self.global_valid_step += 1
            else:
                self.global_test_step += 1

        torch.set_grad_enabled(True)

        # LOGGING loss and metrics at Epoch Level
        esd = eval_epoch_metric_fn(
            self, model=self._model, stage=stage, force_log_epoch=force_log_epoch, **eval_epoch_metric_fn_kwargs
        )
        epoch_summary = SummaryManager(esd)
        loss_dict, num_sample_dict = loss_manager.compute()
        epoch_summary.update(loss_dict)

        if auto_log:
            log_epoch = force_log_epoch if force_log_epoch is not None else self.current_epoch
            self.log_metrics(epoch_summary, epoch=log_epoch)
        update_dict_no_collision(self.get_current_epoch_summary(), epoch_summary)

        self.fabric.call(f"on_{hook_stage_name}_epoch_end", self, model)
        self.fabric.call(f"on_{hook_stage_name}_model_train", self, model)
        return epoch_summary

    def get_current_epoch_summary(self) -> SummaryManager:
        if self.state in (TrainerStatus.TRAINING, TrainerStatus.VALIDATING):
            ces = self.current_fit_epoch_summary
        elif self.state == TrainerStatus.TESTING:
            ces = self.current_test_epoch_summary
        else:
            raise RuntimeError(
                f"{self.__class__.__name__}(state={self.state}) "
                f"is not in an valid state to get `current_[xxx]_epoch_summary`"
            )
        return ces

    def step_scheduler(
            self,
            model: torch.nn.Module,
            scheduler_cfg: Optional[Mapping[str, Union[L.fabric.utilities.types.LRScheduler, bool, str, int]]],
            level: Literal["step", "epoch"],
            current_value: int,
    ) -> None:
        """Steps the learning rate scheduler if necessary.

        Args:
            model: The LightningModule to train
            scheduler_cfg: The learning rate scheduler configuration.
                Have a look at :meth:`lightning.pytorch.LightningModule.configure_optimizers` for supported values.
            level: whether we are trying to step on epoch- or step-level
            current_value: Holds the current_epoch if ``level==epoch``, else holds the ``global_step``

        """
        # no scheduler
        if scheduler_cfg is None:
            return

        scheduler = scheduler_cfg["scheduler"]
        if scheduler is None:
            return

        if scheduler_cfg["interval"] != level:
            return  # wrong interval (step vs. epoch)

        # right interval, but wrong step wrt frequency
        if current_value % cast(int, scheduler_cfg["frequency"]) != 0:
            return

        monitor_key = scheduler_cfg["monitor"]
        # rely on model hook for actual step
        if monitor_key is None:  # monitor nothing and do scheduler
            scheduler.step()
            return

            # assemble potential monitored values
        if level == "step":
            current_return = (
                self._batch_return_dict["train"] if "train" in monitor_key else self._batch_return_dict["valid"]
            )
        elif level == "epoch":
            current_return = self.current_fit_epoch_summary
        else:
            raise ValueError
        monitor_val = current_return.get(monitor_key, None)

        if monitor_val is None:
            scheduler.step()  # type: ignore[call-arg]
        else:  # ReduceLROnPlateau monitors a metric to change the lr
            scheduler.step(monitor_val)

    @property
    def should_validate(self) -> bool:
        """Whether to currently run validation."""
        return (self.current_epoch + 1) % self.check_val_every_n_epoch == 0

    def progbar_wrapper(self, iterable: Iterable, total: int, **kwargs: Any):
        """Wraps the iterable with tqdm for global rank zero.

        Args:
            iterable: the iterable to wrap with tqdm
            total: the total length of the iterable, necessary in case the number of batches was limited.

        """
        if self.fabric.is_global_zero:
            return tqdm(enumerate(iterable), total=total, **kwargs)
        return iterable

    def load(
            self, state: Optional[Union[Mapping, torch.nn.Module]], path: PathStr, strict: bool = True,
            weights_only=False, load_warning: bool = True,
    ) -> None:
        """Loads a checkpoint from a given file into state.

        Args:
            state: a mapping contaning model, optimizer and lr scheduler
            path: the path to load the checkpoint from

        """
        if state is None:  # state = {"model": model, "optim": optimizer, "scheduler": scheduler_cfg}
            state = self._current_state  # load to the internal handling model
        elif isinstance(state, torch.nn.Module):
            state = {"model": state}
        elif isinstance(state, Mapping):
            state = state
        else:
            raise TypeError

        if weights_only:
            state = {"model": state["model"]}

        remainder = self.fabric.load(path, state, strict=strict)

        if not weights_only:  # load beyond weights, usually to resume
            self.global_step = remainder.pop("global_step")
            self.current_epoch = remainder.pop("current_epoch")
            self.num_train_epochs = remainder.pop("num_train_epochs")
            self.num_valid_epochs = remainder.pop("num_valid_epochs")
            self.num_train_epochs = remainder.pop("num_test__epochs")

        if remainder and load_warning and self.fabric.is_global_zero:
            rank_zero_warn(f"Unused Checkpoint Values: {list(remainder.keys())}")

    def save_checkpoint(self, filepath: PathStr, state: Optional[Mapping], only_weights=False) -> None:
        """Saves a checkpoint to the ``checkpoint_dir``

        Args:
            state: A mapping containing model, optimizer and lr scheduler.

        """
        if state is None:
            state = self._current_state
        # state = {"model": model, "optim": optimizer, "scheduler": scheduler_cfg}
        if only_weights:  # only model weights
            state = {"model": state["model"]}

        state.update(global_step=self.global_step, current_epoch=self.current_epoch)
        self.fabric.save(filepath, state)

    @staticmethod
    def get_latest_child(parent_dir: PathStr) -> Optional[Path]:
        """Returns the latest checkpoint from the ``parent_dir``

        Args:
            parent_dir: the directory to search for checkpoints

        """
        parent_dir = Path(parent_dir)
        if not parent_dir.is_dir():
            return None

        # sort with the version number behind "version_"
        def get_vn(path):
            match = re.match(r"version_(\d+)", path.name)
            if match:
                return int(match.group(1))
            return 0

        items = list(parent_dir.iterdir())
        items.sort(key=get_vn)
        if not items:
            return None

        return items[-1].absolute()  # full path

    @staticmethod
    def _format_iterable(
            prog_bar,
            candidates: Optional[Union[torch.Tensor, Mapping[str, Union[torch.Tensor, float, int]]]],
            prefix: str,
            filter_out: Sequence[str] = None,
    ):
        """Adds values as postfix string to progressbar.

        Args:
            prog_bar: a progressbar (on global rank zero) or an iterable (every other rank).
            candidates: the values to add as postfix strings to the progressbar.
            prefix: the prefix to add to each of these values.

        """
        if isinstance(prog_bar, tqdm) and candidates is not None:
            _default_filter_out = ["logit", "y", "label", "features", "aim_mask"]
            filter_out = filter_out or []
            _filter_out = _default_filter_out + list(filter_out)
            postfix_str = ""
            candidates = {k: v for k, v in candidates.items() if k not in _filter_out}
            float_candidates = apply_to_collection(candidates, torch.Tensor, lambda x: x.item())
            if isinstance(candidates, torch.Tensor):
                postfix_str += f" {prefix}_loss: {float_candidates:.3f}"
            elif isinstance(candidates, Mapping):
                for k, v in float_candidates.items():
                    if k not in _filter_out and v is not None:
                        if k.startswith(prefix):
                            postfix_str += f" {k}: {v:.3f}"
                        else:
                            postfix_str += f" {prefix}_{k}: {v:.3f}"
            if postfix_str:
                prog_bar.set_postfix_str(postfix_str)

    @staticmethod
    def info_dict2fmt_str(info_dict):
        info_str = ""
        for k, v in info_dict.items():
            if v is not None and (isinstance(v, (float, int)) or v.numel()) == 1:
                info_str = info_str + f"  {k}: {v:.3f}"
        return info_str

    @property
    def ckpt_path(self):
        path = self.checkpoint_dir / f"epoch={self.current_epoch}.ckpt"
        return path

    def remove_checkpoint(self, path: PathStr) -> None:
        """Remove checkpoint file from the filesystem.

        Args:
            path: Path to checkpoint

        """
        fs = get_filesystem(path)
        if fs.exists(path):
            fs.rm(path, recursive=True)
            if self.ckpt_verbose:
                self.fabric.print(f"Removed checkpoint: {path}")

    @staticmethod
    def get_callback_named(fabric, names):
        if not isinstance(names, list):
            names = [names]
        all_normalized_names = [normalize_string(name) for name in names]
        all_callback_dict = {normalize_string(cb.__class__.__name__): cb for cb in fabric._callbacks}
        query_out = {}
        for i, name in enumerate(all_normalized_names):
            if name in all_callback_dict:
                query_out[name] = all_callback_dict[name]

        return query_out
