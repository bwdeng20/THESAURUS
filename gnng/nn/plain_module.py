from typing import Any, Dict, Optional, Callable, Union, Mapping
from gnng.typing import ParsableOptimizer, ParsableScheduler, ParsableMetric, ParsableLoss, ParsableModel
from functools import cached_property
from lightning import LightningModule
import torch
from torch.nn import ModuleList, ModuleDict
from torchmetrics import MetricCollection
from gnng.nn.resolver_initializer import (
    optimizer_initializer,
    scheduler_initializer,
    metric_initializer,
    loss_initializer,
    model_initializer,
)
from gnng.nn.steps_also4distill import configure_step


class PlainModule(LightningModule):
    def __init__(
        self,
        model: ParsableModel,
        model_kwargs: Optional[Dict[str, Any]] = None,
        optimizer: Optional[ParsableOptimizer] = None,
        optimizer_kwargs: Optional[Dict[str, Any]] = None,
        scheduler: Optional[ParsableScheduler] = None,
        scheduler_kwargs: Optional[Dict[str, Any]] = None,
        pl_scheduler_conf_kwargs: Optional[Dict[str, Any]] = None,
        aim_criterion: Optional[ParsableLoss] = "CrossEntropy",
        aim_criterion_kwargs: Optional[Dict[str, Any]] = None,
        metric: ParsableMetric = "Accuracy",
        metric_kwargs: Optional[Dict[str, Any]] = None,
        customized_step_func: Optional[Callable] = None,
        compile_kwargs: Optional[Union[Dict[str, Any], bool]] = None,
        log_every_train_step: bool = False,
        log_every_eval_step: bool = False,
        num_train_dataloader: int = 1,
        num_val_dataloader: int = 1,
        num_test_dataloader: int = 1,
    ):
        """
        Args:
            model: An initialized GNN or Non-graph model
            optimizer:
                1)The name of a pytorch optimizer;
                2)A pytorch optimizer class (not instance!);
                3)A partial pytorch optimizer, e.g., partial(Adam,lr=0.001). The partial
                    initialized arguments WILL be overridden by "optimizer_args"
                4)If None, return Adam(self.parameters(),lr=0.001) by default
            optimizer_kwargs:
                1) Dict[str,Any], a pytorch optimizer param group like
                    {"lr":0.001,"wd":0.0001}, where the missing parameter key-value
                    pair "param":x will be merged after invoking
                    "LightningModule.configure_optimizers()"
                    to make up {"params":x,"lr":0.001,"wd":0.0001}
                2) List[Dict[str, Any], A list of Dicts in 1).The i-th Dict is the
                    optimizer arguments for the i-th group of parameters
            scheduler:
                1) str: A pytorch learning rate scheduler name
                2) A pytorch scheduler class (not instance!)
                3) A partial pytorch scheduler, e.g., partial(StepLR,step_size=10). The
                    partial initialized arguments WON'T be overridden by "scheduler_args"

            scheduler_kwargs:
                A dict of scheduler arguments, e.g., {"step_size":20, "gamma":0.2} for StepLR

            pl_scheduler_conf_kwargs:
                A dict of args as `lr_scheduler_config` valid to LightningModule, e.g.,
                {
                    "scheduler": "ReduceLROnPlateau",
                    "monitor": "val_loss",
                    "interval": "epoch",
                    "frequency": 1
                }

        """
        super().__init__()

        self.save_hyperparameters(logger=False, ignore=["model"])
        self.model = model_initializer(model, model_kwargs, compile_kwargs)
        self.aim_criterion = loss_initializer(aim_criterion, aim_criterion_kwargs)

        self.tape = getattr(self.model, "tape", False)

        self.step_impl = self.configure_step(customized_step_func)
        self._metric_name = None
        self.metric_module_dict = self.configure_metric()
        self.log_every_step = {"train": log_every_train_step, "val": log_every_eval_step, "test": log_every_eval_step}

    @cached_property
    def aim_criterion_name(self) -> str:
        return self.aim_criterion.__class__.__name__

    @property
    def metric_name(self):
        return self._metric_name

    def get_metric(self, stage, dataloader_idx: Optional[int] = None):
        metric_list = self.metric_module_dict[f"{stage}_metric"]
        wanted = metric_list[dataloader_idx] if dataloader_idx is not None else metric_list
        return wanted, self.metric_module_dict[f"{stage}_metric_all"]

    def reset_metric(self, stage, dataloader_idx=None):
        stage_metric, stage_metric_all = self.get_metric(stage, dataloader_idx)
        [mt.reset() for mt in stage_metric]
        if stage_metric_all is not None:
            stage_metric_all.reset()

    def configure_metric(self):
        hparams = self.hparams
        # TODO: add func or method to infer some necessary args of various Metric when missing
        metric_kwargs = hparams.metric_kwargs or {"task": "multiclass", "num_classes": self.model.out_channels}

        metric = metric_initializer(hparams.metric, metric_kwargs)
        if not isinstance(metric, MetricCollection):  # wrapped with Collection
            metric = MetricCollection(metric)
        self._metric_name = list(metric.keys())
        train_metric = ModuleList(
            [metric.clone(prefix="train/", postfix=f"_dl={i}") for i in range(hparams.num_train_dataloader)]
        )
        val_metric = ModuleList(
            [metric.clone(prefix="val/", postfix=f"_dl={i}") for i in range(hparams.num_val_dataloader)]
        )
        test_metric = ModuleList(
            [metric.clone(prefix="test/", postfix=f"_dl={i}") for i in range(hparams.num_test_dataloader)]
        )

        train_metric_all = (
            metric.clone(prefix="train/", postfix="_dl=all") if hparams.num_train_dataloader > 1 else None
        )
        val_metric_all = metric.clone(prefix="val/", postfix="_dl=all") if hparams.num_val_dataloader > 1 else None
        test_metric_all = metric.clone(prefix="test/", postfix="_dl=all") if hparams.num_test_dataloader > 1 else None
        metric_module_dict = ModuleDict(
            {
                "train_metric": train_metric,
                "val_metric": val_metric,
                "test_metric": test_metric,
                "train_metric_all": train_metric_all,
                "val_metric_all": val_metric_all,
                "test_metric_all": test_metric_all,
            }
        )
        return metric_module_dict

    def configure_step(self, func: Optional[Callable] = None):
        step_impl = configure_step(self.model, func)
        return step_impl

    def forward(self, x, *args, **kwargs):
        out = self.step_impl(self.model, *args, **kwargs)[0]
        return out

    def step(self, data, split):
        logit4aim, y4aim = self.step_impl(self.model, data, split)[:2]
        return logit4aim, y4aim

    def step_aim_loss_metric(self, logit, target, stage, dataloader_idx: int = 0, batch_size=None):
        """
        Compute and log the aim loss and metrics of the target samples in the current step (batch) from
        the current dataloader. Note that the overall Metric only record the pred and target of current
        batch of current dataloader. Please self.log_dict(OverallMetric) at the epoch end.
        Args:
            logit:
            target:
            stage:
            batch_size:
            dataloader_idx:

        Returns:
            Tensor: the computed aim loss for later usage, e.g., backward and optimization
        """
        aim_loss = self.aim_criterion(logit, target)
        stage_metric, stage_metric_all = self.get_metric(stage, dataloader_idx)
        batch_size = batch_size if batch_size is not None else logit.shape[0]
        log_on_step = self.log_every_step[stage]
        self.log(
            f"{stage}/{self.aim_criterion_name}{stage_metric.postfix}",
            aim_loss,
            on_step=log_on_step,
            on_epoch=True,
            add_dataloader_idx=False,
            batch_size=batch_size,
        )

        if log_on_step:
            batch_metric = stage_metric(logit, target)  # forward=compute batch-level and accumulate
            self.log_dict(batch_metric, on_step=True, on_epoch=True, add_dataloader_idx=False, batch_size=batch_size)
        else:  # only update will save some costs
            stage_metric.update(logit, target)

        if stage_metric_all is not None:  # Never Compute or log this metric across all dataloaders
            stage_metric_all.update(logit, target)
        return aim_loss

    def on_train_start(self):
        # by default lightning executes validation step sanity checks before training starts,
        # so it's worth to make sure validation metrics don't store results from these checks
        # self.reset_metric("train")
        # self.reset_metric("test")
        self.reset_metric("val")

    def training_step(self, data, batch_idx, dataloader_idx: int = 0):
        logit, target = self.step(data, split="train")
        loss = self.step_aim_loss_metric(logit, target, "train", dataloader_idx)
        return loss

    def validation_step(self, data, batch_idx, dataloader_idx: int = 0):
        logit, target = self.step(data, split="val")
        loss = self.step_aim_loss_metric(logit, target, "val", dataloader_idx)
        return loss

    def test_step(self, data, batch_idx, dataloader_idx: int = 0):
        logit, target = self.step(data, split="test")
        loss = self.step_aim_loss_metric(logit, target, "test", dataloader_idx)
        return loss

    def epoch_end_log_and_reset_metric(self, stage, reset=True):
        # we need compute the epoch-level metric
        stage_metric_list, stage_metric_all = self.get_metric(stage)
        if not self.log_every_step[stage]:
            for mt in stage_metric_list:
                epoch_metric = mt.compute()
                self.log_dict(epoch_metric, on_step=False, on_epoch=True, add_dataloader_idx=False)

        if stage_metric_all is not None:
            epoch_metric = stage_metric_all.compute()
            self.log_dict(epoch_metric, on_step=False, on_epoch=True, add_dataloader_idx=False)

        if reset:
            self.reset_metric(stage)

    def on_train_epoch_end(self):
        self.epoch_end_log_and_reset_metric("train")

    def on_validation_epoch_end(self):
        self.epoch_end_log_and_reset_metric("val")

    def on_test_epoch_end(self):
        self.epoch_end_log_and_reset_metric("test")

    def configure_optimizers(self):
        opt = optimizer_initializer(self.hparams.optimizer, self.hparams.optimizer_kwargs, self.model)
        sch = scheduler_initializer(
            self.hparams.scheduler, self.hparams.scheduler_kwargs, opt, self.hparams.pl_scheduler_conf_kwargs
        )

        if sch is None:
            return opt
        else:
            return opt, sch

        # TODO: try to always return two lists, which are the list of torch optimizers and
        #       the list of (config dict of) torch schedulers. There is _MockOptimizer in lightning to handle None opt
        #       we may need to add MockScheduler to handle None sch. Such mock-stuff-based protocol will bring about
        #       the most flexible GNN distillation scheme as we can set part of optimizers or schedulers as mock ones
        #       to mute or enable the training of corresponding model.
        #   return [opt], [sch]

    def load_state_dict(self, state_dict: Mapping[str, Any], strict: bool = False, assign: bool = False) -> None:
        # When any of torch.nn.utils.parametrizations, e.g., spectral_norm, is applied to
        # "weight" of a "submodule", the stata_dict contains both a weight named
        # "submodule.weight" and the renamed original learnable parameters handled by
        # "submodule.parametrization.weight.original". Non-strict state_dict loaders is
        # required to support loading state_dict to both the original model and the model
        # with parametrization.
        super().load_state_dict(state_dict, strict, assign)

    @torch.no_grad()
    def inference(self):
        raise NotImplementedError
