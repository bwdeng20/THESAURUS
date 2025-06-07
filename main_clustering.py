import copy
import time
import logging
from dataclasses import asdict
from pprint import pprint
from typing import Optional, Dict, Any, Union, List
from pathlib import Path
import pandas as pd
import torch
import wandb
from tabulate import tabulate
from lightning.fabric import seed_everything
from lightning.pytorch.cli import OptimizerCallable, LRSchedulerCallable
from omegaconf import OmegaConf
from torch_geometric import compile as pyg_compile
from torch_geometric.data import InMemoryDataset
from torch_geometric.transforms import BaseTransform
from gnng.cli import CLI, color_logger
from gnng.data import TVTGraph, DataSplitterBase, Augmentor, UnifiedNodeEdgeLevelDataset
from gnng.data.loader import get_loader_from_cfg
from gnng.metrics import ClusteringSummary
from gnng.nn import GnnGBase, model_initializer, GnnGLoss
from gnng.pyg_ext import LoaderConfig, diagonal
from gnng.training import (
    FabricTrainer,
    FabricTrainerConfig,
    WandbLogger,
    ModelCheckpoint,
    infer_node_mask,
)
from gnng.nn.models.clustering import KMeans

from gnng.utils import report_df2dict, scalar_tensor2pynum
from tools import get_info_df, class_info_from_name_space, get_tabulate_report

from gwclustering.metric_fns import (
    noop_epoch_metric,
    unsup_epoch_metric_fn4clustering_main,
)
from gwclustering.steps import train_step_gw_clustering, eval_step_gw_clustering

logger = logging.getLogger(__name__)
logger = color_logger(logger)


def train_main(
        model: Union[GnnGBase] = None,
        dataset: Union[InMemoryDataset, UnifiedNodeEdgeLevelDataset] = None,
        loader: LoaderConfig = None,
        eval_loader: LoaderConfig = None,  # by default empty
        loader_transform: Optional[BaseTransform] = None,
        trainer: FabricTrainerConfig = None,
        optimizer: OptimizerCallable = None,
        scheduler: LRSchedulerCallable = None,
        augmentors: List[Union[torch.nn.Module, Augmentor]] = None,
        criterion: Union[GnnGLoss] = None,
        splitter: Optional[DataSplitterBase] = None,
        tvt_graph_producer: Optional[TVTGraph] = None,
        compile_kwargs: Optional[Dict[str, Any]] = None,
        monitor: str = None,
        wandb_cfg: Optional[Dict[str, Any]] = None,
        seed: int = None,
        raw_cfg_dict: Optional[Dict[str, Any]] = None,
):
    raw_cfg_dict = copy.deepcopy(raw_cfg_dict)
    raw_cfg_dict["seed"] = seed
    del raw_cfg_dict["seeds"]
    seed_everything(seed)

    # ============ Prepare Train/Val/Test datasets and dataloaders ==================
    if eval_loader is None or eval_loader.name is not None:
        eval_loader = loader
    loader_cfg = asdict(loader)
    eval_loader_cfg = asdict(eval_loader)
    loader_name = loader_cfg["name"]
    eval_loader_name = eval_loader_cfg["name"]

    wandb_cfg = wandb_cfg or {"project": "nc_clustering_test", "tags": ["debug"], "mode": "disabled"}

    data = dataset[0]
    if splitter is not None:
        data = splitter(data)

    transductive_clustering = True
    if tvt_graph_producer is not None:  # generate train/val/test graphs
        data_dict = tvt_graph_producer(data)
        train_data = data_dict["train"]
        val_data = data_dict["val"]
        test_data = data_dict["test"]
        transductive_clustering = "trans" in tvt_graph_producer.piece_rule
    else:  # transductive
        train_data = data.clone()
        val_data = data.clone()
        test_data = data.clone()

    # ============ Prepare WandbLogger and ModelCheckpoint CB for FabricTrainer ==================
    # hack wandb logger and ModelCheckpoint callback, prefer to init them here
    trainer_kwargs = asdict(trainer)  # dataclass to dict
    if trainer_kwargs["callbacks"] is None:
        checkpoint_cb = ModelCheckpoint(
            save_top_k=1,
            save_last=True,
            monitor=monitor,  # monitor the validation Accuracy on the first dataloader/dataset
            mode="min" if (monitor is not None and "loss" in monitor) else "max",
        )
        trainer_kwargs["callbacks"] = [checkpoint_cb]
    else:
        checkpoint_cb = None
        for cb in trainer_kwargs["callbacks"]:
            if isinstance(cb, ModelCheckpoint):
                checkpoint_cb = cb
    assert checkpoint_cb is not None, "Please use checkpoint callback to manage the best model ckpt."

    wdlogger: Optional[WandbLogger] = None
    if wandb_cfg.get("mode", "disabled") != "disabled":
        if trainer_kwargs["loggers"] is None:  # Not initialize logger from Cli or Conf files
            wdlogger: WandbLogger = WandbLogger(**wandb_cfg)
            trainer_kwargs["loggers"] = [wdlogger]
        else:  # wandb_mode != "disabled"
            for tlogger in trainer_kwargs["loggers"]:
                if isinstance(tlogger, WandbLogger):
                    wdlogger = tlogger

    # =========================== Model init & compile  =========================
    # Two kinds of model yaml support:  1) jsonargparse Cli style;
    #                                   2) dict: model_cls(name) + init args
    model.reset_parameters()
    if isinstance(model, dict):
        model_cls = model.pop("model_cls")
        model["num_features"] = data.num_node_features
        model["num_classes"] = data["num_classes"]
        model = model_initializer(model_cls, model_kwargs=model)

    compile_kwargs = compile_kwargs or {}

    if compile_kwargs and loader_name == "neighbor":  # torch2.0 support compile! torch2.1 support dynamic compile
        mandatory4sampling_loader = {"dynamic": True, "fullgraph": True}
        compile_kwargs = {**mandatory4sampling_loader, **compile_kwargs}

    if compile_kwargs:  # do compile if any compile args, but may break for some old torch and dynamic data shape
        model = pyg_compile(model, **compile_kwargs)

    # ========================= Trainer: Init  =========================
    trainer = FabricTrainer(**trainer_kwargs)

    # =================== compose and log human-Readable hyper-parameters =========================
    long_model_name = repr(model)
    tvt_graph_producer_info = repr(tvt_graph_producer) if tvt_graph_producer is not None else "none"
    long_data_name = (
        f"{repr(dataset)}_split={repr(splitter) if splitter is not None else 'default'}"
        f"_tvt={tvt_graph_producer_info}"
    )
    raw_cfg_dict["long_model"] = long_model_name
    raw_cfg_dict["long_data"] = long_data_name

    # ========================= Trainer:  Fit =========================
    input_train_nodes = None if transductive_clustering else infer_node_mask(train_data, "train")
    train_loader = get_loader_from_cfg(
        train_data,
        loader_name,
        input_nodes=input_train_nodes,  # if inductive, None ->  tro+trn; if transductive, None ->  tro+trn + val +test
        transform=loader_transform,
        loader_cfg=loader_cfg,
    )

    infer_nodes4val = None if transductive_clustering else infer_node_mask(val_data, "val")
    val_loader = get_loader_from_cfg(
        val_data,
        eval_loader_name,
        input_nodes=infer_nodes4val,
        transform=loader_transform,
        loader_cfg=loader_cfg,
    )

    infer_nodes4test = None if transductive_clustering else infer_node_mask(test_data, "test")
    test_loader = get_loader_from_cfg(
        test_data,
        eval_loader_name,
        input_nodes=infer_nodes4test,
        transform=loader_transform,
        loader_cfg=loader_cfg,
    )

    if isinstance(model.predictor, KMeans):
        model.predictor.random_state = seed  # KMeans

    time_fit_begin = time.time()
    trainer.fit(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        optimizer=optimizer,
        scheduler=scheduler,
        metric=ClusteringSummary(compute_on_cpu=True),
        train_step_fn=train_step_gw_clustering,
        eval_step_fn=eval_step_gw_clustering,
        criterion=criterion,
        tqdm_frequency=200,
        verbose_frequency=200,
        train_epoch_metric_fn=noop_epoch_metric,
        eval_epoch_metric_fn=(
            unsup_epoch_metric_fn4clustering_main
        ),
        train_step_fn_kwargs={"augmentors": augmentors},
    )
    torch.cuda.current_stream().synchronize()
    time_fit_end = time.time()
    time_fit = time_fit_end - time_fit_begin

    # ========================= Trainer: Test   ===============================
    return_summary = {"time_fit": time_fit}
    monitor = checkpoint_cb.monitor
    best_monitor_value = checkpoint_cb.best_model_score.item()
    best_model_path = checkpoint_cb.best_model_path
    return_summary.update(
        {
            "monitor": monitor,
            "best_monitor_value": best_monitor_value,
            "best_model_path": best_model_path,
            f"{monitor}": best_monitor_value,
        }
    )

    print(
        f"=============> Testing with the best {monitor} ckpt with "
        f"an value of {best_monitor_value}"
        f"\n==> Path: {best_model_path}"
    )

    time_test_begin = time.time()
    test_summary = trainer.test(
        model,
        dataloader=test_loader,
        ckpt_path=best_model_path,
        auto_log=False,
    )
    torch.cuda.current_stream().synchronize()
    time_test_end = time.time()
    time_test = time_test_end - time_test_begin
    test_summary = {f"{k}_ckpt={monitor.replace('/', '_')}": v for k, v in test_summary.items()}
    test_summary.update({"time_test": time_test, "time_fit": time_fit})
    trainer.log_metrics(test_summary)
    test_summary["model"] = "THESAURUS"

    return_summary.update(test_summary)
    return_summary["long_model"] = long_model_name
    return_summary["long_data"] = long_data_name

    if wdlogger is not None:
        wdlogger.log_hyperparams(raw_cfg_dict)
        wdlogger.experiment.finish()

    else:  # write hyperparameters into a local yaml file
        with open(trainer.default_root_dir / "hparams.yaml", "w") as f:
            OmegaConf.save(OmegaConf.create(raw_cfg_dict), f=f)
    return_summary = {k: scalar_tensor2pynum(v) for k, v in return_summary.items()}
    return return_summary


def multirun_train_main(
        model: Union[GnnGBase, Dict[str, Any]] = None,
        dataset: Union[InMemoryDataset, UnifiedNodeEdgeLevelDataset] = None,
        loader: LoaderConfig = LoaderConfig(),
        eval_loader: LoaderConfig = LoaderConfig(),  # by default empty
        loader_transform: Optional[BaseTransform] = None,
        trainer: FabricTrainerConfig = FabricTrainerConfig(),
        optimizer: OptimizerCallable = torch.optim.Adam,
        scheduler: LRSchedulerCallable = torch.optim.lr_scheduler.ConstantLR,
        augmentors: List[Union[torch.nn.Module, Augmentor]] = None,
        criterion: Union[GnnGLoss] = None,
        splitter: Optional[DataSplitterBase] = None,
        tvt_graph_producer: Optional[TVTGraph] = None,
        compile_kwargs: Optional[Dict[str, Any]] = None,
        monitor: str = None,
        wandb_cfg: Optional[Dict[str, Any]] = None,
        seeds: List[int] = None,
        raw_cfg_dict: Optional[Dict[str, Any]] = None,
):

    multi_result_dict = {}
    basic_wandb_cfg = wandb_cfg or {"project": "node_clustering", "tags": ["debug"], "mode": "disabled"}
    basic_wandb_cfg["reinit"] = True
    sweep_run = wandb.init(**basic_wandb_cfg)

    sweep_id = sweep_run.sweep_id or "unknown"
    sweep_url = sweep_run.get_sweep_url()
    project_url = sweep_run.get_project_url()
    sweep_group_url = f"{project_url}/groups/{sweep_id}"

    sweep_run_name = sweep_run.name or sweep_run.id or "unknown_2"
    sweep_run_id = sweep_run.id
    sweep_run.finish()
    wandb.sdk.wandb_setup._setup(_reset=True)

    trainer.default_root_dir = Path(f"fabric_logs/nc_clustering/{WhichData}/{WhichModel}/{sweep_run_id}")
    result_records = []
    for seed in seeds:
        # add sweep run info to group runs
        wandb_cfg = copy.deepcopy(basic_wandb_cfg)
        wandb_cfg["group"] = sweep_id
        wandb_cfg["job_type"] = sweep_run_name
        wandb_cfg["name"] = f"{sweep_run_name}-seed={seed}"

        return_summary = train_main(
            model=copy.deepcopy(model),
            dataset=dataset,
            loader=loader,
            eval_loader=eval_loader,
            loader_transform=loader_transform,
            trainer=trainer,
            optimizer=optimizer,
            scheduler=scheduler,
            augmentors=augmentors,
            criterion=criterion,
            splitter=splitter,
            tvt_graph_producer=tvt_graph_producer,
            compile_kwargs=compile_kwargs,
            monitor=monitor,
            wandb_cfg=wandb_cfg,
            seed=seed,
            raw_cfg_dict=raw_cfg_dict,
        )
        torch.cuda.empty_cache()
        multi_result_dict[seed] = return_summary
        return_summary["seed"] = seed
        return_summary = {k: v for k, v in return_summary.items() if isinstance(v, (torch.Tensor, float, int, str))}

        result_records.append(return_summary)
        raw_cfg_dict["long_model"] = return_summary["long_model"]
        raw_cfg_dict["long_data"] = return_summary["long_data"]

    result_df = pd.DataFrame.from_records(result_records)

    basic_wandb_cfg["id"] = sweep_run_id
    basic_wandb_cfg["resume"] = "must"
    sweep_run = wandb.init(**basic_wandb_cfg)
    sweep_run.config.update(raw_cfg_dict, allow_val_change=True)
    average_result = report_df2dict(result_df)
    sweep_run.log(average_result)
    sweep_run.finish()
    print("*" * 40)
    print("Sweep URL:       ", sweep_url)
    print("Sweep Group URL: ", sweep_group_url)
    print("*" * 40)
    if basic_wandb_cfg["mode"] == "disabled":
        with open(trainer.default_root_dir / "sweep_hparams.yaml", "w") as f:
            OmegaConf.save(OmegaConf.create(raw_cfg_dict), f=f)
    multi_result_dict["report"] = average_result
    multi_result_dict["report"]["sweep_run_id"] = sweep_run_id
    multi_result_dict["report"]["monitor"] = monitor
    multi_result_dict["report"]["seeds"] = seeds
    multi_result_dict["report"]["sweep_run_name"] = sweep_run_name
    # pprint(multi_result_dict)
    return multi_result_dict


def objective_main():
    # prepare raw_cfg_dict to upload to wandb
    cfg2log = cfg.clone().as_dict()

    config_fps = cfg2log["config"]
    cfg2log["cwd"] = config_fps[0]._cwd  # noqa
    cfg2log["config"] = [str(fp) for fp in config_fps]
    # pprint(cfg2log)

    # True run: initialize the classes then run
    del cfg2log["raw_cfg_dict"]
    arg2replace = {"raw_cfg_dict": cfg2log}  # this is somewhat hacking to log cfg to wandb
    multi_result_dict = my_cli.instantiate_and_run(cfg=cfg, init_from=arg2replace)
    #  log (mean +) results here
    report = multi_result_dict["report"]  # metrics and info dumped into Optuna.Trial
    # pprint({k: v for k, v in report.items() if isinstance(v, str)})
    return report


if __name__ == "__main__":
    # ========================== MISC preparing ===========================================
    torch.set_float32_matmul_precision("high")
    seed_everything(2050)
    # objective_main() --> multirun_train_main() --> for seed in seeds: train_main()

    # ================= Meta information from preparsed Cli arguments ======================
    my_cli = CLI(components=multirun_train_main, parser_mode="omegaconf")
    # namespace will be merged into the default? NO! Something wrong, please use another way
    cfg = my_cli.parse_args(namespace=None)
    dt_name = cfg.dataset.init_args.name
    all_data_info = get_info_df()
    if dt_name in ["acm", "dblp", "uat", "wiki"]:
        dt_name = f"{dt_name}_clustering"
    info = all_data_info.loc[all_data_info["Dataset"] == dt_name.lower()]
    print(info)

    WhichModel = f"{cfg.model.class_path.rsplit('.', 1)[1].lower()}"
    WhichData = (
        f"{dt_name}"
        f"_split={class_info_from_name_space(cfg.splitter)}"
        f"_tvt={class_info_from_name_space(cfg.tvt_graph_producer)}"
    ).lower()


    report = objective_main()

    interested_mts = [
        "ClusteringAccuracy",
        "NMI",
        "ARI",
        "ClusteringF1",
    ]
    tb_reproduce = get_tabulate_report(report, interested_mts, monitor=cfg.monitor, prefix="repr2_test")
    print(tabulate(tb_reproduce, headers=interested_mts, tablefmt="simple"))
