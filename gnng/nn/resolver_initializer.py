from typing import Any, Union, Optional, List, Dict, Iterable, Callable
from gnng.typing import Tensor, Optimizer, ModuleList, Module, Metric, MetricCollection, _Loss, LRScheduler
from gnng.typing import (
    ParsableOptimizer,
    ParsableScheduler,
    ParsableMetric,
    ParsableLoss,
    ParsableModel,
    ParsableMultiModel,
    ParsableCkpt,
)
import copy
import inspect
from functools import partial
import torch
from torch.optim.lr_scheduler import ReduceLROnPlateau

from torch_geometric.nn.lr_scheduler import (
    ConstantWithWarmupLR,
    CosineWithWarmupLR,
    CosineWithWarmupRestartsLR,
    LinearWithWarmupLR,
    PolynomialWithWarmupLR,
)
from gnng.resolver_initializer import normalize_string, resolver, initializer
from gnng.nn.optimizer import MockOptimizer
from gnng.nn.utils.ckpt import load_th_or_pl_ckpt2md
from gnng.utils import parse_compile_kwargs
from gnng.nn.losses import GnnGLoss


def model_resolver(query: Union[Any, str] = "GraphSAGE", return_cls=False, *args, **kwargs):
    import gnng.nn.models as gb_models

    base_cls = Module
    mds = [md for md in vars(gb_models).values() if isinstance(md, type) and issubclass(md, base_cls)]
    return resolver(mds, {}, query, base_cls, None, return_cls, *args, **kwargs)


def metric_resolver(query: Union[Any, str] = "Accuracy", return_cls=False, *args, **kwargs):
    import torchmetrics

    base_cls = torchmetrics.Metric
    mts = [mt for mt in vars(torchmetrics).values() if isinstance(mt, type) and issubclass(mt, base_cls)]
    return resolver(mts, {}, query, base_cls, None, return_cls, *args, **kwargs)


def loss_resolver(query: Union[Any, str] = "CrossEntropy", return_cls=False, *args, **kwargs):
    import torch.nn.modules.loss as th_losses

    base_cls = th_losses._Loss
    base_cls_repr = "Loss"
    losses = [loss for loss in vars(th_losses).values() if isinstance(loss, type) and issubclass(loss, base_cls)]
    return resolver(losses, {}, query, base_cls, base_cls_repr, return_cls, *args, **kwargs)


# Activation Resolver #########################################################


def swish(x: Tensor) -> Tensor:
    return x * x.sigmoid()


def activation_resolver(query: Union[Any, str] = "relu", return_cls: bool = False, *args, **kwargs):
    base_cls = torch.nn.Module
    base_cls_repr = "Act"
    acts = [
        act for act in vars(torch.nn.modules.activation).values() if isinstance(act, type) and issubclass(act, base_cls)
    ]
    acts += [
        swish,
    ]
    act_dict = {}
    return resolver(acts, act_dict, query, base_cls, base_cls_repr, return_cls, *args, **kwargs)


# Normalization Resolver ######################################################


def normalization_resolver(query: Union[Any, str], return_cls: bool = False, *args, **kwargs):
    import torch_geometric.nn.norm as norm

    base_cls = torch.nn.Module
    base_cls_repr = "Norm"
    norms = [norm for norm in vars(norm).values() if isinstance(norm, type) and issubclass(norm, base_cls)]
    norm_dict = {}
    return resolver(norms, norm_dict, query, base_cls, base_cls_repr, return_cls, *args, **kwargs)


# Aggregation Resolver ########################################################


def aggregation_resolver(query: Union[Any, str], return_cls: bool = False, *args, **kwargs):
    import torch_geometric.nn.aggr as aggr

    if isinstance(query, (list, tuple)):
        return aggr.MultiAggregation(query, *args, **kwargs)

    base_cls = aggr.Aggregation
    aggrs = [aggr for aggr in vars(aggr).values() if isinstance(aggr, type) and issubclass(aggr, base_cls)]
    aggr_dict = {"add": aggr.SumAggregation}
    return resolver(aggrs, aggr_dict, query, base_cls, None, return_cls, *args, **kwargs)


# Optimizer Resolver ##########################################################


def optimizer_resolver(query: Union[Any, str], return_cls: bool = False, *args, **kwargs):
    base_cls = Optimizer
    optimizers = [
        optimizer
        for optimizer in vars(torch.optim).values()
        if isinstance(optimizer, type) and issubclass(optimizer, base_cls)
    ]
    opt_dict = {"MockOptimizer": MockOptimizer}
    return resolver(optimizers, opt_dict, query, base_cls, None, return_cls, *args, **kwargs)


# Learning Rate Scheduler Resolver ############################################


def lr_scheduler_resolver(
    query: Union[Any, str],
    optimizer: Optimizer,
    warmup_ratio_or_steps: Optional[Union[float, int]] = 0.1,
    num_training_steps: Optional[int] = None,
    return_cls: bool = False,
    **kwargs,
) -> Union[LRScheduler, ReduceLROnPlateau]:
    r"""A resolver to obtain a learning rate scheduler implemented in either PyG or PyTorch from its name or type.

    Args:
        query (Any or str): The query name of the learning rate scheduler.
        optimizer (Optimizer): The optimizer to be scheduled.
        warmup_ratio_or_steps (float or int, optional): The number of warmup
            losses. If given as a `float`, it will act as a ratio that gets
            multiplied with the number of training losses to obtain the number
            of warmup losses. Only required for warmup-based LR schedulers.
            (default: :obj:`0.1`)
        num_training_steps (int, optional): The total number of training losses.
            (default: :obj:`None`)
        return_cls (bool): If True, return the queried class instead of instance
            (default: :obj:`False`)
        **kwargs (optional): Additional arguments of the LR scheduler.

    """
    if not isinstance(query, str):
        return query

    if isinstance(warmup_ratio_or_steps, float):
        if warmup_ratio_or_steps < 0 or warmup_ratio_or_steps > 1:
            raise ValueError(
                f"`warmup_ratio_or_steps` needs to be between "
                f"0.0 and 1.0 when given as a floating point "
                f"number (got {warmup_ratio_or_steps})."
            )
        if num_training_steps is not None:
            warmup_steps = round(warmup_ratio_or_steps * num_training_steps)
    elif isinstance(warmup_ratio_or_steps, int):
        if warmup_ratio_or_steps < 0:
            raise ValueError(
                f"`warmup_ratio_or_steps` needs to be positive "
                f"when given as an integer "
                f"(got {warmup_ratio_or_steps})."
            )
        warmup_steps = warmup_ratio_or_steps
    else:
        raise ValueError(f"Found invalid type of `warmup_ratio_or_steps` " f"(got {type(warmup_ratio_or_steps)})")

    base_cls = LRScheduler
    torch_native_schedulers = [
        scheduler
        for scheduler in vars(torch.optim.lr_scheduler).values()
        if isinstance(scheduler, type) and issubclass(scheduler, base_cls)
    ] + [ReduceLROnPlateau]

    pyg_lr_schedulers = [
        ConstantWithWarmupLR,
        LinearWithWarmupLR,
        CosineWithWarmupLR,
        CosineWithWarmupRestartsLR,
        PolynomialWithWarmupLR,
    ]

    classes = torch_native_schedulers + pyg_lr_schedulers

    query_repr = normalize_string(query)
    base_cls_repr = normalize_string("LR")

    for cls in classes:
        cls_repr = normalize_string(cls.__name__)
        if query_repr in [cls_repr, cls_repr.replace(base_cls_repr, "")]:
            if inspect.isclass(cls) and not return_cls:
                if cls in pyg_lr_schedulers:
                    cls_keys = inspect.signature(cls).parameters.keys()
                    if "num_warmup_steps" in cls_keys:
                        kwargs["num_warmup_steps"] = warmup_steps
                    if "num_training_steps" in cls_keys:
                        kwargs["num_training_steps"] = num_training_steps
                obj = cls(optimizer, **kwargs)
                return obj
            return cls

    choices = {cls.__name__ for cls in classes}
    raise ValueError(f"Could not resolve '{query}' among choices {choices}")


# Initializer ################################################################3
def loss_initializer(loss: Optional[ParsableLoss], loss_kwargs: Optional[Dict[str, Any]]):
    return initializer(loss, Union[_Loss, GnnGLoss], loss_resolver, loss_kwargs)


def metric_initializer(metric: Optional[ParsableMetric], metric_kwargs: Optional[Dict[str, Any]]):
    return initializer(metric, Union[Metric, MetricCollection], metric_resolver, metric_kwargs)


def model_initializer(
    model: Optional[ParsableModel],
    model_kwargs: Optional[Dict[str, Any]] = None,
    compile_kwargs: Optional[Union[Dict[str, Any], bool]] = None,
):
    model = initializer(model, Module, model_resolver, model_kwargs)
    do_compile, compile_kwargs = parse_compile_kwargs(compile_kwargs)
    return model if not do_compile else torch.compile(model, **compile_kwargs)


def many_model_initializer_from_instance(
    model: Union[List[Module], ModuleList, Module],
    model_ckpt: Optional[Union[ParsableCkpt, List[ParsableCkpt]]] = None,
    num_model: Optional[int] = None,
    map_location="cpu",
    compile_kwargs: Optional[Union[Dict[str, Any], bool]] = None,
) -> ModuleList:
    """
    if isinstance(m, Module)
    --------------------------------------------------------------------------------------
    model | model_kwargs | model_ckpt | num_model | result; result with num_model=None
    --------------------------------------------------------------------------------------
    0) m       -              mc           -           [m]  with the first mc ckpts;
    1) m       -              1            -           [m]  with 1 ckpt;
    2) 1       -              mc           nm          [mc] ckpts: [max(mc,nm)] with the first mc ckpts
    3) 1       -              1            nm          [1]; [nm] same models
    Returns:  ModuleList

    """
    model_ckpt = model_ckpt or []
    if not isinstance(model_ckpt, List):
        model_ckpt = [model_ckpt]

    if not isinstance(model, (List, ModuleList)):
        model = [model]

    if len(model) == 1:
        num_model = len(model_ckpt) if num_model is None else max(num_model, len(model_ckpt))
        models = ModuleList([copy.deepcopy(model[0]) for _ in range(num_model)])
    else:
        models = ModuleList(model)

    do_compile, compile_kwargs = parse_compile_kwargs(compile_kwargs)
    if do_compile:
        models = torch.compile(models, **compile_kwargs)

    if len(model_ckpt) > 0:
        ckpts = [model_ckpt] if not isinstance(model_ckpt, (list, tuple)) else model_ckpt
        for i in range(len(ckpts)):  # load ckpt for the first len(model_ckpt) models
            models[i] = load_th_or_pl_ckpt2md(models[i], ckpts[i], map_location=map_location)
    return models


def many_model_initializer(
    model: Union[ParsableMultiModel, ParsableModel],
    model_kwargs: Optional[Union[Dict[str, Any], List[Dict[str, Any]]]] = None,
    model_ckpt: Optional[Union[ParsableCkpt, List[ParsableCkpt]]] = None,
    num_model: Optional[int] = None,
    map_location="cpu",
    compile_kwargs: Optional[Union[Dict[str, Any], bool]] = None,
) -> ModuleList:
    """If isinstance(m, Module) 0)-3): see build_models_from_instance.

    elif isinstance(m, (str, partial, subclass))
    --------------------------------------------------------------------------------------
    model | model_kwargs | model_ckpt | num_model | result; result with num_model=None
    --------------------------------------------------------------------------------------
    4) 1       1              1,0            nm          [1]; [nm] same models
    5) 1       1              mc,0           nm          [mc] with mc ckpts; [nm] with first mc ckpts
    6) 1       mk>1           1,0            nm          [mk] with 1 ckpt
    7) 1       mk>1           mc,0           nm          [mk] mk==mc,every mc --> mk
    8) m       1              1,0            nm            [m] md with 1 ckpt if model is not
    9) m       mk             mc,0           nm            [m] m == mk == mc
    return 1d ModuleList

    """
    if not isinstance(model, (List, ModuleList)):
        model = [model]

    model_kwargs = model_kwargs or [{}]
    if not isinstance(model_kwargs, List):
        model_kwargs = [model_kwargs]

    model_ckpt = model_ckpt or []
    if not isinstance(model_ckpt, List):
        model_ckpt = [model_ckpt]

    if isinstance(model[0], Module):
        return many_model_initializer_from_instance(model, model_ckpt, num_model, map_location, compile_kwargs)

    num_model_kwargs, num_model_ckpt = len(model_kwargs), len(model_ckpt)
    # if not instance, we must pass init args to instantiate the model, so kwargs is required
    assert num_model_kwargs >= 1, "non-instance `model` requires at least one model init args to initialize model."

    if len(model) == 1:  # (4,5,6,7)
        assert num_model_kwargs >= 1
        if num_model_kwargs > 1:  # (6,7)
            num_model = num_model_kwargs
            if num_model_ckpt > 0:
                assert num_model_kwargs == num_model_ckpt
            if num_model_ckpt == 1:
                model_ckpt = [model_ckpt] * num_model

            model_list = ModuleList(
                [model_initializer(model[0], kwargs, compile_kwargs) for i, kwargs in enumerate(model_kwargs)]
            )
            if num_model_ckpt > 0:
                for i, ckpt in enumerate(model_ckpt):
                    model_list[i] = load_th_or_pl_ckpt2md(model_list[i], ckpt, map_location=map_location)
        else:  # (4,5)
            num_model = max(1, num_model_ckpt) if num_model is None else max(1, num_model_ckpt, num_model)
            model_list = ModuleList(
                [model_initializer(model[0], model_kwargs[0], compile_kwargs) for _ in range(num_model)]
            )
            ckpts = model_ckpt * num_model if num_model_ckpt == 1 else model_ckpt
            for i, ckpt in enumerate(ckpts):
                model_list[i] = load_th_or_pl_ckpt2md(model_list[i], ckpt, map_location=map_location)
    else:  # (8,9)
        num_model = len(model)
        if num_model_kwargs != 1 and num_model_ckpt != 1:  # (9)
            assert num_model == num_model_kwargs == num_model_kwargs, "Please input the same # of kwargs and ckpts."
            model_list = ModuleList(
                [model_initializer(model[i], model_kwargs[i], compile_kwargs) for i in range(num_model)]
            )
            for i, ckpt in enumerate(model_ckpt):
                model_list[i] = load_th_or_pl_ckpt2md(model_list[i], ckpt, map_location=map_location)
        elif num_model_kwargs == 1 and num_model_ckpt == 1:  # (8)
            model_list = ModuleList(
                [model_initializer(model[i], model_kwargs[0], compile_kwargs) for i in range(num_model)]
            )
            for i in range(num_model):
                model_list[i] = load_th_or_pl_ckpt2md(model_list[i], model_ckpt[0], map_location=map_location)
        else:
            raise ValueError("If input m>1 models, # len(model_kwargs) == # len(model_ckpt)")
    return model_list


def optimizer_initializer(
    opt: Optional[ParsableOptimizer],
    opt_kwargs: Optional[Dict[str, Any]],
    model: Union[Module, Iterable[torch.Tensor], Iterable[Dict[str, Any]]],
    mock_none: bool = True,
):
    opt_kwargs = opt_kwargs or {}  # None to empty dict
    opt_kwargs = dict(opt_kwargs)
    if opt is None:
        optimizer = MockOptimizer() if mock_none else None
    elif isinstance(opt, Optimizer):
        optimizer = opt
    elif isinstance(opt, str):
        opt_kwargs.update({"params": model.parameters() if isinstance(model, Module) else model})
        optimizer = optimizer_resolver(opt, **opt_kwargs)
    elif isinstance(opt, partial):
        already_init_kwargs = opt.keywords
        cur_init_kwargs = {"params": model.parameters() if isinstance(model, Module) else model, **opt_kwargs}
        # the second will override the first
        optimizer = opt(**{**cur_init_kwargs, **already_init_kwargs})
    elif isinstance(opt, Callable):  # only miss paras
        params = model.parameters() if isinstance(model, Module) else model
        optimizer = opt(params)
    elif issubclass(opt, Optimizer):
        optimizer = opt(model.parameters() if isinstance(model, Module) else model, **opt_kwargs)
    else:
        raise TypeError(f"{type(opt)} is not valid optimizer init type")
    return optimizer


def many_optimizer_initializer(
    opt: Union[Optional[ParsableOptimizer], List[Optional[ParsableOptimizer]], None],
    opt_kwargs: Union[Optional[Dict[str, Any]], List[Optional[Dict[str, Any]]], None],
    models: Union[List[Module], ModuleList],
) -> List[Optional[Optimizer]]:
    num_model = len(models)

    if opt is None:
        return [MockOptimizer()] * num_model
    if isinstance(opt, List) and isinstance(opt_kwargs, List):  # N opt N opt_kwargs
        assert len(opt) == len(opt_kwargs) == num_model
    elif isinstance(opt, List) and not isinstance(opt_kwargs, List):  # N opts with 1 opt_kwargs
        assert len(opt) == num_model
        opt_kwargs = [opt_kwargs] * num_model
    elif not isinstance(opt, List) and isinstance(opt_kwargs, List):  # 1 opt with N opt_kwargs
        assert len(opt_kwargs) == num_model
        assert not isinstance(
            opt, Optimizer
        ), "Use (partial) class instead of instance to broadcast to multiple opt_kwargs dicts."
        opt = [opt] * num_model
    else:  # 1 opt and 1 opt_kwargs for all models
        if not isinstance(opt, Optimizer) and num_model != 1:
            raise TypeError(
                "Use optimizer name or (partial) class instead of instance to "
                "broadcast to multiple opt_kwargs dicts."
            )
        opt_kwargs = [opt_kwargs] * num_model
        opt = [opt] * num_model

    opt_kwargs = [each_opt_kwargs or {} for each_opt_kwargs in opt_kwargs]
    return [optimizer_initializer(each_opt, opt_kwargs[i], models[i]) for i, each_opt in enumerate(opt)]


def scheduler_initializer(
    sch: Optional[ParsableScheduler],
    sch_kwargs: Optional[Dict[str, Any]],
    opt: Optimizer,
    pl_sch_conf_kwargs: Optional[Dict[str, Any]],
):
    sch_kwargs = sch_kwargs or {}
    pl_sch_conf_kwargs = pl_sch_conf_kwargs or {}

    if sch is None or opt is None or isinstance(opt, MockOptimizer):
        scheduler = None
    elif isinstance(sch, LRScheduler):
        scheduler = sch
    elif isinstance(sch, str):
        scheduler = lr_scheduler_resolver(
            sch,
            opt,
            warmup_ratio_or_steps=sch_kwargs.get("warmup_ratio_or_steps", 0),
            num_training_steps=sch_kwargs.get("num_training_steps", None),
            return_cls=False,
            **sch_kwargs,
        )
    elif isinstance(sch, partial):
        already_init_kwargs = sch.keywords
        cur_init_kwargs = {"optimizer": opt, **sch_kwargs}
        # the second will override the first
        scheduler = sch(**{**cur_init_kwargs, **already_init_kwargs})
    elif isinstance(sch, Callable):
        scheduler = sch(opt)

    elif issubclass(sch, LRScheduler):
        scheduler = sch(opt, **sch_kwargs)
    else:
        raise TypeError(f"{type(sch)} is not valid scheduler init type")

    if len(pl_sch_conf_kwargs) > 0:
        scheduler = {"scheduler": scheduler, **pl_sch_conf_kwargs}
    return scheduler


def many_scheduler_initializer(
    sch: Union[List[Optional[ParsableScheduler]], Optional[ParsableScheduler], None],
    opts: Union[List[Optional[Optimizer]], Optional[Optimizer], None],
    sch_kwargs: Union[Optional[Dict[str, Any]], List[Optional[Dict[str, Any]]], None],
    pl_sch_conf_kwargs: Union[Dict[str, Any], List[Dict[str, Any]], None] = None,
) -> List[Optional[LRScheduler]]:
    """

    Args:
        sch:
        opts:
        sch_kwargs:
        pl_sch_conf_kwargs:

    Returns:

    """
    if not isinstance(opts, List) and isinstance(opts, Optimizer):
        opts = [opts]
    num_opt = len(opts)
    if sch is None:
        return [None] * num_opt

    if pl_sch_conf_kwargs is not None and isinstance(pl_sch_conf_kwargs, List):
        if len(pl_sch_conf_kwargs) == 1:  # broadcast one-sized [dict(k,v)] to all optimizers
            pl_sch_conf_kwargs = pl_sch_conf_kwargs * num_opt
        assert len(pl_sch_conf_kwargs) == num_opt
    else:  # wrap one Dict/None into a list to broadcast this sch_kwargs to all optimizers
        pl_sch_conf_kwargs = [pl_sch_conf_kwargs] * num_opt

    if not isinstance(sch_kwargs, List):
        sch_kwargs = [sch_kwargs]
    if not isinstance(sch, List):
        sch = [sch]

    len_sch = len(sch)
    len_sch_kwargs = len(sch_kwargs)
    if len_sch > 1 and len_sch_kwargs > 1:  # N vs N
        assert len_sch == len_sch_kwargs == num_opt
    elif len_sch > 1 and len_sch_kwargs == 1:  # N vs 1
        assert len_sch == num_opt
        sch_kwargs = sch_kwargs * num_opt

    elif len_sch == 1 and len_sch_kwargs > 1:  # 1 vs N
        assert len_sch_kwargs == num_opt
        assert not isinstance(
            sch[0], LRScheduler
        ), "Use (partial) class instead of instance to broadcast to multiple sch_kwargs dicts."
        sch = sch * num_opt
    elif len_sch == len_sch_kwargs == 1:  # 1 vs 1
        pass
    else:
        raise ValueError(f"The number of `sch`/`sch_kwargs` to initialize is invalid {len_sch}/{len_sch_kwargs}")

    sch_kwargs = [each_sch_kwargs or {} for each_sch_kwargs in sch_kwargs]
    return [
        scheduler_initializer(each_sch, sch_kwargs[i], opts[i], pl_sch_conf_kwargs[i]) for i, each_sch in enumerate(sch)
    ]
