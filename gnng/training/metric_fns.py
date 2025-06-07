import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.patches import Rectangle
from lightning.fabric.wrappers import _unwrap_objects


def noop_epoch_metric(trainer, model=None, stage=None, force_log_step=None, *args, **kwargs):
    return {}


def naive_epoch_metric_fn(trainer, model=None, stage=None, force_log_step=None, *args, **kwargs):
    metric_manager = trainer.get_metric_manager(stage)
    epoch_summary_dict = metric_manager.compute()
    return epoch_summary_dict


def unsup_epoch_metric_fn(trainer, model=None, stage=None, force_log_step=None, *args, **kwargs):
    metric_manager = trainer.get_metric_manager(stage)
    batch_out_manger = trainer.get_bo_manager(stage)

    batch_out_dict = batch_out_manger.compute(set_fix=False)
    # fetch all SSL embeddings and target from `metric_manager`
    embeddings = batch_out_dict["preds"]
    target = batch_out_dict["target"]
    old_mask = batch_out_dict["old_mask"]
    stage_mask = batch_out_dict["stage_mask"]

    # Non-parametric prediction based on methods like (Semi-) KMeans. labelled_mask is the available labels
    # CAUTION:
    #       1.  the `labelled_train_mask` from `train_batch_out_manger` has different node indices when the training
    #           step is performed in a node-shuffle way, like what `SAGENeighborLoader` does
    #       2.  So in sampling-based training, such style gives wrong known labels for SupervisedKMeans.
    #       3.  If Full-Batch and NO node permutation between training and validation node indices, this is correct.
    #       4.  Eventually, we record the training mask also from
    # Remedy:  important for inductive or sampling-based training, to implement later.
    #       1. Global node samples concatenation. Concatenate all accessible nodes and the adhered split/mask info

    train_labelled_mask = batch_out_manger.compute(query="train_labelled_mask", set_fix=False)["train_labelled_mask"]
    # you can only use training labels as (semi-)supervised signals
    res = model.predict(z=embeddings, target=target, y_access_mask=train_labelled_mask)
    preds = res["preds"]
    old_mask = old_mask.to(preds.device)
    target = target.to(preds.device)
    stage_mask = stage_mask.to(preds.device)
    metric_manager.reset()  # empty list of batch-level preds and target
    metric_manager.update(
        preds=res["preds"][stage_mask],
        target=target[stage_mask],
        old_mask=old_mask[stage_mask],
        # x=embeddings[stage_mask],
    )
    epoch_summary_dict = metric_manager.compute()
    return epoch_summary_dict


def gcd_unsup_epoch_metric_fn(trainer, model=None, stage=None, force_log_step=None, *args, **kwargs):
    esd = unsup_epoch_metric_fn(trainer, model, stage, force_log_step, *args, **kwargs)
    gcd_plugin(esd, trainer, model, stage, force_log_step, *args, **kwargs)
    return esd


def gcd_naive_epoch_metric_fn(trainer, model=None, stage=None, force_log_step=None, *args, **kwargs):
    esd = naive_epoch_metric_fn(trainer, model, stage, force_log_step, *args, **kwargs)
    gcd_plugin(esd, trainer, model, stage, force_log_step, *args, **kwargs)
    return esd


def gcd_plugin(esd, trainer, model=None, stage=None, force_log_step=None, *args, **kwargs):
    cm = esd.pop(f"{stage}/ConfusionMatrix")
    dcm = esd.pop(f"{stage}/DetectConfusionMatrix")
    old_classes = esd.pop(f"{stage}/old_classes").cpu().tolist()
    new_classes = esd.pop(f"{stage}/new_classes").cpu().tolist()
    loggers = trainer.fabric.loggers
    if len(loggers) > 0 and (trainer.current_epoch % 20 == 0 or stage == "test"):
        from gnng.training.ext_misc.loggers.wandb import cm2wtp_confusion_matrix

        key1 = f"{stage}/ConfusionMatrix"
        key2 = f"{stage}/DetectConfusionMatrix"
        key3 = f"{stage}/AvgPrototypeInfo"
        if stage == "test":
            num_test_epochs = trainer.num_test_epochs
            key1 = f"{key1}_TestEpoch={num_test_epochs}"
            key2 = f"{key2}_TestEpoch={num_test_epochs}"
            key3 = f"{key3}_TestEpoch={num_test_epochs}"

        esd[key1] = cm2wtp_confusion_matrix(cm, title=key1)
        esd[key2] = cm2wtp_confusion_matrix(
            dcm,
            class_names=[f"0_Old:{old_classes}", f"1_New:{new_classes}"],
            title=key2,
        )

        # log prototype adj as heatmap
        native_pt_model = _unwrap_objects([model])[0]
        proto_topo = getattr(native_pt_model.predictor, "proto_topo", None)
        proto_weight = getattr(native_pt_model.predictor, "proto_weight", None)
        if proto_topo is not None and proto_weight is not None:
            proto_weight = proto_weight.detach().cpu().numpy()
            proto_topo = proto_topo.detach().cpu().numpy()
            n_head, n_classes, _ = proto_topo.shape

            normed_proto_topo = normalize_slices(proto_topo)

            class_labels = kwargs.get("class_labels", None)
            # heat = wandb.plots.HeatMap(class_labels, y_labels=y_labels, matrix_values=catted.mean(0), show_text=True)
            # esd[key3] = make_topo_and_marginal_plt_v2(normed_proto_topo, proto_weight, class_labels)
            esd[key3] = make_topo_and_marginal_plt_v3(normed_proto_topo, proto_weight, class_labels, fig_name=key3)


def make_topo_and_marginal_plt(topo, marginal, fig_name=None, width_unit=5):
    topo = topo.cpu().numpy()
    marginal = marginal.cpu().numpy()
    H, C, C = topo.shape
    fig, axes = plt.subplots(2, H, figsize=(width_unit * H, width_unit), gridspec_kw=dict(height_ratios=[1, 9]))
    marginal = marginal.reshape(H, 1, -1)

    for h in range(H):
        if H == 1:
            ax1, ax2 = axes
        else:
            ax1 = axes[0, h]
            ax2 = axes[1, h]
        sns.heatmap(topo[h], annot=True, fmt=".2f", ax=ax1, square=True)
        heatmap = sns.heatmap(marginal[h], annot=True, fmt=".2f", ax=ax2, square=True)
        for i in range(C):
            rect = Rectangle((i, i), 1, 1, linewidth=3, edgecolor="green", fill=False)
            heatmap.add_patch(rect)
    fig_name = fig_name or ""
    fig.suptitle(f"{fig_name}")
    return fig


def make_topo_and_marginal_plt_v2(topo, marginal, class_labels=None, fig_name=None, width_unit=4):
    H, C, C = topo.shape
    marginal = marginal.reshape(H, 1, -1)
    catted = np.concatenate([marginal, np.zeros((H, 1, C)), topo], axis=1)
    class_labels = class_labels or [f"C_{i}" for i in range(C)]
    y_labels = ["marginal_weight", "N/A", *class_labels]
    fig, axes = plt.subplots(1, H, figsize=(width_unit * H, width_unit))
    if isinstance(axes, plt.Axes):
        axes = [axes]
    for h in range(H):
        ax = axes[h]
        heatmap = sns.heatmap(
            catted[h],
            annot=True,
            fmt=".2f",
            ax=ax,
            square=True,
            vmax=1.2,
            vmin=-0.2,
            xticklabels=class_labels,
            yticklabels=y_labels,
        )
        for i in range(C):
            rect = Rectangle((i, i + 2), 1, 1, linewidth=2, edgecolor="green", fill=False)
            heatmap.add_patch(rect)

        rect = Rectangle((0, 0), C, 1, linewidth=2, edgecolor="cyan", fill=False)
        heatmap.add_patch(rect)

    fig_name = fig_name or ""
    fig.suptitle(f"{fig_name}")
    return fig


def make_topo_and_marginal_plt_v3(topo, marginal, class_labels=None, fig_name=None, width_unit=4):
    import plotly.express as px
    import plotly.graph_objects as go

    H, C, C = topo.shape
    marginal = marginal.reshape(H, 1, -1)
    catted = np.concatenate([marginal, np.zeros((H, 1, C)), topo], axis=1)
    class_labels = class_labels or [f"C_{i}" for i in range(C)]
    y_labels = ["marginal_weight", "N/A", *class_labels]
    fig = px.imshow(catted.mean(0), text_auto=True, x=class_labels, y=y_labels)
    fig_name = fig_name or ""
    fig.update_layout(title=fig_name, yaxis_scaleanchor="x")
    return fig


def normalize_slices(array):
    # (H,M,N) 每个MxN数组除以各自的最大值
    max_values = np.max(array, axis=(1, 2), keepdims=True)
    normalized_array = array / max_values

    return normalized_array
