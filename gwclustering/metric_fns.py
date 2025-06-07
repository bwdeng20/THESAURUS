def get_clean_key(k):
    return (
        k.replace("test/", "")
        .replace("_ckpt=valid_ClusteringAccuracy", "")
    )


def naive_epoch_metric_fn4clustering(trainer, model=None, stage=None, force_log_step=None, *args, **kwargs):
    metric_manager = trainer.get_metric_manager(stage)
    epoch_summary_dict = metric_manager.compute()
    return epoch_summary_dict


def noop_epoch_metric(trainer, model=None, stage=None, force_log_step=None, *args, **kwargs):
    return {}


def unsup_epoch_metric_fn4clustering_main(trainer, model=None, stage=None, force_log_step=None, *args, **kwargs):
    metric_manager = trainer.get_metric_manager(stage)
    batch_out_manger = trainer.get_bo_manager(stage)

    batch_out_dict = batch_out_manger.compute(set_fix=False)
    # fetch all SSL embeddings and target from `metric_manager`
    unsup_embeddings = batch_out_dict["preds"]
    target = batch_out_dict["target"]

    res = model.predict(
        z=unsup_embeddings,
        target=target,
    )
    preds = res["preds"]
    target = target.to(preds.device)
    metric_manager.reset()  # empty list of batch-level preds and target
    metric_manager.update(preds=res["preds"], target=target)
    epoch_summary_dict = metric_manager.compute()
    return epoch_summary_dict
