import torch_geometric.transforms as pyg_transform
from torch_geometric.transforms import BaseTransform

from typing import Optional, Union, Any, Dict, List
from gnng.typing import ParsableTransform

from gnng.resolver_initializer import resolver, initializer, normalize_string


def transform_resolver(query: Union[Any, str], return_cls=False, *args, **kwargs):
    import gnng.data.transforms as gnng_transforms

    base_cls = BaseTransform
    base_cls_repr = "pyg-compatible transform resolver"
    ts = [t for t in vars(pyg_transform).values() if isinstance(t, type) and issubclass(t, base_cls)]
    ts_dict = {
        normalize_string(k): t
        for k, t in vars(gnng_transforms).items()
        if isinstance(t, type) and issubclass(t, base_cls)
    }
    return resolver(ts, ts_dict, query, base_cls, base_cls_repr, return_cls, *args, **kwargs)


def transform_initializer(transform: Optional[ParsableTransform], transform_kwargs: Optional[Dict[str, Any]]):
    return initializer(transform, BaseTransform, transform_resolver, transform_kwargs)


def many_transform_initializer(
    transform: Optional[Union[str, ParsableTransform, List[ParsableTransform]]] = None,
    transform_kwargs: Optional[Union[Dict[str, Any], List[Dict[str, Any]]]] = None,
    sep: str = "-",
    return_identity: bool = False,
):
    """Parse a string joining different transform names that come from :module:`torch_geometric.transforms`. For
    example, 'ToSparseTensor,ToUndirected' means a transform composed by
    :class:`torch_geometric.transforms.ToSparseTensor` and :class:`torch_geometric.transforms.ToUndirected`

    Args:
        transform (string, Callable, optional): a string determining a transform
        transform_kwargs ():
        sep (string): the separator symbol of :obj:`transform`
        return_identity (bool): If True, return f(x)=x else None
    Returns:
        composed_transform (torch_geometric.transforms.Compose): the
            transform

    """

    if transform is None:
        if return_identity:
            transform = ["Identity"]
        else:
            return None
    elif isinstance(transform, str):
        transform = transform.split(sep)

    elif isinstance(transform, ParsableTransform):
        transform = [transform]
    elif isinstance(transform, List):
        transform = transform
    else:
        raise TypeError

    if transform_kwargs is None:
        transform_kwargs = [{} for _ in range(len(transform))]

    elif isinstance(transform_kwargs, List):
        assert len(transform_kwargs) == len(transform), (
            f"The kwargs list length {len(transform_kwargs)} " f"should match with the number of transforms {transform}"
        )
    elif isinstance(transform_kwargs, Dict):
        assert (
            len(transform) == 1
        ), f"Only got one transform kwargs dict, but there are {len(transform_kwargs)} transforms to initialize"
        transform_kwargs = [transform_kwargs]
    else:
        raise TypeError("transforms should be None or List of arg dicts")

    if len(transform) > 1:
        transform_classes = [
            transform_initializer(one_trans, transform_kwargs[i]) for i, one_trans in enumerate(transform)
        ]
        transform = pyg_transform.Compose(transform_classes)
    else:
        transform = transform_initializer(transform[0], transform_kwargs[0])
    return transform
