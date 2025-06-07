"""Append `return_cls` arg to control the resolver initialization behaviour.

This file is substantially from `torch_geometric.resolver_initializer.py`

"""

import inspect
from functools import partial
from typing import Any, Dict, List, Optional, Union
from gnng.utils import normalize_string


def resolver(
    classes: Optional[List[Any]],
    class_dict: Dict[str, Any],
    query: Optional[Union[Any, str]],
    base_cls: Optional[Any],
    base_cls_repr: Optional[str],
    return_cls: bool = False,
    *args,
    **kwargs,
):
    if not isinstance(query, str):
        return query

    query_repr = normalize_string(query)
    if base_cls_repr is None:
        base_cls_repr = base_cls.__name__ if base_cls else ""
    base_cls_repr = normalize_string(base_cls_repr)

    for key_repr, cls in class_dict.items():
        if query_repr == normalize_string(key_repr):
            if inspect.isclass(cls) and not return_cls:
                obj = cls(*args, **kwargs)
                return obj
            return cls

    for cls in classes:
        cls_repr = normalize_string(cls.__name__)
        if query_repr in [cls_repr, cls_repr.replace(base_cls_repr, "")]:
            if inspect.isclass(cls) and not return_cls:
                obj = cls(*args, **kwargs)
                return obj
            return cls

    choices = {cls.__name__ for cls in classes} | set(class_dict.keys())
    raise ValueError(f"Could not resolve '{query}' among choices {choices}")


def initializer(query: Any, base_cls: Any, base_cls_resolver: Any, base_cls_kwargs: Optional[Dict[str, Any]]):
    base_cls_kwargs = base_cls_kwargs or {}
    if query is None:
        return None
    elif isinstance(query, base_cls):
        instance = query
    elif isinstance(query, str):
        instance = base_cls_resolver(query, **base_cls_kwargs)
    elif isinstance(query, partial):
        already_init_kwargs = query.keywords
        instance = query(**{**base_cls_kwargs, **already_init_kwargs})
    elif issubclass(query, base_cls):
        instance = query(**base_cls_kwargs)
    else:
        raise TypeError(f"{type(query)} is not valid {base_cls} init type")
    return instance
