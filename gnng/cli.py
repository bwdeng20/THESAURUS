import inspect
from typing import Any, Callable, Dict, List, Optional, Type, Union, Sequence
import logging
import colorlog
from jsonargparse._cli import ComponentType, DictComponentsType, ComponentsType
from jsonargparse._cli import (
    ActionConfigFile,
    _ActionPrintConfig,
    remove_actions,
    ArgumentParser,
    default_config_option_help,
    Namespace,
    dict_to_namespace,
    get_doc_short_description,
    default_config_option_help,
)

from jsonargparse._cli import _add_subcommands, _add_component_to_parser, get_help_str, _run_component


class CLI:
    def __init__(
        self,
        components: ComponentsType = None,
        config_help: str = default_config_option_help,
        set_defaults: Optional[Dict[str, Any]] = None,
        as_positional: bool = True,
        fail_untyped: bool = True,
        parser_class: Type[ArgumentParser] = ArgumentParser,
        **kwargs,
    ):
        caller = inspect.stack()[1][0]

        if components is None:
            module = inspect.getmodule(caller).__name__  # type: ignore
            components = [
                v
                for v in caller.f_locals.values()
                if ((inspect.isclass(v) or callable(v)) and getattr(inspect.getmodule(v), "__name__", None) == module)
            ]
            if len(components) == 0:
                raise ValueError(
                    "Either components parameter must be given or there must be at least one "
                    "function or class among the locals in the context where CLI is called."
                )

        if isinstance(components, list) and len(components) == 1:
            components = components[0]

        elif not components:
            raise ValueError("components parameter expected to be non-empty")

        if isinstance(components, list):
            unexpected = [c for c in components if not (inspect.isclass(c) or callable(c))]
        elif isinstance(components, dict):
            ns = dict_to_namespace(components)
            unexpected = [c for c in ns.values() if not (inspect.isclass(c) or callable(c))] # diff1
        else:
            unexpected = [c for c in [components] if not (inspect.isclass(c) or callable(c))]
        if unexpected:
            raise ValueError(f"Unexpected components, not class or function: {unexpected}")

        parser = parser_class(default_meta=False, **kwargs)
        parser.add_argument("--config", action=ActionConfigFile, help=config_help)

        if not isinstance(components, (list, dict)):
            _add_component_to_parser(components, parser, as_positional, fail_untyped, config_help)
            if set_defaults is not None:
                parser.set_defaults(set_defaults)

            self.single_component = True

        elif isinstance(components, list):
            components = {c.__name__: c for c in components}
            self.single_component = False
            _add_subcommands(components, parser, config_help, as_positional, fail_untyped)
        if set_defaults is not None:
            parser.set_defaults(set_defaults)


        self.components = components
        self.parser = parser
        self._cfg = None

    @property
    def cfg(self):
        return self._cfg

    def update_cfg(self, cfg_from, cfg_to):
        """Now we can hack the raw cfg to change the process from py side instead of config files and cli"""
        return self.parser.merge_config(cfg_from, cfg_to)

    def parse_args(
        self,
        args: Optional[Sequence[str]] = None,
        namespace: Optional[Namespace] = None,
        env: Optional[bool] = None,
        defaults: bool = True,
        with_meta: Optional[bool] = None,
        **kwargs,
    ):

        cfg = self.parser.parse_args(
            args=args, namespace=namespace, env=env, defaults=defaults, with_meta=with_meta, **kwargs
        )
        self._cfg = cfg.clone()
        return cfg

    def instantiate_and_run(self, cfg=None, cfg_from=None, init_from=None):
        """
        cfg:    The cfg to instantiate run. If None, use the default parsed. If you want to change the cfg on the fly,
                This arg is quite handy.
        cfg_from: To merge some new configs to the default `cfg` or `self._cfg`
        init_from: To change the initialized nested Namespace. Caution! Only replace those args not important to log
        """
        cfg = cfg or self._cfg
        if cfg_from is not None:
            cfg = self.update_cfg(cfg_from, cfg_to=cfg)

        # construct the instances to run the given function(s) stored in `self.component`
        init = self.parser.instantiate_classes(cfg)
        init_from = init_from or {}
        for k, v in init_from.items():
            init[k] = v

        if self.single_component:
            return _run_component(self.components, init)

        else:
            components_ns = dict_to_namespace(self.components)
            subcommand = init.get("subcommand")
            while isinstance(init.get(subcommand), Namespace) and isinstance(init[subcommand].get("subcommand"), str):
                subsubcommand = subcommand + "." + init[subcommand].get("subcommand")
                if subsubcommand in components_ns:
                    subcommand = subsubcommand
                else:
                    break
            component = components_ns[subcommand]
            return _run_component(component, init.get(subcommand))


# def _run_component(component, cfg, **kwargs):
#     """
#     kwargs: additional keyword arguments to run components (functions)
#     """
#     cfg.pop("config", None)
#     if not inspect.isclass(component):
#         return component(**cfg, **kwargs)
#     subcommand = cfg.pop("subcommand")
#     if not subcommand:
#         return component(**cfg)
#     subcommand_cfg = cfg.pop(subcommand, {})
#     subcommand_cfg.pop("config", None)
#     component_obj = component(**cfg, **kwargs)
#     return getattr(component_obj, subcommand)(**subcommand_cfg, **kwargs)
#

def color_logger(logger):
    stream_handler = logging.StreamHandler()
    stream_handler.setLevel("DEBUG")
    log_colors = {
        "DEBUG": "white",
        "INFO": "green",
        "WARNING": "yellow",
        "ERROR": "red",
        "CRITICAL": "purple",
    }
    fmt_string = "%(log_color)s[%(asctime)s][%(levelname)s]%(message)s"
    fmt = colorlog.ColoredFormatter(fmt_string, log_colors=log_colors)
    stream_handler.setFormatter(fmt)

    logger.setLevel("DEBUG")
    logger.propagate = 0
    logger.addHandler(stream_handler)
    return logger
