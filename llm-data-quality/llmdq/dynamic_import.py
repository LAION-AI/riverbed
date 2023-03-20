from typing import Dict
import importlib
from copy import deepcopy
from llmdq.config import Config


def instantiate_class_from_config(config: Config) -> Dict[str, list]:
    config = deepcopy(config)

    obj_map = {}
    for component_name in ["scorer", "clustering"]:
        if component_name not in obj_map:
            obj_map[component_name] = []

        for args in getattr(config, component_name):
            impl = args.pop('_impl_')
            try:
                _cls = getattr(importlib.import_module(f'llmdq.{component_name}.implementation'), impl)
            except AttributeError:
                raise Exception(f"{impl} cannot be found in module llmdq.{component_name}.implementation")
            obj_map[component_name].append(_cls(**args))
    return obj_map
