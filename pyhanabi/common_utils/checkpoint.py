import inspect
import importlib
import torch
try:
    from torch.optim.lr_scheduler import LRScheduler # PyTorch 2.0+
except ImportError:
    from torch.optim.lr_scheduler import _LRScheduler as LRScheduler # PyTorch 1.X


def filter_args_call(func, args, kwargs):
    '''Filter out unwanted kwargs and those covered by positional args, then call the function.'''
    sig = inspect.signature(func)
    if any(param.kind == param.VAR_KEYWORD for param in sig.parameters.values()):
        # the signature contains definitions like **kwargs, so there is no need to filter
        return func(*args, **kwargs)
    n_args = len(args)
    filter_keys = [
        param.name for i, param in enumerate(sig.parameters.values())
        if (param.kind == param.POSITIONAL_OR_KEYWORD and i >= n_args) or param.kind == param.KEYWORD_ONLY
    ]
    filtered_kwargs = {key: kwargs[key] for key in filter_keys if key in kwargs}
    return func(*args, **filtered_kwargs)


def build_object_from_config(config: dict, *args, parent_cls=None, **kwargs):
    '''Build an object of class config['cls'], kwargs override config.
    Unrecognized config and kwargs are ignored, so be careful with spelling.'''
    config = config.copy()
    config.update(kwargs)
    pkg, cls_name = config['cls'].rsplit(".", 1)
    cls_type = getattr(importlib.import_module(pkg), cls_name)
    if parent_cls is not None:
        assert issubclass(cls_type, parent_cls), f'| {cls_type} is not subclass of {parent_cls}.'
    return filter_args_call(cls_type, args, config)


def build_lr_scheduler_from_config(scheduler_config, optimizer):
    '''Build recursively and provide optimizer to each scheduler class if needed.'''

    def helper(config):
        if isinstance(config, list):
            return [helper(s) for s in config]
        elif isinstance(config, dict):
            config = {k: helper(v) for k, v in config.items()}
            if 'cls' in config:
                if (
                    config["cls"] == "torch.optim.lr_scheduler.ChainedScheduler"
                    and scheduler_config["cls"] == "torch.optim.lr_scheduler.SequentialLR"
                ):
                    raise ValueError(f"ChainedScheduler cannot be part of a SequentialLR.")
                obj = build_object_from_config(config, optimizer=optimizer, parent_cls=LRScheduler)
                return obj
            return config
        else:
            return config

    result = helper(scheduler_config)
    assert isinstance(result, LRScheduler), '| "cls" not found in scheduler_config.'
    return result


def simulate_lr_scheduler(optimizer_config, scheduler_config, step_count, num_param_groups=1):
    '''Return the state_dict of the scheduler after step_count steps.'''
    optimizer = build_object_from_config(
        optimizer_config,
        [{'params': torch.nn.Parameter(), 'initial_lr': optimizer_config['lr']} for _ in range(num_param_groups)],
        parent_cls=torch.optim.Optimizer,
    )
    scheduler = build_lr_scheduler_from_config(scheduler_config, optimizer)
    scheduler.optimizer._step_count = 1
    for _ in range(step_count):
        scheduler.step()
    return scheduler.state_dict()