from typing import Callable, Type, TypeVar, Union, cast

import torch
from allennlp.common import Params, Registrable
from torch.optim.lr_scheduler import _LRScheduler

T = TypeVar("T", bound="Scheduler")


class Scheduler(Registrable, _LRScheduler):
    @classmethod
    def from_params(cls: Type[T],
                    params: Params,
                    constructor_to_call: Callable[..., T] = None,
                    constructor_to_inspect: Union[Callable[..., T], Callable[[
                        T], None]] = None,
                    **extras):
        as_registrable = cast(Type[Registrable], cls)
        default_to_first_choice = as_registrable.default_implementation is not None
        choice = params.pop_choice(
            "type",
            choices=as_registrable.list_available(),
            default_to_first_choice=default_to_first_choice)
        subclass, constructor_name = as_registrable.resolve_class_name(choice)
        if not constructor_name:
            constructor_to_inspect = subclass.__init__
            constructor_to_call = subclass  # type: ignore
        else:
            constructor_to_inspect = cast(
                Callable[..., T], getattr(subclass, constructor_name))
            constructor_to_call = constructor_to_inspect

        return subclass(**params, **extras)


Registrable._registry[Scheduler] = {
    'step_lr': (torch.optim.lr_scheduler.StepLR, None),
}
