import pickle
from pathlib import Path
from typing import List, Optional

import jax
import optax
from tqdm import tqdm

from fourierflow.callbacks import Callback

from .jax_callback_hook import TrainerCallbackHookMixin


class JAXTrainer(TrainerCallbackHookMixin):
    def __init__(
        self,
        max_epochs,
        limit_train_batches=None,
        callbacks: Optional[List[Callback]] = None,
    ):
        self.max_epochs = max_epochs
        self.limit_train_batches = limit_train_batches
        self.callbacks = callbacks or []

    def fit(self, routine, builder):
        params = routine.init()
        opt_state = routine.optimizer.init(params)

        @jax.jit
        def step(params, opt_state, inputs, outputs):
            loss_value, grads = jax.value_and_grad(
                routine.loss_fn)(params, inputs, outputs)
            updates, opt_state = routine.optimizer.update(
                grads, opt_state, params)
            params = optax.apply_updates(params, updates)
            return params, opt_state, loss_value

        for epoch in range(self.max_epochs):
            train_batches = iter(builder.train_dataloader())
            with tqdm(train_batches, unit="batch") as tepoch:
                for i, (inputs, outputs) in enumerate(tepoch):
                    tepoch.set_description(f"Epoch {epoch}")
                    params, opt_state, loss_value = step(
                        params, opt_state, inputs, outputs)
                    tepoch.set_postfix(loss=loss_value.item())

                    if self.limit_train_batches and i >= self.limit_train_batches:
                        break

            print('Validating...')
            validate_batches = iter(builder.val_dataloader())
            for i, batch in tqdm(enumerate(validate_batches)):
                loss, logs = routine.valid_step(params, **batch)
                print(logs)

            # checkpoint
            path = Path(f'checkpoints/jax/params_{epoch}.pkl')
            path.parent.mkdir(parents=True, exist_ok=True)
            with open(path, 'wb') as f:
                pickle.dump(params, f)
