import typing as t
from collections import deque
from functools import partial

import numpy as np
from stable_baselines3.common.callbacks import BaseCallback
from torch import nn


class Hook:
    """Wrapper for PyTorch forward hook mechanism."""

    def __init__(self, module: nn.Module, func: t.Callable):
        self.hook = None  # PyTorch's hook.
        self.module = module  # PyTorch layer to which the hook is attached to.
        self.func = func  # Function to call on each forward pass.
        self.register()

    def register(self):
        self.activation_data = deque(maxlen=1024)
        self.hook = self.module.register_forward_hook(partial(self.func, self))

    def remove(self):
        self.hook.remove()


def store_activation(hook, module, inp, outp):
    """Function intented to be called by a hook on a forward pass.

    Args:
        hook:    The hook object that generated the call.
        module:  The module on which the hook is registered.
        inp:     Input of the module.
        outp:    Output of the module.
    """
    hook.activation_data.append(outp.data.cpu().numpy())


def get_low_act(data, threshold=0.2):
    """Computes the proportion of activations that have value close to zero."""
    low_activation = ((-threshold <= data) & (data <= threshold))
    return np.count_nonzero(low_activation) / np.size(low_activation)


# Callback for periodic logging to tensorboard.
class LayerActivationMonitoring(BaseCallback):

    def _on_rollout_start(self) -> None:
        """Called after the training phase."""

        hooks = self.model.policy.features_extractor.hooks

        # Remove the hooks so that they don't get called for rollout collection.
        for h in hooks: h.remove()

        # Log last datapoint and statistics to tensorboard.
        for i, hook in enumerate(hooks):
            if len(hook.activation_data) > 0:
                data = hook.activation_data[-1]
                self.logger.record(f'diagnostics/activation_l{i}', data)
                self.logger.record(f'diagnostics/mean_l{i}', np.mean(data))
                self.logger.record(f'diagnostics/std_l{i}', np.std(data))
                self.logger.record(f'diagnostics/low_act_prop_l{i}', get_low_act(data))

    def _on_rollout_end(self) -> None:
        """Called before the training phase."""
        for h in self.model.policy.features_extractor.hooks: h.register()

    def _on_step(self):
        pass


def register_hooks(model):
    model.policy.features_extractor.hooks = [
        Hook(layer, store_activation)
        for layer in model.policy.features_extractor.cnn
        if isinstance(layer, nn.ReLU) or isinstance(layer, nn.LeakyReLU)]

