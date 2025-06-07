from typing import Any
import torch


class Callback:
    """My customized trainer based on Lightning.Fabric supports the following hooks
    - on_train_epoch_start
    - on train_epoch_end
    - on_train_batch_start
    - on_train_batch_end
    - on_before_backward
    - on_after_backward
    - on_before_zero_grad
    - on_before_optimizer_step
    - on_validation_model_eval
    - on_validation_model_train
    - on_validation_epoch_start
    - on_validation_epoch_end
    - on_validation_batch_start
    - on_validation_batch_end
    - on_test_batch_start:
    - on_test_batch_end:
    - on_test_model_train:Sets the model to train during the test  loop.
    - on_test_model_eval: Sets the model to eval during the test loop
    - on_test_epoch_start
    - on_test_epoch_end
    """

    @property
    def state_key(self) -> str:
        """Identifier for the state of the callback.

        Used to store and retrieve a callback's state from the checkpoint dictionary by
        ``checkpoint["callbacks"][state_key]``. Implementations of a callback need to provide a unique state key if 1)
        the callback has state and 2) it is desired to maintain the state of multiple instances of that callback.

        """
        return self.__class__.__qualname__

    def _generate_state_key(self, **kwargs: Any) -> str:
        """Formats a set of key-value pairs into a state key string with the callback class name prefixed. Useful for
        defining a :attr:`state_key`.

        Args:
            **kwargs: A set of key-value pairs. Must be serializable to :class:`str`.

        """
        return f"{self.__class__.__qualname__}{repr(kwargs)}"

    def setup(self, trainer: Any, model: "torch.nn.Module", stage: str) -> None:
        """Called when fit, validate, test, predict, or tune begins."""
