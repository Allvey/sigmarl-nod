"""TorchRL transforms for typed Opinion information tensors."""

from torchrl.data import DiscreteTensorSpec
from torchrl.envs.transforms import DTypeCastTransform


class DiscreteDTypeCastTransform(DTypeCastTransform):
    """Cast a VMAS float info leaf and expose a real discrete TensorSpec.

    TorchRL 0.2.1's :class:`DTypeCastTransform` changes only the dtype of the
    original ``UnboundedContinuousTensorSpec``.  Its ``rand()`` consequently
    calls ``torch.randn`` with an integer/bool dtype.  Collector policy probing
    can reach that path, so replace the spec class as well as casting values.
    """

    def __init__(self, dtype_in, dtype_out, n: int, **kwargs) -> None:
        if type(n) is not int or n < 2:
            raise ValueError("A discrete Opinion info spec requires n >= 2.")
        self.n = n
        super().__init__(dtype_in=dtype_in, dtype_out=dtype_out, **kwargs)

    def _transform_spec(self, spec):
        return DiscreteTensorSpec(
            n=self.n,
            shape=spec.shape,
            device=spec.device,
            dtype=self.dtype_out,
        )
