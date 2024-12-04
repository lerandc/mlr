import torch


### LazyLinear subclasses


class InitiableLazyLinear(torch.nn.LazyLinear):
    def __init__(
        self,
        out_features,
        bias=True,
        device=None,
        dtype=None,
        requires_grad=True,
        init_method="kaiming_normal",
        **init_kwargs,
    ):
        self._requires_grad = requires_grad
        self._init_method = init_method
        self._init_kwargs = init_kwargs
        super().__init__(out_features, bias, device, dtype)

    def reset_parameters(self):
        self.requires_grad_(self._requires_grad)
        match self._init_method:
            case "kaiming_normal":
                torch.nn.init.kaiming_normal_(self.weight, **self._init_kwargs)
            case "orthogonal":
                torch.nn.init.orthogonal_(self.weight, **self._init_kwargs)


class OrthogonalLazyLinear(InitiableLazyLinear):
    def __init__(
        self,
        out_features,
        bias=True,
        device=None,
        dtype=None,
        requires_grad=True,
        **init_kwargs,
    ):
        super().__init__(
            out_features, bias, device, dtype, requires_grad, init_method="orthogonal"
        )
