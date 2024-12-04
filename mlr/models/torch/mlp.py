from functools import reduce

import torch
from torch import nn

from mlr.models.torch.layers import InitiableLazyLinear


class classifier(nn.Module):
    def __init__(
        self,
        configs,
        N_classes,
        activation="relu",
        final_activation="softmax",
        init_method="kaiming_normal",
        device=None,
    ):
        super(classifier, self).__init__()
        match activation:
            case "relu":
                nonlinearity = nn.ReLU
            case _:
                raise NotImplementedError

        self.N_classes = N_classes
        self.N_layers = len(configs)
        self.layers = nn.ModuleList()
        self.layers.extend(
            reduce(
                lambda x, y: x + y,
                [[nn.Linear(**d, device=device), nonlinearity()] for d in configs],
            )
        )

        self.layers.append(
            InitiableLazyLinear(
                N_classes,
                bias=False,
                device=device,
                init_method=init_method,
                nonlinearity=activation,
            )
        )

        match final_activation:
            case "softmax":
                self.layers.append(nn.Softmax(dim=1))
            case _:
                pass

        for m in self.modules():
            match m:
                case InitiableLazyLinear() | nn.LazyLinear():
                    pass
                case nn.Linear():
                    pass
                    # match init_method:
                    #     case "kaiming_normal":
                    #         nn.init.kaiming_normal_(m.weight, nonlinearity=activation)
                    #     case _:
                    #         raise NotImplementedError
                case _:
                    pass

    @property
    def input_shape(self):
        return self.layers[0].in_features

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)

        return x

    def get_latent_features(self, x):
        for layer in self.layers[:-2]:
            x = layer(x)

        return x
