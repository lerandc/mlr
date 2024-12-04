import torch
from torch import nn

from mlr.database.utils import read_metadata
from mlr.models.torch.utils import read_sd_from_h5


class fixed_width_mlp_Encoder(nn.Module):
    def __init__(self, data_dim, latent_dim, hidden_dim=512, N_layers=2):
        super().__init__()
        self.layers = nn.ModuleList()
        self.layers.extend([nn.Linear(data_dim, hidden_dim), nn.ReLU()])

        for _ in range(N_layers - 1):
            self.layers.extend([nn.Linear(hidden_dim, hidden_dim), nn.ReLU()])

        self.layers.extend([nn.Linear(hidden_dim, latent_dim), nn.ReLU()])

        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, nonlinearity="relu")

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)

        return x


class fixed_width_mlp_Decoder(nn.Module):
    def __init__(self, data_dim, latent_dim, gaussian_dim, hidden_dim=512, N_layers=2):
        super().__init__()
        self.layers = nn.ModuleList()
        self.layers.extend(
            [
                nn.Linear(gaussian_dim, latent_dim),
                nn.ReLU(),
                nn.Linear(latent_dim, hidden_dim),
                nn.ReLU(),
            ]
        )

        for _ in range(N_layers - 1):
            self.layers.extend([nn.Linear(hidden_dim, hidden_dim), nn.ReLU()])

        self.layers.extend([nn.Linear(hidden_dim, data_dim)])

        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, nonlinearity="relu")

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)

        return 3*x


class VAE(nn.Module):
    def __init__(
        self,  
        data_dim=128**2,
        latent_dim=200,
        gaussian_dim=2,
        encoder=fixed_width_mlp_Encoder,
        decoder=fixed_width_mlp_Decoder,
        add_sigmoid=False,
        **kwargs,
    ):
        super().__init__()

        # encoder
        self.encoder = encoder(data_dim, latent_dim, **kwargs)
        self.decoder = decoder(data_dim, latent_dim, gaussian_dim, **kwargs)

        # latent mean and variance
        self.mean_layer = nn.Linear(latent_dim, gaussian_dim)
        self.logvar_layer = nn.Linear(latent_dim, gaussian_dim)

        nn.init.kaiming_normal_(self.mean_layer.weight)

        if add_sigmoid:
            self.decoder.layers.append(nn.Sigmoid())

    def encode(self, x):
        x = self.encoder(x)
        mean, logvar = self.mean_layer(x), self.logvar_layer(x)
        return (mean, logvar)

    def reparameterization(self, mean, var):
        epsilon = torch.randn_like(var)
        z = mean + var * epsilon
        return z

    def decode(self, x):
        return self.decoder(x)

    def forward(self, x):
        (mean, logvar) = self.encode(x)
        z = self.reparameterization(mean, logvar)
        x_hat = self.decode(z)
        return (x_hat, mean, logvar)

    @staticmethod
    def loss_function(x, x_hat, mean, log_var):
        reproduction_loss = nn.functional.mse_loss(x_hat, x, reduction="mean")
        KLD = -0.5 * torch.sum(1 + log_var - mean.pow(2) - log_var.exp())

        return reproduction_loss + KLD
    
    @staticmethod
    def kl_divergence(mean, log_var):
        return -0.5 * torch.sum(1 + log_var - mean.pow(2) - log_var.exp())/mean.shape[0]

    @staticmethod
    def reproduction_loss(x, x_hat):
        return nn.functional.mse_loss(x_hat, x, reduction="mean")

    @classmethod
    def load_ray_result(cls, folder, input_dim=256**2, add_sigmoid=False):
        config_path = folder.joinpath("params.json")
        weight_path = folder.joinpath("checkpoint_000000/weights.h5")

        model_config = read_metadata(config_path)
        state_dict = read_sd_from_h5(weight_path)

        arch_config = {
            k: model_config[k]
            for k in ("latent_dim", "gaussian_dim", "hidden_dim", "N_layers")
        }

        model = cls(data_dim=input_dim, **arch_config, add_sigmoid=add_sigmoid)
        model.load_state_dict(state_dict)

        return model
