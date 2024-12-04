from abc import ABC, abstractmethod

import torch
from torch import nn

from mlr.models.torch.layers import OrthogonalLazyLinear
from mlr.models.torch.mlp import classifier



class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size1, kernel_size2):
        super().__init__()
        self.conv1 = nn.Conv2d(
            in_channels, out_channels, kernel_size1, padding=(kernel_size1 - 1) // 2
        )
        self.conv2 = nn.Conv2d(
            out_channels, out_channels, kernel_size2, padding=(kernel_size2 - 1) // 2
        )
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.relu1 = nn.ReLU()
        self.relu2 = nn.ReLU()
        self.conv_add = nn.Conv2d(
            in_channels, out_channels, kernel_size=1
        )  # Expands/contracts input image in the filter dimension for the ReLU skip connection

        # Copied from torchvision to correctly initialize layers
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        # Store skip connection
        skip = self.conv_add(x)

        # Conv sequence 1
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu1(out)

        # Conv sequence 2
        out = self.conv2(out)
        out = self.bn2(out)

        # Add back in skip, ReLU, return
        return self.relu2(out + skip)


class DoubleResidualBlock(nn.Module):
    def __init__(
        self, in_channels, out_channels, kernel_size1, kernel_size2, stride, bias=False
    ):
        super().__init__()

        self.conv1 = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size1,
            stride=stride,
            padding=(kernel_size1 - 1) // 2,
            bias=bias,
        )
        self.conv2 = nn.Conv2d(
            out_channels,
            out_channels,
            kernel_size2,
            padding=(kernel_size2 - 1) // 2,
            bias=bias,
        )

        self.bn1 = nn.BatchNorm2d(in_channels)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.relu1 = nn.ReLU()
        self.relu2 = nn.ReLU()
        self.conv_add = nn.Conv2d(
            in_channels, out_channels, stride=stride, kernel_size=1, bias=bias
        )  # Expands/contracts input image in the filter dimension for the ReLU skip connection

        self.conv3 = nn.Conv2d(
            out_channels,
            out_channels,
            kernel_size1,
            padding=(kernel_size1 - 1) // 2,
            bias=bias,
        )
        self.conv4 = nn.Conv2d(
            out_channels,
            out_channels,
            kernel_size2,
            padding=(kernel_size2 - 1) // 2,
            bias=bias,
        )
        self.bn3 = nn.BatchNorm2d(out_channels)
        self.bn4 = nn.BatchNorm2d(out_channels)
        self.relu3 = nn.ReLU()
        self.relu4 = nn.ReLU()

        # Copied from torchvision to correctly initialize layers
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight)

            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        ## unit 1
        # bn + relu, save out for skip
        # conv2D
        # BN + Relu
        # conv 2D
        out = self.bn1(x)
        out = self.relu1(out)
        skip = self.conv_add(out)
        out = self.conv1(out)

        out = self.bn2(out)
        out = self.relu2(out)
        out = self.conv2(out)

        # add skip, save out for next skip
        out = out + skip
        skip = out

        ## unit 2
        # bn + relu
        # conv2D
        # bn + relu
        # conv2D
        # add skip
        out = self.bn3(out)
        out = self.relu3(out)
        out = self.conv3(out)

        out = self.bn4(out)
        out = self.relu4(out)
        out = self.conv4(out)

        out = out + skip

        return out


class UpsampleBlock(nn.Module):
    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size1,
        kernel_size2,
        pool_size,
        bias=False,
    ):
        super().__init__()
        self.conv1 = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size1,
            padding=(kernel_size1 - 1) // 2,
            bias=bias,
        )
        self.conv2 = nn.Conv2d(
            out_channels,
            out_channels,
            kernel_size2,
            padding=(kernel_size2 - 1) // 2,
            bias=bias,
        )
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.relu1 = nn.ReLU()
        self.relu2 = nn.ReLU()

        self.upsample = nn.Upsample(scale_factor=pool_size)

        # Copied from torchvision to correctly initialize layers
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    ### Upsamples are performed as: BN -> relu -> upsample2D
    def forward(self, x, skip=None):
        out = self.upsample(x)
        if skip is not None:
            out = torch.cat((out, skip), dim=1)

        out = self.conv1(out)
        out = self.bn1(out)
        out = self.relu1(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu2(out)

        return out


class InitialBlock(nn.Module):
    def __init__(self, out_channels, kernel_size, stride=2, **kwargs):
        super().__init__()
        self.bn_data = nn.BatchNorm2d(1)
        self.bn = nn.BatchNorm2d(out_channels)
        self.conv = nn.Conv2d(
            1,
            out_channels,
            kernel_size,
            stride=stride,
            padding=(kernel_size - 1) // 2,
            **kwargs,
        )
        self.relu = nn.ReLU()

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        out = self.bn_data(x)
        out = self.conv(out)
        out = self.bn(out)
        out = self.relu(out)
        return out


class UNet(nn.Module):
    """Version from Katherine's receptive field project."""

    def __init__(self, num_blocks, filter_sequence, max_pool_sequence, num_classes=2):
        # num_blocks: number of residual blocks in network
        # filter_sequence: list of filter sizes

        """
        D0     ->     U2
          D1   ->   U1
            D2 -> U0
               BN


        """

        super(UNet, self).__init__()
        self.downs = nn.ModuleList()
        self.ups = nn.ModuleList()

        self.pools = nn.ModuleList()
        self.upsamples = nn.ModuleList()

        # creates down and pooling layers
        in_channels = 1
        for i in range(num_blocks):
            self.downs.append(ResidualBlock(in_channels, filter_sequence[i], 3, 3))
            in_channels = filter_sequence[i]
            self.pools.append(
                nn.MaxPool2d(
                    kernel_size=max_pool_sequence[i], stride=max_pool_sequence[i]
                )
            )

        # creates up and upsampling layers
        for i in reversed(range(num_blocks)):
            self.ups.append(
                ResidualBlock(
                    filter_sequence[i] + filter_sequence[i + 1],
                    filter_sequence[i],
                    3,
                    3,
                )
            )  # The 2*filters in the input channels refers to the extra channels from the concat layer
            self.upsamples.append(nn.Upsample(scale_factor=max_pool_sequence[i]))

        # "bottleneck" or middle part at bottom of U
        self.bottleneck = ResidualBlock(
            filter_sequence[num_blocks - 1], filter_sequence[num_blocks], 3, 3
        )

        # final convolution with 1x1 kernel
        self.final_conv = nn.Conv2d(filter_sequence[0], num_classes, kernel_size=1)

        self.num_blocks = num_blocks

    def forward(self, x):
        skips = []  # empty array to store skip connections

        for i in range(self.num_blocks):
            x = self.downs[i](x)
            skips.append(x)
            x = self.pools[i](x)

        x = self.bottleneck(x)
        skips = skips[
            ::-1
        ]  # reverse skips array because we want to work with latest one first

        for idx in range(self.num_blocks):
            x = self.upsamples[idx](x)
            skip = skips[idx]
            concat_skip = torch.cat((skip, x), dim=1)
            x = self.ups[idx](concat_skip)

        out = self.final_conv(x)

        return out


class UNet_v2(nn.Module):
    """Altered version of arch. from above, to be able to conform to architecture generated by segmenationmodels, as used
    in the CZST project."""

    def __init__(
        self,
        num_blocks,
        initial_config,
        initial_pool_config,
        encoder_configs,
        decoder_configs,
        final_config,
        final_conv_config,
    ):
        # num_blocks: number of residual blocks in network
        # filter_sequence: list of filter sizes

        """
        Di     ->     U2 Uf
          D0   ->   U1
            D1 -> U0
               D2

        Di -> U3 Uf
        D0 -> U2
        D1 -> U1
        D2 -> U0
           D3

        """

        """
        LRD Feb 26 2024

        Next thing to do: pass in individual configurations to the conv layers in the residual blocks,
        so that the first encoding block can have a different stride size.


        """
        super(UNet_v2, self).__init__()
        self.num_blocks = num_blocks
        self.initial_block = InitialBlock(**initial_config)
        self.pool = nn.MaxPool2d(**initial_pool_config)

        # creates encoder blocks
        self.downs = nn.ModuleList([DoubleResidualBlock(**c) for c in encoder_configs])

        ## creates decoder blocks
        self.ups = nn.ModuleList([UpsampleBlock(**c) for c in decoder_configs])

        ## final decoder block
        self.final_decoder = UpsampleBlock(**final_config)
        self.final_conv = nn.Conv2d(**final_conv_config)
        self.softmax = nn.Softmax(1)

        ## TODO: init final conv block

    def forward(self, x):
        skips = {}  # empty map to store skip connections

        # process initial block and pool
        x = self.initial_block(x)
        skips[self.num_blocks - 1] = x
        x = self.pool(x)
        # process encoding blocks
        for i in range(self.num_blocks):
            x = self.downs[i](x)
            if i < self.num_blocks - 1:
                skips[self.num_blocks - i - 2] = x

        for i in range(self.num_blocks):
            x = self.ups[i](x, skips[i])

        x = self.final_decoder(x)

        x = self.final_conv(x)
        out = self.softmax(x)
        return out


class BaseProjector(nn.Module, ABC):
    def __init__(
        self,
        num_blocks,
        initial_config,
        initial_pool_config,
        encoder_configs,
        mlp_config,
    ):
        nn.Module.__init__(self)
        self.num_blocks = num_blocks
        self.initial_block = InitialBlock(**initial_config)
        self.pool = nn.MaxPool2d(**initial_pool_config)

        # creates encoder blocks
        self.downs = nn.ModuleList([DoubleResidualBlock(**c) for c in encoder_configs])

        self.mlp = classifier(**mlp_config)
        self.orth_projection = OrthogonalLazyLinear(self.mlp.input_shape, bias=False, requires_grad=False)

    @abstractmethod
    def project(self, x):
        pass

    def get_latent_features(self, x):
        x = self.project(x)

    def get_latent_features_from_projections(self, x):
        return self.mlp(x)

    def forward(self, x):
        x = self.project(x)
        return self.mlp(x)

class BottleneckProjector_UNet(BaseProjector, nn.Module):
    """
    Di -> concatenate -> OrthogonalProjection -> MLP
    D0 -> ^
    D1 -> ^
    D2 -> ^
       D3
    """
    def __init__(
        self,
        num_blocks,
        initial_config,
        initial_pool_config,
        encoder_configs,
        mlp_config,
    ):
        super().__init__(num_blocks, initial_config, initial_pool_config, encoder_configs, mlp_config)
        self.final_pool = nn.AvgPool2d(8)

    def project(self, x):
        skips = {}  # empty map to store skip connections

        # process initial block and pool
        x = self.initial_block(x)
        skips[self.num_blocks - 1] = x
        x = self.pool(x)

        # process encoding blocks
        for i in range(self.num_blocks):
            x = self.downs[i](x)
            if i < self.num_blocks - 1:
                skips[self.num_blocks - i - 2] = x

        # reduce size of model with some last pooling
        for k in skips:
            skips[k] = self.final_pool(skips[k])
        x = self.final_pool(x)

        x = torch.concatenate([x.flatten(start_dim=1)] + [v.flatten(start_dim=1) for v in skips.values()], dim=1)
        x = self.orth_projection(x)
        return x


class TailProjector_UNet(BaseProjector):
    """
    Di -> U3 -> Reduction + OrthogonalProjection -> MLP
    D0 -> U2
    D1 -> U1
    D2 -> U0
       D3
    """

    def __init__(
        self,
        num_blocks,
        initial_config,
        initial_pool_config,
        encoder_configs,
        decoder_configs,
        mlp_config,
        pool=True,
    ):
        super().__init__(num_blocks, initial_config, initial_pool_config, encoder_configs, mlp_config)
        ## creates decoder blocks
        self.ups = nn.ModuleList([UpsampleBlock(**c) for c in decoder_configs])
        self.final_pool = nn.AvgPool2d(8)
        

    def project(self, x):
        skips = {}  # empty map to store skip connections

        # process initial block and pool
        x = self.initial_block(x)
        skips[self.num_blocks - 1] = x
        x = self.pool(x)

        # process encoding blocks
        for i in range(self.num_blocks):
            x = self.downs[i](x)
            if i < self.num_blocks - 1:
                skips[self.num_blocks - i - 2] = x

        for i in range(self.num_blocks):
            x = self.ups[i](x, skips[i])

        x = self.final_pool(x)
        x = x.flatten(start_dim=1)
        x = self.orth_projection(x)
        return x