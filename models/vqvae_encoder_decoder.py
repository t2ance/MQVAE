import torch.nn as nn


class ResBlock(nn.Module):
    def __init__(self, dim, act="relu"):
        super().__init__()
        if act == "relu":
            activation = nn.ReLU()
        elif act == "elu":
            activation = nn.ELU()
        self.block = nn.Sequential(
            activation,
            nn.Conv2d(dim, dim, 3, 1, 1),
            nn.BatchNorm2d(dim),
            activation,
            nn.Conv2d(dim, dim, 1),
            nn.BatchNorm2d(dim)
        )

    def forward(self, x):
        return x + self.block(x)


class VQVAEEncoder(nn.Module):
    def __init__(self, dim_z, factor=4, num_rb=6):
        super(VQVAEEncoder, self).__init__()
        assert factor in [4, 8]
        # Convolution layers
        layers_conv = []
        layers_conv.append(nn.Conv2d(3, dim_z // 2, 4, stride=2, padding=1))
        layers_conv.append(nn.BatchNorm2d(dim_z // 2))
        layers_conv.append(nn.ReLU())

        layers_conv.append(nn.Conv2d(dim_z // 2, dim_z, 4, stride=2, padding=1))
        layers_conv.append(nn.BatchNorm2d(dim_z))
        layers_conv.append(nn.ReLU())

        if factor == 8:
            layers_conv.append(nn.Conv2d(dim_z, dim_z, 4, stride=2, padding=1))
            layers_conv.append(nn.BatchNorm2d(dim_z))
            layers_conv.append(nn.ReLU())

        layers_conv.append(nn.Conv2d(dim_z, dim_z, 3, stride=1, padding=1))
        self.conv = nn.Sequential(*layers_conv)
        # Resblocks
        layers_resblocks = []
        for i in range(num_rb - 1):
            layers_resblocks.append(ResBlock(dim_z))
        self.res = nn.Sequential(*layers_resblocks)
        self.res_m = ResBlock(dim_z)

    def forward(self, x):
        out_conv = self.conv(x)
        out_res = self.res(out_conv)
        return self.res_m(out_res)


class VQVAEDecoder(nn.Module):
    def __init__(self, dim_z, factor=4, num_rb=6):
        super(VQVAEDecoder, self).__init__()
        # Resblocks
        layers_resblocks = []
        for i in range(num_rb):
            layers_resblocks.append(ResBlock(dim_z))
        self.res = nn.Sequential(*layers_resblocks)
        # Convolution layers
        layers_convt = []
        layers_convt.append(nn.ConvTranspose2d(dim_z, dim_z, 3, stride=1, padding=1))

        if factor == 8:
            layers_convt.append(nn.BatchNorm2d(dim_z))
            layers_convt.append(nn.ReLU())
            layers_convt.append(nn.ConvTranspose2d(dim_z, dim_z, 4, stride=2, padding=1))

        layers_convt.append(nn.BatchNorm2d(dim_z))
        layers_convt.append(nn.ReLU())
        layers_convt.append(nn.ConvTranspose2d(dim_z, dim_z // 2, 4, stride=2, padding=1))

        layers_convt.append(nn.BatchNorm2d(dim_z // 2))
        layers_convt.append(nn.ReLU())
        layers_convt.append(nn.ConvTranspose2d(dim_z // 2, 3, 4, stride=2, padding=1))

        self.convt = nn.Sequential(*layers_convt)

    def forward(self, z):
        out_res = self.res(z)
        out = self.convt(out_res)
        return out

    def get_last_layer(self):
        return self.convt[-1].weight


if __name__ == '__main__':
    ...
