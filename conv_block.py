class ConvBlock(nn.Module):
    def __init__(
        self,
        in_channels,
        out_channels,
        discriminator=False,
        use_act=True,
        use_bn=True,
        **kwargs,
    ):
        super().__init__()
        self.use_act = use_act
        self.cnn = nn.Conv2d(in_channels, out_channels, **kwargs, bias=not use_bn)
        self.bn = nn.BatchNorm2d(out_channels) if use_bn else nn.Identity()
        self.act = (
            nn.LeakyReLU(0.2, inplace=True)
            if discriminator
            else nn.PReLU(num_parameters=out_channels)
        )

    def forward(self, x):
        return self.act(self.bn(self.cnn(x))) if self.use_act else self.bn(self.cnn(x))

    class ResidualBlock(nn.Module):
        def __init__(self, in_channels):
            super().__init__()
            self.block1 = ConvBlock(
                in_channels,
                in_channels,
                kernel_size=3,
                stride=1,
                padding=1
            )
            self.block2 = ConvBlock(
                in_channels,
                in_channels,
                kernel_size=3,
                stride=1,
                padding=1,
                use_act=False,
            )

        def forward(self, x):
            out = self.block1(x)
            out = self.block2(out)
            return out + x


class generator(nn.Module):
    def __init__(self):
        super().__init__()
        blocks = []
        for i in range(16):
            blocks.append(residual_block())

        self.pre_residual = nn.Sequential(nn.Conv2d(in_channels=3, out_channels=64,
                                                    kernel_size=9, padding=1),
                                          nn.ReLU())
        self.residual = nn.Sequential(blocks)
        self.post_residual = nn.Sequential(nn.Conv2d(in_channels=64, out_channels=64,
                                                     kernel_size=3, padding=1),
                                           nn.BatchNorm2d(64))
        self.upscale = nn.Sequential(upscale(),
                                     upscale(),
                                     nn.Conv2d(in_channels=64, out_channels=3,
                                               kernel_size=9, padding=1))

    def forward(self, x):
        pre_residual_out = self.pre_residual(x)
        out = self.residual(pre_residual_out)
        post_residual_out = self.post_residual(out)
        out = torch.add(pre_residual_out, post_residual_out)
        out = upscale(out)
        return out


class residual_block(nn.Module):
    def __init__(self):
        super().__init__()
        self.residual = nn.Sequential(nn.Conv2d(in_channels=64, out_channels=64,
                                                kernel_size=3, padding=1),
                                      nn.BatchNorm2d(64),
                                      nn.ReLU(),
                                      nn.Conv2d(in_channels=64, out_channels=64,
                                                kernel_size=3, padding=1),
                                      nn.BatchNorm2d(64))

    def forward(self, x):
        out = self.residual(x)
        out = torch.add(out, x)
        return out


class upscale(nn.Module):
    def __init__(self):
        super().__init__()
        self.upscale = nn.Sequential(nn.Conv2d(in_channels=64, out_channels=64*2,
                                               kernel_size=3, padding=1),
                                     nn.PixelShuffle(2),
                                     nn.PReLU())

    def forward(self, x):
        out = self.upscale(x)
        return out

