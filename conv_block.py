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