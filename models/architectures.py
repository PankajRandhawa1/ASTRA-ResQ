"""ASTRA ResQ neural-network architectures matching the supplied checkpoints."""
import torch
import torch.nn as nn
import torch.nn.functional as F

class GANEncoderBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.LeakyReLU(0.2, inplace=True),
        )
    def forward(self, x):
        return self.block(x)


class GANDecoderBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.block = nn.Sequential(
            nn.Upsample(scale_factor=2, mode="nearest"),
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )
    def forward(self, x):
        return self.block(x)


class GANRefinementBlock(nn.Module):
    def __init__(self, channels=256):
        super().__init__()
        self.conv1 = nn.Conv2d(channels, channels, 3, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(channels)
        self.conv2 = nn.Conv2d(channels, channels, 3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(channels)
    def forward(self, x):
        residual = x
        x = F.relu(self.bn1(self.conv1(x)), inplace=True)
        x = self.bn2(self.conv2(x))
        return F.relu(x + residual, inplace=True)


class GANAttentionBlock(nn.Module):
    def __init__(self, channels=256):
        super().__init__()
        self.attention = nn.Sequential(
            nn.Conv2d(channels, 32, kernel_size=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, channels, kernel_size=1),
            nn.Sigmoid(),
        )
    def forward(self, x):
        return x * self.attention(x)


class ASTRA_GAN_Generator(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder1 = GANEncoderBlock(3, 64)
        self.encoder2 = GANEncoderBlock(64, 128)
        self.encoder3 = GANEncoderBlock(128, 256)
        self.refinement = GANRefinementBlock(256)
        self.attention = GANAttentionBlock(256)
        self.decoder1 = GANDecoderBlock(256, 128)
        self.decoder2 = GANDecoderBlock(256, 64)
        self.final_decoder = nn.Sequential(
            nn.Upsample(scale_factor=2, mode="nearest"),
            nn.Conv2d(128, 3, kernel_size=3, padding=1),
            nn.Tanh(),
        )
    def forward(self, x):
        enc1 = self.encoder1(x)
        enc2 = self.encoder2(enc1)
        enc3 = self.encoder3(enc2)
        features = self.attention(self.refinement(enc3))
        dec1 = self.decoder1(features)
        dec2 = self.decoder2(torch.cat([dec1, enc2], dim=1))
        return self.final_decoder(torch.cat([dec2, enc1], dim=1))


class DoubleConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels), nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels), nn.ReLU(inplace=True),
        )
    def forward(self, x):
        return self.block(x)


class FloodUNet(nn.Module):
    """Flood U-Net; output layer name/shape is inferred from the saved checkpoint."""
    def __init__(self, output_layer_name: str, output_channels=1, output_kernel_size=1, output_bias=True):
        super().__init__()
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.enc1 = DoubleConv(3, 32)
        self.enc2 = DoubleConv(32, 64)
        self.enc3 = DoubleConv(64, 128)
        self.enc4 = DoubleConv(128, 256)
        self.bridge = DoubleConv(256, 512)
        self.up4 = nn.ConvTranspose2d(512, 256, 2, 2)
        self.dec4 = DoubleConv(512, 256)
        self.up3 = nn.ConvTranspose2d(256, 128, 2, 2)
        self.dec3 = DoubleConv(256, 128)
        self.up2 = nn.ConvTranspose2d(128, 64, 2, 2)
        self.dec2 = DoubleConv(128, 64)
        self.up1 = nn.ConvTranspose2d(64, 32, 2, 2)
        self.dec1 = DoubleConv(64, 32)
        if "." in output_layer_name:
            raise ValueError(f"Unexpected nested output layer name: {output_layer_name}")
        setattr(self, output_layer_name, nn.Conv2d(32, output_channels, kernel_size=output_kernel_size,
                                                   padding=output_kernel_size // 2, bias=output_bias))
        self.output_layer_name = output_layer_name
    def forward(self, x):
        enc1 = self.enc1(x)
        enc2 = self.enc2(self.pool(enc1))
        enc3 = self.enc3(self.pool(enc2))
        enc4 = self.enc4(self.pool(enc3))
        bridge = self.bridge(self.pool(enc4))
        dec4 = self.dec4(torch.cat([self.up4(bridge), enc4], dim=1))
        dec3 = self.dec3(torch.cat([self.up3(dec4), enc3], dim=1))
        dec2 = self.dec2(torch.cat([self.up2(dec3), enc2], dim=1))
        dec1 = self.dec1(torch.cat([self.up1(dec2), enc1], dim=1))
        return getattr(self, self.output_layer_name)(dec1)


