import torch.nn as nn

class FullyConvNetwork(nn.Module):

    def __init__(self):
        super().__init__()
        def encoder_block(in_channels, out_channels, use_batch_norm=True):
            layers = [
                nn.Conv2d(in_channels, out_channels, kernel_size=4, stride=2, padding=1),
            ]
            if use_batch_norm:
                layers.append(nn.BatchNorm2d(out_channels))
            layers.append(nn.LeakyReLU(0.2, inplace=True))
            return nn.Sequential(*layers)

        def decoder_block(in_channels, out_channels, use_batch_norm=True):
            layers = [
                nn.ConvTranspose2d(in_channels, out_channels, kernel_size=4, stride=2, padding=1),
            ]
            if use_batch_norm:
                layers.append(nn.BatchNorm2d(out_channels))
            layers.append(nn.ReLU(inplace=True))
            return nn.Sequential(*layers)

        # Encoder (Convolutional Layers)
        self.conv1 = encoder_block(3, 64, use_batch_norm=False)
        self.conv2 = encoder_block(64, 128)
        self.conv3 = encoder_block(128, 256)
        self.conv4 = encoder_block(256, 512)
        self.bottleneck = encoder_block(512, 512)

        # Decoder (Deconvolutional Layers)
        self.deconv1 = decoder_block(512, 512)
        self.deconv2 = decoder_block(512, 256)
        self.deconv3 = decoder_block(256, 128)
        self.deconv4 = decoder_block(128, 64)
        self.output_layer = nn.Sequential(
            nn.ConvTranspose2d(64, 3, kernel_size=4, stride=2, padding=1),
            nn.Tanh()
        )

    def forward(self, x):
        # Encoder forward pass
        x1 = self.conv1(x)
        x2 = self.conv2(x1)
        x3 = self.conv3(x2)
        x4 = self.conv4(x3)
        bottleneck = self.bottleneck(x4)

        # Decoder forward pass
        x = self.deconv1(bottleneck)
        x = self.deconv2(x)
        x = self.deconv3(x)
        x = self.deconv4(x)
        output = self.output_layer(x)

        return output
    
