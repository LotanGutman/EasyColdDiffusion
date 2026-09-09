import torch
import torch.nn as nn
import math

class SinusoidalPositionEmbeddings(nn.Module):
    """
    Converts a single integer timestep into a high-dimensional vector representation.
    """
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, time):
        device = time.device
        half_dim = self.dim // 2
        embeddings = math.log(10000) / (half_dim - 1)
        embeddings = torch.exp(torch.arange(half_dim, device=device) * -embeddings)
        embeddings = time[:, None] * embeddings[None, :]
        embeddings = torch.cat((embeddings.sin(), embeddings.cos()), dim=-1)
        return embeddings

class ConvBlock(nn.Module):
    """
    A standard double convolution block that injects the time embedding.
    """
    def __init__(self, in_ch, out_ch, time_emb_dim):
        super().__init__()
        self.time_mlp = nn.Linear(time_emb_dim, out_ch)
        self.conv1 = nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1)
        self.norm1 = nn.BatchNorm2d(out_ch)
        self.conv2 = nn.Conv2d(out_ch, out_ch, kernel_size=3, padding=1)
        self.norm2 = nn.BatchNorm2d(out_ch)
        self.relu = nn.ReLU()

    def forward(self, x, t):
        # First convolution and normalization
        h = self.relu(self.norm1(self.conv1(x)))

        # Project time embedding to match channels and add it to the feature map
        time_emb = self.relu(self.time_mlp(t))
        time_emb = time_emb[(...,) + (None,) * 2]  # Extend to (B, C, 1, 1)
        h = h + time_emb

        # Second convolution and normalization
        h = self.relu(self.norm2(self.conv2(h)))
        return h

class UNet(nn.Module):
    """
    The main U-Net architecture for Cold Diffusion Image Restoration.
    Strictly outputs the restored image ONLY.
    """
    def __init__(self):
        super().__init__()
        image_channels = 3
        down_channels = (64, 128, 256, 512)
        up_channels = (512, 256, 128, 64)
        out_dim = 3
        time_emb_dim = 64

        # Time embedding layer
        self.time_mlp = nn.Sequential(
            SinusoidalPositionEmbeddings(time_emb_dim),
            nn.Linear(time_emb_dim, time_emb_dim),
            nn.ReLU()
        )

        # Initial projection layer
        self.conv0 = nn.Conv2d(image_channels, down_channels[0], kernel_size=3, padding=1)

        # Downsampling path (Encoder)
        self.downs = nn.ModuleList([
            ConvBlock(down_channels[0], down_channels[1], time_emb_dim),
            ConvBlock(down_channels[1], down_channels[2], time_emb_dim),
            ConvBlock(down_channels[2], down_channels[3], time_emb_dim)
        ])
        self.pool = nn.MaxPool2d(2)

        # Bottleneck layer at the bottom of the U
        self.bottleneck = ConvBlock(down_channels[3], down_channels[3], time_emb_dim)

        # Upsampling path (Decoder)
        self.ups = nn.ModuleList([
            ConvBlock(1024, up_channels[1], time_emb_dim),
            ConvBlock(512, up_channels[2], time_emb_dim),
            ConvBlock(256, up_channels[3], time_emb_dim)
        ])

        self.upconvs = nn.ModuleList([
            nn.ConvTranspose2d(down_channels[3], down_channels[3], kernel_size=2, stride=2),
            nn.ConvTranspose2d(up_channels[1], up_channels[1], kernel_size=2, stride=2),
            nn.ConvTranspose2d(up_channels[2], up_channels[2], kernel_size=2, stride=2)
        ])

        # Final projection to RGB image space
        self.final_conv = nn.Conv2d(up_channels[-1], out_dim, kernel_size=1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x, timestep):
        # Process the timestep into an embedding vector
        t = self.time_mlp(timestep)

        # Pass image through initial layer
        x = self.conv0(x)

        # Save skip connections during downsampling
        skip_connections = []
        for down in self.downs:
            x = down(x, t)
            skip_connections.append(x)
            x = self.pool(x)

        # Pass through bottleneck
        x = self.bottleneck(x, t)

        # Restore resolution and concatenate skip connections during upsampling
        skip_connections = skip_connections[::-1]
        for i in range(len(self.ups)):
            x = self.upconvs[i](x)
            skip = skip_connections[i]
            x = torch.cat((x, skip), dim=1)
            x = self.ups[i](x, t)

        # Generate final image
        x = self.final_conv(x)

        # Returns ONLY the restored image
        return self.sigmoid(x)

if __name__ == "__main__":
    # Create a dummy batch of 4 degraded images (4, 3 channels, 128x128)
    dummy_images = torch.rand((4, 3, 128, 128))

    # Create dummy timesteps
    dummy_timesteps = torch.tensor([2, 5, 12, 20], dtype=torch.float32)

    # Initialize model
    model = UNet()

    # Run the dummy data through the model
    cleaned_output = model(dummy_images, dummy_timesteps)

    print(f"Cleaned Image output shape: {cleaned_output.shape}")  # Expected: [4, 3, 128, 128]