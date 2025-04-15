import torch
import torch.nn as nn
import torch.nn.functional as F
from .attention import FeatureAttention, CrossAttention, TimbreTransformer


class InstrumentTimbreDecoder(nn.Module):
    """Decoder for transforming a timbre vector back to audio features"""

    def __init__(self):
        super(InstrumentTimbreDecoder, self).__init__()

        # Feature attention to focus on important timbre aspects
        self.feature_attention = FeatureAttention(128)

        # Fully connected layers to expand from feature vector
        self.fc1 = nn.Linear(128, 256)
        self.fc2 = nn.Linear(256, 512)
        self.fc3 = nn.Linear(512, 256 * 16 * 16)  # Updated to match encoder dimensions

        # Transposed convolutions for upsampling
        self.deconv1 = nn.ConvTranspose2d(
            256, 128, kernel_size=4, stride=2, padding=1
        )  # 16x16 -> 32x32
        self.deconv2 = nn.ConvTranspose2d(
            128, 64, kernel_size=4, stride=2, padding=1
        )  # 32x32 -> 64x64
        self.deconv3 = nn.ConvTranspose2d(
            64, 32, kernel_size=4, stride=2, padding=1
        )  # 64x64 -> 128x128
        self.deconv4 = nn.ConvTranspose2d(
            32,
            1,
            kernel_size=(3, 6),
            stride=(1, 2),
            padding=(1, 2),
            output_padding=(0, 0),
        )  # 128x128 -> 128x256

        # Activation
        self.relu = nn.ReLU()
        self.tanh = nn.Tanh()

    def forward(self, x):
        # Apply feature attention
        x = self.feature_attention(x)

        # Fully connected layers
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        x = self.relu(self.fc3(x))

        # Reshape for deconvolution - use the updated dimensions
        x = x.view(-1, 256, 16, 16)

        # Print shapes for debugging
        # print(f"Decoder after reshape: {x.shape}")

        # Transposed convolutions
        x = self.relu(self.deconv1(x))
        # print(f"Decoder after deconv1: {x.shape}")

        x = self.relu(self.deconv2(x))
        # print(f"Decoder after deconv2: {x.shape}")

        x = self.relu(self.deconv3(x))
        # print(f"Decoder after deconv3: {x.shape}")

        x = self.tanh(self.deconv4(x))
        # print(f"Decoder final output: {x.shape}")

        return x


class EnhancedTimbreDecoder(nn.Module):
    """Advanced decoder with transformer architecture, cross-attention mechanisms,
    and sophisticated upsampling for optimal timbre reconstruction."""

    def __init__(
        self,
        feature_dim=128,
        output_channels=1,
        use_transformer=True,
        with_residual=True,
        output_size=(128, 256),
    ):
        super(EnhancedTimbreDecoder, self).__init__()

        self.feature_dim = feature_dim
        self.output_channels = output_channels
        self.with_residual = with_residual
        self.use_transformer = use_transformer
        self.output_size = output_size

        # Enhanced feature attention with multi-head mechanism
        self.feature_attention = FeatureAttention(feature_dim)

        # Transformer for sequence modeling (optional)
        if use_transformer:
            self.transformer = TimbreTransformer(
                d_model=feature_dim,
                nhead=8,
                num_encoder_layers=3,
                dim_feedforward=feature_dim * 4,
                dropout=0.1,
            )

            # Project sequence back to feature vector
            self.seq_projection = nn.Linear(feature_dim, feature_dim)

        # Advanced feature expansion network with layer normalization and residual connections
        self.feature_expander = nn.Sequential(
            nn.Linear(feature_dim, 256),
            nn.LayerNorm(256),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(256, 512),
            nn.LayerNorm(512),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(512, 1024),
            nn.LayerNorm(1024),
            nn.GELU(),
            nn.Linear(1024, 128 * 16 * 16),
        )

        # Progressive upsampling with residual blocks
        # First stage: 16x16 -> 32x32
        self.upsample1 = self._make_upsample_block(128, 64)

        # Second stage: 32x32 -> 64x64
        self.upsample2 = self._make_upsample_block(64, 32)

        # Third stage: 64x64 -> 128x128
        self.upsample3 = self._make_upsample_block(32, 16)

        # Optional additional upsampling for output_size fine-tuning
        self.final_upsample = None
        if output_size[0] > 128 or output_size[1] > 128:
            self.final_upsample = nn.Sequential(
                nn.Conv2d(16, 16, kernel_size=3, padding=1),
                nn.BatchNorm2d(16),
                nn.GELU(),
                nn.Upsample(size=output_size, mode="bilinear", align_corners=True),
            )

        # Cross-attention for detail enhancement
        self.cross_attention = CrossAttention(
            query_dim=16, key_dim=16, value_dim=16, num_heads=4, dropout=0.1
        )

        # Final refinement and output projection
        self.output_refiner = nn.Sequential(
            nn.Conv2d(16, 16, kernel_size=3, padding=1),
            nn.BatchNorm2d(16),
            nn.GELU(),
            nn.Conv2d(16, output_channels, kernel_size=3, padding=1),
        )

        # Activation for final output
        self.output_activation = nn.Tanh()

        # Initialize weights for better convergence
        self._init_weights()

    def _make_upsample_block(self, in_channels, out_channels):
        """Create an advanced upsampling block with residual connections"""
        return nn.Sequential(
            # Transposed convolution for upsampling
            nn.ConvTranspose2d(
                in_channels, out_channels, kernel_size=4, stride=2, padding=1
            ),
            nn.BatchNorm2d(out_channels),
            nn.GELU(),
            # Refine features with regular convolution
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.GELU(),
            # Spatial attention for focusing on important regions
            self._make_spatial_attention(out_channels),
        )

    def _make_spatial_attention(self, channels):
        """Create a spatial attention mechanism to focus on important regions"""
        return nn.Sequential(
            nn.Conv2d(channels, channels // 8, kernel_size=1),
            nn.BatchNorm2d(channels // 8),
            nn.GELU(),
            nn.Conv2d(channels // 8, 1, kernel_size=1),
            nn.Sigmoid(),
        )

    def _init_weights(self):
        """Initialize weights using He (Kaiming) initialization"""
        for m in self.modules():
            if isinstance(m, (nn.Conv2d, nn.ConvTranspose2d, nn.Linear)):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, (nn.BatchNorm2d, nn.LayerNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, x, encoder_features=None):
        """Forward pass with enhanced reconstruction capabilities

        Args:
            x: Timbre feature vector [batch_size, feature_dim]
            encoder_features: Optional features from the encoder for skip connections

        Returns:
            Reconstructed audio features
        """
        batch_size = x.size(0)

        # Apply feature attention
        x_attended = self.feature_attention(x)

        # Optionally apply transformer for sequence modeling
        if self.use_transformer:
            # Expand to sequence by repeating
            x_seq = x_attended.unsqueeze(1).repeat(1, 16, 1)  # [B, 16, feature_dim]

            # Apply transformer
            x_transformed = self.transformer(x_seq)

            # Project back to feature vector
            x_processed = self.seq_projection(
                x_transformed.mean(dim=1)
            )  # Global pooling
        else:
            x_processed = x_attended

        # Feature expansion network
        x_expanded = self.feature_expander(x_processed)

        # Reshape to spatial features
        x_spatial = x_expanded.view(batch_size, 128, 16, 16)

        # Progressive upsampling
        x_up1 = self.upsample1(x_spatial)  # 16x16 -> 32x32
        x_up2 = self.upsample2(x_up1)  # 32x32 -> 64x64
        x_up3 = self.upsample3(x_up2)  # 64x64 -> 128x128

        # Final upsampling if needed
        if self.final_upsample:
            x_upsampled = self.final_upsample(x_up3)
        else:
            x_upsampled = x_up3

        # Apply cross-attention for detail enhancement if encoder features available
        if encoder_features is not None and self.with_residual:
            # Reshape for cross-attention: [B, C, H, W] -> [B, H*W, C]
            b, c, h, w = x_upsampled.size()
            x_query = x_upsampled.view(b, c, h * w).permute(0, 2, 1)  # [B, H*W, C]

            # Ensure encoder_features match the expected shape
            if encoder_features.size()[-2:] != (h, w):
                encoder_features = F.interpolate(
                    encoder_features, size=(h, w), mode="bilinear", align_corners=True
                )

            # Prepare key/value from encoder features
            if encoder_features.size(1) != c:
                # Channel adapter if needed
                encoder_features = nn.Conv2d(
                    encoder_features.size(1), c, kernel_size=1
                ).to(x.device)(encoder_features)

            e_b, e_c, e_h, e_w = encoder_features.size()
            encoder_kv = encoder_features.view(e_b, e_c, e_h * e_w).permute(
                0, 2, 1
            )  # [B, H*W, C]

            # Apply cross-attention
            enhanced = self.cross_attention(x_query, encoder_kv, encoder_kv)
            enhanced = enhanced.permute(0, 2, 1).view(b, c, h, w)  # [B, C, H, W]

            # Residual connection
            x_upsampled = x_upsampled + enhanced

        # Final refinement
        output = self.output_refiner(x_upsampled)

        # Apply activation
        output = self.output_activation(output)

        return output


class TimbreAutoencoder(nn.Module):
    """Complete end-to-end autoencoder for instrument timbre modeling
    with advanced encoder-decoder architecture and skip connections.
    """

    def __init__(self, input_channels=1, feature_dim=128, use_transformer=True):
        super(TimbreAutoencoder, self).__init__()

        # Advanced encoder with transformer capabilities
        self.encoder = EncoderWithMultiScaleFusion(
            input_channels=input_channels,
            output_dim=feature_dim,
            use_transformer=use_transformer,
        )

        # Sophisticated decoder with cross-attention and skip connections
        self.decoder = EnhancedTimbreDecoder(
            feature_dim=feature_dim,
            output_channels=input_channels,
            use_transformer=use_transformer,
            with_residual=True,
        )

        # Timbre classification head (optional for multi-task learning)
        self.classifier = nn.Sequential(
            nn.Linear(feature_dim, 256),
            nn.LayerNorm(256),
            nn.GELU(),
            nn.Dropout(0.3),
            nn.Linear(256, 128),
            nn.LayerNorm(128),
            nn.GELU(),
            nn.Linear(128, 32),  # Number of instrument classes
        )

    def forward(self, x, return_features=False, classification=False):
        """Forward pass through the full autoencoder

        Args:
            x: Input tensor [batch_size, channels, height, width]
            return_features: Whether to return the latent features
            classification: Whether to return instrument classification logits

        Returns:
            Tuple of (reconstructed_output, features, class_logits) depending on the flags
        """
        # Get encoder features and latent representation
        features, encoder_activations = self.encoder(x, return_activations=True)

        # Decode with encoder features for skip connections
        reconstructed = self.decoder(features, encoder_activations[-1])

        # Prepare return values based on flags
        result = [reconstructed]

        if return_features:
            result.append(features)

        if classification:
            class_logits = self.classifier(features)
            result.append(class_logits)

        return result[0] if len(result) == 1 else tuple(result)


class EncoderWithMultiScaleFusion(nn.Module):
    """Specialized encoder with multi-scale feature fusion and pyramid pooling
    for capturing timbre characteristics at different frequency resolutions.
    """

    def __init__(self, input_channels=1, output_dim=128, use_transformer=True):
        super(EncoderWithMultiScaleFusion, self).__init__()

        self.input_channels = input_channels
        self.output_dim = output_dim
        self.use_transformer = use_transformer

        # Initial multi-resolution feature extraction
        self.multi_scale_extractor = nn.ModuleDict(
            {
                "fine": nn.Sequential(
                    nn.Conv2d(input_channels, 16, kernel_size=3, stride=1, padding=1),
                    nn.BatchNorm2d(16),
                    nn.GELU(),
                    nn.Conv2d(16, 16, kernel_size=3, stride=1, padding=1),
                    nn.BatchNorm2d(16),
                    nn.GELU(),
                ),
                "medium": nn.Sequential(
                    nn.Conv2d(input_channels, 16, kernel_size=5, stride=1, padding=2),
                    nn.BatchNorm2d(16),
                    nn.GELU(),
                    nn.AvgPool2d(kernel_size=2, stride=2),
                    nn.Conv2d(16, 16, kernel_size=3, stride=1, padding=1),
                    nn.BatchNorm2d(16),
                    nn.GELU(),
                    nn.Upsample(scale_factor=2, mode="bilinear", align_corners=True),
                ),
                "coarse": nn.Sequential(
                    nn.Conv2d(input_channels, 16, kernel_size=7, stride=1, padding=3),
                    nn.BatchNorm2d(16),
                    nn.GELU(),
                    nn.AvgPool2d(kernel_size=4, stride=4),
                    nn.Conv2d(16, 16, kernel_size=3, stride=1, padding=1),
                    nn.BatchNorm2d(16),
                    nn.GELU(),
                    nn.Upsample(scale_factor=4, mode="bilinear", align_corners=True),
                ),
            }
        )

        # Feature fusion with attention
        self.fusion = nn.Sequential(
            nn.Conv2d(48, 32, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(32),
            nn.GELU(),
            nn.Conv2d(32, 32, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(32),
            nn.GELU(),
        )

        # Hierarchical feature extraction with pyramid pooling
        self.pyramid_blocks = nn.ModuleList(
            [
                self._make_pyramid_block(32, 64, pool_size=2),
                self._make_pyramid_block(64, 128, pool_size=2),
                self._make_pyramid_block(128, 256, pool_size=2),
            ]
        )

        # Global pooling and feature projection
        self.global_pool = nn.AdaptiveAvgPool2d((1, 1))

        # Final projection to embedding space
        self.projection = nn.Sequential(
            nn.Linear(256, 256),
            nn.LayerNorm(256),
            nn.GELU(),
            nn.Dropout(0.2),
            nn.Linear(256, output_dim),
        )

        # Initialize weights
        self._init_weights()

    def _make_pyramid_block(self, in_channels, out_channels, pool_size=2):
        """Create a feature pyramid block with multiple pooling operations"""
        return nn.Sequential(
            # Regular convolution path
            nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.GELU(),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.GELU(),
            # Pooling for spatial reduction
            nn.MaxPool2d(kernel_size=pool_size, stride=pool_size),
            # Apply frequency attention
            FrequencyAttention(out_channels),
        )

    def _init_weights(self):
        """Initialize weights for better convergence"""
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, (nn.BatchNorm2d, nn.LayerNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, std=0.01)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def forward(self, x, return_activations=False):
        """Forward pass with optional return of intermediate activations for skip connections

        Args:
            x: Input tensor [batch_size, channels, height, width]
            return_activations: Whether to return intermediate activations

        Returns:
            Timbre features and optionally intermediate activations
        """
        # Apply multi-scale feature extraction
        x_fine = self.multi_scale_extractor["fine"](x)
        x_medium = self.multi_scale_extractor["medium"](x)
        x_coarse = self.multi_scale_extractor["coarse"](x)

        # Concatenate multi-scale features
        x_multi = torch.cat([x_fine, x_medium, x_coarse], dim=1)
        x = self.fusion(x_multi)  # [B, 32, H, W]

        # Store intermediate activations if requested
        activations = [x] if return_activations else None

        # Apply pyramid blocks
        for pyramid_block in self.pyramid_blocks:
            x = pyramid_block(x)
            if return_activations:
                activations.append(x)

        # Global pooling
        x_pooled = self.global_pool(x).view(x.size(0), -1)  # [B, 256]

        # Final projection
        features = self.projection(x_pooled)  # [B, output_dim]

        if return_activations:
            return features, activations
        return features
