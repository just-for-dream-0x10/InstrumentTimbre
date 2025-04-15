import torch
import torch.nn as nn
import torch.nn.functional as F
import math


class SelfAttention(nn.Module):
    """Self-attention mechanism for sequential data"""

    def __init__(self, in_channels):
        super(SelfAttention, self).__init__()
        self.query = nn.Conv1d(in_channels, in_channels // 8, kernel_size=1)
        self.key = nn.Conv1d(in_channels, in_channels // 8, kernel_size=1)
        self.value = nn.Conv1d(in_channels, in_channels, kernel_size=1)
        self.gamma = nn.Parameter(torch.zeros(1))  # Learnable weight parameter
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x):
        """
        Input: x: [B, C, L]
        Output: out: [B, C, L]
        """
        batch_size, C, length = x.size()

        # [B, C/8, L]
        proj_query = self.query(x)
        # [B, C/8, L]
        proj_key = self.key(x)
        # [B, C, L]
        proj_value = self.value(x)

        # [B, L, C/8] x [B, C/8, L] -> [B, L, L]
        energy = torch.bmm(proj_query.permute(0, 2, 1), proj_key)
        # [B, L, L]
        attention = self.softmax(energy)
        # [B, L, L] x [B, C, L] -> [B, C, L]
        out = torch.bmm(proj_value, attention.permute(0, 2, 1))

        # Residual connection
        out = self.gamma * out + x

        return out


class FeatureAttention(nn.Module):
    """Feature attention mechanism for vector data"""

    def __init__(self, feature_dim):
        super(FeatureAttention, self).__init__()
        self.attention = nn.Sequential(
            nn.Linear(feature_dim, feature_dim // 4),
            nn.ReLU(),
            nn.Linear(feature_dim // 4, feature_dim),
            nn.Sigmoid(),
        )

    def forward(self, x):
        """
        Input: x: [B, feature_dim]
        Output: out: [B, feature_dim]
        """
        # Calculate attention weights
        weights = self.attention(x)
        # Apply attention weights
        out = x * weights
        return out


class FrequencyAttention(nn.Module):
    """Enhanced frequency attention mechanism for spectral data with gating"""

    def __init__(self, channels, reduction_ratio=8):
        super(FrequencyAttention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d((None, 1))
        self.max_pool = nn.AdaptiveMaxPool2d((None, 1))

        # Enhanced feature extraction path
        reduced_channels = max(
            channels // reduction_ratio, 8
        )  # Avoid too small dimension
        self.fc = nn.Sequential(
            nn.Conv2d(channels * 2, reduced_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(reduced_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(reduced_channels, channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(channels),
        )

        # Gating path
        self.gate_fc = nn.Sequential(
            nn.Conv2d(channels * 2, reduced_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(reduced_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(reduced_channels, channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(channels),
            nn.Sigmoid(),
        )

    def forward(self, x):
        # x shape: [B, C, F, T]
        # Focus on frequency dimension
        avg_out = self.avg_pool(x.transpose(2, 3)).transpose(2, 3)  # [B, C, F, 1]
        max_out = self.max_pool(x.transpose(2, 3)).transpose(2, 3)  # [B, C, F, 1]

        # Concatenate along channel dimension
        cat = torch.cat([avg_out, max_out], dim=1)  # [B, 2C, F, 1]
        att = self.fc(cat)  # [B, C, F, 1]
        gate = self.gate_fc(cat)  # [B, C, F, 1]

        # Apply attention with gating
        return x * gate + att


class MultiHeadAttention(nn.Module):
    """Enhanced Multi-head attention mechanism with improved scaling and dropout"""

    def __init__(self, d_model, num_heads, dropout=0.1):
        super(MultiHeadAttention, self).__init__()
        self.d_model = d_model
        self.num_heads = num_heads
        self.head_dim = d_model // num_heads
        assert (
            self.head_dim * num_heads == d_model
        ), "d_model must be divisible by num_heads"

        self.query = nn.Linear(d_model, d_model)
        self.key = nn.Linear(d_model, d_model)
        self.value = nn.Linear(d_model, d_model)

        self.dropout = nn.Dropout(dropout)
        self.fc_out = nn.Linear(d_model, d_model)

    def forward(self, query, key, value, mask=None, return_attention=False):
        batch_size = query.shape[0]
        seq_length = query.shape[1]

        # Linear projections and reshape for multi-head attention
        Q = (
            self.query(query)
            .view(batch_size, -1, self.num_heads, self.head_dim)
            .permute(0, 2, 1, 3)
        )
        K = (
            self.key(key)
            .view(batch_size, -1, self.num_heads, self.head_dim)
            .permute(0, 2, 1, 3)
        )
        V = (
            self.value(value)
            .view(batch_size, -1, self.num_heads, self.head_dim)
            .permute(0, 2, 1, 3)
        )

        # Attention calculation with improved numerical stability
        attention_scale = 1.0 / math.sqrt(self.head_dim)
        scores = torch.matmul(Q, K.permute(0, 1, 3, 2)) * attention_scale

        if mask is not None:
            scores = scores.masked_fill(mask == 0, -1e9)

        attention_weights = F.softmax(scores, dim=-1)
        attention_probs = self.dropout(attention_weights)

        # Apply attention to values
        output = torch.matmul(attention_probs, V)
        output = (
            output.permute(0, 2, 1, 3).contiguous().view(batch_size, -1, self.d_model)
        )
        output = self.fc_out(output)

        if return_attention:
            return output, attention_weights
        return output


class PositionalEncoding(nn.Module):
    """Positional encoding for transformer architectures"""

    def __init__(self, d_model, max_seq_length=5000, dropout=0.1):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        # Create positional encoding matrix
        pe = torch.zeros(max_seq_length, d_model)
        position = torch.arange(0, max_seq_length, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model)
        )

        # Apply sine to even indices
        pe[:, 0::2] = torch.sin(position * div_term)
        # Apply cosine to odd indices
        pe[:, 1::2] = torch.cos(position * div_term)

        pe = pe.unsqueeze(0)
        self.register_buffer("pe", pe)

    def forward(self, x):
        # x: [batch_size, seq_length, d_model]
        x = x + self.pe[:, : x.size(1), :]
        return self.dropout(x)


class TransformerEncoderLayer(nn.Module):
    """Transformer encoder layer for sequence modeling"""

    def __init__(self, d_model, nhead, dim_feedforward=2048, dropout=0.1):
        super(TransformerEncoderLayer, self).__init__()

        # Multi-head self-attention
        self.self_attn = MultiHeadAttention(d_model, nhead, dropout)

        # Feed forward network
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)

        # Layer normalization
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)

        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)

        self.activation = F.gelu  # Using GELU activation for better performance

    def forward(self, src, src_mask=None):
        # Self attention with residual connection and layer normalization
        src2 = self.norm1(src)
        src2 = self.self_attn(src2, src2, src2, mask=src_mask)
        src = src + self.dropout1(src2)

        # Feed forward network with residual connection and layer normalization
        src2 = self.norm2(src)
        src2 = self.linear2(self.dropout(self.activation(self.linear1(src2))))
        src = src + self.dropout2(src2)

        return src


class TransformerEncoder(nn.Module):
    """Transformer encoder for sequence modeling with multiple layers"""

    def __init__(self, encoder_layer, num_layers, norm=None):
        super(TransformerEncoder, self).__init__()
        self.layers = nn.ModuleList([encoder_layer for _ in range(num_layers)])
        self.norm = norm

    def forward(self, src, mask=None):
        output = src
        for layer in self.layers:
            output = layer(output, src_mask=mask)

        if self.norm is not None:
            output = self.norm(output)

        return output


class TimbreTransformer(nn.Module):
    """Complete transformer model for instrument timbre analysis"""

    def __init__(
        self,
        d_model=512,
        nhead=8,
        num_encoder_layers=6,
        dim_feedforward=2048,
        dropout=0.1,
        activation="gelu",
    ):
        super(TimbreTransformer, self).__init__()

        # Positional encoding
        self.pos_encoder = PositionalEncoding(d_model, dropout=dropout)

        # Create encoder layer
        encoder_layer = TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
        )

        # Create encoder with multiple layers
        self.transformer_encoder = TransformerEncoder(
            encoder_layer, num_layers=num_encoder_layers, norm=nn.LayerNorm(d_model)
        )

        # Initialize parameters with Glorot/fan_avg
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def forward(self, src, src_mask=None):
        # Add positional encoding
        src = self.pos_encoder(src)

        # Pass through transformer encoder
        output = self.transformer_encoder(src, mask=src_mask)

        return output


class CrossAttention(nn.Module):
    """Cross-attention mechanism for mixing information from different modalities or features"""

    def __init__(self, query_dim, key_dim, value_dim, num_heads=8, dropout=0.1):
        super(CrossAttention, self).__init__()
        self.num_heads = num_heads
        self.query_dim = query_dim
        self.key_dim = key_dim
        self.value_dim = value_dim
        self.head_dim = query_dim // num_heads

        # Project queries, keys and values
        self.q_proj = nn.Linear(query_dim, query_dim)
        self.k_proj = nn.Linear(key_dim, query_dim)  # Project keys to query dimension
        self.v_proj = nn.Linear(
            value_dim, query_dim
        )  # Project values to query dimension

        self.dropout = nn.Dropout(dropout)
        self.out_proj = nn.Linear(query_dim, query_dim)

    def forward(self, query, key, value, mask=None):
        batch_size = query.shape[0]

        # Project and reshape
        q = (
            self.q_proj(query)
            .view(batch_size, -1, self.num_heads, self.head_dim)
            .permute(0, 2, 1, 3)
        )
        k = (
            self.k_proj(key)
            .view(batch_size, -1, self.num_heads, self.head_dim)
            .permute(0, 2, 1, 3)
        )
        v = (
            self.v_proj(value)
            .view(batch_size, -1, self.num_heads, self.head_dim)
            .permute(0, 2, 1, 3)
        )

        # Compute attention scores
        attn_weights = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.head_dim)

        if mask is not None:
            attn_weights = attn_weights.masked_fill(mask == 0, -1e9)

        attn_probs = F.softmax(attn_weights, dim=-1)
        attn_probs = self.dropout(attn_probs)

        # Apply attention weights to values
        output = torch.matmul(attn_probs, v)
        output = (
            output.permute(0, 2, 1, 3).contiguous().view(batch_size, -1, self.query_dim)
        )
        output = self.out_proj(output)

        return output


class HybridAttention(nn.Module):
    """Hybrid attention combining CNN features with self-attention for better audio feature extraction"""

    def __init__(self, in_channels, time_steps, freq_bins, heads=4):
        super(HybridAttention, self).__init__()

        # CNN feature extractor
        self.conv_layers = nn.Sequential(
            nn.Conv2d(in_channels, in_channels * 2, kernel_size=3, padding=1),
            nn.BatchNorm2d(in_channels * 2),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(in_channels * 2, in_channels * 2, kernel_size=3, padding=1),
            nn.BatchNorm2d(in_channels * 2),
            nn.LeakyReLU(0.2, inplace=True),
        )

        # Frequency attention mechanism
        self.freq_attention = FrequencyAttention(in_channels * 2)

        # Project to sequence for transformer
        self.d_model = in_channels * 2 * freq_bins  # Flattened frequency dimension
        self.projection = nn.Linear(self.d_model, self.d_model)

        # Self-attention mechanism
        self.self_attention = MultiHeadAttention(self.d_model, heads)

        # Output projection
        self.output_proj = nn.Linear(self.d_model, self.d_model)

        # Layer norms for stability
        self.norm1 = nn.LayerNorm(self.d_model)
        self.norm2 = nn.LayerNorm(self.d_model)

    def forward(self, x):
        # x shape: [batch_size, channels, freq_bins, time_steps]
        batch_size = x.shape[0]

        # Apply CNN feature extraction
        cnn_features = self.conv_layers(x)

        # Apply frequency attention
        freq_attn_features = self.freq_attention(cnn_features)

        # Reshape for self-attention: [batch_size, time_steps, channels*freq_bins]
        reshape_features = freq_attn_features.permute(0, 3, 1, 2).contiguous()
        reshape_features = reshape_features.view(batch_size, -1, self.d_model)

        # Apply self-attention with residual connection
        projected = self.projection(reshape_features)
        norm1 = self.norm1(projected)
        attention_output = self.self_attention(norm1, norm1, norm1)
        residual1 = projected + attention_output

        # Final projection with residual
        norm2 = self.norm2(residual1)
        output = self.output_proj(norm2)
        output = output + residual1

        return output
