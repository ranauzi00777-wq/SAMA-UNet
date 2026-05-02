"""

Created on Thu March  9 22:49:39 2025

@author: drsaq

SAMA-UNet: Self-Adaptive Mamba-like Attention UNet

"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
import math


class DepthwiseSeparableConv(nn.Module):
    """Depthwise Separable Convolution"""
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1):
        super().__init__()
        self.depthwise = nn.Conv2d(in_channels, in_channels, kernel_size, 
                                   stride, padding, groups=in_channels, bias=False)
        self.pointwise = nn.Conv2d(in_channels, out_channels, 1, bias=False)
        self.bn = nn.BatchNorm2d(out_channels)
        
    def forward(self, x):
        x = self.depthwise(x)
        x = self.pointwise(x)
        x = self.bn(x)
        return x


class LocallyEnhancedPE(nn.Module):
    """Locally Enhanced Positional Encoding using convolution"""
    def __init__(self, dim, kernel_size=3):
        super().__init__()
        self.conv = nn.Conv2d(dim, dim, kernel_size, padding=kernel_size//2, groups=dim)
        
    def forward(self, x):
        # x: (B, C, H, W)
        return x + self.conv(x)


class DifferentialAttention(nn.Module):
    """
    Differential Attention mechanism
    Computes: (softmax(Q1*K1^T) - λ*softmax(Q2*K2^T)) * V
    """
    def __init__(self, dim, num_heads=8, qkv_bias=False, lambda_init=0.8):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim ** -0.5
        
        # Split into two sets for differential attention
        self.qkv = nn.Linear(dim, dim * 3 * 2, bias=qkv_bias)  # *2 for two sets
        self.lambda_param = nn.Parameter(torch.tensor(lambda_init))
        self.proj = nn.Linear(dim, dim)
        
    def forward(self, x):
        B, N, C = x.shape
        
        # Generate Q, K, V for two attention heads
        qkv = self.qkv(x).reshape(B, N, 2, 3, self.num_heads, self.head_dim)
        qkv = qkv.permute(2, 3, 0, 4, 1, 5)  # (2, 3, B, num_heads, N, head_dim)
        
        q1, k1, v = qkv[0][0], qkv[0][1], qkv[0][2]
        q2, k2, _ = qkv[1][0], qkv[1][1], qkv[1][2]
        
        # Compute attention scores for both sets
        attn1 = (q1 @ k1.transpose(-2, -1)) * self.scale
        attn1 = F.softmax(attn1, dim=-1)
        
        attn2 = (q2 @ k2.transpose(-2, -1)) * self.scale
        attn2 = F.softmax(attn2, dim=-1)
        
        # Differential attention
        attn_diff = attn1 - self.lambda_param * attn2
        
        # Apply attention to values
        x = (attn_diff @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        
        return x


class DifferentialAggregatedAttention(nn.Module):
    """
    Differential Aggregated Attention for local or global branch
    Includes Group Normalization and Positional Encoding
    """
    def __init__(self, dim, num_heads=8, window_size=7, is_local=True):
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.window_size = window_size
        self.is_local = is_local
        
        self.diff_attn = DifferentialAttention(dim, num_heads)
        self.norm = nn.GroupNorm(num_groups=8, num_channels=dim)
        self.pe = LocallyEnhancedPE(dim)
        
    def forward(self, x):
        """
        x: (B, C, H, W)
        """
        B, C, H, W = x.shape
        
        if self.is_local:
            # Local attention with windowing (unfold operation)
            # For simplicity, we use standard attention on the full feature map
            # In practice, you would implement window-based attention
            x_flat = x.flatten(2).transpose(1, 2)  # (B, H*W, C)
        else:
            # Global attention with average pooling for efficiency
            x_pooled = F.adaptive_avg_pool2d(x, (H//4, W//4))
            x_flat = x_pooled.flatten(2).transpose(1, 2)  # (B, H'*W', C)
        
        # Apply differential attention
        attn_out = self.diff_attn(x_flat)
        
        if self.is_local:
            attn_out = attn_out.transpose(1, 2).reshape(B, C, H, W)
        else:
            attn_out = attn_out.transpose(1, 2).reshape(B, C, H//4, W//4)
            attn_out = F.interpolate(attn_out, size=(H, W), mode='bilinear', align_corners=False)
        
        # Add positional encoding
        attn_out = self.pe(attn_out)
        
        # Group normalization
        attn_out = self.norm(attn_out)
        
        return attn_out


class SAMA_Block(nn.Module):
    """
    Self-Adaptive Mamba-like Aggregated Attention Block
    Implements the architecture shown in Figure 2(c) of the paper
    """
    def __init__(self, dim, num_heads=8, mlp_ratio=4., window_size=7):
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        
        # Initial linear layer and activation
        self.linear1 = nn.Linear(dim, dim)
        self.silu1 = nn.SiLU()
        
        # Depthwise convolution
        self.dconv = nn.Conv2d(dim, dim, kernel_size=3, padding=1, groups=dim)
        
        # Split channels for local and global branches
        self.split_ratio = 0.5
        local_dim = int(dim * self.split_ratio)
        global_dim = dim - local_dim
        
        # Local and Global Differential Aggregated Attention
        self.local_attn = DifferentialAggregatedAttention(local_dim, num_heads//2, window_size, is_local=True)
        self.global_attn = DifferentialAggregatedAttention(global_dim, num_heads//2, window_size, is_local=False)
        
        # Bypass branch
        self.linear_bypass = nn.Linear(dim, dim)
        self.silu_bypass = nn.SiLU()
        
        # Final projection
        self.linear_out = nn.Linear(dim, dim)
        
        # FFN sub-block
        self.norm = nn.LayerNorm(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = nn.Sequential(
            nn.Linear(dim, mlp_hidden_dim),
            nn.GELU(),
            nn.Linear(mlp_hidden_dim, dim)
        )
        
    def forward(self, x):
        """
        x: (B, C, H, W)
        """
        B, C, H, W = x.shape
        identity = x
        
        # Convert to (B, H*W, C) for linear layers
        x_flat = x.flatten(2).transpose(1, 2)
        
        # Initial processing: SiLU(DepthConv(Linear(I)))
        x_processed = self.linear1(x_flat)
        x_processed = x_processed.transpose(1, 2).reshape(B, C, H, W)
        x_processed = self.silu1(x_processed)
        x_processed = self.dconv(x_processed)
        
        # Split into local and global branches
        local_dim = int(C * self.split_ratio)
        x_local, x_global = x_processed.split([local_dim, C - local_dim], dim=1)
        
        # Apply DiffAgg on both branches
        local_out = self.local_attn(x_local)
        global_out = self.global_attn(x_global)
        
        # Concatenate
        x_concat = torch.cat([local_out, global_out], dim=1)  # (B, C, H, W)
        
        # Bypass branch: SiLU(Linear(I))
        x_bypass = self.linear_bypass(x_flat)
        x_bypass = x_bypass.transpose(1, 2).reshape(B, C, H, W)
        x_bypass = self.silu_bypass(x_bypass)
        
        # Element-wise multiplication with bypass
        x_concat_flat = x_concat.flatten(2).transpose(1, 2)
        x_bypass_flat = x_bypass.flatten(2).transpose(1, 2)
        x_mult = x_concat_flat * x_bypass_flat
        
        # Final linear projection
        x_out = self.linear_out(x_mult)
        x_out = x_out.transpose(1, 2).reshape(B, C, H, W)
        
        # Residual connection
        x_out = x_out + identity
        
        # FFN sub-block with residual
        x_out_flat = x_out.flatten(2).transpose(1, 2)
        x_out_flat = x_out_flat + self.mlp(self.norm(x_out_flat))
        x_out = x_out_flat.transpose(1, 2).reshape(B, C, H, W)
        
        return x_out

import torch
import torch.nn as nn
import torch.nn.functional as F

class SSM2D(nn.Module):
    def __init__(self, dim, d_state=16):
        super().__init__()

        self.dim = dim
        self.d_state = d_state

        # state transition
        self.A = nn.Parameter(torch.randn(d_state, d_state) * 0.1)

        # projections
        self.B = nn.Linear(dim, d_state)
        self.C = nn.Linear(d_state, dim)

        # input-dependent step size
        self.delta = nn.Linear(dim, d_state)

        # skip scaling (IMPORTANT: use scalar, not vector)
        self.D = nn.Parameter(torch.ones(1))

        # normalizations (IMPORTANT FIX)
        self.in_norm = nn.LayerNorm(dim)
        self.state_norm = nn.LayerNorm(d_state)

        self.out_norm = nn.LayerNorm(dim)

    def forward(self, x):
        B, L, C = x.shape

        # normalize input first (IMPORTANT STABILITY FIX)
        x = self.in_norm(x)

        # projections
        B_state = self.B(x)
        delta = F.softplus(self.delta(x))

        # init state
        h = torch.zeros(B, self.d_state, device=x.device)

        states = []

        for t in range(L):
            Ah = torch.matmul(h, self.A.T)

            h = h + delta[:, t] * Ah + B_state[:, t]

            # stabilize state
            h = self.state_norm(h)

            states.append(h)

        states = torch.stack(states, dim=1)

        # decode
        y = self.C(states)

        # residual connection (scaled)
        y = y + self.D * x

        # final norm
        y = self.out_norm(y)

        return y


class CausalResonanceMultiScaleModule(nn.Module):
    """
    CR-MSM: Causal-Resonance Multi-Scale Module
    Implements Algorithm 1 from the paper
    """
    def __init__(self, dim, d_state=16):
        super().__init__()
        self.dim = dim
        
        # SSM for processing directional views
        self.ssm = SSM2D(dim, d_state)
        
        # Linear projection for skip connection
        self.proj = nn.Linear(dim, dim)
        
    def forward(self, x):
        """
        x: encoder feature map (B, C, H, W)
        Returns: fused feature for skip connection (B, C, H, W)
        """
        B, C, H, W = x.shape
        
        # Generate four directional views
        x_orig = x  # Original
        x_T = x.transpose(2, 3)  # Transposed
        x_flip = torch.flip(x, dims=[2])  # Flipped (vertical)
        x_flipT = torch.flip(x.transpose(2, 3), dims=[2])  # Flipped-Transposed
        
        views = [x_orig, x_T, x_flip, x_flipT]
        outputs = []
        
        for view in views:
            # Flatten to sequence
            view_flat = view.flatten(2).transpose(1, 2)  # (B, H*W, C)
            
            # Apply SSM
            view_out = self.ssm(view_flat)  # (B, H*W, C)
            
            # Reshape back
            view_out = view_out.transpose(1, 2).reshape(B, C, H, W)
            outputs.append(view_out)
        
        # Restore original orientation for each view
        y1 = outputs[0]
        y2 = outputs[1].transpose(2, 3)
        y3 = torch.flip(outputs[2], dims=[2])
        y4 = torch.flip(outputs[3], dims=[2]).transpose(2, 3)
        
        # Causal-resonance averaging (Equation in paper)
        y_fused = (y1 + y2 + y3 + y4) / 4.0
        
        # Linear projection
        y_fused_flat = y_fused.flatten(2).transpose(1, 2)
        z = self.proj(y_fused_flat)
        z = z.transpose(1, 2).reshape(B, C, H, W)
        
        return z


class PatchEmbedding(nn.Module):
    """
    Patch Embedding with overlapping patches
    """
    def __init__(self, in_channels=3, embed_dim=96, patch_size=4, stride=4):
        super().__init__()
        self.proj = nn.Conv2d(in_channels, embed_dim, 
                             kernel_size=patch_size, stride=stride,
                             padding=patch_size//2)
        self.norm = nn.LayerNorm(embed_dim)
        
    def forward(self, x):
        x = self.proj(x)  # (B, C, H, W)
        B, C, H, W = x.shape
        x = x.flatten(2).transpose(1, 2)  # (B, H*W, C)
        x = self.norm(x)
        x = x.transpose(1, 2).reshape(B, C, H, W)
        return x


class DownSampling(nn.Module):
    """Downsampling module"""
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, 
                             kernel_size=2, stride=2)
        self.norm = nn.BatchNorm2d(out_channels)
        
    def forward(self, x):
        return self.norm(self.conv(x))


class UpSampling(nn.Module):
    """Upsampling module using transpose convolution"""
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.up = nn.ConvTranspose2d(in_channels, out_channels,
                                     kernel_size=2, stride=2)
        self.norm = nn.BatchNorm2d(out_channels)
        
    def forward(self, x):
        return self.norm(self.up(x))


class ResidualConvBlock(nn.Module):
    """Residual Convolution Block for decoder"""
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, 3, padding=1)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, 3, padding=1)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        
        self.shortcut = nn.Sequential()
        if in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, 1),
                nn.BatchNorm2d(out_channels)
            )
            
    def forward(self, x):
        identity = self.shortcut(x)
        
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += identity
        out = self.relu(out)
        
        return out


class SAMA_UNet(nn.Module):
    """
    SAMA-UNet: Complete architecture
    Default configuration for BTCV dataset (224x224 input)
    """
    def __init__(self, 
                 in_channels=1,
                 num_classes=14,
                 embed_dims=[96, 192, 384, 768],
                 depths=[2, 2, 2, 2],
                 num_heads=[3, 6, 12, 24]):
        super().__init__()
        
        self.num_classes = num_classes
        self.depths = depths
        
        # Patch embedding
        self.patch_embed = PatchEmbedding(in_channels, embed_dims[0], patch_size=4, stride=4)
        
        # Encoder stages
        self.encoder_stages = nn.ModuleList()
        self.downsamplers = nn.ModuleList()
        
        for i in range(len(depths)):
            stage = nn.ModuleList([
                SAMA_Block(embed_dims[i], num_heads[i]) 
                for _ in range(depths[i])
            ])
            self.encoder_stages.append(stage)
            
            if i < len(depths) - 1:
                self.downsamplers.append(
                    DownSampling(embed_dims[i], embed_dims[i+1])
                )
        
        # CR-MSM modules for skip connections
        self.cr_msm_modules = nn.ModuleList([
            CausalResonanceMultiScaleModule(embed_dims[i])
            for i in range(len(depths))
        ])
        
        # Decoder stages
        self.upsamplers = nn.ModuleList()
        self.decoder_stages = nn.ModuleList()
        
        for i in range(len(depths)-1, 0, -1):
            self.upsamplers.append(
                UpSampling(embed_dims[i], embed_dims[i-1])
            )
            self.decoder_stages.append(
                ResidualConvBlock(embed_dims[i-1] * 2, embed_dims[i-1])
            )
        
        # Final segmentation head
        self.seg_head = nn.Conv2d(embed_dims[0], num_classes, 1)
        
    def forward(self, x):
        """
        x: (B, C, H, W) where H=W=224 for BTCV
        """
        # Patch embedding
        x = self.patch_embed(x)  # (B, 96, 56, 56) for 224x224 input
        
        # Encoder with skip connections
        encoder_features = []
        
        for i, stage in enumerate(self.encoder_stages):
            for block in stage:
                x = block(x)
            
            # Apply CR-MSM for skip connection
            skip_feat = self.cr_msm_modules[i](x)
            encoder_features.append(skip_feat)
            
            # Downsample (except last stage)
            if i < len(self.encoder_stages) - 1:
                x = self.downsamplers[i](x)
        
        # Decoder
        for i, (upsampler, decoder_block) in enumerate(zip(self.upsamplers, self.decoder_stages)):
            # Upsample
            x = upsampler(x)
            
            # Concatenate with skip connection
            skip_idx = len(encoder_features) - 2 - i
            skip = encoder_features[skip_idx]
            
            x = torch.cat([x, skip], dim=1)
            
            # Decoder block
            x = decoder_block(x)
        
        # Final segmentation
        x = self.seg_head(x)
        
        # Upsample to original resolution if needed
        x = F.interpolate(x, scale_factor=4, mode='bilinear', align_corners=False)
        
        return x


if __name__ == "__main__":
    # Test the model
    model = SAMA_UNet(in_channels=1, num_classes=14)
    x = torch.randn(2, 1, 224, 224)
    
    print(f"Input shape: {x.shape}")
    output = model(x)
    print(f"Output shape: {output.shape}")
    
    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Total parameters: {total_params / 1e6:.2f}M")
