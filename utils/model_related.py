from typing import Optional, Tuple
from requests import patch
import torch
import torch.nn.functional as F
import numpy as np
from scipy import interpolate
from torchvision.transforms import ColorJitter
import torch.nn as nn
from torch.distributions.multivariate_normal import MultivariateNormal


# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
# --------------------------------------------------------
# Position embedding utils
# --------------------------------------------------------

# --------------------------------------------------------
# 2D sine-cosine position embedding
# References:
# Transformer: https://github.com/tensorflow/models/blob/master/official/nlp/transformer/model_utils.py
# MoCo v3: https://github.com/facebookresearch/moco-v3
# --------------------------------------------------------
def sincos_pos_embed(embed_dim, grid_size, cls_token=False, use_both_axes=True):
    """
    Generate multi-scale sine-cosine position embedding for 3D images with two contrast channels (treated as distinct views).

    Args:
        embed_dim: Output dimension for each position.
        grid_size: Tuple for the grid (C, D, H, W) where C is the number of contrast channels.
        cls_token: If True, add a class token embedding.
        use_both_axes: If True, adds embedding to distinguish between the two contrast axes.

    Returns:
        pos_embed: [np.prod(grid_size), embed_dim] or [1+np.prod(grid_size), embed_dim] (with or without cls_token).
    """
    print(f"Grid size: {grid_size}")
    print(f"embed_dim: {embed_dim}")
    if use_both_axes:
        grid_dim = len(grid_size)

        assert grid_dim == 4, "Grid_size should be 4D for contrast and spatial positional embedding (C, D, H, W)"
        assert (embed_dim - 1) % (grid_dim * 2) == 0, "Each dimension must have 2 channels (sin, cos)"

        # Split the grid into contrast-specific segments
        grid_size_contrast1 = (1, *grid_size[1:])  # First contrast
        grid_size_contrast2 = (1, *grid_size[1:])  # Second contrast

        # Generate the meshgrid for spatial dimensions (D, H, W) for the first contrast
        grid_contrast1 = torch.meshgrid(*[torch.arange(s, dtype=torch.float32)
                                          for s in grid_size_contrast1], indexing="ij")  # (1, D, H, W)
        grid_contrast1 = torch.stack(grid_contrast1, dim=0)  # (4, 1, D, H, W)
        pos_embed_contrast1 = get_multi_sincos_pos_embed_from_grid(embed_dim - 1,
                                                                   grid_contrast1)  # (D * H * W, embed_dim - 1)

        # Generate the meshgrid for spatial dimensions (D, H, W) for the second contrast
        grid_contrast2 = torch.meshgrid(*[torch.arange(s, dtype=torch.float32)
                                          for s in grid_size_contrast2], indexing="ij")  # (1, D, H, W)
        grid_contrast2 = torch.stack(grid_contrast2, dim=0)  # (4, 1, D, H, W)
        pos_embed_contrast2 = get_multi_sincos_pos_embed_from_grid(embed_dim - 1,
                                                                   grid_contrast2)  # (D * H * W, embed_dim - 1)

        # Concatenate contrast-specific position embeddings and add a contrast distinguishing embedding
        contrast_indicator = torch.cat([torch.zeros([pos_embed_contrast1.shape[0], 1]),  # Contrast 1
                                        torch.ones([pos_embed_contrast2.shape[0], 1])],
                                       dim=0)  # Contrast 2 (binary indicator)
        pos_embed = torch.cat([pos_embed_contrast1, pos_embed_contrast2], dim=0)  # Concatenate both contrasts
        pos_embed = torch.cat([contrast_indicator, pos_embed], dim=1)  # Concatenate contrast identifier

    else:
        grid_dim = len(grid_size)
        assert grid_dim >= 3, "Grid_size should be at least 3D for positional embedding"
        assert embed_dim % (grid_dim * 2) == 0, "Each dimension has 2 channels (sin, cos)"

        # Generate meshgrid for spatial dimensions
        grid = torch.meshgrid(*[torch.arange(s, dtype=torch.float32) for s in grid_size], indexing="ij")  # (C, D, H, W)
        grid = torch.stack(grid, dim=0)  # (4, C, D, H, W)
        pos_embed = get_multi_sincos_pos_embed_from_grid(embed_dim, grid)

    # Add class token if needed
    if cls_token:
        pos_embed = torch.cat([torch.zeros([1, embed_dim]), pos_embed], dim=0)

    return pos_embed


def get_multi_sincos_pos_embed_from_grid(embed_dim, grid):
    # use half of dimensions to encode grid
    grid_dim = len(grid.shape) - 1
    emb = [get_1d_sincos_pos_embed_from_grid(embed_dim // grid_dim, grid[i]) for i in range(grid.shape[0])]
    emb = torch.concatenate(emb, dim=1)  # [(S*T*H*W, D/4)] -> (S*T*H*W, D)
    return emb


def get_1d_sincos_pos_embed_from_grid(embed_dim, pos):
    """
    embed_dim: output dimension for each position
    pos: a list of positions to be encoded: size (M,)
    out: (M, D)
    """
    assert embed_dim % 2 == 0
    omega = torch.arange(embed_dim // 2, dtype=torch.float)
    omega /= embed_dim / 2.
    omega = 1. / 10000 ** omega  # (D/2,)

    pos = pos.reshape(-1)  # (M,)
    out = torch.einsum("m,d->md", pos, omega)  # (M, D/2), outer product

    emb_sin = torch.sin(out)  # (M, D/2)
    emb_cos = torch.cos(out)  # (M, D/2)

    emb = torch.concatenate([emb_sin, emb_cos], dim=1)  # (M, D)
    return emb


def patchify(im: torch.Tensor, patch_size: list[int, int, int] = [5, 16, 16]):
    """Split image into patches of size patch_size.

    im: [B, S, T, H, W]
    patch_size: a list of 3
    x: [B, L, np.prod(patch_size)] where L = S * T * H * W / np.prod(patch_size)
    """
    assert len(im.shape) == 5
    assert len(patch_size) == 3

    B, S, T, H, W = im.shape
    t, h, w = T // patch_size[0], H // patch_size[1], W // patch_size[2]
    x = im.reshape(B, S, t, patch_size[0], h, patch_size[1], w, patch_size[2])
    x = torch.einsum("bstphqwr->bsthwpqr", x)
    x = x.reshape(B, S * t * h * w, np.prod(patch_size))
    return x


def unpatchify(x: torch.Tensor, im_shape: list[int], patch_size: list[int, int, int] = [5, 16, 16]):
    """Combine patches into image.

    x: [B, L, np.prod(patch_size) or T * np.prod(patch_size)]
    im_shape: [B, S, T, X, Y]
    im: [B, S, T, X, Y] where X = Y
    """
    assert len(x.shape) == 3
    assert len(patch_size) == 3
    assert len(im_shape) == 5

    B, S, T, H, W = im_shape
    t, h, w = T // patch_size[0], H // patch_size[1], W // patch_size[2]
    x = x.reshape(B, S, t, h, w, patch_size[0], patch_size[1], patch_size[2])
    x = torch.einsum("bsthwpqr->bstphqwr", x)
    im = x.reshape(im_shape)
    return im


class PatchEmbed(nn.Module):
    def __init__(self, im_shape: list[int], 
                 in_channels: int = 1, 
                 patch_size: list[int] = [1, 16, 16], 
                 out_channels: int = 256, 
                 flatten: bool = True, 
                 bias: bool = True,
                 norm_layer: Optional[nn.Module] = None):
        super().__init__()
        
        assert len(patch_size) == 3, "Patch size should be 3D"
        assert in_channels == 1, "Only supporting input channel size as 1"
        self.im_shape = im_shape
        if len(im_shape) == 3:
            self.im_shape = im_shape.unsqueeze(1) # (S, H, W) -> (S, 1, H, W)
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.flatten = flatten
        
        self.proj = nn.Conv3d(in_channels, out_channels, kernel_size=patch_size, stride=patch_size, bias=bias)
        self.norm = norm_layer(out_channels) if norm_layer else nn.Identity()
        self.grid_size = (im_shape[0], 
                          im_shape[1] // patch_size[0], 
                          im_shape[2] // patch_size[1], 
                          im_shape[3] // patch_size[2]) # (S, t, h, w)
        self.num_patches = np.prod(self.grid_size)
    
    def forward(self, x):
        """ 
        input: (B, S, T, H, W)
        output: (B * S, out_channels, t, h, w) or (B, num_patches, out_channels) if flatten is True, 
                where num_patches = S * T * H * W / np.prod(patch_size)
        """
        x_ = x.reshape(-1, *self.im_shape[-3:]) # (B*S, T, H, W)
        x_ = x_.unsqueeze(1) # (B*S, 1, T, H, W)
        x_ = self.proj(x_) # (B*S, out_channels, t, h, w)
        
        if self.flatten:
            x__ = x_.flatten(2) # (B*S, out_channels, t*h*w)
            x__ = x__.moveaxis(1, -1) # (B*S, t*h*w, out_channels)
            x_ = x__.reshape(x.shape[0], -1, x__.shape[-1]) # (B, S*t*h*w, out_channels)
        else:
            x_ = x_.moveaxis(1, -1)
            
        x = self.norm(x_)
        return x

    
class Masker:
    def __init__(self, mask_type, mask_ratio, grid_size, **kwargs):
        self.mask_ratio = mask_ratio
        if mask_type == "random_roi":
            self.masking_strategy = self.random_masking_within_roi
        elif mask_type == "random_roi_blockwise":
            self.masking_strategy = self.random_masking_within_roi_blockwise
        elif mask_type == "random":
            self.masking_strategy = self.random_masking
        else:
            raise NotImplementedError
    
    def __call__(self, x, roi_mask=None):
        """
        x: [N, L, D], sequence
        x_masked: [N, L * mask_ratio, D], masked sequence
        mask: [N, L], binary mask
        ids_restore: [N, L], indices to restore the original order
        """
        mask, ids_restore, ids_keep = self.masking_strategy(input_size=x.shape, device=x.device, roi_mask=roi_mask)
        x_masked = torch.gather(x, dim=1, index=ids_keep.unsqueeze(-1).repeat(1, 1, x.shape[-1]))
        return x_masked, mask, ids_restore
    
    def call_masking_fctn(self, x, fctn_name, **kwargs):
        fctn = eval(f"self.{fctn_name}")
        mask, ids_restore, ids_keep = fctn(input_size=x.shape, device=x.device, **kwargs)
        x_masked = torch.gather(x, dim=1, index=ids_keep.unsqueeze(-1).repeat(1, 1, x.shape[-1]))
        return x_masked, mask, ids_restore
        
    def random_masking(self, input_size, device, **kwargs):
        """
        # Reference: https://github.com/facebookresearch/mae.git
        
        Perform per-sample random masking by per-sample shuffling.
        Per-sample shuffling is done by argsort random noise.
        
        input_size: [N, L, D], sequence
        device: torch.device
        mask: [N, L], binary mask
        ids_restore: [N, L], indices to restore the original order
        """
        #print("Ranodm Masking")
        N, L, D = input_size  # batch, length, dim
        len_keep = int(L * (1 - self.mask_ratio))
        
        noise = torch.rand(N, L, device=device)  # noise in [0, 1]
        
        # Sort noise for each sample
        ids_shuffle = torch.argsort(noise, dim=1)  # ascend: small is keep, large is remove
        ids_restore = torch.argsort(ids_shuffle, dim=1)

        # Keep the first subset
        ids_keep = ids_shuffle[:, :len_keep]

        # Generate the binary mask: 0 is keep, 1 is remove
        mask = torch.ones([N, L], device=device)
        mask[:, :len_keep] = 0
        
        # Unshuffle to get the binary mask
        mask = torch.gather(mask, dim=1, index=ids_restore)

        return mask, ids_restore, ids_keep

    def random_masking_plus_given_index(self, input_size, device, given_index, **kwargs):
        """
        Perform per-sample random masking by per-sample shuffling and also mask out the given index.
        input_size: [N, L, D], sequence
        given_index: [L_g], given index to be masked out
        device: torch.device
        mask: [N, L], binary mask
        ids_restore: [N, L], indices to restore the original order
        """
        N, L, D = input_size
        maskout_index_total = np.union1d(given_index, np.random.choice(L, int(L * self.mask_ratio), replace=False))
        len_keep = L - len(maskout_index_total)
        mask = torch.zeros([N, L], device=device)
        mask[:, maskout_index_total] = 1
        ids_shuffle = torch.argsort(mask, dim=1)
        ids_restore = torch.argsort(ids_shuffle, dim=1)
        ids_keep = ids_shuffle[:, :len_keep]
        return mask, ids_restore, ids_keep

    def random_masking_within_roi(self, input_size, device, roi_mask, **kwargs):
        """
        Perform random masking only within the ROI region defined by `roi_mask`.

        Args:
            input_size: Tuple[int, int, int], dimensions of the input sequence (e.g., [N, L, D]).
            device: torch.device.
            roi_mask: [N, L], binary mask indicating ROI patches after patchification.

        Returns:
            mask: [N, L], binary mask (0 is keep, 1 is masked).
            ids_restore: [N, L], indices to restore the original order.
            ids_keep: [N, len_keep], indices of kept patches.
        """
        #print("Random Masking within ROI")

        num_roi_patches = torch.sum(roi_mask, dim=1)

        N, L, D = input_size  # Extract shape directly

        # Ensure roi_mask is binary and matches the input
        assert roi_mask.shape == (N, L), "ROI mask must have shape [N, L]"
        #print(f"ROI mask binary shape: {roi_mask.shape}")  # Expected: [4, 5376]

        # Calculate the total number of spatches to mask
        #num_to_mask = int(L * self.mask_ratio)  # Fixed number of patches to mask per sample
        num_to_mask = int(torch.min(num_roi_patches).item() * self.mask_ratio)  # Mask a fraction of ROI patches
        #num_to_mask = num_roi_patches * self.mask_ratio

        # Initialize mask as all zeros (unmasked)
        mask = torch.zeros([N, L], device=device)

        # For each sample in the batch
        for i in range(N):
            # Get indices of ROI patches
            roi_indices = torch.nonzero(roi_mask[i], as_tuple=True)[0]  # Indices of ROI patches for sample `i`

            if len(roi_indices) == 0:
                raise ValueError(f"Sample {i} has no ROI patches to mask.")

            # Shuffle ROI indices
            shuffled_indices = roi_indices[torch.randperm(len(roi_indices))]

            # Mask the first `num_to_mask` ROI patches
            num_to_mask_i = min(num_to_mask, len(roi_indices))  # Limit to the number of available ROI patches
            mask[i, shuffled_indices[:num_to_mask_i]] = 1  # Set these patches as masked

        # Generate ids_restore for reordering patches
        noise = torch.rand(N, L, device=device)
        ids_shuffle = torch.argsort(noise, dim=1)
        ids_restore = torch.argsort(ids_shuffle, dim=1)

        # Recompute ids_keep: Indices where mask == 0 (patches to keep)
        ids_keep = torch.nonzero(mask == 0, as_tuple=True)[1]
        ids_keep = ids_keep.reshape(N, -1)  # Reshape to [N, len_keep]

        return mask, ids_restore, ids_keep
    
    def random_masking_within_roi_blockwise(self, input_size, device, roi_mask, **kwargs):
        """
        Perform random masking only within the ROI region defined by `roi_mask`.
        Args:
            input_size: Tuple[int, int, int], dimensions of the input sequence (e.g., [N, L, D]).
            device: torch.device.
            roi_mask: [N, 2 * L], binary mask indicating ROI patches for both contrasts.

        Returns:
            mask: [N, 2 * L], binary mask (0 is keep, 1 is masked).
            ids_restore: [N, 2 * L], indices to restore the original order.
            ids_keep: [N, len_keep], indices of kept patches.
        """
        print("Random Masking within ROI blockwise")
        N, L, D = input_size  # Batch size, total patches, embedding dimension
        num_patches_per_channel = L // 2  # Divide total patches equally for two channels

        # Reshape the mask to separate channels
        roi_mask = roi_mask.reshape(N, 2, num_patches_per_channel)  # [N, 2, num_patches_per_channel]


        # Verify that both channels have identical masks
        assert torch.all(roi_mask[:, 0, :] == roi_mask[:, 1, :]), "Masks for both channels must be identical."
        roi_mask_collapsed = roi_mask[:, 0, :]  # Use only one channel's mask

        # Number of ROI patches per sample
        num_roi_patches = torch.sum(roi_mask_collapsed, dim=1)  # [N]

        # Determine the number of patches to mask
        num_to_mask = (num_roi_patches * self.mask_ratio).long()

        # Initialize mask as all zeros (unmasked)
        mask_per_channel = torch.zeros([N, num_patches_per_channel], device=device)

        for i in range(N):
            # Get indices of ROI patches
            roi_indices = torch.nonzero(roi_mask_collapsed[i], as_tuple=True)[0]

            if len(roi_indices) == 0:
                raise ValueError(f"Sample {i} has no ROI patches to mask.")

            # Shuffle ROI indices
            shuffled_indices = roi_indices[torch.randperm(len(roi_indices))]

            # Mask the first `num_to_mask` ROI patches
            num_to_mask_i = min(num_to_mask[i].item(), len(roi_indices))
            mask_per_channel[i, shuffled_indices[:num_to_mask_i]] = 1

        # Repeat the mask for both channels
        mask = mask_per_channel.unsqueeze(1).repeat(1, 2, 1)  # [N, 2, num_patches_per_channel]
        assert mask[:, 0, :].equal(mask[:, 1, :]), "Masks for both channels must be identical."
        mask = mask.reshape(N, L)  # Flatten back to [N, L]

        # Generate ids_restore for reordering patches
        noise = torch.rand(N, L, device=device)
        ids_shuffle = torch.argsort(noise, dim=1)
        ids_restore = torch.argsort(ids_shuffle, dim=1)

        # Recompute ids_keep: Indices where mask == 0 (patches to keep)
        ids_keep = torch.nonzero(mask == 0, as_tuple=True)[1]
        ids_keep = ids_keep.reshape(N, -1)  # Reshape to [N, len_keep]

        return mask, ids_restore, ids_keep




