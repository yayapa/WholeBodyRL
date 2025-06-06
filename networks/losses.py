from __future__ import print_function
from matplotlib.pyplot import xcorr
import torch
from typing import Optional
from medutils.measures import psnr
from monai.losses import DiceLoss
import pandas as pd
from networks.metrics import truncated_concordance_td, auc_td, brier_score as bs
from pycox.evaluation import EvalSurv
import numpy as np




class PSNR(torch.nn.Module):
    def __init__(self, max_value=1.0, magnitude_psnr=True):
        super(PSNR, self).__init__()
        self.max_value = max_value
        self.magnitude_psnr = magnitude_psnr

    def forward(self, u, g, mask=None):
        """

        :param u: noised image
        :param g: ground-truth image
        :param mask: mask for the image
        :return:
        """
        if self.magnitude_psnr:
            u, g = torch.abs(u), torch.abs(g)
        batch_size = u.shape[0]
        diff = (u.reshape(batch_size, -1) - g.reshape(batch_size, -1))
        if mask is not None:
            diff = diff[mask.reshape(batch_size, -1) == 1]
        square = torch.conj(diff) * diff
        max_value = g.abs().max() if self.max_value == "on_fly" else self.max_value
        if square.is_complex():
            square = square.real
        v = torch.mean(20 * torch.log10(max_value / torch.sqrt(torch.mean(square, -1))))
        return v
    

class ContrastiveLoss(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.criterion = torch.nn.CosineSimilarity(dim=1)
        
    # def forward(self, p1, p2, z1, z2):
    #     """y is detached from the computation of the gradient.
    #     This is inspired by paper: Chen and He, et al. 2020. Exploring simple siamese representation learning."""
    #     return -(self.criterion(p1, z2).mean() + self.criterion(p2, z1).mean()) * 0.5
    
    def forward(self, p2, z1):
        """y is detached from the computation of the gradient.
        This is inspired by paper: Chen and He, et al. 2020. Exploring simple siamese representation learning."""
        return -self.criterion(p2, z1).mean()
    
    
class ReconstructionCriterion(torch.nn.Module):
    def __init__(self, loss_types, loss_weights: Optional[list[float]] = None, **kwargs):
        super().__init__()
        self.max_value = 1.0
        self.loss_types = loss_types
        self.loss_weights = loss_weights if loss_weights is not None else [1.0] * len(self.loss_types)
        self.loss_fcts = []
        loss_fct_dict = {"mse": torch.nn.MSELoss(reduction="none"), "cl": ContrastiveLoss()}
        if not isinstance(self.loss_types, list):
            self.loss_fcts = [loss_fct_dict[self.loss_types]]
        else:
            self.loss_fcts += [loss_fct_dict[loss_name] for loss_name in self.loss_types]
        
    def forward(self, x, y, mask: Optional[torch.Tensor] = None, **kwargs):
        """Compute reconstruction loss
        x: [B, L, D] torch.Tensor, reconstructed image
        y: [B, L, D] torch.Tensor, reference image
        mask: [B, L] torch.Tensor, mask for encoder, where 0 is keep, 1 is remove
        mask_dropout: [B, L] torch.Tensor, mask for decoder dropout, where 0 is keep, 1 is remove
        
        loss_dict: dict, loss values
        psnr_value: float, psnr value
        """
        assert x.shape[-1] == y.shape[-1]
        
        B = x.shape[0]
        if mask is not None:
            masked_x = x[mask == 1]
            masked_y = y[mask == 1]
        else:
            masked_x = x
            masked_y = y
        # --------------------------------------------------------------------------
        # Calculate losses
        total_loss = 0.0
        loss_dict = {}
        for i, loss_name in enumerate(self.loss_types):
            self.loss_fct = self.loss_fcts[i]
            if loss_name == "cl":
                p2, z1 = kwargs["p2"], kwargs["z1"]
                loss = self.loss_fct(p2, z1)
                # p1, p2, z1, z2 = kwargs["p1"], kwargs["p2"], kwargs["z1"], kwargs["z2"]
                # loss = self.loss_fct(p1, p2, z1, z2)
            else:    
                loss = self.loss_fct(masked_x, masked_y)
                loss = loss.mean()
            loss_dict[loss_name] = loss
            total_loss += self.loss_weights[i] * loss
        loss_dict["loss"] = total_loss
        # --------------------------------------------------------------------------
        psnr_value = psnr(x.detach().cpu().numpy(), y.detach().cpu().numpy(), axes=(-2, -1))
        return loss_dict, psnr_value
    
    
class RegressionCriterion(torch.nn.Module):
    def __init__(self, loss_types, **kwargs):
        super().__init__()
        assert len(loss_types) == 1
        for type in loss_types:
            if type == "mse":
                self.loss_fct = torch.nn.MSELoss(reduction="mean")
            elif type == "huber":
                self.loss_fct = torch.nn.HuberLoss(reduction="mean")
            else:
                raise NotImplementedError("Loss function {} is not implemented for regression".format(type))
    
    def forward(self, x, y):
        loss = 0.0
        loss = self.loss_fct(x, y)
        mae = torch.abs(x - y).detach().mean().item()
        return loss, mae
    

class SegmentationCriterion(torch.nn.Module):
    def __init__(self, loss_types, loss_weights=None, data_view=2, **kwargs):
        super().__init__()
        self.data_view = data_view
        assert len(loss_types) == 1
        for type in loss_types:
            if type == "dice":
                self.loss_fct = DiceLoss(reduction="none")
            else:
                raise NotImplementedError(f"Loss function {type} is not implemented for segmentation")
    
    def forward(self, pred, target):
        """input size: (B, C, S, T, H, W)"""
        pred = pred.moveaxis(1, 2)
        target = target.moveaxis(1, 2)
        B, S = pred.shape[:2]
        slice_mask = torch.ones((B, S), dtype=torch.bool)
        if self.data_view != 0: # If it is either long-axis or both axes
            slice_mask[:, 1] = 0
        pred_ = pred[slice_mask]
        target_ = target[slice_mask]
        loss = self.loss_fct(pred_, target_)
        loss = loss.squeeze().mean(dim=0)
        dice = 1 - loss.detach()
        return loss.mean(), dice




class FineGrayCriterion(torch.nn.Module):
    """Wrapper to choose between total vs. cause‑specific loss."""

    def __init__(self, loss_types, **kwargs):
        super().__init__()
        assert len(loss_types) == 1
        if "total_nfg" in loss_types:
            self.loss_fct = self._total_loss
            self.needs_weight = True
        elif "cause_specific" in loss_types:
            self.loss_fct = self._total_loss_cs
            self.needs_weight = False
        else:
            raise ValueError("loss_type must be 'total' or 'cause_specific'") 
 
    def forward(self, log_hr, log_balance_sr, e, rep, contrastive_weight):
        if self.needs_weight:
            return self.loss_fct(log_hr, log_balance_sr, e, rep, contrastive_weight)
        return self.loss_fct(log_hr, log_balance_sr, e, rep, contrastive_weight)
    
    def _total_loss(self, log_hr, log_balance_sr, e, rep, contrastive_weight):
        """Full Fine‑Gray negative log‑likelihood with optional contrastive term."""
        err = -torch.logsumexp(log_balance_sr[e == 0], dim=1).sum()
        available_risks = torch.unique(e[e > 0]).tolist()
        available_risks = [int(k) for k in available_risks]

        for k in available_risks:
            err -= (log_balance_sr[e == k][:, k-1] + log_hr[e == k]).sum()
        
        survival_loss = err / len(e)  # it was len(t) before

        if rep is not None:
            print("contrastive loss!")
            labels = (e > 0).long()
            features = rep.unsqueeze(1)
            contrastive_loss = supcon_criterion(features, labels)
            return survival_loss + contrastive_weight * contrastive_loss
        return survival_loss

    @staticmethod
    def _total_loss_cs(self, model, x: torch.Tensor, t: torch.Tensor, e: torch.Tensor):
        log_sr, _, tau = model.forward(x, t)
        log_hr = model.gradient(log_sr, tau, e).log()

        err = torch.tensor(0.0, device=x.device, dtype=x.dtype)
        for k in range(model.risks):
            err -= log_sr[e != (k + 1)][:, k].sum()
            err -= log_hr[e == (k + 1)].sum()
        return err / len(t)
    
        

# -----------------------------------------------------------------------------
# Contrastive loss (SupCon)
# -----------------------------------------------------------------------------
class SupConLoss(torch.nn.Module):
    """Supervised Contrastive Loss (from https://arxiv.org/abs/2004.11362)."""

    def __init__(self, temperature: float = 0.07, contrast_mode: str = "all", base_temperature: float = 0.07):
        super().__init__()
        self.temperature = temperature
        self.contrast_mode = contrast_mode
        self.base_temperature = base_temperature

    def forward(self, features: torch.Tensor, labels: Optional[torch.Tensor] = None, mask: Optional[torch.Tensor] = None):
        if features.ndim < 3:
            raise ValueError("`features` needs to be [bsz, n_views, ...]")
        if features.ndim > 3:
            features = features.view(features.shape[0], features.shape[1], -1)

        device = features.device
        batch_size = features.shape[0]

        if labels is None and mask is None:
            mask = torch.eye(batch_size, dtype=torch.float32, device=device)
        elif labels is not None:
            labels = labels.view(-1, 1)
            mask = torch.eq(labels, labels.T).float().to(device)
        else:
            mask = mask.float().to(device)

        contrast_count = features.shape[1]
        contrast_feature = torch.cat(torch.unbind(features, dim=1), dim=0)

        if self.contrast_mode == "one":
            anchor_feature = features[:, 0]
            anchor_count = 1
        else:
            anchor_feature = contrast_feature
            anchor_count = contrast_count

        logits = torch.div(anchor_feature @ contrast_feature.T, self.temperature)
        logits_max, _ = logits.max(dim=1, keepdim=True)
        logits = logits - logits_max.detach()

        mask = mask.repeat(anchor_count, contrast_count)
        logits_mask = torch.ones_like(mask)
        diag = torch.arange(batch_size * anchor_count, device=device)
        logits_mask.scatter_(1, diag.view(-1, 1), 0)
        mask = mask * logits_mask

        exp_logits = torch.exp(logits) * logits_mask
        log_prob = logits - torch.log(exp_logits.sum(1, keepdim=True))

        pos_pairs = mask.sum(1)
        pos_pairs = torch.where(pos_pairs < 1e-6, torch.ones_like(pos_pairs), pos_pairs)
        mean_log_prob_pos = (mask * log_prob).sum(1) / pos_pairs

        loss = -(self.temperature / self.base_temperature) * mean_log_prob_pos
        return loss.view(anchor_count, batch_size).mean()


supcon_criterion = SupConLoss()