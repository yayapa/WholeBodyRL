import os.path
import time

from IPython import embed
import lightning.pytorch as pl
import numpy as np
import torch
from torch import embedding, nn, optim
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts

from networks.losses import ReconstructionCriterion
from networks.decoders import ViTDecoder
from utils.logging_related import imgs_to_wandb_video, replace_with_gt_wandb_video, CustomWandbLogger
from utils.model_related import Masker, PatchEmbed, sincos_pos_embed, patchify, unpatchify
from timm.models.vision_transformer import Block
from data.dataloaders import RandomDistributedSampler

from utils.train_related import add_weight_decay


class BasicModule(pl.LightningModule):
    def __init__(self, *args, **kwargs):
        super().__init__()

        self.lr = kwargs.get("lr")
        self.train_rate = kwargs.get("train_log_rate")
        self.val_rate = kwargs.get("val_log_rate")
        self.warmup_epochs = kwargs.get("warmup_epochs")
        self.min_lr = kwargs.get("min_lr")
        self.optim_weight_decay = kwargs.get("optim_weight_decay")

        self.train_epoch_start_time = None
        self.val_epoch_start_time = None
        self.test_epoch_start_time = None
        self.batch_size = kwargs.get("batch_size")
        self.train_num_per_epoch = kwargs.get("train_num_per_epoch")
        self.max_lr = kwargs.get("max_lr")
        self.module_logger = CustomWandbLogger()
        self.multi_gpu = kwargs.get("multi_gpu", False)
        
    def configure_optimizers(self):
        # param_groups = add_weight_decay(self.named_parameters(), self.optim_weight_decay)
        # optimizer = optim.Adam(param_groups, lr=self.lr)
        optimizer = optim.Adam(self.parameters(), lr=self.lr)

        # scheduler = CosineAnnealingWarmRestarts(optimizer, T_0=self.warmup_epochs, eta_min=self.min_lr)
        # return {"optimizer": optimizer, "lr_scheduler": {"scheduler": scheduler}}
        return optimizer

    """
    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.lr)

        steps_per_epoch = self.train_num_per_epoch // self.batch_size   # Number of steps per epoch
        scheduler = torch.optim.lr_scheduler.OneCycleLR(
            optimizer,
            max_lr=self.max_lr,  # Peak learning rate
            steps_per_epoch=steps_per_epoch,  # Number of steps per epoch
            epochs=self.trainer.max_epochs,  # Total number of epochs
            pct_start=0.3  # Percentage of the cycle spent increasing LR
        )
        return [optimizer], [{"scheduler": scheduler, "interval": "step"}]
    """
    def on_train_epoch_start(self) -> None:
        self.train_epoch_start_time = time.time()
        self.module_logger.reset_item()
        # ser epoch in sampler of train dataloader
        if self.multi_gpu:
            datamodule = self.trainer.datamodule
            if hasattr(datamodule, '_train_dataloader') and hasattr(datamodule._train_dataloader, 'sampler'):
                if isinstance(datamodule._train_dataloader.sampler, RandomDistributedSampler):
                    datamodule._train_dataloader.sampler.set_epoch(self.current_epoch)

    def on_validation_epoch_end(self) -> None:
        self.module_logger.wandb_log(self.current_epoch, mode="val")

    def on_train_epoch_end(self) -> None:
        epoch_runtime = (time.time() - self.train_epoch_start_time) / 60
        self.module_logger.update_metric_item("train/epoch_runtime", epoch_runtime, mode="train")
        self.module_logger.update_metric_item("train/lr", self.lr, mode="train")
        self.module_logger.wandb_log(self.current_epoch, mode="train")
        

class ReconMAE(BasicModule):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        
        self.mask_type = kwargs.get("mask_type")
        self.enc_embed_dim = kwargs.get("enc_embed_dim")
        self.dec_embed_dim = kwargs.get("dec_embed_dim")
        self.patch_size = kwargs.get("patch_size")
        self.patch_embed_cls = globals()[kwargs.get("patch_embed_cls")]
        val_dataset= kwargs.get("val_dset")
        self.img_shape = val_dataset[0][0].shape
        self.use_both_axes = True if val_dataset.get_view() == 2 else False # For positional embedding
        self.patch_p_num = np.prod(kwargs.get("patch_size")) * kwargs.get("patch_in_channels")
        # --------------------------------------------------------------------------
        # MAE encoder
        print("patch in channels", kwargs.get("patch_in_channels"))
        self.patch_embed = self.patch_embed_cls(self.img_shape, 
                                                in_channels=kwargs.get("patch_in_channels"), 
                                                patch_size=kwargs.get("patch_size"), 
                                                out_channels=kwargs.get("enc_embed_dim"), )
        self.cls_token = nn.Parameter(torch.zeros(1, 1, self.patch_embed.out_channels))
        self.enc_pos_embed = nn.Parameter(torch.zeros(1, 1 + self.patch_embed.num_patches, self.patch_embed.out_channels), 
                                      requires_grad=False)
        self.encoder = nn.ModuleList([Block(dim=self.patch_embed.out_channels, 
                                            num_heads=kwargs.get("enc_num_heads"), 
                                            mlp_ratio=kwargs.get("mlp_ratio"), 
                                            qkv_bias=True,)
                                      for i in range(kwargs.get("enc_depth"))])
        self.encoder_norm = nn.LayerNorm(self.patch_embed.out_channels)
        # --------------------------------------------------------------------------
        # MAE Masker
        self.masker = Masker(mask_type=kwargs.get("mask_type"), 
                             mask_ratio=kwargs.get("mask_ratio"), 
                             grid_size=self.patch_embed.grid_size)

        self.mask_token = nn.Parameter(torch.zeros(1, 1, kwargs.get("dec_embed_dim")), requires_grad=True)
        # --------------------------------------------------------------------------
        # Reconstruction decoder and head
        self.decoder_num_patches = self.patch_embed.num_patches
        self.decoder_embed = nn.Linear(kwargs.get("enc_embed_dim"), kwargs.get("dec_embed_dim"))
        self.dec_pos_embed = nn.Parameter(
            torch.zeros(1, 1 + self.patch_embed.num_patches, kwargs.get("dec_embed_dim")), requires_grad=False)
        self.decoder = ViTDecoder(dim=kwargs.get("dec_embed_dim"), 
                                  num_heads=kwargs.get("dec_num_heads"),
                                  depth=kwargs.get("dec_depth"),
                                  mlp_ratio=kwargs.get("mlp_ratio"),)
        self.recon_head = nn.Linear(in_features=kwargs.get("dec_embed_dim"),
                                    out_features=self.patch_p_num,)
        self.reconstruction_criterion = ReconstructionCriterion(**kwargs)
        self.generate_embeddings = kwargs.get("generate_embeddings", None)
        print("generate_embeddings", self.generate_embeddings)
        if self.generate_embeddings is not None:
            self.all_embeddings = []
            self.all_idx = [] 
        self.initialize_parameters()
        self.save_hyperparameters()

    
    def initialize_parameters(self):        
        # Initialize (and freeze) pos_embed by sin-cos embedding
        enc_pos_embed = sincos_pos_embed(self.enc_embed_dim, self.patch_embed.grid_size, cls_token=True,
                                         use_both_axes=self.use_both_axes)
        self.enc_pos_embed.data.copy_(enc_pos_embed.unsqueeze(0))
        dec_pos_embed = sincos_pos_embed(self.dec_embed_dim, self.patch_embed.grid_size, cls_token=True,
                                         use_both_axes=self.use_both_axes)
        self.dec_pos_embed.data.copy_(dec_pos_embed.unsqueeze(0))
        # Initialize patch_embed like nn.Linear (instead of nn.Conv2d)
        w = self.patch_embed.proj.weight.data
        torch.nn.init.xavier_uniform_(w.view([w.shape[0], -1]))
        # timm"s trunc_normal_(std=.02) is effectively normal_(std=0.02) as cutoff is too big (2.)
        if hasattr(self, "cls_token"):
            torch.nn.init.normal_(self.cls_token, std=.02)
        if hasattr(self, "mask_token"):
            torch.nn.init.normal_(self.mask_token, std=.02)

        # Initialize nn.Linear and nn.LayerNorm
        self.apply(self._init_weights) # TODO
        
    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            # we use xavier_uniform following official JAX ViT:
            torch.nn.init.xavier_uniform_(m.weight)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
    
    def forward_encoder_eval(self, x):
        """Forward pass of encoder
        input: [B, S, T, H, W] torch.Tensor
        output:
            latent: [B, 1 + length * mask_ratio, embed_dim] torch.Tensor
            mask: [B, 1 + length * mask_ratio] torch.Tensor
            ids_restore: [B, 1 + length * mask_ratio] torch.Tensor
        """
        # Embed patches: (B, S, T, H, W) -> (B, S * T * num_patches, embed_dim)
        x = self.patch_embed(x)
        
        # Add positional embedding: (B, S * T * num_patches, embed_dim)
        enc_pos_embed = self.enc_pos_embed.repeat(x.shape[0], 1, 1)
        x = x + enc_pos_embed[:, 1:, :]
        
        # Append cls token: (B, 1 + length * mask_ratio, embed_dim)
        cls_token = self.cls_token + enc_pos_embed[:, :1, :] # (1, 1, embed_dim)
        cls_tokens = cls_token.expand(x.shape[0], -1, -1) # (B, 1, embed_dim)
        x = torch.cat((cls_tokens, x), dim=1)
        
        # Apply transformer encoder
        for blk in self.encoder:
            x = blk(x)
        x = self.encoder_norm(x)
        # return only cls_token
        return x[:, :1, :]
    
    def forward_encoder(self, x, roi_mask=None):
        """Forward pass of encoder
        input: [B, S, T, H, W] torch.Tensor
        input: [B, S, T, H, W] torch.Tensor
        output:
            latent: [B, 1 + length * mask_ratio, embed_dim] torch.Tensor
            mask: [B, 1 + length * mask_ratio] torch.Tensor
            ids_restore: [B, 1 + length * mask_ratio] torch.Tensor
        """
        # Embed patches: (B, S, T, H, W) -> (B, S * T * num_patches, embed_dim)
        x = self.patch_embed(x)
        
        # Add positional embedding: (B, S * T * num_patches, embed_dim)
        enc_pos_embed = self.enc_pos_embed.repeat(x.shape[0], 1, 1)
        x = x + enc_pos_embed[:, 1:, :]

        # Patchify binary mask to align with patch embeddings

        roi_mask_patchified = patchify(roi_mask, patch_size=self.patch_size)  # [B, L]

        roi_mask_patchified = torch.any(roi_mask_patchified > 0, dim=-1).float()  # Collapse patch volume to [N, L], 1 if any element > 0

        x, mask, ids_restore = self.masker(x, roi_mask=roi_mask_patchified)

        # Append cls token: (B, 1 + length * mask_ratio, embed_dim)
        cls_token = self.cls_token + enc_pos_embed[:, :1, :] # (1, 1, embed_dim)
        cls_tokens = cls_token.expand(x.shape[0], -1, -1) # (B, 1, embed_dim)
        x = torch.cat((cls_tokens, x), dim=1)
        
        # Apply transformer encoder
        for blk in self.encoder:
            x = blk(x)
        x = self.encoder_norm(x)
        return x, mask, ids_restore
    
    def forward_decoder(self, x, ids_restore):
        """Forward pass of reconstruction decoder
        input:
            x: [B, 1 + length * mask_ratio, embed_dim] torch.Tensor
            ids_restore: [B, 1 + length * mask_ratio] torch.Tensor
        output:
            pred: [B, length, embed_dim] torch.Tensor
        """
        # Embed tokens
        x = self.decoder_embed(x)
        
        # Append mask tokens and add positional embedding in schuffled order
        # No mask tokens; directly proceed with embedding
        #dec_pos_embed = self.dec_pos_embed.repeat(x.shape[0], 1, 1)
        #x_restore_ = x[:, 1:, :] + dec_pos_embed[:, 1:, :]  # Exclude cls token for normal patches
        #cls_tok_ = x[:, :1, :] + dec_pos_embed[:, :1, :]

        mask_token_n = ids_restore.shape[1] + 1 - x.shape[1]
        mask_tokens = self.mask_token.repeat(x.shape[0], mask_token_n, 1)
        x_shuffle = torch.cat([x[:, 1:, :], mask_tokens], dim=1)
        x_restore = torch.gather(x_shuffle, dim=1, index=ids_restore.unsqueeze(-1).repeat(1, 1, x_shuffle.shape[-1]))
        dec_pos_embed = self.dec_pos_embed.repeat(x.shape[0], 1, 1)
        x_restore_ = x_restore + dec_pos_embed[:, 1:, :]
        cls_tok_ = x[:, :1, :] + dec_pos_embed[:, :1, :]
        
        # Reconstruction decoder
        x = torch.cat([cls_tok_, x_restore_], dim=1) # add class token
        x = self.decoder(x) # apply transformer decoder
        
        # Reconstruction head
        x = self.recon_head(x)
        x = x[:, 1:, :] # remove cls token
        x = torch.sigmoid(x) # scale x to [0, 1] for reconstruction task
        return x
    
    def forward(self, imgs, roi_masks=None):
        latent, mask, ids_restore = self.forward_encoder(imgs, roi_masks)
        pred = self.forward_decoder(latent, ids_restore)
        return pred, mask
        
    def training_step(self, batch, batch_idx, mode="train"):
        imgs, _, sub_idx, body_masks = batch

        roi_mask_patchified = patchify(body_masks, patch_size=self.patch_size)  # [B, L]
        roi_mask_patchified = torch.any(roi_mask_patchified > 0, dim=-1).float()  # Collapse patch volume to [N, L], 1 if any element > 0

        pred_patches, mask = self.forward(imgs, roi_masks=body_masks)
        imgs_patches = patchify(imgs, patch_size=self.patch_size)
        #loss_dict, psnr_value = self.reconstruction_criterion(pred_patches, imgs_patches, mask)
        loss_dict, psnr_value = self.reconstruction_criterion(pred_patches, imgs_patches, roi_mask_patchified)
        # Logging metrics and median
        self.log_recon_metrics(loss_dict, psnr_value, mode=mode)
        if mode == "train" or mode == "val":
            if mode == "val":
                self.log_dict({f"{mode}_PSNR": psnr_value}) # For checkpoint tracking
                
            log_rate = eval(f"self.{mode}_rate")
            if self.current_epoch > 0 and ((self.current_epoch + 1) % log_rate == 0):
                if (sub_idx == 0).any():
                    i = (sub_idx == 0).argwhere().squeeze().item()
                    self.log_recon_videos(mask[i], pred_patches[i], imgs[i], sub_idx[i], mode=mode)
                if (sub_idx == 1).any():
                    i = (sub_idx == 1).argwhere().squeeze().item()
                    self.log_recon_videos(mask[i], pred_patches[i], imgs[i], sub_idx[i], mode=mode)
            return loss_dict["loss"]
        
        elif mode == "test": # TODO
            if (sub_idx == 0).any():
                i = (sub_idx == 0).argwhere().squeeze().item()
                self.log_recon_videos(mask[i], pred_patches[i], imgs[i], sub_idx[i], mode=mode)
            if (sub_idx == 1).any():
                i = (sub_idx == 1).argwhere().squeeze().item()
                self.log_recon_videos(mask[i], pred_patches[i], imgs[i], sub_idx[i], mode=mode)
        
        return loss_dict["loss"]
    
    @torch.no_grad()
    def validation_step(self, batch, batch_idx):
        _ = self.training_step(batch, batch_idx, mode="val")

    @torch.no_grad()
    def test_step(self, batch, batch_idx):
        imgs, _, sub_idx, body_masks = batch
        if self.generate_embeddings is not None:
            embeddings = self.forward_encoder_eval(imgs)
            embeddings = embeddings.cpu().numpy()
            sub_idx = sub_idx.cpu().numpy()
            save_dir = os.path.dirname(self.generate_embeddings)
            os.makedirs(save_dir, exist_ok=True)
            for emb, idx in zip(embeddings, sub_idx):
                file_path = os.path.join(save_dir, f"{idx}.npy")
                np.save(file_path, emb)
            #self.all_embeddings.append(embeddings)
            #self.all_idx.append(sub_idx)
        else:
            _ = self.training_step(batch, batch_idx, mode="test")
    """
    def on_test_end(self):
        if self.generate_embeddings is not None:
            save_dir = os.path.dirname(self.generate_embeddings)
            os.makedirs(save_dir, exist_ok=True)

            all_embeddings = np.concatenate(self.all_embeddings, axis=0)
            all_idx = np.concatenate(self.all_idx, axis=0)
            np.savez_compressed(self.generate_embeddings, embeddings=all_embeddings, idx=all_idx)
            print(f"Embeddings saved to {self.generate_embeddings}")
    """
    def log_recon_metrics(self, loss_dict, psnr_value, mode="train"):
        for loss_name, loss_value in loss_dict.items():
            self.module_logger.update_metric_item(f"{mode}/recon_{loss_name}", loss_value.detach().item(), mode=mode)
        self.module_logger.update_metric_item(f"{mode}/recon_psnr", psnr_value, mode=mode)
               
    def log_recon_videos(self, mask, pred_patches, gt_imgs, sub_idx, mode="train"):
        sub_id = eval(f"self.trainer.datamodule.{mode}_dset").labels["eid"].iloc[sub_idx.item()]

        # Extend batch dimension for calculation
        mask = mask[None]
        pred_patches = pred_patches[None]
        gt_imgs = gt_imgs[None]

        mask_patches = mask.unsqueeze(-1).repeat(1, 1, np.prod(self.patch_size))

        mask_imgs = unpatchify(mask_patches, patch_size=self.patch_size, im_shape=gt_imgs.shape)

        pred_imgs = unpatchify(pred_patches, im_shape=gt_imgs.shape, patch_size=self.patch_size)

        mask_vis_path = "/u/home/sdm/GitHub/WholeBodyRL/mask_vis_path"



        #self.save_nifti(gt_imgs[0][0], np.eye(4), os.path.join(mask_vis_path, "gt.nii.gz"))
        #self.save_nifti(mask_imgs[0][0], np.eye(4), os.path.join(mask_vis_path, "masked.nii.gz"))

        gt_wat_img_log = imgs_to_wandb_video(gt_imgs[0][0]).cpu().numpy()
        pred_wat_img_log = imgs_to_wandb_video(pred_imgs[0][0]).cpu().numpy()
        mask_img_wat_log = imgs_to_wandb_video(mask_imgs[0][0]).cpu().numpy()
        self.module_logger.update_video_item(f"{mode}_video/gt_wat", sub_id, gt_wat_img_log, mode=mode)
        self.module_logger.update_video_item(f"{mode}_video/pred_wat", sub_id, pred_wat_img_log, mode=mode)
        self.module_logger.update_video_item(f"{mode}_video/mask_wat", sub_id, mask_img_wat_log, mode=mode)

        
        if self.use_both_axes:
            gt_fat_img_log = imgs_to_wandb_video(gt_imgs[0][1]).cpu().numpy()
            pred_fat_img_log = imgs_to_wandb_video(pred_imgs[0][1]).cpu().numpy()
            mask_img_fat_log = imgs_to_wandb_video(mask_imgs[0][1]).cpu().numpy()
            
            
            self.module_logger.update_video_item(f"{mode}_video/gt_fat", sub_id, gt_fat_img_log, mode=mode)
            self.module_logger.update_video_item(f"{mode}_video/pred_fat", sub_id, pred_fat_img_log, mode=mode)
            self.module_logger.update_video_item(f"{mode}_video/mask_fat", sub_id, mask_img_fat_log, mode=mode)



    def save_nifti(self, data, affine, filepath):
        import nibabel as nib
        """
        Save a 3D or 4D tensor as a NIfTI file.
        Args:
            data: numpy array to save.
            affine: affine transformation matrix for the NIfTI file.
            filepath: path to save the NIfTI file.
        """
        affine[0, 0] = 2.23  # Voxel size in x-direction
        affine[1, 1] = 3.00  # Voxel size in y-direction
        affine[2, 2] = 2.23  # Voxel size in z-direction
        #data = data.permute(1, 2, 3, 0)
        #data
        data = data.cpu().numpy()
        nifti_img = nib.Nifti1Image(data, affine)
        nib.save(nifti_img, filepath)