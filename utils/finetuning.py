from lightning.pytorch.callbacks import BaseFinetuning

class FreezeEncoder(BaseFinetuning):
    def __init__(self, unfreeze_at=10, lr_scale=0.1):
        super().__init__()
        self.unfreeze_at = unfreeze_at
        self.lr_scale = lr_scale

    def freeze_before_training(self, pl_module):
        # freeze encoder pieces
        for m in [pl_module.patch_embed, pl_module.encoder, pl_module.encoder_norm]:
            self.freeze(m)

    def finetune_function(self, pl_module, current_epoch, optimizer):
        if current_epoch == self.unfreeze_at:
            for m in [pl_module.patch_embed, pl_module.encoder, pl_module.encoder_norm]:
                self.unfreeze_and_add_param_group(
                    modules=m, optimizer=optimizer,
                    lr=optimizer.param_groups[0]["lr"] * self.lr_scale
                )
