import argparse
from dataclasses import asdict
from datetime import datetime
import os
from pathlib import Path

from matplotlib.dates import set_epoch
import torch
import wandb

from data.dataloaders import WBDataModule
from models.reconstruction_models import ReconMAE
from models.regression_models import RegrMAE, ResNet18Module, ResNet18Module3D, ResNet50Module
from models.segmentation_models import SegMAE
from utils.data_related import get_data_paths
from utils.params import load_config_from_yaml

from lightning import Trainer, seed_everything
from lightning.pytorch.loggers import CSVLogger
from lightning.pytorch.callbacks import ModelCheckpoint, BaseFinetuning
from dotenv import load_dotenv

from pytorch_lightning.callbacks import Callback
from data.dataloaders import SetEpochCallback
import torch.distributed as dist



def parser_command_line():
    "Define the arguments required for the script"
    parser = argparse.ArgumentParser(description="Masked Autoencoder Downstream Tasks",
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    #parser.add_argument("--labels_file", type=str, default="labels.csv", help="Path to the labels file")
    
    subparser = parser.add_subparsers(dest="pipeline", help="pipeline to run")
    # Arguments for training
    parser_train = subparser.add_parser("train", help="train the model")
    parser_train.add_argument("-c", "--config", help="config file (.yml) containing the ¢hyper-parameters for inference.")
    parser_train.add_argument("-g", "--wandb_group_name", default=None, help="specify the name of the group")
    parser_train.add_argument("-n", "--wandb_run_name", default=None, help="specify the name of the experiment")
    parser_train.add_argument("-m", "--multi_gpu", default=False, action="store_true", help="use multiple GPUs")
    parser_train.add_argument("--labels_file", type=str, default="labels.csv", help="Path to the labels file")
    # Arguments for evaluation
    parser_eval = subparser.add_parser("eval", help="evaluate the model")
    parser_eval.add_argument("-c", "--config", help="config file (.yml) containing the hyper-parameters for inference.")
    parser_eval.add_argument("-g", "--wandb_group_name", default=None, help="specify the name of the group")
    parser_eval.add_argument("-n", "--wandb_run_name", default=None, help="specify the name of the experiment")
    parser_eval.add_argument("--generate_embeddings", type=str, default=None, 
                             help="Generate and save embeddings during testing. Specify output file (e.g., embeddings.npz).")
    parser_eval.add_argument("--generate_predictions", type=str, default=None, help="Generate and save predictions for test dataset in csv file. Specify output file (e.g., predictions.csv).")
    parser_eval.add_argument("--labels_file", type=str, default="labels.csv", help="Path to the labels file")
    return parser.parse_args()


def main():
    load_dotenv()
    torch.backends.cudnn.enabled = True
    torch.set_float32_matmul_precision("medium")
    
    args = parser_command_line() # Load the arguments from the command line
    try:
        config_path = args.config
    except AttributeError:
        config_path = None
    params = load_config_from_yaml(config_path)
    if args.pipeline == "train" and args.multi_gpu:
        multi_gpu = True
    else:
        multi_gpu = False
    if args.pipeline == "eval":
        print("generate_embeddings", args.generate_embeddings)
        params.module.training_params.__dict__["generate_embeddings"] = args.generate_embeddings
        params.module.training_params.__dict__["generate_predictions"] = args.generate_predictions
        # Remove invalid argument if present
        #params.trainer.__dict__.pop("generate_embeddings", None)
        #print("params.module.training_params.__dict__", params.module.training_params.__dict__["generate_embeddings"])
    paths = get_data_paths()
    os.environ["WANDB_DISABLED"] = params.general.wandb_disabled
    #os.environ["CUDA_VISIBLE_DEVICES"] = "0" # Configure accelerator and devices
    seed_everything(params.general.seed, workers=False) # Sets seeds for numpy, torch and python.random.

    # Initialize wandb logging
    wandb_kwargs = dict()
    wandb_kwargs["entity"] = params.general.wandb_entity
    wandb_kwargs["group"] = args.wandb_group_name if args.wandb_group_name is not None else params.general.wandb_group_name
    wandb_run_name = args.wandb_run_name if args.wandb_run_name is not None else params.general.wandb_run_name
    time_now = datetime.now().strftime("%d-%m-%Y_%H-%M-%S")
    wandb_kwargs["name"] = f"{wandb_run_name}_{time_now}"
    wandb_kwargs["resume"] = "allow" # Resume if the run_id is provided and identical to a previous run otherwise, start a new run
    if params.general.wandb_run_id is not None:
        wandb_kwargs["id"] = params.general.wandb_run_id
    logger = CSVLogger(paths.log_folder)

    if multi_gpu and dist.is_initialized():
        print("Multi GPU WANDB")
        rank = dist.get_rank()
        if rank == 0:
            wandb.init(
                project="WholeBodyRL",
                config=asdict(params),
                **wandb_kwargs,
            )
        else:
            os.environ["WANDB_MODE"] = "disabled"  # Disable WandB for other ranks
    else:
        wandb.init(project="WholeBodyRL", config=asdict(params), **wandb_kwargs,)
    

    data_module = WBDataModule(
        load_dir=paths.dataset_folder,
        labels_folder=paths.labels_folder,
        body_mask_dir=paths.body_mask_folder,
        multi_gpu=multi_gpu,
        labels_file=args.labels_file,
        **params.data.__dict__
    )
    data_module.setup("fit")

    # Initialze lighting module
    module_LUT = {"reconstruction": [ReconMAE],
                  "regression": [RegrMAE, ResNet18Module, ResNet50Module, ResNet18Module3D],
                  "segmentation": [SegMAE]}
    if params.module.task_idx == 0:
        module_cls = module_LUT["reconstruction"][params.module.module_idx]
        module_params = {**params.module.training_params.__dict__, **params.module.recon_hparams.__dict__}
    elif params.module.task_idx == 1:
        module_cls = module_LUT["regression"][params.module.module_idx]
        module_params = {**params.module.training_params.__dict__, **params.module.regr_hparams.__dict__}
    elif params.module.task_idx == 2:
        module_cls = module_LUT["segmentation"][params.module.module_idx]
        module_params = {**params.module.training_params.__dict__, **params.module.seg_hparams.__dict__}
    else:
        raise NotImplementedError

    print("Before model load")
    model = module_cls(val_dset=data_module.val_dset, **module_params)
    print("After model load")
    print("ckpt path:", params.general.ckpt_path)
    
    # Check the resuming and loading of the checkpoints
    resume_ckpt_path = None
    if params.general.resume_training:  # Resume training
        assert params.general.ckpt_path != None, "The path for checkpoint is not provided."
        resume_ckpt_path = params.general.ckpt_path
    
    if params.general.load_encoder: # Load pretraining encoder
        assert params.general.ckpt_path != None, "The path for checkpoint is not provided."
        ckpt = torch.load(params.general.ckpt_path)
        pretrained_dict = ckpt["state_dict"]
        processed_dict = {}
        pretrained_params = ["cls_token", "enc_pos_embed", "mask_token", "patch_embed", "encoder", "encoder_norm"]
        for k in model.state_dict().keys():
            decomposed_k = k.split(".")
            if decomposed_k[0] in pretrained_params:
                processed_dict[k] = pretrained_dict[k]
        model.load_state_dict(processed_dict, strict=False)
    
    if params.general.freeze_encoder: # Freeze encoder
        BaseFinetuning.freeze([model.patch_embed, model.encoder, model.encoder_norm])
                
    # Monitor foreground dice for segmentation. When reconstruction, monitor PSNR. MAE for regression.
    if params.general.resume_training:
        ckpt_dir = Path(resume_ckpt_path).parent
    else:
        ckpt_dir = os.path.join(f"{paths.log_folder}/checkpoints/{wandb_run_name}/{time_now}")
    monitor_LUT = [
        ("val_PSNR", "model-{epoch:03d}-{val_PSNR:.2f}", "max"), # Reconstruction
        ("val_MAE", "model-{epoch:03d}-{val_MAE:.2f}", "min"), # Regression
        ("val_Dice_FG", "model-{epoch:03d}-{val_Dice_FG:.2f}", "max"), # Segmentation
    ]
    monitor_metric, ckpt_filename, monitor_mode = monitor_LUT[params.module.task_idx]
    os.makedirs(ckpt_dir, exist_ok=True)
    checkpoint_callback = ModelCheckpoint(dirpath=ckpt_dir, filename=ckpt_filename, monitor=monitor_metric, 
                                          mode=monitor_mode, save_top_k=5, save_last=True, verbose=True,)
    set_epoch_callback = SetEpochCallback()

    print("Before trainer")
    # Initialize trainer
    if multi_gpu:
        trainer = Trainer(
            default_root_dir=paths.log_folder,
            logger=logger,
            callbacks=[checkpoint_callback],
            fast_dev_run=False,
            limit_train_batches=1.0,
            limit_val_batches=1.0,
            num_sanity_val_steps=1,
            benchmark=True,
            devices='auto',  # Automatically select all available GPUs or specify a number like `devices=2`
            strategy="ddp",  # Distributed Data Parallel (DDP) strategy for multi-GPU training
            num_nodes=1,
            use_distributed_sampler=False,
            **params.trainer.__dict__,
        )
        print("MUTLI GPU")
        print("num gpus:", torch.cuda.device_count())  # Should return 2
        print("CUDA  available:", torch.cuda.is_available())
        #import torch.distributed as dist
        #print(f"DDP Backend: {dist.get_backend()}")
    else:
        trainer = Trainer(
            default_root_dir=paths.log_folder,
            logger=logger,
            callbacks=[checkpoint_callback],
            fast_dev_run=False,
            limit_train_batches=1.0,
            limit_val_batches=1.0,
            num_sanity_val_steps=2,
            benchmark=True,
            **params.trainer.__dict__,
        )
        #if args.pipeline == "eval":
        #    model.generate_embeddings = args.generate_embeddings
        #    model.all_embeddings = []
        #    model.all_idx = []
    print("Before fit")

    if args.pipeline == "train":
        trainer.fit(model, datamodule=data_module, ckpt_path=resume_ckpt_path)
        trainer.test(model, datamodule=data_module)
    elif args.pipeline == "eval":
        trainer.test(model, datamodule=data_module, ckpt_path=params.general.ckpt_path)
    wandb.finish() # Finish logging


if __name__ == "__main__":
    main()
