import pandas as pd
import torch
from torch import nn
from timm.models.vision_transformer import Block
from models.reconstruction_models import BasicModule
from networks.losses import FineGrayCriterion
from utils.model_related import sincos_pos_embed, PatchEmbed
from torch.autograd import grad
import numpy as np

from networks.metrics import truncated_concordance_td, brier_score as bs
from pycox.evaluation import EvalSurv
from pycox.models.loss import DeepHitLoss
from pycox.models.data import pair_rank_mat
import os
from utils.deephit import LabTransform, CauseSpecificNet

from torch.autograd import grad
import numpy as np
from typing import List, Optional, Tuple

# -----------------------------------------------------------------------------
# Helper layers & builders (ported from NeuralFineGrayTorch)
# -----------------------------------------------------------------------------
class PositiveLinear(nn.Module):
    """Linear layer whose weights are constrained to be positive via a squared
    parameterisation.  Identical behaviour to original PositiveLinear but
    rewritten for clarity."""

    def __init__(self, in_features: int, out_features: int, bias: bool = False):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.log_weight = nn.Parameter(torch.Tensor(out_features, in_features))
        if bias:
            self.bias = nn.Parameter(torch.Tensor(out_features))
        else:
            self.register_parameter("bias", None)
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.xavier_uniform_(self.log_weight)
        if self.bias is not None:
            fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.log_weight)
            bound = np.sqrt(1 / np.sqrt(fan_in))
            nn.init.uniform_(self.bias, -bound, bound)
        # ensure positivity at initialisation
        self.log_weight.data.abs_().sqrt_()

    def forward(self, input: torch.Tensor) -> torch.Tensor:  # noqa: D401
        return nn.functional.linear(input, self.log_weight ** 2, self.bias)


def create_representation_positive(inputdim: int,
                                   layers: List[int],
                                   dropout: float = 0.0) -> nn.Sequential:
    """Stack of PositiveLinear + activation (Softplus at the end)."""
    modules: List[nn.Module] = []
    act = nn.Tanh()
    prev_dim = inputdim
    for hidden in layers:
        modules.append(PositiveLinear(prev_dim, hidden, bias=True))
        if dropout > 0.0:
            modules.append(nn.Dropout(p=dropout))
        modules.append(act)
        prev_dim = hidden
    modules[-1] = nn.Softplus()
    return nn.Sequential(*modules)


def create_representation(inputdim: int,
                           layers: List[int],
                           activation: str,
                           dropout: float = 0.0,
                           last: Optional[nn.Module] = None) -> List[nn.Module]:
    if activation == "ReLU6":
        act: nn.Module = nn.ReLU6()
    elif activation == "ReLU":
        act = nn.ReLU()
    elif activation == "Tanh":
        act = nn.Tanh()
    else:
        raise ValueError(f"Unsupported activation {activation}")

    modules: List[nn.Module] = []
    prev_dim = inputdim
    for hidden in layers:
        modules.append(nn.Linear(prev_dim, hidden, bias=True))
        #modules.append(nn.BatchNorm1d(hidden))
        if dropout > 0.0:
            modules.append(nn.Dropout(p=dropout))
        modules.append(act)
        prev_dim = hidden
    if last is not None:
        modules[-1] = last
    return modules

# -----------------------------------------------------------------------------
# NFG‑MAE  –  MAE ViT encoder + Fine‑Gray decoder with cls/linear projector
# -----------------------------------------------------------------------------
class NFGMAE(BasicModule):
    """Masked‑Autoencoder encoder + Neural Fine‑Gray competing‑risk decoder.

    *Exactly mirrors RegrMAE's bottleneck:*  we insert `dec_embed` and
    `dec_pos_embed`, then expose two "representation_type" options:
      • **"cls_token"** – global representation is the [CLS] vector only.
      • **"linear"**    – drop CLS, flatten all patch tokens, pass through
        a small Linear to return to `dec_embed_dim`.

    The output vector is consumed by the Neural Fine‑Gray heads.
    """

    # ---------------------- initialisation ------------------------------
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        # -------------------- Encoder (unmodified MAE ViT) --------------
        self.ckpt_dir = kwargs.get("ckpt_dir", "")
        self.enc_embed_dim = kwargs.get("enc_embed_dim")
        self.dec_embed_dim = kwargs.get("dec_embed_dim", self.enc_embed_dim)
        self.patch_embed_cls = globals()[kwargs.get("patch_embed_cls")]
        val_dataset = kwargs.get("val_dset")
        #self.img_shape = val_dataset[0][0].shape  # type: ignore[index]
        self.img_shape = val_dataset[0]["image"].shape  # type: ignore[index]
        self.use_both_axes = bool(getattr(val_dataset, "get_view", lambda: 1)() == 2)

        self.patch_embed = self.patch_embed_cls(
            self.img_shape,
            in_channels=kwargs.get("patch_in_channels"),
            patch_size=kwargs.get("patch_size"),
            out_channels=self.enc_embed_dim,
        )
        self.cls_token = nn.Parameter(torch.zeros(1, 1, self.patch_embed.out_channels))
        self.enc_pos_embed = nn.Parameter(
            torch.zeros(1, 1 + self.patch_embed.num_patches, self.patch_embed.out_channels),
            requires_grad=False,
        )
        self.encoder = nn.ModuleList([
            Block(dim=self.patch_embed.out_channels,
                  num_heads=kwargs.get("enc_num_heads"),
                  mlp_ratio=kwargs.get("mlp_ratio"),
                  qkv_bias=True)
            for _ in range(kwargs.get("enc_depth"))
        ])
        self.encoder_norm = nn.LayerNorm(self.patch_embed.out_channels)

        # -------------------- Bottleneck (RegrMAE‑style) ----------------
        self.dec_embed = nn.Linear(self.enc_embed_dim, self.dec_embed_dim, bias=True)
        self.dec_pos_embed = nn.Parameter(
            torch.zeros(1, 1 + self.patch_embed.num_patches, self.dec_embed_dim),
            requires_grad=False,
        )
        self.representation_type = kwargs.get("representation_type", "cls_token")
        assert self.representation_type in {"cls_token", "linear"}

        # Linear aggregator used only when representation_type == "linear"
        self._flatten_fc: Optional[nn.Linear] = None

        # -------------------- Fine‑Gray decoder -------------------------


        self.risks = kwargs.get("risks", 1)

        
        rep_layers = kwargs.get("rep_layers", [128])
        surv_layers = kwargs.get("surv_layers", [128])
        act = kwargs.get("act", "Tanh")  # Tanh activation as in the original paper
        dropout = kwargs.get("dropout", 0.0)
        self.multihead = kwargs.get("multihead", True)

        inputdim = self.dec_embed_dim  # after bottleneck

        self.embed = nn.Identity()
        self.balance = nn.Sequential(*create_representation(inputdim,
                                                            rep_layers + [self.risks],
                                                            act))

        if self.multihead:
            self.outcome = nn.ModuleList([
                create_representation_positive(inputdim + 1,
                                                surv_layers + [1],
                                                dropout=dropout)
                for _ in range(self.risks)
            ])
        else:
            self.outcome = create_representation_positive(inputdim + 1,
                                                          surv_layers + [self.risks],
                                                          dropout=dropout)
        self.softlog = nn.LogSoftmax(dim=1)

        # save hparams for Lightning etc.
        self.save_hyperparameters()

        #self.initialize_parameters()

        self.criterion = FineGrayCriterion(**kwargs)

        loss_types = kwargs.get("loss_types", None)
        if "cause_specific" in loss_types:
            self.cause_specific = True
        else:
            self.cause_specific = False

        self.norm_uniform = kwargs.get("norm_uniform", True)
        self.times = kwargs.get("times", 100)  # Number of time points for survival analysis
        t = kwargs.get("durations", None)
        if t is None:
            print("We need all t's for predict_survival")
        self.times = np.linspace(t.min(), t.max(), self.times) if isinstance(self.times, int) else self.times
        self.time = None

        self.contrastive_weight = kwargs.get("contrastive_weight", 0.0)
        self._val_preds, self._val_t, self._val_e = [], [], []
        self._test_preds, self._test_t, self._test_e = [], [], []
        self._df_columns = None

        self.e_train = np.asarray(kwargs["events"], dtype=float)
        self.t_train = kwargs.get("durations", None)  # For normalisation
        self.time = np.sort(self.t_train)
        self.max_time = self.time.max()



    # ---------------------- parameter init -----------------------------
    def initialize_parameters(self):
        # sin‑cos positional embeddings for decoder space
        dec_pos = sincos_pos_embed(self.dec_embed_dim,
                                   self.patch_embed.grid_size,
                                   cls_token=True,
                                   use_both_axes=self.use_both_axes)
        self.dec_pos_embed.data.copy_(dec_pos.unsqueeze(0))

        # generic Linear/LayerNorm init as in MAE impl
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            nn.init.xavier_uniform_(m.weight)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    # ---------------------- encoder forward ----------------------------
    def forward_encoder(self, imgs: torch.Tensor) -> torch.Tensor:
        """Return *all* encoder tokens (CLS + patches)."""
        x = self.patch_embed(imgs)                         # (B, N, Denc)
        pos = self.enc_pos_embed.repeat(x.size(0), 1, 1)
        x = x + pos[:, 1:, :]
        cls_tok = self.cls_token + pos[:, :1, :]
        x = torch.cat([cls_tok.expand(x.size(0), -1, -1), x], dim=1)  # (B,1+N,Denc)
        for blk in self.encoder:
            x = blk(x)
        x = self.encoder_norm(x)
        return x  # keep all tokens

    # ---------------------- token → vector projector -------------------
    def _project_tokens(self, tok: torch.Tensor) -> torch.Tensor:
        """Convert (B,1+N,Ddec) tokens into one vector per sample."""
        if self.representation_type == "cls_token":
            #print("cls_token")
            return tok[:, 0, :]  # (B, Ddec)
        # lazy initialise aggregator the first time we need it
        if self._flatten_fc is None:
            self._flatten_fc = nn.Linear((tok.size(1) - 1) * tok.size(2), self.dec_embed_dim)
        patches_flat = tok[:, 1:, :].flatten(1)  # (B, (N)*Ddec)
        return self._flatten_fc(patches_flat)

    # ---------------------- Fine‑Gray passes ---------------------------
    def _forward_multihead(self, z: torch.Tensor, horizon: torch.Tensor, return_rep: bool):
        #print("shape of z", z.shape)
        rep = self.embed(z)
        #print("shape of rep", rep.shape)
        log_beta = self.softlog(self.balance(rep))
        #print("shape of log_beta", log_beta.shape)
        #rep = rep.detach().requires_grad_(True)
        sr = []
        tau_outcome = horizon.clone().detach().cuda().requires_grad_(True).unsqueeze(1)
        #print("shape of tau_outcome", tau_outcome.shape)

        for outcome_competing in self.outcome:
            sr.append(-tau_outcome * outcome_competing(torch.cat([rep, tau_outcome], dim=1)))
        sr = torch.cat(sr, dim=1)  # (B,K)
        if return_rep:
            return sr, log_beta, tau_outcome, rep
        return sr, log_beta, tau_outcome

    def _forward_single(self, z: torch.Tensor, horizon: torch.Tensor):
        rep = self.embed(z)
        log_beta = self.softlog(self.balance(rep))
        tau = horizon.clone().detach().requires_grad_(True).unsqueeze(1)
        out = tau * self.outcome(torch.cat([rep, tau], dim=1))
        return -out, log_beta, tau

    # ---------------------- public forward -----------------------------
    def forward(self, imgs: torch.Tensor, horizon: torch.Tensor, return_rep: bool = False):
        #print("shape of imgs", imgs.shape)
        tok = self.forward_encoder(imgs)      
        #print("shape after forward_encoder", tok.shape)                 # (B,1+N,Denc)
        tok = self.dec_embed(tok)
        #print("shape after dec_embed", tok.shape)                              # (B,1+N,Ddec)
        tok = tok + self.dec_pos_embed.repeat(tok.size(0), 1, 1)
        #print("shape after dec_pos_embed", tok.shape)                        # (B,1+N,Ddec)
        z = self._project_tokens(tok)
        #print("shape after _project_tokens", z.shape)                          # (B,Ddec)
        if self.multihead:
            return self._forward_multihead(z, horizon, return_rep)
        return self._forward_single(z, horizon)

    # ---------------------- gradient helper ----------------------------
    @staticmethod
    def gradient(outcomes: torch.Tensor, horizon: torch.Tensor, e: torch.Tensor, create_graph: bool = True):
        return grad([ -outcomes[:, r][e == (r + 1)].sum() for r in range(outcomes.size(1)) ],
                    horizon,
                    create_graph=create_graph)[0].clamp_(1e-10)[:, 0]
    
    def training_step(self, batch, batch_idx, mode="train"):
        #imgs, (t, e), sub_idx, _ = batch  # actually (t, e)
        imgs = batch["image"]
        (t, e) = batch["target_value"]
        sub_idx = batch["sub_idx"]

        imgs = imgs.float()  # Convert to float
        t = t.float()
        e = e.float()
        t = self._normalise(t)  

        if self.contrastive_weight > 0.0:
            pred_values = self.forward(imgs, t, return_rep=True)  # check if horizon is t!
            log_sr, log_b, tau, rep = pred_values
        else: 
            # Use the forward method to get the predictions
            pred_values = self.forward(imgs, t, return_rep=False)
            log_sr, log_b, tau = pred_values
            rep = None


        log_hr = self.gradient(log_sr, tau, e).log()
        log_balance_sr = log_b + log_sr



        loss = self.criterion.forward(log_hr, log_balance_sr, e, rep, self.contrastive_weight)
                # Metric logging via the ModuleLogger helper of BasicModule
        self.module_logger.update_metric_item("train/training_step", 1, mode="train")
        self.module_logger.update_metric_item("train/surv_loss", loss.detach().item(), mode="train")
        self.log_dict({f"{mode}_surv_loss": loss.detach().item()})
        #print("loss", loss.detach().item())

        return loss
    
        # ---------------------------------------------------------------
    # helper — put anywhere inside the module, e.g. right after _project_tokens
    def _batch_survival_dataframe(self,
                                z: torch.Tensor,
                                times: np.ndarray,
                                risk = None) -> pd.DataFrame:
        """
        Build a (B, K·T) survival matrix from a **single** horizon-independent
        representation `z` by looping over evaluation times *without* re-encoding
        the images.
        """
        B = z.size(0)
        device, dtype = z.device, z.dtype

        cols_risk, cols_time, rows = [], [], []

        for t_val in times:
            tau = torch.full((B,), t_val, dtype=dtype, device=device)

            if self.multihead:
                log_sr, log_beta, _ = self._forward_multihead(z, tau, return_rep=False)
            else:
                log_sr, log_beta, _ = self._forward_single(z, tau)

            beta = 1.0 if self.cause_specific else torch.exp(log_beta)
            S = 1.0 - beta * (1.0 - torch.exp(log_sr))           # (B, K)

            if risk is not None:
                S = S[:, risk-1 : risk]                          # keep one column

            rows.append(S.cpu())                                 # [B, K] → list
            k = S.size(1)
            cols_risk.extend(range(1, k+1))
            cols_time.extend([t_val] * k)

        surv = torch.cat([r.reshape(B, -1) for r in rows], dim=1).numpy()
        columns = pd.MultiIndex.from_arrays([cols_risk, cols_time],
                                            names=["risk", "time"])
        return pd.DataFrame(surv, columns=columns)
    # ---------------------------------------------------------------

    @torch.no_grad()
    def _collect_survival(self, imgs):
        """Second encoder pass (no grads) + survival curves."""
        tok = self.forward_encoder(imgs)
        tok = self.dec_embed(tok) + self.dec_pos_embed[:, :tok.size(1), :]
        z   = self._project_tokens(tok)                 # (B, dec_embed_dim)
        return self._batch_survival_dataframe(z, self.times)
    # --------------------------------------------------------------
    #@torch.no_grad()
    def validation_step(self, batch, batch_idx):
        #imgs, (t, e), sub_idx, _ = batch  # actually (t, e)
        imgs = batch["image"]
        (t, e) = batch["target_value"]
        sub_idx = batch["sub_idx"]

        imgs = imgs.float()  # Convert to float
        t = t.float()
        e = e.float()
        t = self._normalise(t)

        rep = None

        with torch.set_grad_enabled(True):
            t.requires_grad_(True)             # horizon must track grads

            if self.contrastive_weight > 0.0:
                log_sr, log_b, tau, rep = self.forward(imgs, t, return_rep=True)
            else:
                log_sr, log_b, tau      = self.forward(imgs, t, return_rep=False)

            # make sure tau can receive grads if it comes from the net
            tau.requires_grad_(True)

            # first-order derivative is enough for evaluation
            log_hr = self.gradient(log_sr, tau, e, create_graph=False).log()

        log_hr = log_hr.detach()               # drop the graph to save memory
        log_balance_sr = log_b + log_sr
        loss = self.criterion.forward(
            log_hr, log_balance_sr, e, rep, self.contrastive_weight
        )

        print("VALIDATION loss", loss.detach().item())

        self.module_logger.update_metric_item("val/surv_loss", loss.detach().item(), mode="val")
        self.log_dict({"val_surv_loss": loss.detach().item()})

        with torch.no_grad():
            df_surv = self._collect_survival(imgs)   # helper from previous answer
        
        
        self._val_preds.append(df_surv.values)

        if self._df_columns is None:
            self._df_columns = df_surv.columns
        self._val_t.append(t.cpu().numpy())
        self._val_e.append(e.cpu().numpy())

    @torch.no_grad()
    def test_step(self, batch, batch_idx):
        #imgs, (t, e), sub_idx, _ = batch  # actually (t, e)
        imgs = batch["image"]
        (t, e) = batch["target_value"]

        imgs = imgs.float()  # Convert to float
        t = t.float()
        e = e.float()
        t = self._normalise(t)

        rep = None
        
        df_surv = self._collect_survival(imgs)   # helper from previous answer
        
        
        self._test_preds.append(df_surv.values)
        self._test_t.append(t.cpu().numpy())
        self._test_e.append(e.cpu().numpy())

    def on_test_epoch_end(self):

        curves = np.vstack(self._test_preds)              # (N, K·T)

        df_surv_all = pd.DataFrame(curves,
                                columns=self._df_columns)

        t_all = np.concatenate(self._test_t)  # (N,)
        e_all = np.concatenate(self._test_e)  # (N,)

        horizons_q = [0.25, 0.50, 0.75]
        times_eval = np.quantile(t_all[e_all > 0], horizons_q)

        metrics_df = self.evaluate_single_fold(
            survival=df_surv_all,
            t=t_all,
            e=e_all,
            times_eval=times_eval,
        )
        print("metrics_df", metrics_df)

        os.makedirs(os.path.join(self.ckpt_dir, "survival_files"), exist_ok=True)
        df_surv_all.to_csv(os.path.join(self.ckpt_dir, "survival_files", "survival_single_fold_test.csv"))
        metrics_df.to_csv(os.path.join(self.ckpt_dir, "survival_files", "metrics_single_fold_test.csv"))
        print("\nTest metrics saved in", os.path.join(self.ckpt_dir, "survival_files"))

        self.log_overall_metrics(metrics_df, mode="test")


        # -------------------------------------------------------
        # tidy-up for next epoch
        # -------------------------------------------------------
        self._test_preds.clear()
        self._test_t.clear()
        self._test_e.clear()
        self.module_logger.wandb_log(self.current_epoch, mode="test")

    def log_overall_metrics(
        self,
        metrics_df: pd.DataFrame,
        mode: str = "val",          # "val"  or  "test"
        ):
        """
        Log one scalar per risk:

            {mode}_CIS_risk_{r}
            {mode}_IBS_risk_{r}

        Parameters
        ----------
        metrics_df : pd.DataFrame
            Rows  = Risk (index)
            Cols  must contain "CIS" and "IBS".
        mode : str
            Tag prefix; choose "val" during validation, "test" at test-time.
        """
        assert mode in {"val", "test"}, "mode must be 'val' or 'test'"

        print("metrics_df", metrics_df)

        for risk_id, row in metrics_df.iterrows():     # iterate by *rows*
            cis = float(row.get("CIS", np.nan))
            brs = float(row.get("BRS", np.nan))
            print(f"Risk {risk_id}: CIS={cis:.3f}, BRS={brs:.3f}")

            self.module_logger.update_metric_item(
                f"{mode}/CIS_risk_{int(risk_id)}", cis, mode=mode)
            self.module_logger.update_metric_item(
                f"{mode}/BRS_risk_{int(risk_id)}", brs, mode=mode)
            

    def on_validation_epoch_end(self):
        if not self._val_preds:         # safety for first epoch
            return

        curves = np.vstack(self._val_preds)              # (N, K·T)

        df_surv_all = pd.DataFrame(curves,
                                columns=self._df_columns)

        t_all = np.concatenate(self._val_t)  # (N,)
        e_all = np.concatenate(self._val_e)  # (N,)

        horizons_q = [0.25, 0.50, 0.75]
        times_eval = np.quantile(t_all[e_all > 0], horizons_q)

        metrics_df = self.evaluate_single_fold(
            survival=df_surv_all,
            t=t_all,
            e=e_all,
            times_eval=times_eval,
        )
        print("metrics_df", metrics_df)

        self.log_overall_metrics(metrics_df, mode="val")


        self._val_preds.clear()
        self._val_t.clear()
        self._val_e.clear()
        self.module_logger.wandb_log(self.current_epoch, mode="val")

    def evaluate_single_fold(
        self,
        survival,
        t,
        e,
        times_eval
    ):
        """
        Parameters
        ----------
        survival  : DataFrame (patients × (risk,time))
                    Values = survivor S_r(t).
        t, e      : 1-D arrays with durations and event indicators.
        times_eval: horizons for truncated metrics (None -> skip them).

        Returns
        -------
        pd.DataFrame  (rows = Risk, columns = metrics)
        """
        # tidy column index -------------------------------------------------
        surv = survival.copy()
        if "Use" in surv.columns:
            surv = surv.drop(columns="Use")

        surv.columns = pd.MultiIndex.from_frame(
            pd.DataFrame(index=surv.columns).reset_index().astype(float)
        )

        time_grid = surv.columns.get_level_values(1).unique()
        risks = surv.columns.get_level_values(0).unique()

        results = {}
        for r in risks:
            surv_r = surv[r]       
            dummy = pd.DataFrame(
                np.ones((len(time_grid), len(self.t_train))),
                index=time_grid
            )                        # (N , |grid|)
            km = EvalSurv(dummy, self.t_train, self.e_train != 0, censor_surv="km")
            ev      = EvalSurv(surv_r.T, t, e == int(r), censor_surv=km)


                    # ---------------- overall CIS / IBS ---------------------------
            try:
                cis_overall = ev.concordance_td()
            except ZeroDivisionError:       # no comparable pairs
                cis_overall = np.nan

            res = {"CIS": cis_overall}

            try:
                brs_overall = ev.integrated_brier_score(time_grid.to_numpy())
            except Exception:
                print("Exception in integrated_brier_score")
                brs_overall = np.nan

            res["BRS"] = brs_overall

            km = (self.e_train, self.t_train)

            # horizons ------------------------------------------------------
            if times_eval is not None:
                for te in times_eval:
                    try:
                        cis_trunc, km_pair = truncated_concordance_td(
                            e, t, 1 - surv_r.values, time_grid, te,
                            km=km, competing_risk=int(r)
                        )
                        brs_trunc = bs(
                            e, t, 1 - surv_r.values, time_grid, te,
                            km=km_pair, competing_risk=int(r)
                        )[0]
                        res[f"CIS@{te:.0f}"] = cis_trunc
                        res[f"BRS@{te:.0f}"] = brs_trunc
                    except Exception:
                        continue

            results[r] = pd.Series(res)

        return pd.DataFrame(results).T.set_index(pd.Index(risks, name="Risk"))

    
    def _normalise(self, time):
        if self.norm_uniform:
            if self.time is None:
                print("Time init is not initialised in the training step. Is it ok?")
            ecdf = lambda x: (np.searchsorted(self.time, x.cpu().item() if hasattr(x, 'cpu') else x, side='right') + 1) / len(self.time)
            uniform_data = torch.Tensor([ecdf(t) for t in time])
            return uniform_data + 1e-5 # Avoid 0
        else:
            time = time + 1 # Do not want event at time 0
            return time / self.max_time # Normalise time between 0 and 1
        

class DeepHitMAE(BasicModule):
    """Masked‑Autoencoder encoder + DeepHit competing‑risk decoder.

    The encoder/bottleneck is copied 1‑to‑1 from *RegrMAE* / *NFGMAE*.
    Instead of the Fine‑Gray heads we attach a discrete‑time DeepHit head
    implemented with the user‑provided *CauseSpecificNet*.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        # ---------- Encoder (exactly the same as in NFGMAE) ------------
        self.ckpt_dir = kwargs.get("ckpt_dir", "")
        self.enc_embed_dim = kwargs.get("enc_embed_dim")
        self.dec_embed_dim = kwargs.get("dec_embed_dim", self.enc_embed_dim)
        print("dec_embed_dim", self.dec_embed_dim)
        self.patch_embed_cls = globals()[kwargs.get("patch_embed_cls")]
        val_dataset = kwargs.get("val_dset")
        self.img_shape = val_dataset[0]['image'].shape  # type: ignore[index]
        self.use_both_axes = bool(getattr(val_dataset, "get_view", lambda: 1)() == 2)

        self.patch_embed = self.patch_embed_cls(
            self.img_shape,
            in_channels=kwargs.get("patch_in_channels"),
            patch_size=kwargs.get("patch_size"),
            out_channels=self.enc_embed_dim,
        )
        self.cls_token = nn.Parameter(torch.zeros(1, 1, self.patch_embed.out_channels))
        self.enc_pos_embed = nn.Parameter(
            torch.zeros(1, 1 + self.patch_embed.num_patches, self.patch_embed.out_channels),
            requires_grad=False,
        )
        self.encoder = nn.ModuleList(
            [
                Block(
                    dim=self.patch_embed.out_channels,
                    num_heads=kwargs.get("enc_num_heads"),
                    mlp_ratio=kwargs.get("mlp_ratio"),
                    qkv_bias=True,
                )
                for _ in range(kwargs.get("enc_depth"))
            ]
        )
        self.encoder_norm = nn.LayerNorm(self.patch_embed.out_channels)

        # ---------- Bottleneck (RegrMAE‑style) -------------------------
        self.dec_embed = nn.Linear(self.enc_embed_dim, self.dec_embed_dim, bias=True)
        self.dec_pos_embed = nn.Parameter(
            torch.zeros(1, 1 + self.patch_embed.num_patches, self.dec_embed_dim),
            requires_grad=False,
        )
        self.representation_type = kwargs.get("representation_type", "cls_token")
        assert self.representation_type in {"cls_token", "linear"}
        self._flatten_fc: Optional[nn.Linear] = None  # for representation_type == "linear"

        # -----------------------------------------------------------------
        # DeepHit‑specific -------------------------------------------------
        # -----------------------------------------------------------------
        self.risks: int = kwargs.get("risks", 1)

        # --- Discrete time grid (labtrans) -----------------------------
        self.t_train = np.asarray(kwargs["durations"], dtype=float)
        self.e_train = np.asarray(kwargs["events"], dtype=float)
        labtrans_n = int(kwargs["labtrans_n"])
        num_eval_times = 100
        eval_times = np.linspace(0.0, self.t_train.max(), num_eval_times)
        self.times = eval_times  # used later for prediction helpers
        self.times = np.asarray(self.times, dtype=float)
        discretize_times = np.linspace(0.0, self.t_train.max(), labtrans_n)
        self.labtrans = LabTransform(discretize_times.tolist())

        # --- Decoder networks -----------------------------------------
        surv_layers = tuple(kwargs.get("surv_layers", [128]))
        dropout = kwargs.get("dropout", 0.0)
        self.hazard_net = CauseSpecificNet(
            in_features=self.dec_embed_dim,
            num_nodes_indiv=surv_layers,
            num_risks=self.risks,
            out_features=labtrans_n,
            batch_norm=True,
            dropout=dropout,
        )

        # --- Loss ------------------------------------------------------
        self.alpha = kwargs.get("dh_alpha", 1.0)
        self.gamma = kwargs.get("dh_gamma", 0.2)
        self.sigma = kwargs.get("dh_sigma", 0.1)

        #self._virtual_B = kwargs.get("virtual_B", 32)
        self.fold_id = kwargs.get("fold_id", 0)
        self.criterion = DeepHitLoss(alpha=self.alpha, sigma=self.sigma) # no gamma!

        # Misc bookkeeping ---------------------------------------------
        self.save_hyperparameters()

        # Storage for validation / test epoch‑level metrics -------------
        self._val_preds, self._val_t, self._val_e, self._val_idx = [], [], [],  []
        self._test_preds, self._test_t, self._test_e, self._test_idx = [], [], [], []

        # --------------------------------------------------------------------
        #self.virtual_B = kwargs.get("virtual_B", 4)   # desired ranking batch
        #self._buf_logits, self._buf_d, self._buf_e = [], [], []

    # ------------------------------------------------------------------
    # Helper initialisation (sin‑cos PE etc.) --------------------------
    # ------------------------------------------------------------------

    def initialize_parameters(self):
        dec_pos = sincos_pos_embed(
            self.dec_embed_dim,
            self.patch_embed.grid_size,
            cls_token=True,
            use_both_axes=self.use_both_axes,
        )
        self.dec_pos_embed.data.copy_(dec_pos.unsqueeze(0))
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            nn.init.xavier_uniform_(m.weight)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    # ------------------------------------------------------------------
    # Encoder forward (unchanged) --------------------------------------
    # ------------------------------------------------------------------

    def forward_encoder(self, imgs: torch.Tensor) -> torch.Tensor:
        x = self.patch_embed(imgs)
        pos = self.enc_pos_embed.repeat(x.size(0), 1, 1)
        x = x + pos[:, 1:, :]
        cls_tok = self.cls_token + pos[:, :1, :]
        x = torch.cat([cls_tok.expand(x.size(0), -1, -1), x], dim=1)
        for blk in self.encoder:
            x = blk(x)
        x = self.encoder_norm(x)
        return x

    # ------------------------------------------------------------------
    # Token‑to‑vector projector (unchanged) -----------------------------
    # ------------------------------------------------------------------

    def _project_tokens(self, tok: torch.Tensor) -> torch.Tensor:
        if self.representation_type == "cls_token":
            return tok[:, 0, :]  # here was tok[:, :1, :]
        if self._flatten_fc is None:
            self._flatten_fc = nn.Linear((tok.size(1) - 1) * tok.size(2), self.dec_embed_dim)
        patches_flat = tok[:, 1:, :].flatten(1)
        return self._flatten_fc(patches_flat)

    # ------------------------------------------------------------------
    # Decoder forward --------------------------------------------------
    # ------------------------------------------------------------------

    def forward(self, imgs: torch.Tensor) -> torch.Tensor:
        tok = self.forward_encoder(imgs)
        tok = self.dec_embed(tok)
        tok = tok + self.dec_pos_embed.repeat(tok.size(0), 1, 1)
        z = self._project_tokens(tok)  # (B, Ddec)
        logits = self.hazard_net(z)    # (B, K, T)
        #hazards = torch.softmax(logits, dim=-1)  # (B, K, T) softmax is in loss
        #hazards = hazards.permute(0, 2, 1)
        logits = logits.contiguous() # (B, T, K) do we need it? 
        #print("logits", logits.shape)
        return logits

    # ------------------------------------------------------------------
    # Lightning steps --------------------------------------------------
    # ------------------------------------------------------------------

    def _discretise_labels(self, durations: torch.Tensor, events: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        dur_np = durations.detach().cpu().numpy()
        evt_np = events.detach().cpu().numpy()
        d_idx, e_idx = self.labtrans.transform(dur_np, evt_np)
        d_idx = torch.from_numpy(d_idx).long().to(self.device)
        e_idx = torch.from_numpy(e_idx).long().to(self.device)

        return d_idx, e_idx
    
    def _make_rank_mat(self, d_idx: torch.Tensor, e_idx: torch.Tensor) -> torch.Tensor:
        mat = pair_rank_mat(d_idx.cpu().numpy(), e_idx.cpu().numpy())  # (B,B) float32 CPU
        return torch.from_numpy(mat).to(self.device)


    def training_step(self, batch, batch_idx, mode: str = "train"):
        #imgs, (t, e), sub_idx, _ = batch
        imgs = batch["image"]
        (t, e) = batch["target_value"]
        sub_idx = batch["sub_idx"]

        imgs = imgs.float()
        t = t.float()
        e = e.long()
        
        durations_idx, events_idx = self._discretise_labels(t, e)
        logits = self.forward(imgs)

        if self.alpha == 1.0: 
            rank_mat_stub = torch.zeros((durations_idx.size(0), events_idx.size(0)), device=self.device) # dummy
        else:
            rank_mat_stub = torch.from_numpy(
                pair_rank_mat(
                    durations_idx.cpu().numpy(),        # durations *after* discretisation
                    events_idx.cpu().numpy()         # 0 = cens, 1..R = risks
                )
            ).to(logits.device)                 # (B , B) float32

        loss = self.criterion(logits, durations_idx, events_idx, rank_mat_stub)

        # Metric logging via the ModuleLogger helper of BasicModule
        self.module_logger.update_metric_item("train/training_step", 1, mode="train")
        self.module_logger.update_metric_item("train/surv_loss", loss.detach().item(), mode="train")
        self.log_dict({f"{mode}_surv_loss": loss.detach().item()})
        return loss

    def pad_col(self, input, val=0, where='end'):
        """Addes a column of `val` at the start of end of `input`."""
        if len(input.shape) != 2:
            raise ValueError(f"Only works for `phi` tensor that is 2-D.")
        pad = torch.zeros_like(input[:, :1])
        if val != 0:
            pad = pad + val
        if where == 'end':
            return torch.cat([input, pad], dim=1)
        elif where == 'start':
            return torch.cat([pad, input], dim=1)
        raise ValueError(f"Need `where` to be 'start' or 'end', got {where}")

    def predict_cif(self, logits):
        # logits : (B, R, T)

        # 1. flatten risk×time so each patient has one long logit vector
        flat = logits.view(logits.size(0), -1)          # (B, R*T)

        # 2. add zero-logit “no-event” column, softmax row-wise, drop it
        pmf_flat = self.pad_col(flat).softmax(1)[:, :-1]  # (B, R*T)

        # 3. reshape back and swap axes twice
        pmf = pmf_flat.view(logits.shape)      # (B, R, T)
        pmf = pmf.transpose(0, 1).transpose(1, 2)   # (R, T, B)

        # 4. cumsum over the time axis (dim=1)
        cif = pmf.cumsum(dim=1)                # (R, T, B) 
        return cif

    def predict_surv(self, logits, r):
        survival = 1 - self.predict_cif(logits)[r - 1]
        return survival

    @torch.no_grad()
    def validation_step(self, batch, batch_idx):
        #imgs, (t, e), sub_idx, _ = batch
        imgs = batch["image"]
        (t, e) = batch["target_value"]
        sub_idx = batch["sub_idx"]

        imgs = imgs.float()
        t = t.float()
        e = e.long()
        
        durations_idx, events_idx = self._discretise_labels(t, e)
        logits = self.forward(imgs)

        if self.alpha == 1.0: 
            rank_mat_stub = torch.zeros((durations_idx.size(0), events_idx.size(0)), device=self.device) # dummy
        else:
            rank_mat_stub = torch.from_numpy( #BETTER USE DEFAULT
                pair_rank_mat(
                    durations_idx.cpu().numpy(),        
                    events_idx.cpu().numpy()        
                )
            ).to(logits.device)                 # (B , B)
        
        loss = self.criterion(logits, durations_idx, events_idx, rank_mat_stub)

        self.module_logger.update_metric_item("val/surv_loss", loss.detach().item(), mode="val")
        self.log_dict({"val_surv_loss": loss.detach().item()})

        self._val_preds.append(self.predict_cif(logits).detach().cpu().numpy())
        self._val_t.append(t.cpu().numpy())
        self._val_e.append(e.cpu().numpy())
        self._val_idx.append(sub_idx.cpu().numpy())

    def make_rank_matrix(
            self,
            durations_idx: torch.Tensor,   # (B,)
            events_idx:    torch.Tensor,   # (B,)
            num_risks:     int,
            device=None,
        ) -> torch.Tensor:
        """
        Return a binary tensor rank_mat (B , B , R)
        rank_mat[i,j,r] = 1  if  t_i < t_j  and  events_idx[i] == r (>0)
        """
        device = device or durations_idx.device
        B = durations_idx.size(0)
        # (B, B) pairwise comparison t_i < t_j
        tmat = durations_idx.unsqueeze(1) < durations_idx.unsqueeze(0)   # bool
        # (B, R) one-hot of observed event type  (0 = censored)
        event_onehot = torch.zeros(B, num_risks, device=device, dtype=torch.bool)
        for r in range(1, num_risks + 1):
            event_onehot[:, r - 1] = events_idx == r

        # broadcast: (B,1)×(B,R) -> (B,B,R)
        rank_mat = (tmat.unsqueeze(2) & event_onehot.unsqueeze(0)).float()
        return rank_mat

    @torch.no_grad()
    def test_step(self, batch, batch_idx):
        #imgs, (t, e), sub_idx, _ = batch
        imgs = batch["image"]
        (t, e) = batch["target_value"]
        sub_idx = batch["sub_idx"]

        print("shape of imgs", imgs.shape)
        print("shape of t", t.shape)
        print("shape of e", e.shape)

        imgs = imgs.float()
        t = t.float()
        e = e.long()

        logits = self.forward(imgs)  # (B, T, K)

        self._test_preds.append(self.predict_cif(logits).detach().cpu().numpy())
        self._test_t.append(t.cpu().numpy())
        self._test_e.append(e.cpu().numpy())
        self._test_idx.append(sub_idx.cpu().numpy())

    def on_test_epoch_end(self):
        #logits_all = torch.cat(self._test_preds, dim=0)
        cif = np.concatenate(self._test_preds, axis=2)  # (R, T, N)

        t_all = np.concatenate(self._test_t)
        e_all = np.concatenate(self._test_e)
        idx_all = np.concatenate(self._test_idx)

        #cif = self.predict_cif(logits_all)  # (R, T, N)
        
        #cif = cif.cpu().numpy()

        surv_df = self._cif_to_survival_df(cif, idx_all)  # (N, R·K)

        
        horizons_q = [0.25, 0.50, 0.75]
        times_eval = np.quantile(t_all[e_all > 0], horizons_q)

        metrics_df = self.evaluate_single_fold(
            survival=surv_df,
            t=t_all,
            e=e_all,
            times_eval=times_eval,
        )


        # optional persistence
        surv_df["Use"] = self.fold_id
        # create survival_files directory if it does not exist
        os.makedirs(os.path.join(self.ckpt_dir, "survival_files"), exist_ok=True)
        surv_df.to_csv(os.path.join(self.ckpt_dir, "survival_files", "survival_single_fold_test.csv"))
        metrics_df.to_csv(os.path.join(self.ckpt_dir, "survival_files", "metrics_single_fold_test.csv"))
        print("\nTest metrics saved in", os.path.join(self.ckpt_dir, "survival_files"))

        # log scalars: one overall + three horizons per risk
        self.log_overall_metrics(metrics_df, mode="test")

        # clean buffers for next epoch
        self._test_preds.clear()
        self._test_t.clear()
        self._test_e.clear()
        self._test_idx.clear()
        
        self.module_logger.wandb_log(self.current_epoch, mode="test")

    


    def log_overall_metrics(
        self,
        metrics_df: pd.DataFrame,
        mode: str = "val",          # "val"  or  "test"
    ):
        """
        Log one scalar per risk:

            {mode}_CIS_risk_{r}
            {mode}_IBS_risk_{r}

        Parameters
        ----------
        metrics_df : pd.DataFrame
            Rows  = Risk (index)
            Cols  must contain "CIS" and "IBS".
        mode : str
            Tag prefix; choose "val" during validation, "test" at test-time.
        """
        assert mode in {"val", "test"}, "mode must be 'val' or 'test'"

        print("metrics_df", metrics_df)

        for risk_id, row in metrics_df.iterrows():     # iterate by *rows*
            cis = float(row.get("CIS", np.nan))
            brs = float(row.get("BRS", np.nan))
            print(f"Risk {risk_id}: CIS={cis:.3f}, BRS={brs:.3f}")

            self.module_logger.update_metric_item(
                f"{mode}/CIS_risk_{int(risk_id)}", cis, mode=mode)
            self.module_logger.update_metric_item(
                f"{mode}/BRS_risk_{int(risk_id)}", brs, mode=mode)
            

            #self.log(f"{mode}_CIS_risk_{int(risk_id)}",
            #        cis, prog_bar=True,  on_epoch=True,  sync_dist=True)

            #self.log(f"{mode}_IBS_risk_{int(risk_id)}",
            #        ibs, prog_bar=False, on_epoch=True,  sync_dist=True)


    def on_validation_epoch_end(self):
        #logits_all = torch.cat(self._val_preds, dim=0)
        #logits_all = np.concatenate(self._val_preds, axis=0)  # (R, T, N)
        cif = np.concatenate(self._val_preds, axis=2)  # (R, T, N)
        t_all = np.concatenate(self._val_t)
        e_all = np.concatenate(self._val_e)
        idx_all = np.concatenate(self._val_idx)

        #cif = self.predict_cif(logits_all)  # (R, T, N)
        
        #cif = cif.cpu().numpy()

        surv_df = self._cif_to_survival_df(cif, idx_all)  # (N, R·K)

        
        horizons_q = [0.25, 0.50, 0.75]
        times_eval = np.quantile(t_all[e_all > 0], horizons_q)

        metrics_df = self.evaluate_single_fold(
            survival=surv_df,
            t=t_all,
            e=e_all,
            times_eval=times_eval,
        )



        #print("\nValidation metrics (single fold)")
        #print(metrics_df)

        # optional persistence
        surv_df["Use"] = self.fold_id
        #surv_df.to_csv("survival_single_fold.csv")
        #metrics_df.to_csv("metrics_single_fold.csv")

        # log scalars: one overall + three horizons per risk
        self.log_overall_metrics(metrics_df, mode="val")

        # clean buffers for next epoch
        self._val_preds.clear()
        self._val_t.clear()
        self._val_e.clear()
        self._val_idx.clear()
        self.module_logger.wandb_log(self.current_epoch, mode="val")




    def evaluate_single_fold(
        self,
        survival,
        t,
        e,
        times_eval
    ):
        """
        Parameters
        ----------
        survival  : DataFrame (patients × (risk,time))
                    Values = survivor S_r(t).
        t, e      : 1-D arrays with durations and event indicators.
        times_eval: horizons for truncated metrics (None -> skip them).

        Returns
        -------
        pd.DataFrame  (rows = Risk, columns = metrics)
        """
        # tidy column index -------------------------------------------------
        surv = survival.copy()
        if "Use" in surv.columns:
            surv = surv.drop(columns="Use")

        surv.columns = pd.MultiIndex.from_frame(
            pd.DataFrame(index=surv.columns).reset_index().astype(float)
        )

        time_grid = surv.columns.get_level_values(1).unique()
        risks = surv.columns.get_level_values(0).unique()

        results = {}
        for r in risks:
            surv_r = surv[r]           
            dummy = pd.DataFrame(
                np.ones((len(time_grid), len(self.t_train))),
                index=time_grid
            )                        # (N , |grid|)
            km = EvalSurv(dummy, self.t_train, self.e_train != 0, censor_surv="km")
            ev      = EvalSurv(surv_r.T, t, e == int(r), censor_surv=km)

                    # ---------------- overall CIS / IBS ---------------------------
            try:
                cis_overall = ev.concordance_td()
            except ZeroDivisionError:       # no comparable pairs
                cis_overall = np.nan

            res = {"CIS": cis_overall}

            try:
                brs_overall = ev.integrated_brier_score(time_grid.to_numpy())
            except Exception:
                print("Exception in integrated_brier_score")
                brs_overall = np.nan

            res["BRS"] = brs_overall


            km = (self.e_train, self.t_train)

            # horizons ------------------------------------------------------
            if times_eval is not None:
                for te in times_eval:
                    try:
                        cis_trunc, km_pair = truncated_concordance_td(
                            e, t, 1 - surv_r.values, time_grid, te,
                            km=km, competing_risk=int(r)
                        )
                        brs_trunc = bs(
                            e, t, 1 - surv_r.values, time_grid, te,
                            km=km_pair, competing_risk=int(r)
                        )[0]
                        res[f"CIS@{te:.0f}"] = cis_trunc
                        res[f"BRS@{te:.0f}"] = brs_trunc
                    except Exception:
                        continue

            results[r] = pd.Series(res)

        return pd.DataFrame(results).T.set_index(pd.Index(risks, name="Risk"))




    def _cif_to_survival_df(self, cif, patient_ids):
        """
        Convert a CIF tensor (R,T,N) into a DataFrame shaped
            rows    = patients  (N)
            columns = MultiIndex (risk_id , time)  for *each* risk, NO all-cause.

        Parameters
        ----------
        cif : torch.Tensor or np.ndarray
            Shape (R, T, N)  –  already on CPU or still on GPU.
        patient_ids : 1-D sequence of length N
            Labels for the DataFrame rows.

        Returns
        -------
        pd.DataFrame
        """
        # --- ensure NumPy on CPU ------------------------------------------

        R, T, N = cif.shape
        cuts  = self.labtrans.cuts.astype("float64")          # (T,)
        times = np.asarray(self.times, dtype="float64")       # (K,)
        cols = pd.Index(patient_ids, dtype="object")        # canonicalised

        dfs = []
        for r in range(R):
            # cause-specific survivor  S_r(t) = 1 − F_r(t)
            surv_r = 1.0 - cif[r]                    # (T, N)
            # Step-function interpolation onto `times`
            df = pd.DataFrame(
                surv_r, index=cuts, columns=cols  # rows=T, cols=N
            ).astype("float32")
            pad = pd.DataFrame(np.nan, index=times, columns=cols)
            df = (df.reindex(np.union1d(cuts, times))   # add missing rows
                    .bfill()
                    .ffill()
                    .loc[times]                         # keep evaluation grid only
                    .T                                  # (N, K)
        )

            df.columns = pd.MultiIndex.from_product([[r + 1], times])
            dfs.append(df)

        return pd.concat(dfs, axis=1)     # (N , R·K)

                # stash tensors on CPU to limit GPU footprint
        #self._buf_logits.append(logits.cpu())
        #self._buf_d.append(durations_idx.cpu())
        #self._buf_e.append(events_idx.cpu())

        #cur_B = sum(x.size(0) for x in self._buf_d)      # accumulated size
        #if cur_B < self.virtual_B:                       # not yet full
        #    return None                                  # Lightning skips backward
        
        #logits_big = torch.cat(self._buf_logits).to(self.device)
        #d_big      = torch.cat(self._buf_d).to(self.device)
        #e_big      = torch.cat(self._buf_e).to(self.device)

        #rank_mat = (
        #torch.zeros(1, device=self.device) if self.alpha == 1.0   # NLL-only
        #    else self._make_rank_mat(d_big, e_big)                    # (B,B)
        #)

        #loss = self.criterion(logits_big, d_big, e_big, rank_mat)

        # normalise so lr is calibrated: divide by #mini-batches merged
        #loss = loss / (cur_B / imgs.size(0))

        # clear buffers for next cycle
        #self._buf_logits.clear(); self._buf_d.clear(); self._buf_e.clear()

        #self.log("train/surv_loss", loss, prog_bar=True, on_epoch=True)

        #for lst in (self._buf_logits, self._buf_d, self._buf_e):
        #    lst.clear()                     # drops Python references
        #torch.cuda.empty_cache()            # returns freed blocks to the pool
        #return loss