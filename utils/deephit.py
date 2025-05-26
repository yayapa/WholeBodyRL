import torch
import torchtuples as tt
from pycox.preprocessing.label_transforms import LabTransDiscreteTime
from typing import Optional, Tuple


class LabTransform(LabTransDiscreteTime):
    """Keep original event codes when transforming to discrete time."""

    def transform(self, durations, events):
        durations, is_event = super().transform(durations, events > 0)
        events[is_event == 0] = 0
        return durations, events.astype("int64")


class CauseSpecificNet(torch.nn.Module):
    """Simple cause‑specific network without shared trunk."""

    def __init__(
        self,
        in_features: int,
        num_nodes_indiv: Tuple[int, ...],
        num_risks: int,
        out_features: int,
        batch_norm: bool = True,
        dropout: Optional[float] = None,
    ) -> None:
        super().__init__()

        self.risk_nets = torch.nn.ModuleList(
            [
                tt.practical.MLPVanilla(
                    in_features,
                    list(num_nodes_indiv),
                    out_features,
                    batch_norm=batch_norm,
                    dropout=dropout,
                )
                for _ in range(num_risks)
            ]
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # (B, D) → (B, K, T)
        out = [net(x) for net in self.risk_nets]  # list of (B, T)
        out = torch.stack(out, dim=1)  # (B, K, T)
        return out
