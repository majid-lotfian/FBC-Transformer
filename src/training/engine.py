from __future__ import annotations

from typing import Dict

import torch
from torch import nn

from ..objectives.objective_manager import ObjectiveManager


def move_batch_to_device(batch: Dict, device: torch.device) -> Dict:
    moved = {}

    for key, value in batch.items():
        if torch.is_tensor(value):
            moved[key] = value.to(device)
        else:
            moved[key] = value

    return moved


def train_step(
    model: nn.Module,
    batch: Dict,
    objective_manager: ObjectiveManager,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    *,
    grad_clip_norm: float | None = None,
) -> Dict[str, float]:
    model.train()
    batch = move_batch_to_device(batch, device)

    model_inputs = objective_manager.get_model_inputs(batch)






    
    optimizer.zero_grad(set_to_none=True)

    model_outputs = model(**model_inputs)



    objective_outputs = objective_manager.compute_total_loss(model_outputs, batch)

    loss = objective_outputs["loss"]



    loss.backward()

    if grad_clip_norm is not None:
        torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip_norm)

    optimizer.step()

    return {
        key: float(value.detach().cpu().item())
        for key, value in objective_outputs.items()
    }


'''@torch.no_grad()
def validation_step(
    model: nn.Module,
    batch: Dict,
    objective_manager: ObjectiveManager,
    device: torch.device,
) -> Dict[str, float]:
    model.eval()
    batch = move_batch_to_device(batch, device)

    model_inputs = objective_manager.get_model_inputs(batch)
    model_outputs = model(**model_inputs)
    objective_outputs = objective_manager.compute_total_loss(model_outputs, batch)

    return {
        key: float(value.detach().cpu().item())
        for key, value in objective_outputs.items()
    }'''
@torch.no_grad()
def validation_step(
    model: nn.Module,
    batch: Dict,
    objective_manager: ObjectiveManager,
    device: torch.device,
) -> Dict[str, float]:
    model.eval()
    batch = move_batch_to_device(batch, device)

    model_inputs = objective_manager.get_model_inputs(batch)
    model_outputs = model(**model_inputs)
    objective_outputs = objective_manager.compute_total_loss(model_outputs, batch)

    # TEMP DEBUG: catch rare exploding validation batches
    loss_value = float(objective_outputs["loss"].detach().cpu().item())
    if loss_value > 1000:
        print("\n=== EXTREME VALIDATION BATCH DETECTED ===")
        print("loss:", loss_value)
        print(
            "reconstruction_loss:",
            float(objective_outputs["reconstruction_loss"].detach().cpu().item()),
        )
        print(
            "reconstruction_mae:",
            float(objective_outputs["reconstruction_mae"].detach().cpu().item()),
        )

        preds = model_outputs["reconstruction"]
        targets = batch["masked_targets"]
        prediction_mask = batch["prediction_mask"]

        valid_mask = prediction_mask & torch.isfinite(targets)
        valid_targets = targets[valid_mask]
        valid_preds = preds[valid_mask]

        print("n_valid:", int(valid_targets.numel()))

        if valid_targets.numel() > 0:
            print("target min:", float(valid_targets.min().detach().cpu().item()))
            print("target max:", float(valid_targets.max().detach().cpu().item()))
            print("pred min:", float(valid_preds.min().detach().cpu().item()))
            print("pred max:", float(valid_preds.max().detach().cpu().item()))
            print(
                "abs diff max:",
                float((valid_preds - valid_targets).abs().max().detach().cpu().item()),
            )

    return {
        key: float(value.detach().cpu().item())
        for key, value in objective_outputs.items()
    }
