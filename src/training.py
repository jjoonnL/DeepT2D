import random

import numpy as np
import torch
import torch.nn.functional as F
from torch.optim import AdamW
from torch.optim.lr_scheduler import StepLR
from torch.utils.data import DataLoader, TensorDataset

from src.models import build_mic


def set_deterministic_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def make_loader(data, batch_size, shuffle, seed, drop_last):
    dataset = TensorDataset(
        data["genotype"],
        data["proteome"],
        data["metabolite"],
        data["clinical"],
    )
    generator = torch.Generator().manual_seed(seed)
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        drop_last=drop_last,
        generator=generator,
        pin_memory=torch.cuda.is_available(),
    )


def build_optimizer(model, params, config_module):
    optimizer = AdamW(
        model.parameters(),
        lr=config_module.LEARNING_RATE,
        weight_decay=float(params["weight_decay"]),
    )
    scheduler = StepLR(
        optimizer,
        step_size=config_module.SCHEDULER_STEP_SIZE,
        gamma=config_module.SCHEDULER_GAMMA,
    )
    return optimizer, scheduler


def train_epoch(model, loader, optimizer, device):
    model.train()
    running_loss = 0.0
    running_n = 0

    for genotype, proteome, metabolite, clinical in loader:
        genotype = genotype.to(device, non_blocking=True)
        proteome = proteome.to(device, non_blocking=True)
        metabolite = metabolite.to(device, non_blocking=True)
        clinical = clinical.to(device, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)
        loss = F.mse_loss(model(genotype, proteome, metabolite), clinical)
        loss.backward()
        optimizer.step()

        running_loss += float(loss.item()) * len(clinical)
        running_n += len(clinical)

    if running_n == 0:
        raise ValueError("No training samples were used; check batch size and drop_last")
    return running_loss / running_n


def reconstruction_loss(model, data, config_module, device):
    loader = make_loader(
        data,
        batch_size=config_module.BATCH_SIZE,
        shuffle=False,
        seed=config_module.RANDOM_STATE,
        drop_last=False,
    )
    model.eval()
    total_loss = 0.0
    total_n = 0

    with torch.no_grad():
        for genotype, proteome, metabolite, clinical in loader:
            genotype = genotype.to(device, non_blocking=True)
            proteome = proteome.to(device, non_blocking=True)
            metabolite = metabolite.to(device, non_blocking=True)
            clinical = clinical.to(device, non_blocking=True)
            loss = F.mse_loss(model(genotype, proteome, metabolite), clinical)
            total_loss += float(loss.item()) * len(clinical)
            total_n += len(clinical)

    return total_loss / total_n


def train_with_early_stopping(train_data, validation_data, input_dims, params,
                              config_module, device, seed):
    set_deterministic_seed(seed)
    model = build_mic(input_dims, params, config_module).to(device)
    optimizer, scheduler = build_optimizer(model, params, config_module)
    loader = make_loader(
        train_data,
        batch_size=config_module.BATCH_SIZE,
        shuffle=True,
        seed=seed,
        drop_last=True,
    )

    best_loss = np.inf
    best_state = None
    best_epoch = 0
    best_train_loss = np.nan
    epochs_without_improvement = 0
    history = []

    for epoch in range(1, config_module.MAX_EPOCHS + 1):
        train_loss = train_epoch(model, loader, optimizer, device)
        validation_loss = reconstruction_loss(
            model, validation_data, config_module, device
        )
        history.append({
            "epoch": epoch,
            "train_loss": train_loss,
            "validation_loss": validation_loss,
            "learning_rate": optimizer.param_groups[0]["lr"],
        })

        if validation_loss < best_loss - config_module.EARLY_STOPPING_MIN_DELTA:
            best_loss = validation_loss
            best_epoch = epoch
            best_train_loss = train_loss
            best_state = {
                key: value.detach().clone()
                for key, value in model.state_dict().items()
            }
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1

        scheduler.step()
        if epochs_without_improvement >= config_module.EARLY_STOPPING_PATIENCE:
            break

    model.load_state_dict(best_state)
    return model, {
        "best_epoch": best_epoch,
        "stopped_epoch": epoch,
        "best_train_loss": float(best_train_loss),
        "best_validation_loss": float(best_loss),
        "history": history,
    }


def train_fixed_epochs(train_data, input_dims, params, epochs, config_module,
                       device, seed, shuffle=True, drop_last=True):
    set_deterministic_seed(seed)
    model = build_mic(input_dims, params, config_module).to(device)
    optimizer, scheduler = build_optimizer(model, params, config_module)
    loader = make_loader(
        train_data,
        batch_size=config_module.BATCH_SIZE,
        shuffle=shuffle,
        seed=seed,
        drop_last=drop_last,
    )
    history = []

    for epoch in range(1, epochs + 1):
        train_loss = train_epoch(model, loader, optimizer, device)
        scheduler.step()
        history.append({
            "epoch": epoch,
            "train_loss": train_loss,
            "learning_rate": optimizer.param_groups[0]["lr"],
        })

    return model, {
        "epochs": int(epochs),
        "final_train_loss": float(history[-1]["train_loss"]),
        "history": history,
    }


def extract_latent(model, data, config_module, device):
    loader = make_loader(
        data,
        batch_size=config_module.BATCH_SIZE,
        shuffle=False,
        seed=config_module.RANDOM_STATE,
        drop_last=False,
    )
    model.eval()
    latent = []

    with torch.no_grad():
        for genotype, proteome, metabolite, _ in loader:
            encoded = model.encode(
                genotype.to(device, non_blocking=True),
                proteome.to(device, non_blocking=True),
                metabolite.to(device, non_blocking=True),
            )
            latent.append(encoded.cpu().numpy())

    return np.concatenate(latent, axis=0)
