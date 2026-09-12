import os
import numpy as np
import torch
from torch.utils.data import DataLoader

from dataset import discover_iovnbd_pairs, split_by_driver, CachedIOVNBDSlidingWindow
from model import MultiTaskInertialNet, RobustInertialLoss

EPOCHS = 40
BATCH_SIZE = 64
LR = 1e-3
BASE_DIR = "dataset/Synchronised V and S datasets"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

print(f"[*] Starting Stateful Model 1 Training on {DEVICE}...")

all_pairs = discover_iovnbd_pairs(BASE_DIR)
train_pairs, val_pairs, test_pairs = split_by_driver(all_pairs)

if len(train_pairs) == 0 or len(val_pairs) == 0:
    raise ValueError("Missing training or validation pairs. Verify BASE_DIR.")

print("\n[*] Initializing 10Hz Datasets (20 samples per window)...")
train_dataset = CachedIOVNBDSlidingWindow(
    train_pairs, window_size=20, stride=2, is_train=True
)
val_dataset = CachedIOVNBDSlidingWindow(
    val_pairs,
    window_size=20,
    stride=2,
    mean=train_dataset.mean,
    std=train_dataset.std,
    is_train=False,
)

train_loader = DataLoader(
    train_dataset, batch_size=BATCH_SIZE, shuffle=True, drop_last=True
)
val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)

model = MultiTaskInertialNet().to(DEVICE)
criterion = RobustInertialLoss(lambda_lean=0.2)
optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=1e-4)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
    optimizer, T_max=EPOCHS, eta_min=1e-6
)

best_val_mae = float("inf")

print("\n[*] Training Loop Started...")
for epoch in range(EPOCHS):
    model.train()
    running_loss = 0.0

    for windows, v_prev, v_gt, phi_gt in train_loader:
        windows = windows.to(DEVICE)
        v_prev = v_prev.to(DEVICE)
        v_gt = v_gt.to(DEVICE)
        phi_gt = phi_gt.to(DEVICE)

        optimizer.zero_grad()

        v_pred, log_var_v, phi_pred, log_var_phi = model(windows, v_prev)
        loss = criterion(v_pred, log_var_v, v_gt, phi_pred, log_var_phi, phi_gt)

        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=2.0)
        optimizer.step()

        running_loss += loss.item()

    scheduler.step()

    # Validation Phase
    model.eval()
    val_v_errors = []
    val_phi_errors = []

    with torch.no_grad():
        for windows, v_prev, v_gt, phi_gt in val_loader:
            windows = windows.to(DEVICE)
            v_prev = v_prev.to(DEVICE)
            v_gt = v_gt.to(DEVICE)
            phi_gt = phi_gt.to(DEVICE)

            v_pred, _, phi_pred, _ = model(windows, v_prev)

            val_v_errors.extend(torch.abs(v_pred - v_gt).cpu().numpy())
            val_phi_errors.extend(torch.abs(phi_pred - phi_gt).cpu().numpy())

    mae_v = np.mean(val_v_errors)
    rmse_v = np.sqrt(np.mean(np.square(val_v_errors)))
    mae_phi_deg = np.rad2deg(np.mean(val_phi_errors))

    print(
        f"Epoch [{epoch+1:02d}/{EPOCHS}] | "
        f"Loss: {running_loss/len(train_loader):.4f} | "
        f"Val Speed MAE: {mae_v:.2f} m/s ({mae_v*3.6:.1f} km/h) | "
        f"Val Speed RMSE: {rmse_v:.2f} m/s | "
        f"Val Lean MAE: {mae_phi_deg:.1f}°"
    )

    if mae_v < best_val_mae:
        best_val_mae = mae_v
        torch.save(model.state_dict(), "model1_best.pth")
        if train_dataset.mean is None or train_dataset.std is None:
            raise ValueError(
                "Training dataset normalization statistics are unavailable."
            )
        np.savez(
            "normalization_stats.npz", mean=train_dataset.mean, std=train_dataset.std
        )

print("\n[SUCCESS] Training Complete. Best model saved as 'model1_best.pth'.")
