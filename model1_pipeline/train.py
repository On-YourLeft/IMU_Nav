import os
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

# Import the custom classes from the files we just made
from dataset import discover_iovnbd_pairs, split_by_driver, CachedIOVNBDSlidingWindow
from model import MultiTaskInertialNet, RobustInertialLoss


# ---------------------------------------------------------
# 1. Configuration & Hyperparameters
# ---------------------------------------------------------
EPOCHS = 40
BATCH_SIZE = 64
LR = 1e-3
BASE_DIR = "dataset/Synchronised V and S datasets"  # Path to your root data folder
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

print(f"[*] Starting Model 1 Training Pipeline on {DEVICE}...")

# ---------------------------------------------------------
# 2. Data Discovery & Loading
# ---------------------------------------------------------
# Crawl the directory and split strictly by Driver ID to prevent leakage
all_pairs = discover_iovnbd_pairs(BASE_DIR)
train_pairs, val_pairs, test_pairs = split_by_driver(all_pairs)

if len(train_pairs) == 0 or len(val_pairs) == 0:
    raise ValueError(
        "Could not find training/validation pairs. Check your BASE_DIR path."
    )

print(
    "\n[*] Initializing PyTorch Datasets (This may take a minute to slice windows)..."
)
train_dataset = CachedIOVNBDSlidingWindow(train_pairs, is_train=True)
val_dataset = CachedIOVNBDSlidingWindow(
    val_pairs, mean=train_dataset.mean, std=train_dataset.std, is_train=False
)

train_loader = DataLoader(
    train_dataset, batch_size=BATCH_SIZE, shuffle=True, drop_last=True
)
val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)

# ---------------------------------------------------------
# 3. Model, Loss, and Optimizer Setup
# ---------------------------------------------------------
model = MultiTaskInertialNet().to(DEVICE)
criterion = RobustInertialLoss(lambda_lean=0.2)
optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=1e-4)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
    optimizer, T_max=EPOCHS, eta_min=1e-6
)

best_val_mae = float("inf")

# ---------------------------------------------------------
# 4. Training Loop
# ---------------------------------------------------------
print("\n[*] Commencing Training...")
for epoch in range(EPOCHS):
    model.train()
    running_loss = 0.0

    for windows, v_gt, phi_gt in train_loader:
        windows = windows.to(DEVICE)
        v_gt = v_gt.to(DEVICE)
        phi_gt = phi_gt.to(DEVICE)

        optimizer.zero_grad()

        # Forward pass
        v_pred, log_var_v, phi_pred, log_var_phi = model(windows)

        # Calculate Heteroscedastic NLL Loss
        loss = criterion(v_pred, log_var_v, v_gt, phi_pred, log_var_phi, phi_gt)

        # Backward pass & optimize
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=2.0)
        optimizer.step()

        running_loss += loss.item()

    scheduler.step()

    # ---------------------------------------------------------
    # 5. Validation Phase
    # ---------------------------------------------------------
    model.eval()
    val_v_errors = []
    val_phi_errors = []

    with torch.no_grad():
        for windows, v_gt, phi_gt in val_loader:
            windows = windows.to(DEVICE)
            v_gt = v_gt.to(DEVICE)
            phi_gt = phi_gt.to(DEVICE)

            v_pred, _, phi_pred, _ = model(windows)

            val_v_errors.extend(torch.abs(v_pred - v_gt).cpu().numpy())
            val_phi_errors.extend(torch.abs(phi_pred - phi_gt).cpu().numpy())

    mae_v = np.mean(val_v_errors)
    rmse_v = np.sqrt(np.mean(np.square(val_v_errors)))
    mae_phi_deg = np.rad2deg(np.mean(val_phi_errors))

    print(
        f"Epoch [{epoch+1:02d}/{EPOCHS}] | "
        f"Loss: {running_loss/len(train_loader):.4f} | "
        f"Val Speed MAE: {mae_v:.2f} m/s ({mae_v*3.6:.1f} km/h) | "
        f"Val Lean MAE: {mae_phi_deg:.1f}°"
    )

    # Save best checkpoint
    if mae_v < best_val_mae:
        best_val_mae = mae_v
        torch.save(model.state_dict(), "model1_best.pth")
        if train_dataset.mean is None or train_dataset.std is None:
            raise RuntimeError("Training dataset normalization statistics are unavailable.")
        np.savez(
            "normalization_stats.npz", mean=train_dataset.mean, std=train_dataset.std
        )

# ---------------------------------------------------------
# 6. Export to ONNX for Mobile (10 Hz) & Edge (200 Hz)
# ---------------------------------------------------------
print("\n[*] Exporting final optimized model to ONNX...")
model.load_state_dict(
    torch.load("model1_best.pth", map_location="cpu", weights_only=True)
)
model.to("cpu")
model.eval()

dummy_input = torch.randn(1, 6, 20, dtype=torch.float32)
torch.onnx.export(
    model,
    (dummy_input,),
    "model1_velocity_lean.onnx",
    export_params=True,
    opset_version=14,
    do_constant_folding=True,
    input_names=["imu_window"],
    output_names=["v_x", "log_var_v", "phi", "log_var_phi"],
    dynamic_axes={"imu_window": {0: "batch_size"}},
)
print("[SUCCESS] Exported 'model1_velocity_lean.onnx'. Pipeline Complete.")
