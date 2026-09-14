import os
import copy
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader

from model2 import Model2


# ============================================================
# CONFIGURATION
# ============================================================

BATCH_SIZE = 64
LEARNING_RATE = 1e-3
WEIGHT_DECAY = 1e-4

MAX_EPOCHS = 150
PATIENCE = 20

MODEL_PATH = "../models/model2_best.pth"

TRAIN_X = "../data/processed/model2_X_train.npy"
TRAIN_Y = "../data/processed/model2_Y_train.npy"

VAL_X = "../data/processed/model2_X_val.npy"
VAL_Y = "../data/processed/model2_Y_val.npy"

TEST_X = "../data/processed/model2_X_test.npy"
TEST_Y = "../data/processed/model2_Y_test.npy"

METADATA = "../data/processed/model2_metadata.csv"


# ============================================================
# DEVICE
# ============================================================

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

print("=" * 60)
print("MODEL 2 TRAINING")
print("=" * 60)
print("Device:", device)


# ============================================================
# LOAD DATA
# ============================================================

X_train = np.load(TRAIN_X)
Y_train = np.load(TRAIN_Y)

X_val = np.load(VAL_X)
Y_val = np.load(VAL_Y)

X_test = np.load(TEST_X)
Y_test = np.load(TEST_Y)

metadata = pd.read_csv(METADATA)

print("\nDataset shapes:")
print("Train X:", X_train.shape)
print("Train Y:", Y_train.shape)
print("Val   X:", X_val.shape)
print("Val   Y:", Y_val.shape)
print("Test  X:", X_test.shape)
print("Test  Y:", Y_test.shape)


# ============================================================
# GPS MASK
# ============================================================
# αR is only meaningful when GPS measurements exist.
#
# During a complete GPS outage:
#
#       R does not participate in the EKF measurement update.
#
# Therefore we do NOT force the neural network to learn a
# meaningless αR label during GPS-free windows.
# ============================================================

gps_updates = metadata["gps_updates"].values

train_end = len(X_train)
val_end = train_end + len(X_val)

train_gps = gps_updates[:train_end]
val_gps = gps_updates[train_end:val_end]
test_gps = gps_updates[val_end:val_end + len(X_test)]

train_r_mask = torch.tensor(train_gps > 0, dtype=torch.bool)
val_r_mask = torch.tensor(val_gps > 0, dtype=torch.bool)
test_r_mask = torch.tensor(test_gps > 0, dtype=torch.bool)

print("\nGPS-active windows:")
print("Train:", train_r_mask.sum().item(), "/", len(train_r_mask))
print("Val  :", val_r_mask.sum().item(), "/", len(val_r_mask))
print("Test :", test_r_mask.sum().item(), "/", len(test_r_mask))


# ============================================================
# PYTORCH DATASETS
# ============================================================

train_dataset = TensorDataset(
    torch.tensor(X_train, dtype=torch.float32),
    torch.tensor(Y_train, dtype=torch.float32)
)

val_dataset = TensorDataset(
    torch.tensor(X_val, dtype=torch.float32),
    torch.tensor(Y_val, dtype=torch.float32)
)

test_dataset = TensorDataset(
    torch.tensor(X_test, dtype=torch.float32),
    torch.tensor(Y_test, dtype=torch.float32)
)

train_loader = DataLoader(
    train_dataset,
    batch_size=BATCH_SIZE,
    shuffle=True
)

val_loader = DataLoader(
    val_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False
)

test_loader = DataLoader(
    test_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False
)


# ============================================================
# MODEL
# ============================================================

model = Model2(input_features=X_train.shape[2]).to(device)

print("\nModel:")
print(model)

print(
    "\nTrainable parameters:",
    sum(p.numel() for p in model.parameters() if p.requires_grad)
)


# ============================================================
# OPTIMIZER
# ============================================================

optimizer = torch.optim.AdamW(
    model.parameters(),
    lr=LEARNING_RATE,
    weight_decay=WEIGHT_DECAY
)

scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
    optimizer,
    mode="min",
    factor=0.5,
    patience=7,
    min_lr=1e-6
)


# ============================================================
# LOSS FUNCTION
# ============================================================
#
# αQ and αR range from 0.1 to 30.
#
# Instead of comparing:
#
#       predicted α - target α
#
# we compare:
#
#       log(predicted α) - log(target α)
#
# This prevents large values such as 30 from dominating the loss.
# ============================================================

huber = nn.SmoothL1Loss(reduction="none")


def covariance_loss(pred, target, r_mask):
    """
    pred   : [batch, 2]
             [:,0] = alpha_Q
             [:,1] = alpha_R

    target : [batch, 2]

    r_mask : [batch]
             True when GPS is available.
    """

    pred = torch.clamp(pred, min=1e-6)
    target = torch.clamp(target, min=1e-6)

    log_pred = torch.log(pred)
    log_target = torch.log(target)

    losses = huber(log_pred, log_target)

    # αQ is always relevant.
    q_loss = losses[:, 0].mean()

    # αR is only relevant when GPS is available.
    if r_mask.any():
        r_loss = losses[r_mask, 1].mean()
    else:
        r_loss = torch.tensor(0.0, device=pred.device)

    return q_loss + r_loss, q_loss.detach(), r_loss.detach()


# ============================================================
# TRAINING
# ============================================================

best_val_loss = float("inf")
best_state = None
epochs_without_improvement = 0

history = []

print("\n" + "=" * 60)
print("STARTING TRAINING")
print("=" * 60)

for epoch in range(1, MAX_EPOCHS + 1):

    # --------------------------------------------------------
    # TRAIN
    # --------------------------------------------------------

    model.train()

    train_loss_total = 0.0
    train_batches = 0

    for batch_x, batch_y in train_loader:

        batch_x = batch_x.to(device)
        batch_y = batch_y.to(device)

        # Determine which samples have GPS.
        # Dataset is shuffled, so use the target's position
        # indirectly through batch ordering is NOT possible.
        #
        # Therefore αR masking is handled using the fact that
        # gps_update is included as feature 12. A window is
        # considered GPS-active if any GPS update exists.
        gps_mask = batch_x[:, :, 12].max(dim=1).values > 0

        optimizer.zero_grad()

        prediction = model(batch_x)

        loss, q_loss, r_loss = covariance_loss(
            prediction,
            batch_y,
            gps_mask
        )

        loss.backward()

        torch.nn.utils.clip_grad_norm_(
            model.parameters(),
            max_norm=1.0
        )

        optimizer.step()

        train_loss_total += loss.item()
        train_batches += 1

    train_loss = train_loss_total / train_batches

    # --------------------------------------------------------
    # VALIDATION
    # --------------------------------------------------------

    model.eval()

    val_loss_total = 0.0
    val_batches = 0

    with torch.no_grad():

        for batch_x, batch_y in val_loader:

            batch_x = batch_x.to(device)
            batch_y = batch_y.to(device)

            gps_mask = batch_x[:, :, 12].max(dim=1).values > 0

            prediction = model(batch_x)

            loss, _, _ = covariance_loss(
                prediction,
                batch_y,
                gps_mask
            )

            val_loss_total += loss.item()
            val_batches += 1

    val_loss = val_loss_total / val_batches

    scheduler.step(val_loss)

    current_lr = optimizer.param_groups[0]["lr"]

    history.append(
        {
            "epoch": epoch,
            "train_loss": train_loss,
            "val_loss": val_loss,
            "learning_rate": current_lr
        }
    )

    print(
        f"Epoch {epoch:03d} | "
        f"Train Loss: {train_loss:.5f} | "
        f"Val Loss: {val_loss:.5f} | "
        f"LR: {current_lr:.2e}"
    )

    # --------------------------------------------------------
    # EARLY STOPPING
    # --------------------------------------------------------

    if val_loss < best_val_loss:

        best_val_loss = val_loss
        best_state = copy.deepcopy(model.state_dict())

        epochs_without_improvement = 0

        print("   ✓ New best model")

    else:

        epochs_without_improvement += 1

    if epochs_without_improvement >= PATIENCE:

        print("\nEarly stopping triggered.")

        break


# ============================================================
# SAVE BEST MODEL
# ============================================================

if best_state is not None:

    model.load_state_dict(best_state)

os.makedirs("../models", exist_ok=True)

checkpoint = {
    "model_state_dict": model.state_dict(),
    "input_features": X_train.shape[2],
    "best_val_loss": best_val_loss,
}

torch.save(checkpoint, MODEL_PATH)

print("\n" + "=" * 60)
print("BEST MODEL SAVED")
print("=" * 60)
print(MODEL_PATH)


# ============================================================
# TEST EVALUATION
# ============================================================

model.eval()

all_predictions = []
all_targets = []

with torch.no_grad():

    for batch_x, batch_y in test_loader:

        batch_x = batch_x.to(device)

        prediction = model(batch_x)

        all_predictions.append(
            prediction.cpu().numpy()
        )

        all_targets.append(
            batch_y.numpy()
        )

predictions = np.concatenate(all_predictions)
targets = np.concatenate(all_targets)


# ============================================================
# METRICS
# ============================================================

alpha_q_mae = np.mean(
    np.abs(predictions[:, 0] - targets[:, 0])
)

alpha_r_mae_all = np.mean(
    np.abs(predictions[:, 1] - targets[:, 1])
)

if test_r_mask.any():

    mask = test_r_mask.numpy()

    alpha_r_mae_gps = np.mean(
        np.abs(
            predictions[mask, 1] -
            targets[mask, 1]
        )
    )

else:

    alpha_r_mae_gps = float("nan")


print("\n" + "=" * 60)
print("TEST RESULTS")
print("=" * 60)

print(f"αQ MAE              : {alpha_q_mae:.4f}")
print(f"αR MAE (all)        : {alpha_r_mae_all:.4f}")
print(f"αR MAE (GPS active) : {alpha_r_mae_gps:.4f}")


# ============================================================
# PREDICTION DISTRIBUTION
# ============================================================

print("\nPrediction statistics:")

print(
    f"αQ prediction: "
    f"min={predictions[:,0].min():.3f}, "
    f"mean={predictions[:,0].mean():.3f}, "
    f"max={predictions[:,0].max():.3f}"
)

print(
    f"αR prediction: "
    f"min={predictions[:,1].min():.3f}, "
    f"mean={predictions[:,1].mean():.3f}, "
    f"max={predictions[:,1].max():.3f}"
)


# ============================================================
# SAVE PREDICTIONS
# ============================================================

prediction_df = pd.DataFrame(
    {
        "true_alpha_q": targets[:, 0],
        "pred_alpha_q": predictions[:, 0],
        "true_alpha_r": targets[:, 1],
        "pred_alpha_r": predictions[:, 1],
        "gps_active": test_r_mask.numpy()
    }
)

prediction_path = "../data/processed/model2_test_predictions.csv"

prediction_df.to_csv(
    prediction_path,
    index=False
)

print("\nSaved predictions:")
print(prediction_path)


# ============================================================
# SAVE TRAINING HISTORY
# ============================================================

history_df = pd.DataFrame(history)

history_path = "../results/model2_training_history.csv"

os.makedirs("../results", exist_ok=True)

history_df.to_csv(
    history_path,
    index=False
)

print("Saved training history:")
print(history_path)

print("\n" + "=" * 60)
print("MODEL 2 TRAINING COMPLETE")
print("=" * 60)