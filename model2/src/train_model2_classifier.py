import os
import json
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader

from model2_classification import Model2Classifier

# ============================================================
# SETTINGS
# ============================================================
X_FILE = "../data/processed/model2_X.npy"
Y_FILE = "../data/processed/model2_Y.npy"

MODEL_FILE = "../models/model2_classifier_best.pth"

ALPHA_GRID = np.array([
    0.10, 0.15, 0.20, 0.30,
    0.50, 0.70, 1.00, 1.50,
    2.00, 3.00, 5.00, 7.00,
    10.0, 15.0, 20.0, 30.0
], dtype=np.float32)

EPOCHS = 60
BATCH_SIZE = 64
LR = 7e-4
PATIENCE = 12

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

print("=" * 70)
print("MODEL 2 - CLASSIFICATION / REGIME TRAINING")
print("=" * 70)
print("Device:", device)

X = np.load(X_FILE).astype(np.float32)
Y = np.load(Y_FILE).astype(np.float32)

n = len(X)
train_end = int(0.70 * n)
val_end = int(0.85 * n)

X_train, Y_train = X[:train_end], Y[:train_end]
X_val, Y_val = X[train_end:val_end], Y[train_end:val_end]
X_test, Y_test = X[val_end:], Y[val_end:]

def to_class(y):
    # Exact labels come from the same grid used to generate the dataset.
    idx = np.argmin(
        np.abs(y[..., None] - ALPHA_GRID[None, :]),
        axis=-1
    )
    return idx.astype(np.int64)

Q_train = to_class(Y_train[:, 0])
R_train = to_class(Y_train[:, 1])
Q_val = to_class(Y_val[:, 0])
R_val = to_class(Y_val[:, 1])

print("\nShapes:")
print("Train:", X_train.shape, Y_train.shape)
print("Val  :", X_val.shape, Y_val.shape)
print("Test :", X_test.shape, Y_test.shape)

def class_weights(labels):
    counts = np.bincount(
        labels,
        minlength=len(ALPHA_GRID)
    ).astype(np.float32)

    # sqrt inverse-frequency weighting is deliberately less aggressive
    # than full inverse-frequency weighting.
    weights = np.sqrt(
        counts.sum() / np.maximum(counts, 1.0)
    )

    # Prevent a rare class from completely dominating training.
    weights = np.clip(weights, 0.5, 6.0)

    # Normalize around 1.
    weights /= weights.mean()

    return torch.tensor(weights, dtype=torch.float32)

q_weights = class_weights(Q_train).to(device)
r_weights = class_weights(R_train).to(device)

train_ds = TensorDataset(
    torch.from_numpy(X_train),
    torch.from_numpy(Q_train),
    torch.from_numpy(R_train),
)

val_ds = TensorDataset(
    torch.from_numpy(X_val),
    torch.from_numpy(Q_val),
    torch.from_numpy(R_val),
)

train_loader = DataLoader(
    train_ds,
    batch_size=BATCH_SIZE,
    shuffle=True
)

val_loader = DataLoader(
    val_ds,
    batch_size=BATCH_SIZE,
    shuffle=False
)

model = Model2Classifier(
    input_features=X.shape[2],
    num_classes=len(ALPHA_GRID)
).to(device)

criterion_q = nn.CrossEntropyLoss(weight=q_weights)
criterion_r = nn.CrossEntropyLoss(weight=r_weights)

optimizer = torch.optim.AdamW(
    model.parameters(),
    lr=LR,
    weight_decay=1e-4
)

scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
    optimizer,
    mode="min",
    factor=0.5,
    patience=4
)

best_loss = float("inf")
best_epoch = 0
patience_count = 0

for epoch in range(1, EPOCHS + 1):

    model.train()
    train_loss = 0.0

    for xb, qb, rb in train_loader:
        xb = xb.to(device)
        qb = qb.to(device)
        rb = rb.to(device)

        optimizer.zero_grad()

        q_logits, r_logits = model(xb)

        loss_q = criterion_q(q_logits, qb)
        loss_r = criterion_r(r_logits, rb)

        loss = loss_q + loss_r

        loss.backward()

        torch.nn.utils.clip_grad_norm_(
            model.parameters(),
            max_norm=2.0
        )

        optimizer.step()

        train_loss += loss.item()

    train_loss /= max(len(train_loader), 1)

    model.eval()
    val_loss = 0.0

    with torch.no_grad():
        for xb, qb, rb in val_loader:
            xb = xb.to(device)
            qb = qb.to(device)
            rb = rb.to(device)

            q_logits, r_logits = model(xb)

            val_loss += (
                criterion_q(q_logits, qb)
                + criterion_r(r_logits, rb)
            ).item()

    val_loss /= max(len(val_loader), 1)

    scheduler.step(val_loss)

    lr_now = optimizer.param_groups[0]["lr"]

    print(
        f"Epoch {epoch:03d} | "
        f"train {train_loss:.4f} | "
        f"val {val_loss:.4f} | "
        f"lr {lr_now:.2e}"
    )

    if val_loss < best_loss - 1e-5:
        best_loss = val_loss
        best_epoch = epoch
        patience_count = 0

        torch.save({
            "model_state_dict": model.state_dict(),
            "input_features": int(X.shape[2]),
            "num_classes": len(ALPHA_GRID),
            "alpha_grid": ALPHA_GRID,
            "best_val_loss": best_loss,
            "epoch": epoch,
        }, MODEL_FILE)

    else:
        patience_count += 1

    if patience_count >= PATIENCE:
        print("Early stopping.")
        break

# ============================================================
# TEST
# ============================================================
checkpoint = torch.load(
    MODEL_FILE,
    map_location=device,
    weights_only=False
)

model.load_state_dict(checkpoint["model_state_dict"])
model.eval()

q_pred = []
r_pred = []

with torch.no_grad():
    for start in range(0, len(X_test), BATCH_SIZE):
        xb = torch.from_numpy(
            X_test[start:start + BATCH_SIZE]
        ).to(device)

        q_logits, r_logits = model(xb)

        q_prob = torch.softmax(q_logits, dim=1)
        r_prob = torch.softmax(r_logits, dim=1)

        # Geometric expectation: appropriate for covariance scales
        # spanning orders of magnitude.
        log_grid = torch.log(
            torch.tensor(ALPHA_GRID, device=device)
        )

        q_value = torch.exp(
            (q_prob * log_grid).sum(dim=1)
        )

        r_value = torch.exp(
            (r_prob * log_grid).sum(dim=1)
        )

        q_pred.append(q_value.cpu().numpy())
        r_pred.append(r_value.cpu().numpy())

q_pred = np.concatenate(q_pred)
r_pred = np.concatenate(r_pred)

q_true = Y_test[:, 0]
r_true = Y_test[:, 1]

print("\n" + "=" * 70)
print("TEST RESULTS")
print("=" * 70)

print(f"Q MAE: {np.mean(np.abs(q_pred - q_true)):.4f}")
print(f"R MAE: {np.mean(np.abs(r_pred - r_true)):.4f}")

print(
    "Q correlation:",
    np.corrcoef(q_pred, q_true)[0, 1]
)

print(
    "R correlation:",
    np.corrcoef(r_pred, r_true)[0, 1]
)

np.savetxt(
    "../data/processed/model2_classifier_test_predictions.csv",
    np.column_stack([q_true, q_pred, r_true, r_pred]),
    delimiter=",",
    header="q_true,q_pred,r_true,r_pred",
    comments=""
)

print("\nBest epoch:", best_epoch)
print("Best validation loss:", best_loss)
print("Saved:", MODEL_FILE)
