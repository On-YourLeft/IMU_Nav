import torch
import torch.nn as nn


class MultiTaskInertialNet(nn.Module):
    def __init__(self, in_channels=6, gru_hidden=64, gru_layers=2):
        """
        Input x: [Batch, 6, 20]
        Input v_prev: [Batch]
        """
        super().__init__()

        # 1. 1D-CNN: Filter bank for 10 Hz with dropout
        self.feature_extractor = nn.Sequential(
            nn.Conv1d(in_channels, 32, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm1d(32),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Dropout1d(p=0.15),
            nn.Conv1d(32, 64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm1d(64),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Dropout1d(p=0.15),
            nn.Conv1d(64, 128, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm1d(128),
            nn.LeakyReLU(0.1, inplace=True),
        )

        # 2. Recurrent Core
        self.gru = nn.GRU(
            input_size=128,
            hidden_size=gru_hidden,
            num_layers=gru_layers,
            batch_first=True,
            dropout=0.15 if gru_layers > 1 else 0.0,
        )

        # 3. Head A: Stateful Velocity Head (gru_hidden + 1 for v_prev)
        self.vel_head = nn.Sequential(
            nn.Linear(gru_hidden + 1, 32),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Linear(32, 2),
        )

        # 4. Head B: Roll / Lean Regressor (Unchanged to protect bike dynamics)
        self.lean_head = nn.Sequential(
            nn.Linear(gru_hidden, 32), nn.LeakyReLU(0.1, inplace=True), nn.Linear(32, 2)
        )

    def forward(self, x, v_prev):
        # x: [Batch, 6, 20]
        feat = self.feature_extractor(x)  # [Batch, 128, 20]
        feat = feat.permute(0, 2, 1)  # [Batch, 20, 128]

        gru_out, _ = self.gru(feat)  # [Batch, 20, 64]
        last_step = gru_out[:, -1, :]  # [Batch, 64]

        # Concatenate v_prev only into velocity input
        v_prev_expanded = v_prev.unsqueeze(1)  # [Batch, 1]
        vel_input = torch.cat([last_step, v_prev_expanded], dim=1)  # [Batch, 65]

        vel_out = self.vel_head(vel_input)  # [Batch, 2]
        lean_out = self.lean_head(last_step)  # [Batch, 2]

        v_x = vel_out[:, 0]
        log_var_v = torch.clamp(vel_out[:, 1], min=-4.0, max=4.0)

        phi = lean_out[:, 0]
        log_var_phi = torch.clamp(lean_out[:, 1], min=-4.0, max=4.0)

        return v_x, log_var_v, phi, log_var_phi


class RobustInertialLoss(nn.Module):
    def __init__(self, lambda_lean=0.2):
        super().__init__()
        self.lambda_lean = lambda_lean
        self.huber = nn.SmoothL1Loss(reduction="none")

    def forward(self, v_pred, log_var_v, v_gt, phi_pred, log_var_phi, phi_gt):
        huber_v = self.huber(v_pred, v_gt)
        loss_v = torch.exp(-log_var_v) * huber_v + 0.5 * log_var_v

        huber_phi = self.huber(phi_pred, phi_gt)
        loss_phi = torch.exp(-log_var_phi) * huber_phi + 0.5 * log_var_phi

        return torch.mean(loss_v) + self.lambda_lean * torch.mean(loss_phi)
