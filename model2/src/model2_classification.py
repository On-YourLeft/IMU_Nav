import torch
import torch.nn as nn


class Model2Classifier(nn.Module):
    """
    Model 2: Adaptive Covariance Tuner.

    Same approach as before:
        temporal sensor/innovation window -> adaptive Q/R

    Difference:
        Q and R scales are treated as discrete covariance regimes
        rather than ordinary continuous regression targets.

    This avoids regression-to-the-mean on the strongly bimodal alpha_R
    labels and makes rare high-noise regimes learnable.
    """

    def __init__(self, input_features=16, num_classes=16):
        super().__init__()

        self.features = nn.Sequential(
            nn.Conv1d(input_features, 32, kernel_size=5, padding=2),
            nn.BatchNorm1d(32),
            nn.GELU(),

            nn.Conv1d(32, 48, kernel_size=5, padding=4, dilation=2),
            nn.BatchNorm1d(48),
            nn.GELU(),

            nn.Conv1d(48, 64, kernel_size=5, padding=8, dilation=4),
            nn.BatchNorm1d(64),
            nn.GELU(),

            nn.Conv1d(64, 64, kernel_size=3, padding=2, dilation=2),
            nn.BatchNorm1d(64),
            nn.GELU(),
        )

        self.pool = nn.AdaptiveAvgPool1d(1)

        self.shared = nn.Sequential(
            nn.Linear(64, 64),
            nn.GELU(),
            nn.Dropout(0.10),
        )

        self.q_head = nn.Linear(64, num_classes)
        self.r_head = nn.Linear(64, num_classes)

    def forward(self, x):
        # x: [batch, time, features]
        x = x.transpose(1, 2)
        x = self.features(x)
        x = self.pool(x).squeeze(-1)
        x = self.shared(x)

        return self.q_head(x), self.r_head(x)
