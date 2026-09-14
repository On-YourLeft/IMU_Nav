import torch
import torch.nn as nn


# ============================================================
# MODEL 2
# Adaptive Covariance & Sensor-Fusion Network
# ============================================================

class Model2(nn.Module):
    """
    Temporal Convolutional Network for adaptive EKF covariance.

    Input:
        batch × 50 time steps × 16 features

    Output:
        batch × 2

        output[0] -> alpha_Q
        output[1] -> alpha_R
    """

    def __init__(self, input_features=16):
        super().__init__()

        # ----------------------------------------------------
        # Temporal feature extraction
        # ----------------------------------------------------

        self.temporal_layers = nn.Sequential(

            # 16 -> 32 features
            nn.Conv1d(
                in_channels=input_features,
                out_channels=32,
                kernel_size=3,
                padding=1
            ),

            nn.ReLU(),

            nn.BatchNorm1d(32),

            # 32 -> 32 features
            nn.Conv1d(
                in_channels=32,
                out_channels=32,
                kernel_size=3,
                padding=1
            ),

            nn.ReLU(),

            nn.BatchNorm1d(32),

            # ------------------------------------------------
            # Dilated convolution
            #
            # Allows the model to see a larger time context
            # without making the network very large.
            # ------------------------------------------------

            nn.Conv1d(
                in_channels=32,
                out_channels=64,
                kernel_size=3,
                padding=2,
                dilation=2
            ),

            nn.ReLU(),

            nn.BatchNorm1d(64),

            nn.Conv1d(
                in_channels=64,
                out_channels=64,
                kernel_size=3,
                padding=4,
                dilation=4
            ),

            nn.ReLU(),

            nn.BatchNorm1d(64)
        )

        # ----------------------------------------------------
        # Convert temporal representation into one vector
        # ----------------------------------------------------

        self.global_pool = nn.AdaptiveAvgPool1d(1)

        # ----------------------------------------------------
        # Fully connected decision layer
        # ----------------------------------------------------

        self.dense = nn.Sequential(

            nn.Linear(64, 32),

            nn.ReLU(),

            nn.Dropout(0.10)
        )

        # ----------------------------------------------------
        # Output layer
        #
        # Two outputs:
        #
        # 0 -> alpha_Q
        # 1 -> alpha_R
        # ----------------------------------------------------

        self.output_layer = nn.Linear(
            32,
            2
        )

        # ----------------------------------------------------
        # Softplus ensures covariance scaling remains positive.
        # ----------------------------------------------------

        self.softplus = nn.Softplus()

    def forward(self, x):

        # ----------------------------------------------------
        # Input arrives as:
        #
        # batch × time × features
        #
        # Conv1D expects:
        #
        # batch × features × time
        # ----------------------------------------------------

        x = x.transpose(
            1,
            2
        )

        # ----------------------------------------------------
        # Temporal convolutions
        # ----------------------------------------------------

        x = self.temporal_layers(x)

        # ----------------------------------------------------
        # Global temporal pooling
        #
        # batch × 64 × time
        #
        # becomes:
        #
        # batch × 64 × 1
        # ----------------------------------------------------

        x = self.global_pool(x)

        # Remove final dimension
        x = x.squeeze(-1)

        # ----------------------------------------------------
        # Dense layer
        # ----------------------------------------------------

        x = self.dense(x)

        # ----------------------------------------------------
        # Output
        # ----------------------------------------------------

        x = self.output_layer(x)

        # ----------------------------------------------------
        # Positive covariance scaling
        # ----------------------------------------------------

        x = self.softplus(x)

        return x


# ============================================================
# QUICK MODEL TEST
# ============================================================

if __name__ == "__main__":

    print("=" * 70)
    print("MODEL 2 ARCHITECTURE TEST")
    print("=" * 70)

    # --------------------------------------------------------
    # Create model
    # --------------------------------------------------------

    model = Model2(
        input_features=16
    )

    print("\nModel:")
    print(model)

    # --------------------------------------------------------
    # Count parameters
    # --------------------------------------------------------

    total_parameters = sum(
        p.numel()
        for p in model.parameters()
        if p.requires_grad
    )

    print(
        f"\nTrainable parameters: "
        f"{total_parameters:,}"
    )

    # --------------------------------------------------------
    # Create fake input matching our real dataset
    #
    # Batch = 4
    # Time = 50
    # Features = 16
    # --------------------------------------------------------

    test_input = torch.randn(
        4,
        50,
        16
    )

    # --------------------------------------------------------
    # Forward pass
    # --------------------------------------------------------

    with torch.no_grad():

        output = model(
            test_input
        )

    print(
        f"\nInput shape : "
        f"{test_input.shape}"
    )

    print(
        f"Output shape: "
        f"{output.shape}"
    )

    print("\nExample predictions:")

    print(output)

    # --------------------------------------------------------
    # Verify output is positive
    # --------------------------------------------------------

    if torch.all(output > 0):

        print(
            "\n✓ All covariance scaling values "
            "are positive."
        )

    else:

        print(
            "\n✗ ERROR: Negative covariance "
            "scaling detected."
        )

    # --------------------------------------------------------
    # Final check
    # --------------------------------------------------------

    if output.shape == (4, 2):

        print(
            "✓ Output shape is correct."
        )

    else:

        print(
            "✗ ERROR: Output shape is incorrect."
        )

    print("\n")
    print("=" * 70)
    print("MODEL 2 TEST COMPLETE")
    print("=" * 70)