import numpy as np
import pandas as pd
from pathlib import Path
from tqdm import tqdm


# ============================================================
# PATHS
# ============================================================

ROOT = Path(__file__).resolve().parents[1]

PROCESSED_PATH = (
    ROOT / "data" / "processed" / "M_driver_B_processed.csv"
)

INNOVATION_PATH = (
    ROOT / "data" / "processed" / "M_driver_B_innovation.csv"
)

REFERENCE_PATH = (
    ROOT / "data" / "processed" / "M_driver_B_reference.csv"
)

OUTPUT_PATH = (
    ROOT / "data" / "processed" / "M_driver_B_covariance_labels.csv"
)


# ============================================================
# SETTINGS
# ============================================================

WINDOW_SIZE = 50          # 50 samples ≈ 5 seconds
STEP_SIZE = 50            # non-overlapping windows

# ------------------------------------------------------------
# NEW DENSE SEARCH GRID
# ------------------------------------------------------------
#
# Previous version:
# [0.1, 0.3, 1, 3, 10, 30]
#
# This version gives the optimizer more choices between
# the extremes.

ALPHA_Q_VALUES = np.array([
    0.10,
    0.15,
    0.20,
    0.30,
    0.50,
    0.70,
    1.00,
    1.50,
    2.00,
    3.00,
    5.00,
    7.00,
    10.00,
    15.00,
    20.00,
    30.00
])

ALPHA_R_VALUES = np.array([
    0.10,
    0.15,
    0.20,
    0.30,
    0.50,
    0.70,
    1.00,
    1.50,
    2.00,
    3.00,
    5.00,
    7.00,
    10.00,
    15.00,
    20.00,
    30.00
])


# ============================================================
# BASELINE COVARIANCES
# ============================================================

Q0 = np.diag([
    0.1,       # position x
    0.1,       # position y
    0.5,       # velocity
    0.01,      # heading
    0.001      # gyro bias
])

R0 = np.diag([
    10.0,      # GPS x
    10.0,      # GPS y
    2.0        # GPS speed
])


# ============================================================
# HELPER FUNCTIONS
# ============================================================

def latlon_to_local_xy(lat, lon):
    """
    Convert latitude/longitude to local x/y coordinates.

    x = east-west distance in meters
    y = north-south distance in meters
    """

    lat = np.asarray(lat)
    lon = np.asarray(lon)

    lat0 = lat[0]
    lon0 = lon[0]

    R_EARTH = 6371000.0

    x = np.deg2rad(lon - lon0) * R_EARTH * np.cos(
        np.deg2rad(lat0)
    )

    y = np.deg2rad(lat - lat0) * R_EARTH

    return x, y


def normalize_angle(angle):
    """
    Normalize angle to [-pi, pi].
    """

    return (angle + np.pi) % (2 * np.pi) - np.pi


def angle_difference(a, b):
    """
    Smallest angular difference a-b.
    """

    return normalize_angle(a - b)


# ============================================================
# EKF WINDOW EVALUATION
# ============================================================

def run_window(
    processed,
    innovation,
    reference,
    start,
    end,
    alpha_q,
    alpha_r
):
    """
    Run a small EKF over one 5-second window.

    IMPORTANT:
    The reference state is used ONLY to initialize the
    window for offline label generation.

    The reference is NOT used as an input feature to Model 2.
    """

    # --------------------------------------------------------
    # Initial reference state
    # --------------------------------------------------------

    x = np.array([
        reference["reference_x"].iloc[start],
        reference["reference_y"].iloc[start],
        reference["reference_velocity"].iloc[start],
        np.deg2rad(reference["reference_heading"].iloc[start]),
        0.0
    ], dtype=float)

    P = np.diag([
        1.0,
        1.0,
        1.0,
        np.deg2rad(5.0) ** 2,
        0.01
    ])

    # --------------------------------------------------------
    # Scaled covariances
    # --------------------------------------------------------

    Q = alpha_q * Q0
    R = alpha_r * R0

    position_errors = []
    velocity_errors = []
    heading_errors = []
    nis_values = []

    gps_updates = 0

    # --------------------------------------------------------
    # EKF loop
    # --------------------------------------------------------

    for i in range(start, end):

        dt = float(processed["dt"].iloc[i])

        if not np.isfinite(dt) or dt <= 0:
            dt = 0.1

        # ----------------------------------------------------
        # Inputs
        # ----------------------------------------------------

        acceleration = float(
            innovation["predicted_acceleration"].iloc[i]
        )

        yaw_rate = float(
            innovation["predicted_yaw_rate"].iloc[i]
        )

        if not np.isfinite(acceleration):
            acceleration = 0.0

        if not np.isfinite(yaw_rate):
            yaw_rate = 0.0

        # ----------------------------------------------------
        # Prediction
        # ----------------------------------------------------

        px = x[0]
        py = x[1]
        velocity = x[2]
        heading = x[3]
        gyro_bias = x[4]

        corrected_yaw_rate = yaw_rate - gyro_bias

        new_heading = normalize_angle(
            heading + corrected_yaw_rate * dt
        )

        new_velocity = velocity + acceleration * dt

        # Prevent physically impossible negative speed.
        new_velocity = max(new_velocity, 0.0)

        new_px = (
            px
            + velocity * np.sin(heading) * dt
        )

        new_py = (
            py
            + velocity * np.cos(heading) * dt
        )

        x_pred = np.array([
            new_px,
            new_py,
            new_velocity,
            new_heading,
            gyro_bias
        ])

        # ----------------------------------------------------
        # Jacobian
        # ----------------------------------------------------

        F = np.eye(5)

        F[0, 2] = np.sin(heading) * dt
        F[0, 3] = velocity * np.cos(heading) * dt

        F[1, 2] = np.cos(heading) * dt
        F[1, 3] = -velocity * np.sin(heading) * dt

        F[3, 4] = -dt

        P_pred = F @ P @ F.T + Q

        x = x_pred
        P = P_pred

        # ----------------------------------------------------
        # GPS UPDATE
        # ----------------------------------------------------

        gps_update = int(
            innovation["gps_update"].iloc[i]
        )

        if gps_update == 1:

            gps_x = float(
                processed["_gps_x"].iloc[i]
            )

            gps_y = float(
                processed["_gps_y"].iloc[i]
            )

            gps_speed = float(
                processed["gps_speed"].iloc[i]
            )

            if (
                np.isfinite(gps_x)
                and np.isfinite(gps_y)
                and np.isfinite(gps_speed)
            ):

                z = np.array([
                    gps_x,
                    gps_y,
                    gps_speed
                ])

                H = np.zeros((3, 5))

                H[0, 0] = 1.0
                H[1, 1] = 1.0
                H[2, 2] = 1.0

                innovation_vector = (
                    z - H @ x
                )

                S = (
                    H @ P @ H.T
                    + R
                )

                try:

                    S_inv = np.linalg.inv(S)

                    nis = float(
                        innovation_vector.T
                        @ S_inv
                        @ innovation_vector
                    )

                    if np.isfinite(nis):
                        nis_values.append(nis)

                    K = (
                        P
                        @ H.T
                        @ S_inv
                    )

                    x = (
                        x
                        + K @ innovation_vector
                    )

                    # Joseph-form covariance update
                    I = np.eye(5)

                    P = (
                        (I - K @ H)
                        @ P
                        @ (I - K @ H).T
                        + K @ R @ K.T
                    )

                    x[3] = normalize_angle(x[3])

                    gps_updates += 1

                except np.linalg.LinAlgError:
                    pass

        # ----------------------------------------------------
        # ERROR AGAINST REFERENCE
        # ----------------------------------------------------

        ref_x = float(
            reference["reference_x"].iloc[i]
        )

        ref_y = float(
            reference["reference_y"].iloc[i]
        )

        ref_velocity = float(
            reference["reference_velocity"].iloc[i]
        )

        ref_heading = np.deg2rad(
            reference["reference_heading"].iloc[i]
        )

        position_error = np.sqrt(
            (x[0] - ref_x) ** 2
            + (x[1] - ref_y) ** 2
        )

        velocity_error = abs(
            x[2] - ref_velocity
        )

        heading_error = abs(
            np.rad2deg(
                angle_difference(
                    x[3],
                    ref_heading
                )
            )
        )

        position_errors.append(position_error)
        velocity_errors.append(velocity_error)
        heading_errors.append(heading_error)

    # --------------------------------------------------------
    # Window metrics
    # --------------------------------------------------------

    return {
        "position_mae": np.mean(position_errors),
        "velocity_mae": np.mean(velocity_errors),
        "heading_mae": np.mean(heading_errors),
        "mean_nis": (
            np.mean(nis_values)
            if len(nis_values) > 0
            else np.nan
        ),
        "gps_updates": gps_updates
    }


# ============================================================
# MAIN
# ============================================================

print("=" * 70)
print("GENERATING DENSE COVARIANCE LABELS FOR MODEL 2")
print("=" * 70)

print("\nLoading data...")

processed = pd.read_csv(
    PROCESSED_PATH
)

innovation = pd.read_csv(
    INNOVATION_PATH
)

reference = pd.read_csv(
    REFERENCE_PATH
)

print("Processed shape:", processed.shape)
print("Innovation shape:", innovation.shape)
print("Reference shape:", reference.shape)


# ============================================================
# CHECK ALIGNMENT
# ============================================================

if not (
    len(processed)
    == len(innovation)
    == len(reference)
):
    raise ValueError(
        "Processed, innovation and reference data "
        "do not have the same number of rows."
    )


# ============================================================
# CREATE GPS LOCAL COORDINATES
# ============================================================

print("\nCreating local GPS coordinates...")

gps_x, gps_y = latlon_to_local_xy(
    processed["gps_lat"].values,
    processed["gps_lon"].values
)

processed["_gps_x"] = gps_x
processed["_gps_y"] = gps_y


# ============================================================
# WINDOW SETUP
# ============================================================

n = len(processed)

windows = []

for start in range(
    0,
    n - WINDOW_SIZE + 1,
    STEP_SIZE
):

    end = start + WINDOW_SIZE

    windows.append(
        (start, end)
    )

print("\nWindow configuration:")
print("Window size:", WINDOW_SIZE)
print("Step size:", STEP_SIZE)
print("Number of windows:", len(windows))

print("\nDense search:")
print("Alpha Q values:", len(ALPHA_Q_VALUES))
print("Alpha R values:", len(ALPHA_R_VALUES))
print(
    "Combinations per window:",
    len(ALPHA_Q_VALUES)
    * len(ALPHA_R_VALUES)
)

print(
    "Maximum filter evaluations:",
    len(windows)
    * len(ALPHA_Q_VALUES)
    * len(ALPHA_R_VALUES)
)


# ============================================================
# LABEL GENERATION
# ============================================================

results = []

total_evaluations = 0

print("\nGenerating labels...\n")

for window_number, (start, end) in enumerate(
    tqdm(windows)
):

    # --------------------------------------------------------
    # Determine whether this window contains GPS.
    # --------------------------------------------------------

    gps_mask = (
        innovation["gps_update"]
        .iloc[start:end]
        .values
        == 1
    )

    has_gps = bool(
        np.any(gps_mask)
    )

    # --------------------------------------------------------
    # Best result
    # --------------------------------------------------------

    best_score = np.inf
    best_result = None

    # --------------------------------------------------------
    # Search αQ
    # --------------------------------------------------------

    for alpha_q in ALPHA_Q_VALUES:

        # ----------------------------------------------------
        # If GPS is available, search αR normally.
        # ----------------------------------------------------

        if has_gps:

            r_values = ALPHA_R_VALUES

        # ----------------------------------------------------
        # If there is NO GPS, αR has no effect.
        #
        # We only need one neutral value.
        # ----------------------------------------------------

        else:

            r_values = [1.0]

        for alpha_r in r_values:

            metrics = run_window(
                processed,
                innovation,
                reference,
                start,
                end,
                alpha_q,
                alpha_r
            )

            total_evaluations += 1

            # ------------------------------------------------
            # Combined objective
            # ------------------------------------------------
            #
            # Position is the main objective.
            #
            # Velocity and heading are included with
            # smaller weights so that we don't select a
            # covariance setting that gives good position
            # while completely destroying motion estimation.
            #
            # All quantities are normalized approximately
            # to comparable scales.

            score = (
                metrics["position_mae"]
                + 5.0 * metrics["velocity_mae"]
                + 0.10 * metrics["heading_mae"]
            )

            if score < best_score:

                best_score = score

                best_result = {
                    "alpha_q": alpha_q,
                    "alpha_r": alpha_r,
                    "position_mae": metrics[
                        "position_mae"
                    ],
                    "velocity_mae": metrics[
                        "velocity_mae"
                    ],
                    "heading_mae": metrics[
                        "heading_mae"
                    ],
                    "mean_nis": metrics[
                        "mean_nis"
                    ],
                    "gps_updates": metrics[
                        "gps_updates"
                    ],
                    "has_gps": has_gps
                }

    # --------------------------------------------------------
    # Store label
    # --------------------------------------------------------

    result = {
        "window": window_number,
        "start_index": start,
        "end_index": end,
        "time": float(
            processed["time"].iloc[start]
        ),

        "alpha_q": best_result["alpha_q"],
        "alpha_r": best_result["alpha_r"],

        "position_mae": best_result[
            "position_mae"
        ],

        "velocity_mae": best_result[
            "velocity_mae"
        ],

        "heading_mae": best_result[
            "heading_mae"
        ],

        "mean_nis": best_result[
            "mean_nis"
        ],

        "gps_updates": best_result[
            "gps_updates"
        ],

        "has_gps": int(
            best_result["has_gps"]
        )
    }

    results.append(result)


# ============================================================
# SAVE
# ============================================================

labels = pd.DataFrame(results)

OUTPUT_PATH.parent.mkdir(
    parents=True,
    exist_ok=True
)

labels.to_csv(
    OUTPUT_PATH,
    index=False
)


# ============================================================
# SUMMARY
# ============================================================

print("\n" + "=" * 70)
print("LABEL GENERATION COMPLETE")
print("=" * 70)

print("\nNumber of labelled windows:", len(labels))

print(
    "Total filter evaluations:",
    total_evaluations
)

print(
    "Saved to:",
    OUTPUT_PATH
)


# ============================================================
# ALPHA Q DISTRIBUTION
# ============================================================

print("\n" + "=" * 70)
print("ALPHA Q DISTRIBUTION")
print("=" * 70)

print(
    labels["alpha_q"]
    .value_counts()
    .sort_index()
    .to_string()
)


# ============================================================
# ALPHA R DISTRIBUTION
# ============================================================

print("\n" + "=" * 70)
print("ALPHA R DISTRIBUTION")
print("=" * 70)

print(
    labels["alpha_r"]
    .value_counts()
    .sort_index()
    .to_string()
)


# ============================================================
# GPS ACTIVE DISTRIBUTION
# ============================================================

gps_active = labels[
    labels["has_gps"] == 1
]

gps_inactive = labels[
    labels["has_gps"] == 0
]

print("\n" + "=" * 70)
print("GPS WINDOW SUMMARY")
print("=" * 70)

print(
    "GPS-active windows:",
    len(gps_active)
)

print(
    "No-GPS windows:",
    len(gps_inactive)
)

if len(gps_active) > 0:

    print(
        "\nGPS-active Alpha R distribution:"
    )

    print(
        gps_active["alpha_r"]
        .value_counts()
        .sort_index()
        .to_string()
    )

    print(
        "\nGPS-active Alpha Q distribution:"
    )

    print(
        gps_active["alpha_q"]
        .value_counts()
        .sort_index()
        .to_string()
    )


# ============================================================
# AVERAGE ERRORS
# ============================================================

print("\n" + "=" * 70)
print("AVERAGE LABEL PERFORMANCE")
print("=" * 70)

print(
    f"Position MAE: "
    f"{labels['position_mae'].mean():.3f} m"
)

print(
    f"Velocity MAE: "
    f"{labels['velocity_mae'].mean():.3f} m/s"
)

print(
    f"Heading MAE: "
    f"{labels['heading_mae'].mean():.3f} deg"
)


# ============================================================
# FIRST 15 WINDOWS
# ============================================================

print("\n" + "=" * 70)
print("FIRST 15 LABELS")
print("=" * 70)

print(
    labels[
        [
            "window",
            "time",
            "alpha_q",
            "alpha_r",
            "gps_updates",
            "position_mae"
        ]
    ]
    .head(15)
    .to_string(index=False)
)


print("\n" + "=" * 70)
print("DONE")
print("=" * 70)