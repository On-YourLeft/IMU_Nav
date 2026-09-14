import os
import numpy as np
import pandas as pd


# ============================================================
# CONFIGURATION
# ============================================================

ROOT = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..")
)

PROCESSED_FILE = os.path.join(
    ROOT,
    "data",
    "processed",
    "M_driver_B_processed.csv"
)

INNOVATION_FILE = os.path.join(
    ROOT,
    "data",
    "processed",
    "M_driver_B_innovation.csv"
)

LABEL_FILE = os.path.join(
    ROOT,
    "data",
    "processed",
    "M_driver_B_covariance_labels.csv"
)

OUTPUT_DIR = os.path.join(
    ROOT,
    "data",
    "processed"
)

OUTPUT_X = os.path.join(
    OUTPUT_DIR,
    "model2_X.npy"
)

OUTPUT_Y = os.path.join(
    OUTPUT_DIR,
    "model2_Y.npy"
)

OUTPUT_META = os.path.join(
    OUTPUT_DIR,
    "model2_metadata.csv"
)

WINDOW_SIZE = 50


# ============================================================
# LOAD DATA
# ============================================================

print("=" * 70)
print("MODEL 2 DATASET PREPARATION")
print("=" * 70)

print("\nLoading data...")

processed = pd.read_csv(
    PROCESSED_FILE
)

innovation = pd.read_csv(
    INNOVATION_FILE
)

labels = pd.read_csv(
    LABEL_FILE
)

print(
    f"Processed data : {processed.shape}"
)

print(
    f"Innovation data: {innovation.shape}"
)

print(
    f"Labels         : {labels.shape}"
)


# ============================================================
# CHECK DATA
# ============================================================

if len(processed) != len(innovation):

    raise ValueError(
        "Processed and innovation data "
        "have different lengths."
    )


# ============================================================
# CHECK LABEL COLUMNS
# ============================================================

required_label_columns = [
    "window",
    "start_index",
    "end_index",
    "time",
    "alpha_q",
    "alpha_r",
    "position_mae",
    "velocity_mae",
    "heading_mae",
    "mean_nis",
    "gps_updates",
    "has_gps"
]

missing_labels = [
    column
    for column in required_label_columns
    if column not in labels.columns
]

if missing_labels:

    raise ValueError(
        "Missing label columns: "
        + ", ".join(missing_labels)
    )


# ============================================================
# FEATURES
# ============================================================

FEATURE_COLUMNS = [

    # --------------------------------------------------------
    # IMU
    # --------------------------------------------------------

    "acc_x",
    "acc_y",
    "acc_z",

    "gyro_yaw",
    "gyro_pitch",
    "gyro_roll",

    # --------------------------------------------------------
    # Calibrated motion model
    # --------------------------------------------------------

    "predicted_acceleration",
    "predicted_yaw_rate",

    # --------------------------------------------------------
    # EKF innovation
    # --------------------------------------------------------

    "innovation_x",
    "innovation_y",
    "innovation_speed",

    # --------------------------------------------------------
    # GPS / uncertainty information
    # --------------------------------------------------------

    "nis",
    "gps_update",
    "gps_accuracy",
    "gps_satellites",
    "gps_speed",
]


print("\nFeatures")
print("-" * 70)

for i, feature in enumerate(
    FEATURE_COLUMNS
):

    print(
        f"{i:2d}. {feature}"
    )


# ============================================================
# CREATE FEATURE DATAFRAME
# ============================================================

feature_data = pd.DataFrame(
    index=processed.index
)

for column in FEATURE_COLUMNS:

    if column in processed.columns:

        feature_data[column] = (
            processed[column]
        )

    elif column in innovation.columns:

        feature_data[column] = (
            innovation[column]
        )

    else:

        raise ValueError(
            f"Required feature not found: {column}"
        )


# ============================================================
# HANDLE GPS UPDATE
# ============================================================

feature_data["gps_update"] = (
    pd.to_numeric(
        feature_data["gps_update"],
        errors="coerce"
    )
    .fillna(0.0)
    .astype(float)
)


# ============================================================
# HANDLE NIS
# ============================================================

# NIS is unavailable during GPS outages.
# Represent this as zero while retaining gps_update=0.

feature_data["nis"] = (
    pd.to_numeric(
        feature_data["nis"],
        errors="coerce"
    )
)

feature_data["nis"] = (
    feature_data["nis"]
    .replace(
        [np.inf, -np.inf],
        np.nan
    )
)

feature_data["nis"] = (
    feature_data["nis"]
    .fillna(0.0)
)

# Compress very large NIS values.
feature_data["nis"] = np.log1p(
    np.maximum(
        feature_data["nis"].values,
        0.0
    )
)


# ============================================================
# HANDLE GPS ACCURACY
# ============================================================

feature_data["gps_accuracy"] = (
    pd.to_numeric(
        feature_data["gps_accuracy"],
        errors="coerce"
    )
)

feature_data["gps_accuracy"] = (
    feature_data["gps_accuracy"]
    .replace(
        [np.inf, -np.inf],
        np.nan
    )
    .fillna(10.0)
)

feature_data["gps_accuracy"] = np.clip(
    feature_data["gps_accuracy"],
    0.0,
    100.0
)


# ============================================================
# HANDLE GPS SATELLITES
# ============================================================

feature_data["gps_satellites"] = (
    pd.to_numeric(
        feature_data["gps_satellites"],
        errors="coerce"
    )
)

feature_data["gps_satellites"] = (
    feature_data["gps_satellites"]
    .replace(
        [np.inf, -np.inf],
        np.nan
    )
    .fillna(0.0)
)


# ============================================================
# HANDLE ALL REMAINING VALUES
# ============================================================

feature_data = feature_data.replace(
    [np.inf, -np.inf],
    np.nan
)

feature_data = feature_data.interpolate(
    limit_direction="both"
)

feature_data = feature_data.fillna(
    0.0
)


# ============================================================
# EXTRACT FEATURE MATRIX
# ============================================================

features = feature_data[
    FEATURE_COLUMNS
].values.astype(
    np.float32
)


# ============================================================
# NORMALIZATION
# ============================================================

print("\nNormalizing features...")
print("-" * 70)

feature_mean = np.mean(
    features,
    axis=0
)

feature_std = np.std(
    features,
    axis=0
)

# Prevent division by zero.
feature_std[
    feature_std < 1e-6
] = 1.0

features_normalized = (
    features - feature_mean
) / feature_std


# ============================================================
# CREATE TEMPORAL WINDOWS
# ============================================================

print("\nCreating temporal windows...")
print("-" * 70)

X = []
Y = []
metadata = []


for _, row in labels.iterrows():

    # --------------------------------------------------------
    # NEW LABEL FORMAT
    # --------------------------------------------------------

    window_id = int(
        row["window"]
    )

    start = int(
        row["start_index"]
    )

    end = int(
        row["end_index"]
    ) 


    # --------------------------------------------------------
    # Safety check
    # --------------------------------------------------------

    if end - start != WINDOW_SIZE:

        print(
            f"Skipping window {window_id}: "
            f"size={end-start}"
        )

        continue

    if (
        start < 0
        or end > len(features_normalized)
    ):

        print(
            f"Skipping window {window_id}: "
            f"invalid indices {start}:{end}"
        )

        continue


    # --------------------------------------------------------
    # INPUT WINDOW
    # --------------------------------------------------------

    window = features_normalized[
        start:end
    ]


    # --------------------------------------------------------
    # TARGET
    # --------------------------------------------------------

    alpha_q = float(
        row["alpha_q"]
    )

    alpha_r = float(
        row["alpha_r"]
    )


    # --------------------------------------------------------
    # STORE
    # --------------------------------------------------------

    X.append(
        window
    )

    Y.append([
        alpha_q,
        alpha_r
    ])


    # --------------------------------------------------------
    # METADATA
    # --------------------------------------------------------

    metadata.append({

        "window_id": window_id,

        "start_index": start,

        "end_index": end - 1,

        "start_time": float(
            row["time"]
        ),

        "end_time": float(
            row["time"]
            + (WINDOW_SIZE - 1) * 0.1
        ),

        "alpha_q": alpha_q,

        "alpha_r": alpha_r,

        "gps_updates": int(
            row["gps_updates"]
        ),

        "has_gps": int(
            row["has_gps"]
        ),

        "mean_nis": row[
            "mean_nis"
        ],

        "position_mae": row[
            "position_mae"
        ],

        "velocity_mae": row[
            "velocity_mae"
        ],

        "heading_mae_deg": row[
            "heading_mae"
        ]
    })


# ============================================================
# CONVERT TO NUMPY
# ============================================================

X = np.asarray(
    X,
    dtype=np.float32
)

Y = np.asarray(
    Y,
    dtype=np.float32
)

metadata_df = pd.DataFrame(
    metadata
)


# ============================================================
# CHECK SHAPES
# ============================================================

print("\nDataset shapes")
print("-" * 70)

print(
    f"X shape: {X.shape}"
)

print(
    f"Y shape: {Y.shape}"
)

print(
    "Expected X: "
    "(number_of_windows, 50, 16)"
)

print(
    "Expected Y: "
    "(number_of_windows, 2)"
)


# ============================================================
# TRAIN / VALIDATION / TEST SPLIT
# ============================================================

# This is time-series data.
#
# DO NOT randomly shuffle.
#
# Earlier windows -> training
# Middle windows  -> validation
# Later windows   -> testing

n = len(X)

train_end = int(
    0.70 * n
)

val_end = int(
    0.85 * n
)


X_train = X[
    :train_end
]

Y_train = Y[
    :train_end
]


X_val = X[
    train_end:val_end
]

Y_val = Y[
    train_end:val_end
]


X_test = X[
    val_end:
]

Y_test = Y[
    val_end:
]


# ============================================================
# SAVE DATA
# ============================================================

os.makedirs(
    OUTPUT_DIR,
    exist_ok=True
)


np.save(
    OUTPUT_X,
    X
)

np.save(
    OUTPUT_Y,
    Y
)

metadata_df.to_csv(
    OUTPUT_META,
    index=False
)


# ============================================================
# SAVE NORMALIZATION PARAMETERS
# ============================================================

np.save(
    os.path.join(
        OUTPUT_DIR,
        "model2_feature_mean.npy"
    ),
    feature_mean.astype(
        np.float32
    )
)

np.save(
    os.path.join(
        OUTPUT_DIR,
        "model2_feature_std.npy"
    ),
    feature_std.astype(
        np.float32
    )
)


# ============================================================
# SAVE SPLITS
# ============================================================

np.save(
    os.path.join(
        OUTPUT_DIR,
        "model2_X_train.npy"
    ),
    X_train
)

np.save(
    os.path.join(
        OUTPUT_DIR,
        "model2_Y_train.npy"
    ),
    Y_train
)

np.save(
    os.path.join(
        OUTPUT_DIR,
        "model2_X_val.npy"
    ),
    X_val
)

np.save(
    os.path.join(
        OUTPUT_DIR,
        "model2_Y_val.npy"
    ),
    Y_val
)

np.save(
    os.path.join(
        OUTPUT_DIR,
        "model2_X_test.npy"
    ),
    X_test
)

np.save(
    os.path.join(
        OUTPUT_DIR,
        "model2_Y_test.npy"
    ),
    Y_test
)


# ============================================================
# FINAL SUMMARY
# ============================================================

print("\n")
print("=" * 70)
print("MODEL 2 DATASET PREPARATION COMPLETE")
print("=" * 70)

print("\nTotal windows:")
print(n)

print("\nTraining windows:")
print(len(X_train))

print("\nValidation windows:")
print(len(X_val))

print("\nTesting windows:")
print(len(X_test))

print("\nInput shape:")
print(X_train.shape)

print("\nTarget shape:")
print(Y_train.shape)

print("\nFeatures:")
print(len(FEATURE_COLUMNS))

print("\nTargets:")
print("1. alpha_q")
print("2. alpha_r")

print("\nFiles saved:")
print("------------------------------")
print("model2_X.npy")
print("model2_Y.npy")
print("model2_metadata.csv")
print("model2_feature_mean.npy")
print("model2_feature_std.npy")
print("model2_X_train.npy")
print("model2_Y_train.npy")
print("model2_X_val.npy")
print("model2_Y_val.npy")
print("model2_X_test.npy")
print("model2_Y_test.npy")

print("\n")
print("=" * 70)