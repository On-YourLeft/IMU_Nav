from pathlib import Path
import numpy as np
import pandas as pd


# ============================================================
# PROJECT PATHS
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[1]

DATA_DIR = (
    PROJECT_ROOT
    / "data"
    / "IO-VNBD"
    / "Synchronised V abd S datasets"
    / "Categorised IOVNB Dataset"
    / "M (Driver B)"
)

SMARTPHONE_FILE = DATA_DIR / "S-M.csv"
VEHICLE_FILE = DATA_DIR / "V-M.csv"


# ============================================================
# LOAD DATA
# ============================================================

print("=" * 70)
print("LOADING DATA")
print("=" * 70)

smartphone = pd.read_csv(
    SMARTPHONE_FILE,
    encoding="latin1"
)

vehicle = pd.read_csv(
    VEHICLE_FILE,
    encoding="latin1"
)

# Remove accidental whitespace from column names
smartphone.columns = smartphone.columns.str.strip()
vehicle.columns = vehicle.columns.str.strip()
def to_numeric(series):
    return pd.to_numeric(
        series,
        errors="coerce"
    )
print("\nCHECKING NUMERIC CONVERSIONS")
print("-" * 70)

for column in vehicle.columns:
    try:
        vehicle[column].astype(float)
    except ValueError as e:
        print("\nFAILED COLUMN:", column)
        print("ERROR:", e)

print("Smartphone shape:", smartphone.shape)
print("Vehicle shape:   ", vehicle.shape)


# ============================================================
# CHECK SYNCHRONIZATION
# ============================================================

if len(smartphone) != len(vehicle):
    raise ValueError(
        "Smartphone and vehicle datasets do not have the same number of rows."
    )

print("\nRow synchronization:")
print("Rows:", len(smartphone))
print("Synchronized row pairing: YES")


# ============================================================
# CREATE PROCESSING DATAFRAME
# ============================================================

data = pd.DataFrame()

# ------------------------------------------------------------
# Time
# ------------------------------------------------------------

# IMPORTANT:
# We do NOT use smartphone TIME SINCE START because it resets
# at row 44226.
#
# Instead, use the vehicle sample period as the initial
# processing clock.

dt = vehicle["Sample period (seconds)"].astype(float)

# Replace the rare invalid/unusual values with 0.1 s
dt = dt.where(
    dt.between(0.09, 0.11),
    0.1
)

data["dt"] = dt

data["time"] = data["dt"].cumsum()
data["time"] = data["time"] - data["time"].iloc[0]


# ============================================================
# SMARTPHONE IMU
# ============================================================

data["acc_x"] = smartphone[
    "ACCELEROMETER X (m/s²)"
].astype(float)

data["acc_y"] = smartphone[
    "ACCELEROMETER Y (m/s²)"
].astype(float)

data["acc_z"] = smartphone[
    "ACCELEROMETER Z (m/s²)"
].astype(float)


data["gyro_yaw"] = smartphone[
    "GYROSCOPE Yaw (rad/s)"
].astype(float)

data["gyro_pitch"] = smartphone[
    "GYROSCOPE Pitch (rad/s)"
].astype(float)

data["gyro_roll"] = smartphone[
    "GYROSCOPE Roll (rad/s)"
].astype(float)


# ============================================================
# SMARTPHONE GPS
# ============================================================

data["gps_lat"] = smartphone[
    "GPS LATITUDE (degrees)"
].astype(float)

data["gps_lon"] = smartphone[
    "GPS LONGITUDE (degrees)"
].astype(float)

data["gps_alt"] = smartphone[
    "GPS ALTITUDE (m)"
].astype(float)

# km/h → m/s
data["gps_speed"] = (
    smartphone["GPS SPEED (Kmh)"].astype(float) / 3.6
)

data["gps_accuracy"] = smartphone[
    "GPS ACCURACY (m)"
].astype(float)

# GPS satellites may be stored as strings such as:
# "18 / 19"
#
# We use the first number as the number of satellites
# currently being used/available for the navigation estimate.

data["gps_satellites"] = (
    smartphone["GPS SATELLITES IN RANGE"]
    .astype(str)
    .str.extract(r"(\d+)", expand=False)
    .astype(float)
)


# ============================================================
# VEHICLE REFERENCE DATA
# ============================================================

# Helper function:
# Convert a vehicle column to numeric values.
# Invalid strings such as "18 / 19" become NaN and will
# be handled by the cleaning step below.

def numeric_column(column_name):
    return pd.to_numeric(
        vehicle[column_name],
        errors="coerce"
    )


# ------------------------------------------------------------
# Vehicle velocity
# ------------------------------------------------------------

# km/h → m/s
data["ref_speed"] = (
    numeric_column("Velocity (km/hr)") / 3.6
)


# ------------------------------------------------------------
# Vehicle yaw rate
# ------------------------------------------------------------

# deg/s → rad/s
data["ref_yaw_rate"] = np.deg2rad(
    numeric_column("Yaw Rate (deg/sec)")
)


# ------------------------------------------------------------
# Vehicle longitudinal acceleration
# ------------------------------------------------------------

# g → m/s²
data["ref_acc_long"] = (
    numeric_column(
        "Indicated Longitudinal Acceleration (g)"
    ) * 9.80665
)


# ------------------------------------------------------------
# Vehicle lateral acceleration
# ------------------------------------------------------------

# g → m/s²
data["ref_acc_lat"] = (
    numeric_column(
        "Indicated Lateral Acceleration (g)"
    ) * 9.80665
)


# ============================================================
# WHEEL SPEEDS
# ============================================================

data["wheel_fl"] = numeric_column(
    "Wheel Speed Front Left (rad/sec)"
)

data["wheel_fr"] = numeric_column(
    "Wheel Speed Front Right (rad/sec)"
)

data["wheel_rl"] = numeric_column(
    "Wheel Speed Rear Left (rad/sec)"
)

data["wheel_rr"] = numeric_column(
    "Wheel Speed Rear Right (rad/sec)"
)

# ============================================================
# CLEAN INVALID VALUES
# ============================================================

print("\nMissing values before cleaning:")
print(data.isna().sum())

data = data.replace(
    [np.inf, -np.inf],
    np.nan
)

data = data.interpolate(
    method="linear",
    limit_direction="both"
)

print("\nMissing values after cleaning:")
print(data.isna().sum().sum())


# ============================================================
# SAVE PROCESSED DATA
# ============================================================

OUTPUT_DIR = PROJECT_ROOT / "data" / "processed"

OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True
)

OUTPUT_FILE = OUTPUT_DIR / "M_driver_B_processed.csv"

data.to_csv(
    OUTPUT_FILE,
    index=False
)


# ============================================================
# SUMMARY
# ============================================================

print("\n" + "=" * 70)
print("PREPROCESSING COMPLETE")
print("=" * 70)

print("\nProcessed shape:", data.shape)

print("\nColumns:")
for column in data.columns:
    print(" -", column)

print("\nSaved to:")
print(OUTPUT_FILE)

print("\n" + "=" * 70)