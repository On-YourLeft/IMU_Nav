import joblib
import pandas as pd
import numpy as np


# ============================================================
# PATHS
# ============================================================

DATA_PATH = "data/processed/M_driver_B_processed.csv"

ACC_MODEL_PATH = "models/imu_acceleration_model.pkl"
YAW_MODEL_PATH = "models/imu_yaw_model.pkl"

OUTPUT_PATH = "data/processed/M_driver_B_innovation.csv"


# ============================================================
# LOAD DATA AND MODELS
# ============================================================

df = pd.read_csv(DATA_PATH)

acc_model = joblib.load(ACC_MODEL_PATH)
yaw_model = joblib.load(YAW_MODEL_PATH)

n = len(df)


# ============================================================
# GPS → LOCAL EAST/NORTH COORDINATES
# ============================================================

lat = df["gps_lat"].values
lon = df["gps_lon"].values

lat0 = lat[0]
lon0 = lon[0]

EARTH_RADIUS = 6371000.0

gps_x = (
    np.deg2rad(lon - lon0)
    * EARTH_RADIUS
    * np.cos(np.deg2rad(lat0))
)

gps_y = (
    np.deg2rad(lat - lat0)
    * EARTH_RADIUS
)


# ============================================================
# SENSOR DATA
# ============================================================

gps_speed = df["gps_speed"].values
gps_accuracy = df["gps_accuracy"].values
gps_satellites = df["gps_satellites"].values

acc_x = df["acc_x"].values
acc_y = df["acc_y"].values
acc_z = df["acc_z"].values

gyro_yaw = df["gyro_yaw"].values
gyro_pitch = df["gyro_pitch"].values
gyro_roll = df["gyro_roll"].values

dt = df["dt"].values


# ============================================================
# DETECT NEW GPS FIX
#
# GPS values are held for ~9 seconds in this dataset.
# We therefore update the filter only when a new GPS
# measurement actually appears.
# ============================================================

new_gps = np.zeros(n, dtype=bool)

new_gps[0] = True

new_gps[1:] = (
    (np.abs(np.diff(lat)) > 1e-10)
    |
    (np.abs(np.diff(lon)) > 1e-10)
    |
    (np.abs(np.diff(gps_speed)) > 1e-6)
)


# ============================================================
# ANGLE WRAPPING
# ============================================================

def wrap_angle(angle):

    return (
        (angle + np.pi)
        % (2.0 * np.pi)
    ) - np.pi


# ============================================================
# INITIAL HEADING
#
# Find first sufficiently large GPS displacement.
# Navigation convention:
# 0° = North
# 90° = East
# ============================================================

initial_heading = 0.0

for k in range(1, n):

    dx = gps_x[k] - gps_x[0]
    dy = gps_y[k] - gps_y[0]

    distance = np.sqrt(
        dx ** 2 + dy ** 2
    )

    if distance > 5.0:

        initial_heading = np.arctan2(
            dx,
            dy
        )

        break


# ============================================================
# STATE
#
# x =
# [position_x,
#  position_y,
#  velocity,
#  heading]
# ============================================================

x = np.array([
    gps_x[0],
    gps_y[0],
    gps_speed[0],
    initial_heading
], dtype=float)


# ============================================================
# INITIAL COVARIANCE
# ============================================================

P = np.diag([
    25.0,
    25.0,
    9.0,
    np.deg2rad(20.0) ** 2
])


# ============================================================
# BASE PROCESS NOISE
# ============================================================

Q = np.diag([
    0.05,
    0.05,
    0.5,
    0.01
])


# ============================================================
# STORAGE
# ============================================================

predicted_x = np.zeros(n)
predicted_y = np.zeros(n)

predicted_velocity = np.zeros(n)
predicted_heading = np.zeros(n)

innovation_x = np.full(n, np.nan)
innovation_y = np.full(n, np.nan)
innovation_speed = np.full(n, np.nan)

nis = np.full(n, np.nan)

gps_update_flags = np.zeros(n, dtype=bool)

# Store model predictions too
predicted_acceleration = np.zeros(n)
predicted_yaw_rate = np.zeros(n)


# ============================================================
# EKF LOOP
# ============================================================

for k in range(n):

    dtk = dt[k]

    # ========================================================
    # 1. CREATE IMU FEATURE VECTOR
    # ========================================================

    imu_features = np.array([[
        acc_x[k],
        acc_y[k],
        acc_z[k],
        gyro_yaw[k],
        gyro_pitch[k],
        gyro_roll[k]
    ]])


    # ========================================================
    # 2. ML SENSOR CALIBRATION
    # ========================================================

    acceleration = acc_model.predict(
        imu_features
    )[0]

    yaw_rate = yaw_model.predict(
        imu_features
    )[0]

    predicted_acceleration[k] = acceleration
    predicted_yaw_rate[k] = yaw_rate


    # ========================================================
    # 3. PREDICTION
    # ========================================================

    theta = x[3]
    velocity = x[2]

    # Position

    x[0] += (
        velocity
        * np.sin(theta)
        * dtk
    )

    x[1] += (
        velocity
        * np.cos(theta)
        * dtk
    )

    # Velocity

    x[2] += (
        acceleration
        * dtk
    )

    # Heading

    x[3] = wrap_angle(
        x[3]
        +
        yaw_rate
        * dtk
    )


    # ========================================================
    # 4. STATE TRANSITION JACOBIAN
    # ========================================================

    F = np.eye(4)

    F[0, 2] = (
        np.sin(theta)
        * dtk
    )

    F[0, 3] = (
        velocity
        * np.cos(theta)
        * dtk
    )

    F[1, 2] = (
        np.cos(theta)
        * dtk
    )

    F[1, 3] = (
        -velocity
        * np.sin(theta)
        * dtk
    )


    # ========================================================
    # 5. COVARIANCE PREDICTION
    # ========================================================

    P = (
        F
        @ P
        @ F.T
        +
        Q * dtk
    )


    # ========================================================
    # STORE PREDICTION
    # ========================================================

    predicted_x[k] = x[0]
    predicted_y[k] = x[1]

    predicted_velocity[k] = x[2]
    predicted_heading[k] = x[3]


    # ========================================================
    # 6. GPS UPDATE
    # ========================================================

    if new_gps[k]:

        gps_update_flags[k] = True

        # ----------------------------------------------------
        # GPS measurement
        # ----------------------------------------------------

        z = np.array([
            gps_x[k],
            gps_y[k],
            gps_speed[k]
        ])


        # ----------------------------------------------------
        # Measurement matrix
        # ----------------------------------------------------

        H = np.zeros((3, 4))

        H[0, 0] = 1.0
        H[1, 1] = 1.0
        H[2, 2] = 1.0


        # ----------------------------------------------------
        # GPS measurement uncertainty
        # ----------------------------------------------------

        accuracy = gps_accuracy[k]

        if not np.isfinite(accuracy):

            accuracy = 10.0


        position_variance = max(
            accuracy ** 2,
            4.0
        )

        speed_variance = 4.0


        R = np.diag([
            position_variance,
            position_variance,
            speed_variance
        ])


        # ----------------------------------------------------
        # Innovation
        # ----------------------------------------------------

        innovation = (
            z
            -
            H @ x
        )


        innovation_x[k] = innovation[0]
        innovation_y[k] = innovation[1]
        innovation_speed[k] = innovation[2]


        # ----------------------------------------------------
        # Innovation covariance
        # ----------------------------------------------------

        S = (
            H
            @ P
            @ H.T
            +
            R
        )


        # ----------------------------------------------------
        # NIS
        # ----------------------------------------------------

        try:

            nis[k] = (
                innovation.T
                @ np.linalg.solve(
                    S,
                    innovation
                )
            )

        except np.linalg.LinAlgError:

            nis[k] = np.nan


        # ----------------------------------------------------
        # Kalman update
        # ----------------------------------------------------

        try:

            K = (
                P
                @ H.T
                @ np.linalg.inv(S)
            )

            x = (
                x
                +
                K @ innovation
            )

            x[3] = wrap_angle(
                x[3]
            )


            # ------------------------------------------------
            # Joseph-form covariance update
            # ------------------------------------------------

            I = np.eye(4)

            P = (
                (I - K @ H)
                @ P
                @ (I - K @ H).T
                +
                K
                @ R
                @ K.T
            )

        except np.linalg.LinAlgError:

            pass


# ============================================================
# SAVE RESULTS
# ============================================================

output = pd.DataFrame({

    "time":
        df["time"].values,

    # --------------------------------------------------------
    # EKF prediction
    # --------------------------------------------------------

    "predicted_x":
        predicted_x,

    "predicted_y":
        predicted_y,

    "predicted_velocity":
        predicted_velocity,

    "predicted_heading_deg":
        np.rad2deg(
            predicted_heading
        ),

    # --------------------------------------------------------
    # Innovation
    # --------------------------------------------------------

    "innovation_x":
        innovation_x,

    "innovation_y":
        innovation_y,

    "innovation_speed":
        innovation_speed,

    "nis":
        nis,

    # --------------------------------------------------------
    # GPS information
    # --------------------------------------------------------

    "gps_update":
        gps_update_flags,

    "gps_accuracy":
        gps_accuracy,

    "gps_satellites":
        gps_satellites,

    "gps_speed":
        gps_speed,

    # --------------------------------------------------------
    # IMU
    # --------------------------------------------------------

    "acc_x":
        acc_x,

    "acc_y":
        acc_y,

    "acc_z":
        acc_z,

    "gyro_yaw":
        gyro_yaw,

    "gyro_pitch":
        gyro_pitch,

    "gyro_roll":
        gyro_roll,

    # --------------------------------------------------------
    # ML sensor predictions
    # --------------------------------------------------------

    "predicted_acceleration":
        predicted_acceleration,

    "predicted_yaw_rate":
        predicted_yaw_rate
})


output.to_csv(
    OUTPUT_PATH,
    index=False
)


# ============================================================
# ANALYSIS
# ============================================================

valid_nis = nis[
    np.isfinite(nis)
]


# ============================================================
# PRINT RESULTS
# ============================================================

print("=" * 70)
print("FINAL INNOVATION ANALYSIS")
print("=" * 70)


print()
print("GPS")
print("-" * 70)

print(
    f"Total samples : {n}"
)

print(
    f"GPS updates   : "
    f"{gps_update_flags.sum()}"
)

print(
    f"GPS update %  : "
    f"{100 * gps_update_flags.mean():.2f}%"
)


print()
print("INNOVATION")
print("-" * 70)

print(
    f"X mean abs     : "
    f"{np.nanmean(np.abs(innovation_x)):.3f} m"
)

print(
    f"Y mean abs     : "
    f"{np.nanmean(np.abs(innovation_y)):.3f} m"
)

print(
    f"Speed mean abs : "
    f"{np.nanmean(np.abs(innovation_speed)):.3f} m/s"
)


print()
print("NIS")
print("-" * 70)

if len(valid_nis) > 0:

    print(
        f"Mean   : "
        f"{np.mean(valid_nis):.3f}"
    )

    print(
        f"Median : "
        f"{np.median(valid_nis):.3f}"
    )

    print(
        f"Maximum: "
        f"{np.max(valid_nis):.3f}"
    )


    print()
    print("NIS PERCENTILES")
    print("-" * 70)

    for p in [50, 75, 90, 95, 99]:

        print(
            f"{p:2d}th percentile : "
            f"{np.percentile(valid_nis, p):.3f}"
        )


print()
print("OUTPUT")
print("-" * 70)

print(
    f"Saved to: {OUTPUT_PATH}"
)


print()
print("=" * 70)
print("FINAL INNOVATION ANALYSIS COMPLETE")
print("=" * 70)