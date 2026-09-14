import asyncio
import json
import math
import os
import sys
from collections import deque

import numpy as np
import torch
import websockets
import onnxruntime as ort
import joblib


# ============================================================
# PATHS
# ============================================================

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)

M1 = os.path.join(
    ROOT,
    "IMU_Nav",
    "model1_pipeline",
    "model1_velocity_lean.onnx"
)

M1S = os.path.join(
    ROOT,
    "IMU_Nav",
    "model1_pipeline",
    "normalization_stats.npz"
)

M2 = os.path.join(
    ROOT,
    "models",
    "model2_classifier_best.pth"
)

FMEAN = os.path.join(
    ROOT,
    "data",
    "processed",
    "model2_feature_mean.npy"
)

FSTD = os.path.join(
    ROOT,
    "data",
    "processed",
    "model2_feature_std.npy"
)

ACC = os.path.join(
    ROOT,
    "models",
    "imu_acceleration_model.pkl"
)

YAW = os.path.join(
    ROOT,
    "models",
    "imu_yaw_model.pkl"
)


# ============================================================
# NETWORK
# ============================================================

UDP_PORT = 5555
WS_PORT = 8765


# ============================================================
# MODEL PARAMETERS
# ============================================================

W1 = 20
W2 = 50

M2_EVERY = 50

NIS_THRESHOLD = 25.0


# ============================================================
# BASE COVARIANCES
# ============================================================

Q0 = np.diag([
    0.1,
    0.1,
    0.5,
    0.01,
    0.001
])

R0 = np.diag([
    10.0,
    10.0,
    2.0
])


# ============================================================
# GYRO ALIGNMENT
# ============================================================

GYRO_YAW_COEF = 0.065717
GYRO_PITCH_COEF = 0.628869
GYRO_ROLL_COEF = 0.333411
GYRO_BIAS = -0.011072


# ============================================================
# GLOBAL STATE
# ============================================================

clients = set()

b1 = deque(maxlen=W1)
b2 = deque(maxlen=W2)

vprev = 0.0

aq = 1.0
ar = 1.0

count = 0
m2updates = 0


# ============================================================
# EKF STATE
#
# [x, y, velocity, heading, gyro_bias]
# ============================================================

x = np.zeros(5)

P = np.diag([
    10.0,
    10.0,
    4.0,
    np.deg2rad(30.0) ** 2,
    0.05
])


# ============================================================
# GPS STATE
# ============================================================

origin_lat = None
origin_lon = None

last_lat = None
last_lon = None

gps_speed = 0.0
gps_acc = 10.0

gps_source = "NONE"

last_gps_update = 0

gps_fix_counter = 0


# ============================================================
# GPS / FILTER DIAGNOSTICS
# ============================================================

last_nis = np.nan

last_innov = np.zeros(3)

last_gps_used = 0

last_time = None


# ============================================================
# HELPER
# ============================================================

def finite(v, default=0.0):

    try:

        v = float(v)

        if np.isfinite(v):
            return v

        return default

    except Exception:

        return default


# ============================================================
# ANGLE WRAP
# ============================================================

def wrap(a):

    return (a + np.pi) % (2.0 * np.pi) - np.pi


# ============================================================
# GPS → LOCAL X/Y
# ============================================================

def gpsxy(lat, lon):

    global origin_lat
    global origin_lon

    if lat is None or lon is None:
        return None

    lat = float(lat)
    lon = float(lon)

    if not np.isfinite(lat) or not np.isfinite(lon):
        return None

    if origin_lat is None:

        origin_lat = np.deg2rad(lat)
        origin_lon = np.deg2rad(lon)

    x_local = (
        6371000.0
        *
        (
            np.deg2rad(lon)
            -
            origin_lon
        )
        *
        np.cos(origin_lat)
    )

    y_local = (
        6371000.0
        *
        (
            np.deg2rad(lat)
            -
            origin_lat
        )
    )

    return np.array(
        [x_local, y_local],
        dtype=np.float64
    )


# ============================================================
# STARTUP
# ============================================================

print("=" * 70)

print(
    "SIH 26168 LIVE MODEL 1 + MODEL 2 + ADAPTIVE EKF"
)

print("=" * 70)


# ============================================================
# CHECK FILES
# ============================================================

required_files = [

    M1,
    M1S,
    M2,
    FMEAN,
    FSTD

]

for f in required_files:

    if not os.path.exists(f):

        raise FileNotFoundError(
            f"\nRequired file not found:\n{f}"
        )


# ============================================================
# LOAD MODEL 1
# ============================================================

stats = np.load(M1S)

m1mean = stats["mean"].astype(
    np.float32
)

m1std = np.maximum(
    stats["std"].astype(np.float32),
    1e-7
)


m1 = ort.InferenceSession(

    M1,

    providers=[
        "CPUExecutionProvider"
    ]

)


m1_inputs = [
    i.name
    for i in m1.get_inputs()
]


# ============================================================
# LOAD MODEL 2
# ============================================================

sys.path[:0] = [

    ROOT,
    os.path.join(ROOT, "src"),
    HERE

]


from model2_classification import Model2Classifier


checkpoint = torch.load(

    M2,

    map_location="cpu",

    weights_only=False

)


alpha_grid = np.asarray(

    checkpoint["alpha_grid"],

    dtype=np.float32

)


m2 = Model2Classifier(

    input_features=int(
        checkpoint["input_features"]
    ),

    num_classes=int(
        checkpoint["num_classes"]
    )

)


m2.load_state_dict(

    checkpoint[
        "model_state_dict"
    ]

)


m2.eval()


# ============================================================
# MODEL 2 NORMALIZATION
# ============================================================

fmean = np.load(
    FMEAN
).astype(
    np.float32
)


fstd = np.maximum(

    np.load(
        FSTD
    ).astype(
        np.float32
    ),

    1e-8

)


# ============================================================
# CALIBRATION MODELS
# ============================================================

acc_model = None
yaw_model = None


if os.path.exists(ACC):

    acc_model = joblib.load(
        ACC
    )


if os.path.exists(YAW):

    yaw_model = joblib.load(
        YAW
    )


# ============================================================
# STARTUP INFORMATION
# ============================================================

print()

print("M1:")
print(M1)

print()

print("M2:")
print(M2)

print()

print("WebSocket:")
print(f"ws://localhost:{WS_PORT}")

print()

print("UDP:")
print(f"port {UDP_PORT}")

print()

print("GPS:")
print("Browser/Laptop GPS ENABLED")

print("=" * 70)


# ============================================================
# PHONE UDP PARSER
#
# Expected minimum:
#
# ax,ay,az,gyro_yaw,gyro_pitch,gyro_roll
#
# Optional GPS:
#
# ax,ay,az,gyro_yaw,gyro_pitch,gyro_roll,
# latitude,longitude,gps_speed_kmh,gps_accuracy,satellites
# ============================================================

def parse_udp(line):

    parts = line.split(",")

    if len(parts) < 6:

        return None

    try:

        values = np.array(
            [
                float(z)
                for z in parts
            ],
            dtype=np.float32
        )

    except Exception:

        return None


    result = {

        "imu":
            values[:6],

        "lat":
            None,

        "lon":
            None,

        "gs":
            None,

        "ga":
            None,

        "sat":
            None

    }


    # --------------------------------------------------------
    # Optional phone GPS
    # --------------------------------------------------------

    if len(values) >= 8:

        result["lat"] = float(
            values[6]
        )

        result["lon"] = float(
            values[7]
        )


    if len(values) >= 9:

        result["gs"] = (
            float(values[8])
            / 3.6
        )


    if len(values) >= 10:

        result["ga"] = float(
            values[9]
        )


    if len(values) >= 11:

        result["sat"] = float(
            values[10]
        )


    return result


# ============================================================
# MODEL 1
# ============================================================

def model1(imu):

    global vprev


    b1.append(
        np.asarray(
            imu,
            dtype=np.float32
        )
    )


    if len(b1) < W1:

        return vprev, 0.0


    imu_window = np.asarray(
        b1,
        dtype=np.float32
    )


    z = (

        imu_window
        -
        m1mean

    ) / m1std


    z = z.T.reshape(
        1,
        6,
        W1
    ).astype(
        np.float32
    )


    vp = np.array(
        [vprev],
        dtype=np.float32
    )


    feed = {}


    for name in m1_inputs:

        lname = name.lower()


        if (
            "imu" in lname
            or
            "window" in lname
        ):

            feed[name] = z


        elif (
            "prev" in lname
            or
            "velocity" in lname
        ):

            feed[name] = vp


    if len(feed) != 2:

        feed = {

            m1_inputs[0]: z,

            m1_inputs[1]: vp

        }


    outputs = m1.run(
        None,
        feed
    )


    velocity = float(

        np.clip(

            np.asarray(
                outputs[0]
            ).reshape(-1)[0],

            0.0,

            60.0

        )

    )


    lean = float(

        np.clip(

            np.degrees(

                np.asarray(
                    outputs[2]
                ).reshape(-1)[0]

            ),

            -60.0,

            60.0

        )

    )


    vprev = velocity

    b1.popleft()


    return velocity, lean


# ============================================================
# IMU CALIBRATION
# ============================================================

def calibrate_imu(imu):

    z = imu.reshape(
        1,
        -1
    )


    # --------------------------------------------------------
    # Acceleration
    # --------------------------------------------------------

    try:

        if acc_model is not None:

            acceleration = float(

                acc_model.predict(
                    z
                ).reshape(-1)[0]

            )

        else:

            acceleration = float(
                imu[0]
            )

    except Exception:

        acceleration = float(
            imu[0]
        )


    # --------------------------------------------------------
    # Yaw rate
    # --------------------------------------------------------

    try:

        if yaw_model is not None:

            yaw_rate = float(

                yaw_model.predict(
                    z
                ).reshape(-1)[0]

            )

        else:

            yaw_rate = (

                GYRO_YAW_COEF
                * imu[3]

                +

                GYRO_PITCH_COEF
                * imu[4]

                +

                GYRO_ROLL_COEF
                * imu[5]

                +

                GYRO_BIAS

            )

    except Exception:

        yaw_rate = (

            GYRO_YAW_COEF
            * imu[3]

            +

            GYRO_PITCH_COEF
            * imu[4]

            +

            GYRO_ROLL_COEF
            * imu[5]

            +

            GYRO_BIAS

        )


    acceleration = float(

        np.clip(
            acceleration,
            -30.0,
            30.0
        )

    )


    yaw_rate = float(

        np.clip(
            yaw_rate,
            -8.0,
            8.0
        )

    )


    return acceleration, yaw_rate


# ============================================================
# MODEL 2
# ============================================================

def model2(features):

    global aq
    global ar
    global m2updates


    b2.append(

        np.asarray(
            features,
            dtype=np.float32
        )

    )


    if len(b2) < W2:

        return


    if count % M2_EVERY != 0:

        return


    window = np.asarray(
        b2,
        dtype=np.float32
    )


    # --------------------------------------------------------
    # Replace invalid values
    # --------------------------------------------------------

    window = np.nan_to_num(
        window,
        nan=0.0,
        posinf=0.0,
        neginf=0.0
    )


    # --------------------------------------------------------
    # Training normalization
    # --------------------------------------------------------

    z = (

        window
        -
        fmean

    ) / fstd


    tensor = torch.from_numpy(
        z
    ).unsqueeze(
        0
    )


    # --------------------------------------------------------
    # MODEL 2 INFERENCE
    # --------------------------------------------------------

    with torch.no_grad():

        q_logits, r_logits = m2(
            tensor
        )


        q_prob = torch.softmax(
                q_logits,
                dim=1
            )


        r_prob = torch.softmax(
                r_logits,
                dim=1
            )


        log_grid = torch.log(
                torch.tensor(
                    alpha_grid,
                    dtype=torch.float32
                )
            )


        aq_value = torch.exp(

            (
                q_prob
                *
                log_grid
            ).sum(
                dim=1
            )

        ).item()


        ar_value = torch.exp(

            (
                r_prob
                *
                log_grid
            ).sum(
                dim=1
            )

        ).item()


    aq = float(

        np.clip(
            aq_value,
            0.05,
            30.0
        )

    )


    ar = float(

        np.clip(
            ar_value,
            0.05,
            30.0
        )

    )


    m2updates += 1


# ============================================================
# EKF
# ============================================================

def ekf(
    dt,
    velocity,
    yaw_rate,
    gxy,
    gps_velocity,
    gps_accuracy
):

    global x
    global P
    global last_nis
    global last_innov
    global last_gps_used


    dt = float(

        np.clip(
            dt,
            0.005,
            1.0
        )

    )


    velocity = max(
        0.0,
        float(velocity)
    )


    # ========================================================
    # PREDICTION
    # ========================================================

    heading = wrap(

        x[3]
        +
        (
            yaw_rate
            -
            x[4]
        )
        *
        dt

    )


    speed = max(
        0.0,
        velocity
    )


    xp = np.array(

        [

            x[0]
            +
            speed
            *
            np.sin(heading)
            *
            dt,

            x[1]
            +
            speed
            *
            np.cos(heading)
            *
            dt,

            speed,

            heading,

            x[4]

        ]

    )


    # ========================================================
    # STATE JACOBIAN
    # ========================================================

    F = np.eye(
        5
    )


    F[0, 2] = np.sin(heading) * dt


    F[0, 3] = speed * np.cos(heading) * dt


    F[1, 2] = np.cos(heading) * dt


    F[1, 3] = -speed * np.sin(heading) * dt


    F[3, 4] = -dt


    # ========================================================
    # COVARIANCE PREDICTION
    # ========================================================

    P = (

        F
        @ P
        @ F.T

        +

        aq
        *
        Q0

    )


    x = xp


    P = (
        P + P.T
    ) / 2.0


    # ========================================================
    # DEFAULT GPS STATUS
    # ========================================================

    last_nis = np.nan

    last_innov[:] = 0.0

    last_gps_used = 0


    # ========================================================
    # NO GPS
    # ========================================================

    if gxy is None:

        return


    # ========================================================
    # GPS MEASUREMENT
    #
    # z = [x, y, speed]
    # ========================================================

    gps_velocity = max(

        0.0,

        float(
            gps_velocity
        )

    )


    z = np.array(

        [

            gxy[0],

            gxy[1],

            gps_velocity

        ]

    )


    # ========================================================
    # MEASUREMENT MATRIX
    # ========================================================

    H = np.zeros(
        (3, 5)
    )


    H[0, 0] = 1.0

    H[1, 1] = 1.0

    H[2, 2] = 1.0


    # ========================================================
    # GPS ACCURACY → R
    # ========================================================

    accuracy = float(

        np.clip(

            finite(
                gps_accuracy,
                10.0
            ),

            1.0,

            100.0

        )

    )


    R = (
        ar
        *
        R0
    ).copy()


    # Position measurement variance

    R[0, 0] = max(

        R[0, 0],

        accuracy ** 2

    )


    R[1, 1] = max(

        R[1, 1],

        accuracy ** 2

    )


    # ========================================================
    # INNOVATION
    # ========================================================

    innovation = (

        z
        -
        H @ x

    )


    last_innov = innovation.copy()


    # ========================================================
    # INNOVATION COVARIANCE
    # ========================================================

    S = (

        H
        @ P
        @ H.T

        +

        R

    )


    try:

        S_inv = np.linalg.inv(
            S
        )

    except Exception:

        S_inv = np.linalg.pinv(
            S
        )


    # ========================================================
    # NIS
    # ========================================================

    last_nis = float(

        innovation
        @
        S_inv
        @
        innovation

    )


    # ========================================================
    # GATE GPS
    # ========================================================

    if (

        np.isfinite(last_nis)

        and

        last_nis <= NIS_THRESHOLD

    ):


        # ----------------------------------------------------
        # Kalman gain
        # ----------------------------------------------------

        K = (

            P
            @
            H.T
            @
            S_inv

        )


        # ----------------------------------------------------
        # State update
        # ----------------------------------------------------

        x = (

            x
            +
            K @ innovation

        )


        x[3] = wrap(
                x[3]
            )


        # ----------------------------------------------------
        # Joseph covariance update
        # ----------------------------------------------------

        I = np.eye(
            5
        )


        A = (
            I
            -
            K @ H
        )


        P = (

            A
            @ P
            @ A.T

            +

            K
            @ R
            @ K.T

        )


        P = (
            P + P.T
        ) / 2.0


        last_gps_used = 1


# ============================================================
# GPS → LAT/LON FROM EKF LOCAL POSITION
# ============================================================

def local_to_gps(local_xy):

    if (
        origin_lat is None
        or
        origin_lon is None
        or
        local_xy is None
    ):

        return None, None


    lat = (

        np.degrees(
            origin_lat
        )

        +

        np.degrees(
            local_xy[1]
            /
            6371000.0
        )

    )


    lon = (

        np.degrees(
            origin_lon
        )

        +

        np.degrees(

            local_xy[0]
            /
            (
                6371000.0
                *
                np.cos(
                    origin_lat
                )
            )

        )

    )


    return float(lat), float(lon)


# ============================================================
# DASHBOARD PAYLOAD
# ============================================================

def make_payload(
    model1_speed,
    lean,
    gps_update
):


    ekf_lat = None
    ekf_lon = None


    if origin_lat is not None:

        ekf_lat, ekf_lon = local_to_gps(

            [
                x[0],
                x[1]
            ]

        )


    return {

        "type":
            "telemetry",


        # ----------------------------------------------------
        # Model 1
        # ----------------------------------------------------

        "speed_kmh":
            round(
                model1_speed * 3.6,
                2
            ),


        "lean_deg":
            round(
                lean,
                2
            ),


        # ----------------------------------------------------
        # EKF
        # ----------------------------------------------------

        "estimated_speed_kmh":
            round(
                x[2] * 3.6,
                2
            ),


        "heading_deg":
            round(
                np.degrees(
                    x[3]
                ) % 360.0,
                2
            ),


        "x_m":
            round(
                x[0],
                2
            ),


        "y_m":
            round(
                x[1],
                2
            ),


        # ----------------------------------------------------
        # EKF GPS position
        # ----------------------------------------------------

        "latitude":
            ekf_lat,


        "longitude":
            ekf_lon,


        # ----------------------------------------------------
        # Raw GPS
        # ----------------------------------------------------

        "gps_latitude":
            last_lat,


        "gps_longitude":
            last_lon,


        "gps_speed_kmh":
            round(
                gps_speed * 3.6,
                2
            ),


        "gps_accuracy_m":
            round(
                gps_acc,
                2
            ),


        # ----------------------------------------------------
        # GPS source
        # ----------------------------------------------------

        "gps_source":
            gps_source,


        # ----------------------------------------------------
        # GPS update
        # ----------------------------------------------------

        "gps_update":
            int(
                gps_update
            ),


        "gps_used":
            int(
                last_gps_used
            ),


        # ----------------------------------------------------
        # Model 2
        # ----------------------------------------------------

        "alpha_q":
            round(
                aq,
                4
            ),


        "alpha_r":
            round(
                ar,
                4
            ),


        "model2_updates":
            m2updates,


        # ----------------------------------------------------
        # NIS
        # ----------------------------------------------------

        "nis":

            None

            if not np.isfinite(
                last_nis
            )

            else round(
                last_nis,
                3
            ),


        # ----------------------------------------------------
        # Innovation
        # ----------------------------------------------------

        "innovation_x":
            round(
                last_innov[0],
                2
            ),


        "innovation_y":
            round(
                last_innov[1],
                2
            ),


        "innovation_speed":
            round(
                last_innov[2],
                2
            ),


        # ----------------------------------------------------
        # Samples
        # ----------------------------------------------------

        "samples":
            count

    }


# ============================================================
# BROWSER GPS HANDLER
#
# THIS IS THE IMPORTANT FIX.
#
# dashboard.html sends:
#
# {
#   type: "browser_gps",
#   latitude: ...,
#   longitude: ...,
#   accuracy: ...,
#   speed: ...
# }
#
# We now actually consume it.
# ============================================================

async def handle_browser_gps(message):

    global last_lat
    global last_lon
    global gps_speed
    global gps_acc
    global gps_source
    global last_gps_update
    global gps_fix_counter


    try:

        lat = float(
            message.get(
                "latitude"
            )
        )

        lon = float(
            message.get(
                "longitude"
            )
        )

        accuracy = float(
            message.get(
                "accuracy",
                10.0
            )
        )

        speed = float(
            message.get(
                "speed",
                0.0
            )
        )

    except Exception:

        return


    if not (
        np.isfinite(lat)
        and
        np.isfinite(lon)
    ):

        return


    # --------------------------------------------------------
    # Sanity check
    # --------------------------------------------------------

    if not (
        -90.0 <= lat <= 90.0
        and
        -180.0 <= lon <= 180.0
    ):

        return


    # --------------------------------------------------------
    # Determine whether this is a new GPS fix
    # --------------------------------------------------------

    changed = (

        last_lat is None

        or

        abs(
            lat - last_lat
        ) > 1e-8

        or

        abs(
            lon - last_lon
        ) > 1e-8

    )


    if changed:

        last_gps_update = 1

        gps_fix_counter += 1

    else:

        last_gps_update = 0


    # --------------------------------------------------------
    # Save GPS
    # --------------------------------------------------------

    last_lat = lat

    last_lon = lon


    gps_acc = float(

        np.clip(
            accuracy,
            1.0,
            100.0
        )

    )


    gps_speed = max(
        0.0,
        speed
    )


    gps_source = "LAPTOP"


    # --------------------------------------------------------
    # Print GPS information
    # --------------------------------------------------------

    print(

        "\n[GPS] LAPTOP FIX",

        f"| Lat: {lat:.6f}",

        f"| Lon: {lon:.6f}",

        f"| Accuracy: {gps_acc:.1f} m",

        f"| Speed: {gps_speed * 3.6:.2f} km/h",

        f"| New: {changed}"

    )


# ============================================================
# WEBSOCKET HANDLER
# ============================================================

async def ws_handler(websocket):

    clients.add(
        websocket
    )


    print(
        "\n[WS] Dashboard connected."
    )


    try:

        async for message in websocket:


            # ------------------------------------------------
            # Dashboard GPS
            # ------------------------------------------------

            try:

                data = json.loads(
                    message
                )

            except Exception:

                continue


            if (
                isinstance(data, dict)
                and
                data.get("type")
                ==
                "browser_gps"
            ):

                await handle_browser_gps(
                    data
                )


    except Exception as e:

        print(
            "\n[WS] Client error:",
            repr(e)
        )


    finally:

        clients.discard(
            websocket
        )


        print(
            "\n[WS] Dashboard disconnected."
        )


# ============================================================
# BROADCAST
# ============================================================

async def broadcast(payload):

    if not clients:

        return


    message = json.dumps(
        payload
    )


    disconnected = []


    for client in list(
        clients
    ):

        try:

            await client.send(
                message
            )

        except Exception:

            disconnected.append(
                client
            )


    for client in disconnected:

        clients.discard(
            client
        )


# ============================================================
# PROCESS ONE IMU SAMPLE
# ============================================================

async def process_imu(
    imu,
    phone_gps=None
):

    global count
    global last_time
    global last_lat
    global last_lon
    global gps_speed
    global gps_acc
    global gps_source
    global last_gps_update


    # ========================================================
    # SAMPLE COUNT
    # ========================================================

    count += 1


    # ========================================================
    # TIME STEP
    # ========================================================

    now = asyncio.get_running_loop().time()


    if last_time is None:

        dt = 0.1

    else:

        dt = np.clip(

            now - last_time,

            0.005,

            1.0

        )


    last_time = now


    # ========================================================
    # PHONE GPS
    # ========================================================

    gps_update = 0


    if phone_gps is not None:

        lat = phone_gps.get(
            "lat"
        )

        lon = phone_gps.get(
            "lon"
        )


        if (
            lat is not None
            and
            lon is not None
            and
            np.isfinite(lat)
            and
            np.isfinite(lon)
        ):


            changed = (

                last_lat is None

                or

                abs(
                    lat - last_lat
                ) > 1e-8

                or

                abs(
                    lon - last_lon
                ) > 1e-8

            )


            last_lat = lat
            last_lon = lon


            if changed:

                gps_update = 1


            if phone_gps.get(
                "gs"
            ) is not None:

                gps_speed = max(

                    0.0,

                    float(
                        phone_gps["gs"]
                    )

                )


            if phone_gps.get(
                "ga"
            ) is not None:

                gps_acc = float(

                    np.clip(

                        phone_gps["ga"],

                        1.0,

                        100.0

                    )

                )


            gps_source = "PHONE"


    # ========================================================
    # GPS LOCAL POSITION
    # ========================================================

    gxy = gpsxy(
        last_lat,
        last_lon
    )


    # ========================================================
    # MODEL 1
    # ========================================================

    model1_speed, lean = model1(
        imu
    )


    # ========================================================
    # CALIBRATED IMU
    # ========================================================

    predicted_acceleration, predicted_yaw_rate = (
        calibrate_imu(
            imu
        )
    )


    # ========================================================
    # LIVE INNOVATION FOR MODEL 2
    #
    # IMPORTANT:
    # This uses actual GPS now.
    # ========================================================

    innovation = np.zeros(
        3,
        dtype=np.float32
    )


    if gxy is not None:

        innovation[0] = (
            gxy[0]
            -
            x[0]
        )


        innovation[1] = (
            gxy[1]
            -
            x[1]
        )


        innovation[2] = (
            gps_speed
            -
            x[2]
        )


    # ========================================================
    # APPROXIMATE NIS FEATURE
    #
    # Used as live Model 2 input.
    # ========================================================

    if gxy is not None:

        position_variance = max(
            gps_acc ** 2,
            1.0
        )


        nis0 = (

            (
                innovation[0]
                /
                gps_acc
            )
            ** 2

            +

            (
                innovation[1]
                /
                gps_acc
            )
            ** 2

            +

            (
                innovation[2]
                /
                2.0
            )
            ** 2

        )

    else:

        nis0 = 0.0


    # ========================================================
    # MODEL 2 FEATURE VECTOR
    #
    # EXACT 16 FEATURES
    #
    # 0-2   raw acceleration
    # 3-5   raw gyro
    # 6     calibrated acceleration
    # 7     calibrated yaw
    # 8-10  innovation x/y/speed
    # 11    log NIS
    # 12    GPS update
    # 13    GPS accuracy
    # 14    satellites placeholder = 0
    # 15    GPS speed
    # ========================================================

    model2_features = [

        float(imu[0]),
        float(imu[1]),
        float(imu[2]),

        float(imu[3]),
        float(imu[4]),
        float(imu[5]),

        float(
            predicted_acceleration
        ),

        float(
            predicted_yaw_rate
        ),

        float(
            innovation[0]
        ),

        float(
            innovation[1]
        ),

        float(
            innovation[2]
        ),

        float(
            math.log1p(
                max(
                    nis0,
                    0.0
                )
            )
        ),

        float(
            gps_update
        ),

        float(
            gps_acc
        ),

        0.0,

        float(
            gps_speed
        )

    ]


    # ========================================================
    # MODEL 2
    # ========================================================

    model2(
        model2_features
    )


    # ========================================================
    # EKF
    #
    # Apply GPS only when there is a NEW fix.
    # ========================================================

    ekf(

        dt,

        model1_speed,

        predicted_yaw_rate,

        gxy
        if gps_update
        else None,

        gps_speed,

        gps_acc

    )


    # ========================================================
    # RESET ONE-SAMPLE GPS UPDATE FLAG
    # ========================================================

    browser_update = (
        last_gps_update
    )


    last_gps_update = 0


    # ========================================================
    # PAYLOAD
    # ========================================================

    payload = make_payload(

        model1_speed,

        lean,

        gps_update
        or
        browser_update

    )


    # ========================================================
    # DASHBOARD
    # ========================================================

    await broadcast(
        payload
    )


    # ========================================================
    # TERMINAL OUTPUT
    # ========================================================

    gps_text = (

        "USED"
        if last_gps_used
        else
        "FIX"
        if (
            gps_source != "NONE"
            and
            gxy is not None
        )
        else
        "--"

    )


    print(

        f"\r"
        f"Speed {model1_speed * 3.6:5.1f} km/h"
        f" | Lean {lean:5.1f}°"
        f" | Heading {np.degrees(x[3]) % 360:5.1f}°"
        f" | Q {aq:5.2f}"
        f" | R {ar:5.2f}"
        f" | NIS {last_nis if np.isfinite(last_nis) else 0:7.1f}"
        f" | GPS {gps_text}"
        f" | {gps_source}",

        end="",

        flush=True

    )


# ============================================================
# UDP PROTOCOL
# ============================================================

class UDPProtocol(
    asyncio.DatagramProtocol
):


    def connection_made(
        self,
        transport
    ):

        self.transport = transport


        print(
            "\n[UDP] Listening on port",
            UDP_PORT
        )


    def datagram_received(
        self,
        data,
        addr
    ):

        try:

            line = data.decode(
                "utf-8",
                errors="ignore"
            ).strip()


            packet = parse_udp(
                line
            )


            if packet is None:

                return


            imu = packet[
                "imu"
            ]


            if not np.all(
                np.isfinite(imu)
            ):

                return


            # ------------------------------------------------
            # Run async processing
            # ------------------------------------------------

            asyncio.create_task(

                process_imu(

                    imu,

                    packet

                )

            )


        except Exception as e:

            print(

                "\n[UDP ERROR]",

                repr(e)

            )


# ============================================================
# MAIN
# ============================================================

async def main():


    # ========================================================
    # WEBSOCKET SERVER
    # ========================================================

    await websockets.serve(

        ws_handler,

        "0.0.0.0",

        WS_PORT

    )


    print(
        f"\n[WS] ws://localhost:{WS_PORT}"
    )


    # ========================================================
    # UDP SERVER
    # ========================================================

    loop = asyncio.get_running_loop()


    transport, protocol = (
        await loop.create_datagram_endpoint(

            lambda:
                UDPProtocol(),

            local_addr=(
                "0.0.0.0",
                UDP_PORT
            )

        )
    )


    print(
        f"[UDP] port {UDP_PORT}"
    )


    print(
        "[GPS] Waiting for browser/laptop GPS..."
    )


    print(
        "[GPS] Phone GPS is also supported."
    )


    print(
        "\nREADY.\n"
    )


    try:

        await asyncio.Future()

    finally:

        transport.close()


# ============================================================
# RUN
# ============================================================

if __name__ == "__main__":

    try:

        asyncio.run(
            main()
        )

    except KeyboardInterrupt:

        print(
            "\n\nStopped."
        )