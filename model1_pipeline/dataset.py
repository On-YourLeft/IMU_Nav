import os
import re
import glob
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset


# ---------------------------------------------------------
# 1. Directory Crawler & Pair Matching
# ---------------------------------------------------------
def discover_iovnbd_pairs(base_dir):
    """
    Crawls the IO-VNBD structure to find matching Smartphone (S-) and Vehicle (V-) files.
    Returns a list of dictionaries containing paths and driver metadata.
    """
    categorised_path = os.path.join(base_dir, "Categorised IOVNB Dataset")
    uncategorised_path = os.path.join(base_dir, "Uncategorised IOVNB Dataset")

    pairs = []

    # Parse Categorised Folders (Organized by Driver folders like "M (Driver B)")
    if os.path.exists(categorised_path):
        for driver_dir in os.listdir(categorised_path):
            driver_full = os.path.join(categorised_path, driver_dir)
            if not os.path.isdir(driver_full):
                continue

            # Find all S- files (can be .txt or .csv)
            s_files = glob.glob(
                os.path.join(driver_full, "**", "S-*.txt"), recursive=True
            ) + glob.glob(os.path.join(driver_full, "**", "S-*.csv"), recursive=True)

            for s_path in s_files:
                # Corresponding vehicle file swaps prefix 'S-' with 'V-'
                dirname, fname = os.path.split(s_path)
                v_fname = re.sub(r"^S-", "V-", fname)
                v_path = os.path.join(dirname, v_fname)

                if os.path.exists(v_path):
                    pairs.append(
                        {
                            "driver": driver_dir,
                            "category": "categorised",
                            "s_path": s_path,
                            "v_path": v_path,
                        }
                    )

    # Parse Uncategorised Folders (Continuous tracks for testing)
    if os.path.exists(uncategorised_path):
        s_folder = os.path.join(uncategorised_path, "S-Dataset")
        v_folder = os.path.join(uncategorised_path, "V-Dataset")

        if os.path.exists(s_folder) and os.path.exists(v_folder):
            s_files = glob.glob(os.path.join(s_folder, "*.txt")) + glob.glob(
                os.path.join(s_folder, "*.csv")
            )
            for s_path in s_files:
                fname = os.path.basename(s_path)
                v_fname = re.sub(r"^S-", "V-", fname)
                v_path = os.path.join(v_folder, v_fname)

                if os.path.exists(v_path):
                    pairs.append(
                        {
                            "driver": "Uncategorised",
                            "category": "continuous",
                            "s_path": s_path,
                            "v_path": v_path,
                        }
                    )

    print(f"[DATASET] Discovered {len(pairs)} synchronized S-V pairs.")
    return pairs


# ---------------------------------------------------------
# 2. Driver-Aware Splitting (Prevents Data Leakage)
# ---------------------------------------------------------
def split_by_driver(pairs):
    """
    Separates the dataset strictly by the driver/session to ensure the CNN-GRU
    learns general physics, not a specific person's driving habits.
    """
    train_pairs, val_pairs, test_pairs = [], [], []

    for p in pairs:
        driver = p["driver"]
        # Train on diverse profiles with various maneuvers (e.g., Driver A, Driver E)
        if "Driver A" in driver or "Driver E" in driver:
            train_pairs.append(p)
        # Validate on an unseen human driver (e.g., Driver D)
        elif "Driver D" in driver:
            val_pairs.append(p)
        # Test benchmark on specific routes or continuous uncut data
        else:
            test_pairs.append(p)

    print(
        f"[SPLIT] {len(train_pairs)} Train | {len(val_pairs)} Val | {len(test_pairs)} Test"
    )
    return train_pairs, val_pairs, test_pairs


# ---------------------------------------------------------
# 3. PyTorch Dataset Loader & Window Generator
# ---------------------------------------------------------# Change the default arguments:
class CachedIOVNBDSlidingWindow(Dataset):
    def __init__(
        self, pair_list, window_size=20, stride=2, mean=None, std=None, is_train=True
    ):
        # window_size=20 (2.0s at 10 Hz)
        # stride=2 (0.2s step, providing dense temporal coverage)
        """
        Loads the synchronized IO-VNBD files, extracts the 6 IMU channels + Vehicle Speed,
        synthesizes the motorcycle lean angle, and prepares the 200-sample sliding windows.
        """
        self.window_size = window_size
        self.stride = stride
        self.windows = []
        self.targets = []

        raw_imu_accum = []
        parsed_trips = []
        g_gravity = 9.80665

        for p in pair_list:
            try:
                # Read files (handling potential whitespace in CSV headers)
                # Change these lines:
                # df_s = pd.read_csv(p["s_path"], skipinitialspace=True)
                # df_v = pd.read_csv(p["v_path"], skipinitialspace=True)

                # To this:
                df_s = pd.read_csv(
                    p["s_path"], skipinitialspace=True, encoding="latin1"
                )
                df_v = pd.read_csv(
                    p["v_path"], skipinitialspace=True, encoding="latin1"
                )

                # Trim to the length of the shorter file to maintain strict synchronization
                min_len = min(len(df_s), len(df_v))
                if min_len < window_size:
                    continue

                df_s = df_s.iloc[:min_len]
                df_v = df_v.iloc[:min_len]

                # Dynamically locate columns regardless of exact naming variations in IO-VNBD
                accel_cols = [c for c in df_s.columns if "ACCELEROMETER" in c.upper()][
                    :3
                ]
                gyro_cols = [c for c in df_s.columns if "GYROSCOPE" in c.upper()][:3]
                spd_cols = [
                    c
                    for c in df_v.columns
                    if "SPEED" in c.upper() or "VELOCITY" in c.upper()
                ]

                if not accel_cols or not gyro_cols or not spd_cols:
                    print(f"[WARN] Missing required columns in {p['s_path']}")
                    continue

                # Extract IMU inputs (6 channels)
                accel = df_s[accel_cols].values.astype(np.float32)
                gyro = df_s[gyro_cols].values.astype(np.float32)
                imu_6axis = np.hstack([accel, gyro])

                # Extract Ground Truth Speed and convert from km/h to m/s
                speed_mps = df_v[spd_cols[0]].to_numpy(dtype=np.float32) / 3.6

                # Synthesize 2-wheeler lean angle: phi = arctan(v * omega_z / g)
                yaw_rate = gyro[:, 2]  # Assuming Z is the vertical axis for yaw
                lean_angle_rad = np.arctan((speed_mps * yaw_rate) / g_gravity).astype(
                    np.float32
                )

                raw_imu_accum.append(imu_6axis)
                parsed_trips.append((imu_6axis, speed_mps, lean_angle_rad))

            except Exception as e:
                print(f"[ERROR] Failed processing {p['s_path']}: {e}")

        # Compute Z-score normalization statistics globally over the training set
        if is_train and raw_imu_accum:
            all_imu = np.vstack(raw_imu_accum)
            self.mean = np.mean(all_imu, axis=0)
            self.std = np.std(all_imu, axis=0) + 1e-7
        else:
            self.mean = mean
            self.std = std

        # Slice each parsed trip into sliding windows independently to prevent boundary leakage
        for imu_raw, speed_mps, lean_rad in parsed_trips:
            # Apply normalization
            imu_norm = (imu_raw - self.mean) / self.std
            n_samples = len(imu_norm)

            for start in range(0, n_samples - window_size, stride):
                end = start + window_size

                # Shape: [6, window_size] (Channels-first format required by PyTorch Conv1D)
                window_slice = imu_norm[start:end].T

                # Targets are tied to the state at the end of the 2-second window
                target_v = speed_mps[end - 1]
                target_phi = lean_rad[end - 1]

                self.windows.append(window_slice)
                self.targets.append([target_v, target_phi])

        self.windows = np.array(self.windows, dtype=np.float32)
        self.targets = np.array(self.targets, dtype=np.float32)
        print(
            f"[LOAD COMPLETE] Built {len(self.windows)} windows of shape ({window_size}, 6)."
        )

    def __len__(self):
        return len(self.windows)

    def __getitem__(self, idx):
        return (
            torch.from_numpy(self.windows[idx]),
            torch.tensor(self.targets[idx][0]),  # Changed from [idx, 0]
            torch.tensor(self.targets[idx][1]),  # Changed from [idx, 1]
        )
