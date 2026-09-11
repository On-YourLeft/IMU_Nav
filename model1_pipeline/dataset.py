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
    categorised_path = os.path.join(base_dir, "Categorised IOVNB Dataset")
    uncategorised_path = os.path.join(base_dir, "Uncategorised IOVNB Dataset")
    
    pairs = []

    if os.path.exists(categorised_path):
        for driver_dir in os.listdir(categorised_path):
            driver_full = os.path.join(categorised_path, driver_dir)
            if not os.path.isdir(driver_full):
                continue
            
            s_files = glob.glob(os.path.join(driver_full, "**", "S-*.txt"), recursive=True) + \
                      glob.glob(os.path.join(driver_full, "**", "S-*.csv"), recursive=True)
            
            for s_path in s_files:
                dirname, fname = os.path.split(s_path)
                v_fname = re.sub(r'^S-', 'V-', fname)
                v_path = os.path.join(dirname, v_fname)

                if os.path.exists(v_path):
                    pairs.append({
                        "driver": driver_dir,
                        "category": "categorised",
                        "s_path": s_path,
                        "v_path": v_path
                    })

    if os.path.exists(uncategorised_path):
        s_folder = os.path.join(uncategorised_path, "S-Dataset")
        v_folder = os.path.join(uncategorised_path, "V-Dataset")

        if os.path.exists(s_folder) and os.path.exists(v_folder):
            s_files = glob.glob(os.path.join(s_folder, "*.txt")) + glob.glob(os.path.join(s_folder, "*.csv"))
            for s_path in s_files:
                fname = os.path.basename(s_path)
                v_fname = re.sub(r'^S-', 'V-', fname)
                v_path = os.path.join(v_folder, v_fname)

                if os.path.exists(v_path):
                    pairs.append({
                        "driver": "Uncategorised",
                        "category": "continuous",
                        "s_path": s_path,
                        "v_path": v_path
                    })

    print(f"[DATASET] Discovered {len(pairs)} synchronized S-V pairs.")
    return pairs

# ---------------------------------------------------------
# 2. Driver-Aware Splitting (Prevents Data Leakage)
# ---------------------------------------------------------
def split_by_driver(pairs):
    train_pairs, val_pairs, test_pairs = [], [], []

    for p in pairs:
        driver = p["driver"]
        if "Driver A" in driver or "Driver E" in driver:
            train_pairs.append(p)
        elif "Driver D" in driver:
            val_pairs.append(p)
        else:  
            test_pairs.append(p)

    print(f"[SPLIT] {len(train_pairs)} Train | {len(val_pairs)} Val | {len(test_pairs)} Test")
    return train_pairs, val_pairs, test_pairs

# ---------------------------------------------------------
# 3. PyTorch Dataset Loader with Stateful Velocity v_{t-1}
# ---------------------------------------------------------
class CachedIOVNBDSlidingWindow(Dataset):
    def __init__(self, pair_list, window_size=20, stride=2, mean=None, std=None, is_train=True):
        """
        window_size=20 (2.0s at 10 Hz)
        stride=2 (0.2s step)
        """
        self.window_size = window_size
        self.stride = stride
        self.windows = []
        self.v_prevs = []
        self.targets = []

        raw_imu_accum = []
        parsed_trips = []
        g_gravity = 9.80665

        for p in pair_list:
            try:
                # latin1 encoding prevents crashes on degree/squared symbols
                df_s = pd.read_csv(p["s_path"], skipinitialspace=True, encoding='latin1')
                df_v = pd.read_csv(p["v_path"], skipinitialspace=True, encoding='latin1')

                min_len = min(len(df_s), len(df_v))
                if min_len <= window_size:
                    continue

                df_s = df_s.iloc[:min_len]
                df_v = df_v.iloc[:min_len]

                accel_cols = [c for c in df_s.columns if 'ACCELEROMETER' in c.upper()][:3]
                gyro_cols = [c for c in df_s.columns if 'GYROSCOPE' in c.upper()][:3]
                spd_cols = [c for c in df_v.columns if 'SPEED' in c.upper() or 'VELOCITY' in c.upper()]

                if not accel_cols or not gyro_cols or not spd_cols:
                    continue

                accel = df_s[accel_cols].values.astype(np.float32)
                gyro = df_s[gyro_cols].values.astype(np.float32)
                imu_6axis = np.hstack([accel, gyro])

                # PYLANCE FIX: Explicitly cast to numpy array before division
                speed_mps = (df_v[spd_cols[0]].to_numpy(dtype=np.float32) / 3.6)

                # Synthesize 2-wheeler lean angle: phi = arctan(v * omega_z / g)
                yaw_rate = gyro[:, 2]
                lean_angle_rad = np.arctan((speed_mps * yaw_rate) / g_gravity).astype(np.float32)

                raw_imu_accum.append(imu_6axis)
                parsed_trips.append((imu_6axis, speed_mps, lean_angle_rad))
                
            except Exception as e:
                print(f"[ERROR] Failed processing {p['s_path']}: {e}")

        if is_train and raw_imu_accum:
            all_imu = np.vstack(raw_imu_accum)
            self.mean = np.mean(all_imu, axis=0)
            self.std = np.std(all_imu, axis=0) + 1e-7
        else:
            self.mean = mean
            self.std = std

        # Generate windows and attach entering velocity v_{t-1}
        for imu_raw, speed_mps, lean_rad in parsed_trips:
            imu_norm = (imu_raw - self.mean) / self.std
            n_samples = len(imu_norm)

            for start in range(0, n_samples - window_size, stride):
                end = start + window_size
                
                # Window: [6, 20]
                window_slice = imu_norm[start:end].T
                
                # Prior speed right before window start (Option B Stateful Input)
                v_prior = speed_mps[start]
                
                target_v = speed_mps[end - 1]
                target_phi = lean_rad[end - 1]

                self.windows.append(window_slice)
                self.v_prevs.append(v_prior)
                self.targets.append([target_v, target_phi])

        self.windows = np.array(self.windows, dtype=np.float32)
        self.v_prevs = np.array(self.v_prevs, dtype=np.float32)
        self.targets = np.array(self.targets, dtype=np.float32)
        print(f"[LOAD COMPLETE] Built {len(self.windows)} windows of shape ({window_size}, 6).")

    def __len__(self):
        return len(self.windows)

    def __getitem__(self, idx):
        # PYLANCE FIX: Using chained [idx][0] indexing to bypass tuple typing errors
        return (
            torch.from_numpy(self.windows[idx]),
            torch.tensor(self.v_prevs[idx], dtype=torch.float32),
            torch.tensor(self.targets[idx][0], dtype=torch.float32),
            torch.tensor(self.targets[idx][1], dtype=torch.float32)
        )