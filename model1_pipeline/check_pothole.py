import numpy as np
import pandas as pd
import torch
import matplotlib.pyplot as plt
from model import MultiTaskInertialNet

S_FILE = "dataset/Synchronised V and S datasets/Categorised IOVNB Dataset/Vta (Driver E)/Vta27/S-Vta27.csv"
V_FILE = "dataset/Synchronised V and S datasets/Categorised IOVNB Dataset/Vta (Driver E)/Vta27/V-vta27.csv"

print("[*] Loading Model & Normalization Stats...")
stats = np.load("normalization_stats.npz")
mean, std = stats['mean'], stats['std']

model = MultiTaskInertialNet()
model.load_state_dict(torch.load("model1_best.pth", map_location="cpu", weights_only=True))
model.eval()

df_s = pd.read_csv(S_FILE, skipinitialspace=True, encoding='latin1')
df_v = pd.read_csv(V_FILE, skipinitialspace=True, encoding='latin1')

min_len = min(len(df_s), len(df_v))
df_s, df_v = df_s.iloc[:min_len], df_v.iloc[:min_len]

accel_cols = [c for c in df_s.columns if 'ACCELEROMETER' in c.upper()][:3]
gyro_cols = [c for c in df_s.columns if 'GYROSCOPE' in c.upper()][:3]
spd_cols = [c for c in df_v.columns if 'SPEED' in c.upper() or 'VELOCITY' in c.upper()]

accel = df_s[accel_cols].values.astype(np.float32)
gyro = df_s[gyro_cols].values.astype(np.float32)
imu_raw = np.hstack([accel, gyro])
speed_gt_mps = (df_v[spd_cols[0]].to_numpy(dtype=np.float32) / 3.6)

imu_norm = (imu_raw - mean) / std

window_size = 20
predictions = []
ground_truths = []

# Initial seed speed from the ground truth start
running_v_est = float(speed_gt_mps[0])

print("[*] Running Stateful Autoregressive Inference...")
with torch.no_grad():
    for start in range(0, len(imu_norm) - window_size, 2):  # 10 Hz stride
        end = start + window_size
        window_slice = imu_norm[start:end].T
        
        tensor_in = torch.from_numpy(window_slice).unsqueeze(0)
        v_prev_tensor = torch.tensor([running_v_est], dtype=torch.float32)
        
        v_x, _, _, _ = model(tensor_in, v_prev_tensor)
        
        running_v_est = max(0.0, v_x.item())
        predictions.append(running_v_est)
        ground_truths.append(speed_gt_mps[end - 1])

# Plotting
plt.figure(figsize=(12, 6))
plt.plot(ground_truths, label="Actual Ground Truth Speed (CAN Bus)", color="blue", linewidth=2)
plt.plot(predictions, label="AI Predicted Speed (Stateful Learned Integrator)", color="red", linestyle="dashed", linewidth=2)
plt.title("Stateful Model 1: Momentum Preserved vs Reality")
plt.xlabel("Steps (0.2s increments)")
plt.ylabel("Speed (m/s)")
plt.legend()
plt.grid(True)
plt.tight_layout()

# --- THE FIX IS HERE ---
print("[*] Saving Graph to 'pothole_test.png'...")
plt.savefig("pothole_test.png", dpi=300, bbox_inches='tight')
print("[SUCCESS] Graph saved! Open 'pothole_test.png' to view it.")