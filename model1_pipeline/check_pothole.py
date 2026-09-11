import numpy as np
import pandas as pd
import torch
import matplotlib.pyplot as plt
from model import MultiTaskInertialNet

# 1. Paths to a specific bumpy route (Driver E, Route Vta27 has potholes and gravel)
# Update these paths if your folder structure differs slightly!
S_FILE = "dataset/Synchronised V and S datasets/Categorised IOVNB Dataset/Vta (Driver E)/Vta27/S-Vta27.csv"
V_FILE = "dataset/Synchronised V and S datasets/Categorised IOVNB Dataset/Vta (Driver E)/Vta27/V-vta27.csv"

print("[*] Loading Model and Normalization Stats...")
# Load stats saved during training
stats = np.load("normalization_stats.npz")
mean, std = stats['mean'], stats['std']

# Load Model
model = MultiTaskInertialNet()
model.load_state_dict(torch.load("model1_best.pth", map_location="cpu", weights_only=True))
model.eval()

print(f"[*] Processing files:\n   {S_FILE}")
# 2. Load the CSVs (using latin1 for the degree symbols)
df_s = pd.read_csv(S_FILE, skipinitialspace=True, encoding='latin1')
df_v = pd.read_csv(V_FILE, skipinitialspace=True, encoding='latin1')

min_len = min(len(df_s), len(df_v))
df_s, df_v = df_s.iloc[:min_len], df_v.iloc[:min_len]

# Extract Columns
accel_cols = [c for c in df_s.columns if 'ACCELEROMETER' in c.upper()][:3]
gyro_cols = [c for c in df_s.columns if 'GYROSCOPE' in c.upper()][:3]
spd_cols = [c for c in df_v.columns if 'SPEED' in c.upper() or 'VELOCITY' in c.upper()]

accel = df_s[accel_cols].values.astype(np.float32)
gyro = df_s[gyro_cols].values.astype(np.float32)
imu_raw = np.hstack([accel, gyro])
speed_gt_mps = (df_v[spd_cols[0]].to_numpy(dtype=np.float32) / 3.6)

# Normalize
imu_norm = (imu_raw - mean) / std

# 3. Run Sliding Window Inference
window_size = 20
predictions = []
ground_truths = []

print("[*] Running AI Inference...")
with torch.no_grad():
    for start in range(0, len(imu_norm) - window_size, 1): # 10 Hz stride
        end = start + window_size
        window_slice = imu_norm[start:end].T
        
        # Convert to tensor [1, 6, 20]
        tensor_in = torch.from_numpy(window_slice).unsqueeze(0)
        
        # Predict
        v_x, _, _, _ = model(tensor_in)
        
        predictions.append(v_x.item())
        ground_truths.append(speed_gt_mps[end - 1])

# 4. Plot the Results
plt.figure(figsize=(12, 6))
plt.plot(ground_truths, label="Actual Ground Truth Speed (CAN Bus)", color="blue", linewidth=2)
plt.plot(predictions, label="AI Predicted Speed (Smartphone IMU)", color="red", linestyle="dashed", linewidth=2)
plt.title("AI Speed Prediction vs. Reality (Over Bumps/Potholes)")
plt.xlabel("Time (10 Hz steps)")
plt.ylabel("Speed (m/s)")
plt.legend()
plt.grid(True)
plt.tight_layout()

print("[*] Saving Graph to 'pothole_test.png'...")
plt.savefig("pothole_test.png", dpi=300, bbox_inches='tight')
print("[SUCCESS] Graph saved! Open 'pothole_test.png' to view it.")