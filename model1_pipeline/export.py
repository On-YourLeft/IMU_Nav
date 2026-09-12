import torch
from model import MultiTaskInertialNet

print("[*] Loading trained weights...")
model = MultiTaskInertialNet()
model.load_state_dict(torch.load("model1_best.pth", map_location="cpu", weights_only=True))
model.eval()

print("[*] Exporting Stateful Model to ONNX...")

# 1. Dummy IMU Window: [Batch, Channels, Window_Size] -> [1, 6, 20]
dummy_x = torch.randn(1, 6, 20, dtype=torch.float32)

# 2. Dummy Previous Velocity (v_prev): [Batch] -> [1]
dummy_v_prev = torch.randn(1, dtype=torch.float32)

# Export passing BOTH inputs as a tuple
torch.onnx.export(
    model,
    args=(dummy_x, dummy_v_prev), 
    f="model1_velocity_lean.onnx",
    export_params=True,
    opset_version=14,
    do_constant_folding=True,
    input_names=['imu_window', 'v_prev'],
    output_names=['v_x', 'log_var_v', 'phi', 'log_var_phi'],
    dynamic_axes={
        'imu_window': {0: 'batch_size'},
        'v_prev': {0: 'batch_size'}
    }
)
print("[SUCCESS] Exported 'model1_velocity_lean.onnx'!")