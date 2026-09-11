import torch
from model import MultiTaskInertialNet

print("[*] Loading trained weights...")
model = MultiTaskInertialNet()
# Load the weights you already saved during training
model.load_state_dict(
    torch.load("model1_best.pth", map_location="cpu", weights_only=True)
)
model.eval()

print("[*] Exporting to ONNX...")
dummy_input = torch.randn(1, 6, 200, dtype=torch.float32)
torch.onnx.export(
    model,
    (dummy_input,),
    "model1_velocity_lean.onnx",
    export_params=True,
    opset_version=14,
    do_constant_folding=True,
    input_names=["imu_window"],
    output_names=["v_x", "log_var_v", "phi", "log_var_phi"],
    dynamic_axes={"imu_window": {0: "batch_size"}},
)
print("[SUCCESS] Exported 'model1_velocity_lean.onnx'!")
