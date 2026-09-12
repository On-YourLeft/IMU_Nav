import json
import asyncio
import websockets
import numpy as np
import onnxruntime as ort
from collections import deque

# Load ONNX Model and Normalization Stats
ort_session = ort.InferenceSession("model1_velocity_lean.onnx")
stats = np.load("normalization_stats.npz")
mean, std = stats["mean"], stats["std"]

# Configuration
UDP_PORT = 5555
WS_HOST = "0.0.0.0"
WS_PORT = 8765

window_size = 20
imu_buffer = deque(maxlen=window_size)
v_prev = 0.0  # Initial stateful velocity

# Keep track of connected WebSocket clients
connected_clients = set()


async def ws_handler(websocket):
    """Manages connecting and disconnecting HTML Dashboard clients."""
    connected_clients.add(websocket)
    print(f"\n[*] Dashboard Connected. Total clients: {len(connected_clients)}")
    try:
        async for _ in websocket:
            pass  # Keep connection open (we only send, we don't receive)
    finally:
        connected_clients.remove(websocket)
        print(f"\n[*] Dashboard Disconnected. Total clients: {len(connected_clients)}")


class UDPBridgeProtocol(asyncio.DatagramProtocol):
    def connection_made(self, transport):
        self.transport = transport

    def datagram_received(self, data, addr):
        global v_prev
        try:
            line = data.decode("utf-8", errors="ignore").strip()
            parts = [float(x) for x in line.split(",") if x]

            if len(parts) >= 6:
                imu_buffer.append(parts[:6])

            # When we have a full 2-second window (20 samples at 10Hz)
            if len(imu_buffer) == window_size:
                imu_raw = np.array(imu_buffer, dtype=np.float32)
                imu_norm = (imu_raw - mean) / std

                # Format inputs for ONNX: [Batch, Channels, Window]
                x_input = imu_norm.T.reshape(1, 6, 20).astype(np.float32)
                v_input = np.array([v_prev], dtype=np.float32)

                # Run Inference
                outputs = ort_session.run(
                    None, {"imu_window": x_input, "v_prev": v_input}
                )
                v_x = max(0.0, float(np.asarray(outputs[0]).reshape(-1)[0]))
                phi_rad = float(np.asarray(outputs[2]).reshape(-1)[0])
                phi_deg = np.degrees(phi_rad)

                # Autoregressive state update
                v_prev = v_x

                # Clear 1 sample to slide the window forward
                imu_buffer.popleft()

                # Broadcast to all connected HTML Dashboards
                if connected_clients:
                    payload = json.dumps({"speed_kmh": v_x * 3.6, "lean_deg": phi_deg})
                    websockets.broadcast(connected_clients, payload)

                # Optional: Print to terminal to verify background processing
                print(f"\rSpeed: {v_x * 3.6:5.1f} km/h | Lean: {phi_deg:5.1f}°", end="")

        except Exception as e:
            pass  # Ignore malformed packets


async def main():
    # Start WebSocket Server
    ws_server = await websockets.serve(ws_handler, WS_HOST, WS_PORT)
    print(f"[*] WebSocket Server running on ws://localhost:{WS_PORT}")

    # Start UDP Listener to catch phone stream independently
    loop = asyncio.get_running_loop()
    transport, protocol = await loop.create_datagram_endpoint(
        lambda: UDPBridgeProtocol(), local_addr=("0.0.0.0", UDP_PORT)
    )
    print(f"[*] UDP Listener catching HyperIMU on port {UDP_PORT}")

    await asyncio.Future()  # Run forever


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\n[*] Server stopped.")
