import time
import subprocess

flag = 0
def check_gps_signal():
    try:
        result = subprocess.run(
            ["adb", "shell", "settings", "get", "secure", "location_mode"],
            capture_output=True, text=True, timeout=2
        )
        mode = result.stdout.strip()
        return mode != "0" and mode != ""
    except (subprocess.TimeoutExpired, FileNotFoundError):
        return False

def run_gps_monitor():
    print("Starting 1-Second GPS Monitor... (Press Ctrl+C to stop)")
    try:
        while True:
            gps_present = check_gps_signal()
            if gps_present:
                flag = 0
                print(flag)
                print("connection present")
            else:
                flag = 1
                print(flag)
                print("connection lost")
            time.sleep(0.2)
    except KeyboardInterrupt:
        print("\nMonitoring stopped by user.")

if __name__ == "__main__":
    run_gps_monitor()