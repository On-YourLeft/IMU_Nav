from pathlib import Path
import pandas as pd
import numpy as np


# ============================================================
# PROJECT PATHS
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[1]

RAW_DATA = (
    PROJECT_ROOT
    / "data"
    / "IO-VNBD"
    / "Synchronised V abd S datasets"
    / "Categorised IOVNB Dataset"
    / "M (Driver B)"
    / "V-M.csv"
)

OUTPUT_DIR = PROJECT_ROOT / "data" / "processed"
OUTPUT_FILE = OUTPUT_DIR / "M_driver_B_reference.csv"


# ============================================================
# LOAD VEHICLE DATA
# ============================================================

print("=" * 70)
print("CREATING VEHICLE REFERENCE")
print("=" * 70)

print("\nLoading vehicle dataset...")

vehicle = pd.read_csv(
    RAW_DATA,
    encoding="latin1"
)

# Remove accidental spaces from column names
vehicle.columns = vehicle.columns.str.strip()

print(f"Loaded {len(vehicle)} rows")
print(f"Loaded {len(vehicle.columns)} columns")


# ============================================================
# SHOW AVAILABLE COLUMNS
# ============================================================

print("\nVehicle columns:")
for i, column in enumerate(vehicle.columns):
    print(f"{i}: {column}")


# ============================================================
# EXTRACT REQUIRED DATA
# ============================================================

print("\nExtracting reference measurements...")

latitude = vehicle["Latitude (degrees)"].astype(float)
longitude = vehicle["Longitude (degrees)"].astype(float)

velocity = (
    vehicle["Velocity (km/hr)"].astype(float)
    / 3.6
)

heading = (
    vehicle["Heading (degrees)"].astype(float)
)

yaw_rate = (
    vehicle["Yaw Rate (deg/sec)"].astype(float)
    * np.pi
    / 180.0
)

dt = vehicle["Sample period (seconds)"].astype(float)

# Protect against unusual sample periods
dt = dt.where(
    (dt >= 0.09) & (dt <= 0.11),
    0.1
)

time = dt.cumsum() - dt.iloc[0]


# ============================================================
# CONVERT LAT/LON TO LOCAL X/Y
# ============================================================

print("Converting GPS latitude/longitude to local coordinates...")

lat0 = latitude.iloc[0]
lon0 = longitude.iloc[0]

earth_radius = 6371000.0

lat_rad = np.deg2rad(latitude)
lon_rad = np.deg2rad(longitude)

lat0_rad = np.deg2rad(lat0)
lon0_rad = np.deg2rad(lon0)

reference_x = (
    earth_radius
    * (lon_rad - lon0_rad)
    * np.cos(lat0_rad)
)

reference_y = (
    earth_radius
    * (lat_rad - lat0_rad)
)


# ============================================================
# CREATE REFERENCE DATAFRAME
# ============================================================

reference = pd.DataFrame({
    "time": time,
    "reference_x": reference_x,
    "reference_y": reference_y,
    "reference_velocity": velocity,
    "reference_heading": heading,
    "reference_yaw_rate": yaw_rate,
})


# ============================================================
# CLEAN DATA
# ============================================================

reference = reference.replace(
    [np.inf, -np.inf],
    np.nan
)

reference = reference.interpolate(
    limit_direction="both"
)

reference = reference.reset_index(drop=True)


# ============================================================
# SAVE
# ============================================================

OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True
)

reference.to_csv(
    OUTPUT_FILE,
    index=False
)


# ============================================================
# SUMMARY
# ============================================================

print("\n" + "=" * 70)
print("REFERENCE DATA CREATED")
print("=" * 70)

print(f"\nOutput file:")
print(OUTPUT_FILE)

print(f"\nRows: {len(reference)}")
print(f"Columns: {len(reference.columns)}")

print("\nColumns created:")
for column in reference.columns:
    print(f" - {column}")

print("\nFirst 5 rows:")
print(reference.head())

print("\nLast 5 rows:")
print(reference.tail())

print("\n" + "=" * 70)
print("REFERENCE CREATION COMPLETE")
print("=" * 70)