
from pathlib import Path
import pandas as pd

from src.risk.thresholds import (
    compute_station_thresholds,
    update_thresholds_config,
)

PROJECT_ROOT = Path(__file__).resolve().parent
STATION_ID = "TN_CAU_TRICHY"

data_dir = PROJECT_ROOT / "data" / "processed"

files = sorted(
    data_dir.glob(f"{STATION_ID}_aligned_*.csv")
)

if not files:
    raise FileNotFoundError(
        f"No aligned CSV files found in {data_dir} "
        f"for station {STATION_ID}"
    )

# Use the latest aligned dataset
data_file = files[-1]
df = pd.read_csv(data_file)

if "river_discharge" not in df.columns:
    raise ValueError(
        f"'river_discharge' column not found in {data_file}"
    )

# Use valid historical discharge values
discharge = pd.to_numeric(
    df["river_discharge"], errors="coerce"
).dropna()

discharge = discharge[discharge >= 0]

if discharge.empty:
    raise ValueError(
        "No valid non-negative discharge observations were found."
    )

threshold_data = compute_station_thresholds(discharge)

update_thresholds_config(
    station_id=STATION_ID,
    threshold_data=threshold_data,
)

print(f"Threshold calibration completed for {STATION_ID}")
print(f"Source file: {data_file}")
print(f"Valid observations: {len(discharge)}")
print("Calculated percentile thresholds:")
print(threshold_data["percentile_basis"])
print("Updated configs/thresholds.yaml")

