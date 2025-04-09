import polars as pl
from scipy.interpolate import CubicSpline
import numpy as np
import matplotlib.pyplot as plt

## Rename columns. This is for the sparkfun IMU
cols = ['time', 'voltage', 'state_of_charge', 'charge_rate', #Renames all the columns with these headers which are easier to type
        'xAcc', 'yAcc', 'zAcc', 
        'xGyro', 'yGyro', 'zGyro', 'ISM330.temp', 
       'xBField', 'yBField', 'zBField', 'MMC5983.temp']

# Reads a CSV, and sets everything to float32. The first 10k rows are used to infer the schema (probably unneccessary because they're reset anyway), and errors are ignored.
df = pl.read_csv("Runs/DataLogger0009.txt",infer_schema_length=10000,ignore_errors=True).with_columns(pl.all().cast(pl.Float32, strict=False))
# df
# df.columns = cols

## Check the beginning and start time in the file and update the values in row 19 accordingly!! np.arange(t0, t1, 0.05)
df["time"][:100]
df["time"][-100:]

# Fit a spline to the data to get regular interval data out
cs = CubicSpline(df["time"].to_numpy()/1000, df.drop("time").to_numpy())
time = np.arange(7, 152, 0.05) # New time columns
new_df = pl.DataFrame(cs(time)) # New data frame with non-time columns
new_df.insert_column(0, pl.Series(time).alias("time")) #Insert time column
new_df.columns = cols # Rename columns to match the original ones
new_df.write_parquet("Runs/GyroCalibration.parquet") # Write to new parquet file