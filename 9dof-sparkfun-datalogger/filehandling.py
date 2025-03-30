import polars as pl
from scipy.interpolate import CubicSpline
import numpy as np
import matplotlib.pyplot as plt

cols = ['time', 'voltage', 'state_of_charge', 'charge_rate', #Renames all the columns with these headers which are easier to type
        'xAcc', 'yAcc', 'zAcc', 
        'xGyro', 'yGyro', 'zGyro', 'ISM330.temp', 
       'xBField', 'yBField', 'zBField', 'MMC5983.temp']

df = pl.read_csv("Runs/DataLogger0009.txt",infer_schema_length=10000,ignore_errors=True).with_columns(pl.all().cast(pl.Float32, strict=False))
df
df.columns = cols

# df["time"][:100]
# df["time"][-100:]
cs = CubicSpline(df["time"].to_numpy()/1000, df.drop("time").to_numpy())
time = np.arange(88.5, 138.5, 0.05)
new_df = pl.DataFrame(cs(time))
new_df.insert_column(0, pl.Series(time).alias("time")) #Insert time column
new_df.columns = cols

new_df.write_parquet("Runs/GyroCalibration.parquet")