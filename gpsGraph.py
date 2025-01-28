## This file contains the code originally used to create a plot using gps coordinates to create
## a heat map. Currently has updated functioning graphs for rpm, torque, and braking. 2025-01-20 (last update by Nathaniel)
## Made and commented by Nathaniel Platt
import polars as pl
import matplotlib.pyplot as plt
from heatGraph import colored_line

df = pl.read_parquet("Parquet/2024-12-02-Part1-100Hz.pq")
# df = pl.read_csv("Temp/2024-12-02-Part1-100Hz.csv",infer_schema_length=0).with_columns(pl.all().cast(pl.Float32, strict=False))
# df1 = pl.read_csv("Temp/2024-12-02-Part2-100Hz.csv",infer_schema_length=0).with_columns(pl.all().cast(pl.Float32, strict=False))
df1 = pl.read_parquet("Parquet/2024-12-02-Part2-100Hz.pq")

df.columns

time1 = 1400
# time1 = 1000
time2 = 1650
# time2 = 1900
lat = "VDM_GPS_Latitude"
long = "VDM_GPS_Longitude"
speed = "SME_TRQSPD_Speed"
busCurrent = "SME_TEMP_BusCurrent"
tsCurrent = "TS_Current"
torque = "SME_THROTL_TorqueDemand"
brakes = "Brakes"
df.columns
short = pl.DataFrame(df.filter(pl.col("Seconds") >= time1).filter(pl.col("Seconds") <= time2)).filter(pl.col("VDM_GPS_Latitude") != 0).filter(pl.col("VDM_GPS_Longitude") != 0)


# df.drop_nulls().select(lat).mean()

# df.select(lat).filter(pl.col("VDM_GPS_Latitude") != 0)
# df.filter(pl.col("Seconds") == 498.199).select([lat,long])\
# fig = plt.figure()
# fig.add_subplot(1,1,1)
# ax = plt.figure().add_subplot(1,1,1)
# ax.pcolorfast(-1*short[long],-1*short[lat],a)
# ax.plot(-1*short[long],-1*short[lat])
# ax.axis('scaled')
# plt.show()
# df.columns

import warnings

import matplotlib.pyplot as plt
import numpy as np

from matplotlib.collections import LineCollection

fig1 = plt.figure()
ax1 = fig1.add_subplot(1,1,1)
lines = colored_line(short[lat], short[long], short[busCurrent], ax1, linewidth=1, cmap="plasma")
fig1.colorbar(lines)  # add a color legend
ax1.axis('scaled')
ax1.set_title("Bus Current (A)")
plt.show()

# Create a figure and plot the line on it
fig1 = plt.figure()
ax1 = fig1.add_subplot(1,3,1)
lines = colored_line(short[lat], short[long], short[speed]/7500*109, ax1, linewidth=1, cmap="plasma")
fig1.colorbar(lines)  # add a color legend
ax1.axis('scaled')
ax1.set_title("RPM (motor)")

ax1 = fig1.add_subplot(1,3,2)
lines = colored_line(short[lat], short[long], short[torque]/30000*7500, ax1, linewidth=1, cmap="viridis")
fig1.colorbar(lines)  # add a color legend
ax1.axis('scaled')
ax1.set_title("Torque (Nm)")

ax1 = fig1.add_subplot(1,3,3)
lines = colored_line(short[lat], short[long], (short[brakes]-0.1)*2000, ax1, linewidth=1, cmap="inferno")
fig1.colorbar(lines)  # add a color legend
ax1.axis('scaled')
ax1.set_title("Braking (psi)")

plt.show()

fig1 = plt.figure()
ax1 = fig1.add_subplot(1,3,1)
lines = colored_line(short[lat], short[long], short[busCurrent], ax1, linewidth=1, cmap="plasma")
fig1.colorbar(lines)  # add a color legend
ax1.axis('scaled')
ax1.set_title("Motor Controller Current (A)")

ax1 = fig1.add_subplot(1,3,2)
lines = colored_line(short[lat], short[long], short[tsCurrent], ax1, linewidth=1, cmap="viridis")
fig1.colorbar(lines)  # add a color legend
ax1.axis('scaled')
ax1.set_title("Accumulator Current (A)")

ax1 = fig1.add_subplot(1,3,3)
lines = colored_line(short[lat], short[long], (short[brakes]-0.1)*2000, ax1, linewidth=1, cmap="inferno")
fig1.colorbar(lines)  # add a color legend
ax1.axis('scaled')
ax1.set_title("Braking (psi)")

plt.show()