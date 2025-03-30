## Implementations of Numerical Integration and Derivation (Not over continuous functions but rather with discrete data)
## Created by Nathaniel Platt. Riemann Sum for Integration, 5 point derivative for derivation.

import polars as pl
import numpy as np
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt


# lat = "VDM_GPS_Latitude"
# long = "VDM_GPS_Longitude"

# df = pl.read_parquet("Parquet/2024-12-02-Part1-100Hz.pq")
# dfP2 = pl.read_parquet("Parquet/2024-12-02-Part2-100Hz.pq")


# df = df[40000:]
# df.write_parquet("Parquet/2024-12-02-Part1-100Hz.pq")


# df.columns

# frT = "TELEM_FR_SUSTRAVEL"
# flT = "TELEM_FL_SUSTRAVEL"
# brT = "TELEM_BR_SUSTRAVEL"
# blT = "TELEM_BL_SUSTRAVEL"
# lat = "VDM_GPS_Latitude"
# long = "VDM_GPS_Longitude"
# course = "VDM_GPS_TRUE_COURSE"
# xA = "VDM_X_AXIS_ACCELERATION"
# yA = "VDM_Y_AXIS_ACCELERATION"
# zA = "VDM_Z_AXIS_ACCELERATION"
# xG = "VDM_X_AXIS_YAW_RATE"
# yG = "VDM_Y_AXIS_YAW_RATE"
# zG = "VDM_Z_AXIS_YAW_RATE"
# rpm = "SME_TRQSPD_Speed"
# speed = "VDM_GPS_SPEED"
# tsC = "TS_Current"
# time = "Seconds"
# frT = "TELEM_FR_SUSTRAVEL"
# flT = "TELEM_FL_SUSTRAVEL"
# brT = "TELEM_BR_SUSTRAVEL"
# blT = "TELEM_BL_SUSTRAVEL"
# lat = "VDM_GPS_Latitude"
# long = "VDM_GPS_Longitude"
# course = "VDM_GPS_TRUE_COURSE"
# xA = "VDM_X_AXIS_ACCELERATION"
# yA = "VDM_Y_AXIS_ACCELERATION"
# zA = "VDM_Z_AXIS_ACCELERATION"
# xG = "VDM_X_AXIS_YAW_RATE"
# yG = "VDM_Y_AXIS_YAW_RATE"
# zG = "VDM_Z_AXIS_YAW_RATE"
# rpm = "SME_TRQSPD_Speed"
# speed = "VDM_GPS_SPEED"
# tsC = "TS_Current"
# xA_mps = "IMU_XAxis_Acceleration_mps"
# yA_mps = "IMU_YAxis_Acceleration_mps"
# zA_mps = "IMU_ZAxis_Acceleration_mps"
# speed_mps = "VMD_GPS_Speed_mps"
# index = "index"

# rpm_to_mph = 11/40*2*np.pi*0.0001342162*60

def mag (a, b, c):
    # print(f"mag = {np.sqrt(a**2 + b**2 + c**2)}")
    return np.sqrt(a**2 + b**2 + c**2)

def in_place_integrate (derivative): #Rimann sum that returns an array of the intergral at that point
    if len(derivative.shape) == 1:
        width = 1
    else:
        width = derivative.shape[1]
    container = derivative[0]
    out = np.zeros((derivative.shape[0], width))
    out[0] = container
    for i in range(1, derivative.shape[0]):
        container += 0.01*derivative[i]
        # print(f"added {0.01*derivative[i]} to {container}")
        out[i] = container
    # print(f"out of integral is {out}")
    return out

def in_place_integrate_20Hz (derivative): #Rimann sum that returns an array of the intergral at that point
    if len(derivative.shape) == 1:
        width = 1
    else:
        width = derivative.shape[1]
    container = derivative[0]
    out = np.zeros((derivative.shape[0], width))
    out[0] = container
    for i in range(1, derivative.shape[0]):
        container += 0.05*derivative[i]
        # print(f"added {0.01*derivative[i]} to {container}")
        out[i] = container
    # print(f"out of integral is {out}")
    return out


##This ended up just overwriting the data with a value from another column. Sort and filter first
# def in_place_integrate_filtered (dataframe, col, vector_cols): #Riemann sum that returns an array of the intergral at that point
#     vector = dataframe.select(vector_cols)
#     derivative = dataframe.select(col)
#     if len(derivative.shape) == 1:
#         width = 1
#     else:
#         width = derivative.shape[1]
#     container = derivative[0]
#     out = np.zeros((derivative.shape[0], width))
#     out[0] = container
#     out2 = np.zeros((derivative.shape[0], width))
#     for i in range(1, derivative.shape[0]):
#         # print(f"type of container is {type(container)}")
#         # print(f"type of dataframe[speed_mps][i] is {type(dataframe[speed_mps][i])}")
#         if (abs(container.item() - dataframe[speed_mps][i]) > 0.2):
#             container = container*0.5 + dataframe[speed_mps][i]*0.5
#             # out2[i] = -1
#         else:
#             b = dataframe[index][i]
#             a = 0.01*derivative[i] - 0.01*( 8.07919772e-02 + b*1.67367686e-05+-2.58170300e-16*b**3+5.10215428e-23*b**5)
#             container += a
#             # out2[i] = a
#         #(not (9.6 < mag(vector[vector_cols[0]][i], vector[vector_cols[1]][i], vector[vector_cols[2]][i]) < 10))
#         # print(9 < mag(vector[vector_cols[0]][i], vector[vector_cols[1]][i], vector[vector_cols[2]][i]) < 10.5)
        
#         # print(f"adding {a}")
#         out[i] = container
#     # print(f"out of integral is {out}")
#     return out#, out2

def derivative_at_point (dataFrame, i):
    rows = dataFrame.shape[0]
    if i > 1 and i < (rows - 2):
        return (dataFrame[i-2] - 8*dataFrame[i-1] + 8*dataFrame[i+1] - dataFrame[i+2])/(12*0.01)
    elif i < 2:
        return (dataFrame[0] - 8*dataFrame[0] + 8*dataFrame[i+1] - dataFrame[i+2])/(12*0.01)
    elif i > rows - 3:
        return (dataFrame[i-2] - 8*dataFrame[i-1] + 8*dataFrame[rows-1] - dataFrame[rows-1])/(12*0.01)

def in_place_derive (integral):
    if len(integral.shape) == 1:
        width = 1
    else:
        width = integral.shape[1]
    out = np.zeros((integral.shape[0], width))
    for i in range(integral.shape[0]):
        out[i] = derivative_at_point(integral, i)
    # print(f"out of derivative is {out}")
    return out