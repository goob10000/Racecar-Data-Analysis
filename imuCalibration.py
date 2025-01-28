# Structure of this file:
# Working code
# Comment block explaining ideas and thought process
# Commented out non-working code
# Made and commented by Nathaniel Platt

## Change .pq to .parquet!!!
## FIX THE BIAS FOR EACH FUN INDIVIDUALLYYYYYY

# from scipy.integrate import solve_ivp
# import functools
import math
from ahrs.filters import madgwick as mg
from ahrs.common import quaternion as qt
import polars as pl
import numpy as np
import scipy.interpolate as itp
import scipy.optimize as opt
import matplotlib.pyplot as plt
from scipy.interpolate import CubicSpline
from heatGraph import colored_line
from integralsAndDerivatives import *
from scipy import fftpack
from fftTools import *


df = pl.read_parquet("Parquet/2024-12-02-Part1-100Hz.pq")[40000:]
df = pl.read_parquet("Nathaniel_IMU_Data/Rectangle2x_Smoothed.pq")
df = pl.read_parquet("Parquet/2024-12-02-Part2-100Hz.pq")
df = pl.read_parquet("Nathaniel_IMU_Data/Orientation.pq")
df = pl.read_parquet("Parquet/2025-01-21-IMU-Calibration.parquet")

car_imu_args = np.asarray([ 0.96324388,  0.98596328,  0.96496715,  0.14328537, -0.25890404,
       -0.11429213,  0.12046681,  0.27326771, -0.08613647,  0.03362676,
       -0.02482835, -0.02720422])
Sx, Sy, Sz, a1, a2, a3, a4, a5, a6, b1, b2, b3 = 0.96324388,  0.98596328,  0.96496715,  0.14328537, -0.25890404, -0.11429213,  0.12046681,  0.27326771, -0.08613647,  0.03362676, -0.02482835, -0.02720422

# df.columns
df.columns = ['time', 'voltage', 'state_of_charge', 'charge_rate', 
              'xA_uncorrected', 'yA_uncorrected', 'zA_uncorrected', 
              'xG', 'yG', 'zG','ISM330.temp', 
              'xBField', 'yBField', 'zBField', 'MMC5983.temp']

xA_uncorrected = "xA_uncorrected"
yA_uncorrected = "yA_uncorrected"
zA_uncorrected = "zA_uncorrected"
xA = "xA"
yA = "yA"
zA = "zA"
xG = "xG"
yG = "yG"
zG = "zG"
time = "time"
imuT = "IMU_temp"

time = "Seconds"
frT = "TELEM_FR_SUSTRAVEL"
flT = "TELEM_FL_SUSTRAVEL"
brT = "TELEM_BR_SUSTRAVEL"
blT = "TELEM_BL_SUSTRAVEL"
lat = "VDM_GPS_Latitude"
long = "VDM_GPS_Longitude"
course = "VDM_GPS_TRUE_COURSE"
xA = "xA"
yA = "yA"
zA = "zA"
xA_uncorrected = "VDM_X_AXIS_ACCELERATION"
yA_uncorrected = "VDM_Y_AXIS_ACCELERATION"
zA_uncorrected = "VDM_Z_AXIS_ACCELERATION"
xG = "VDM_X_AXIS_YAW_RATE"
yG = "VDM_Y_AXIS_YAW_RATE"
zG = "VDM_Z_AXIS_YAW_RATE"
rpm = "SME_TRQSPD_Speed"
speed = "VDM_GPS_SPEED"
tsC = "TS_Current"
xA_mps = "IMU_XAxis_Acceleration_mps"
yA_mps = "IMU_YAxis_Acceleration_mps"
zA_mps = "IMU_ZAxis_Acceleration_mps"
speed_mps = "VMD_GPS_Speed_mps"
index = "index"

rpm_to_mph = 11/40*2*np.pi*0.0001342162*60
df.insert_column(0, df.select(time).map_rows(lambda x : round(x[0]*100))['map'].alias("index"))
df.insert_column(df.get_column_index(xA), ((df.select(xA))[xA]*9.81).alias("IMU_XAxis_Acceleration_mps"))
df.insert_column(df.get_column_index(yA), ((df.select(yA))[yA]*9.81).alias("IMU_YAxis_Acceleration_mps"))
df.insert_column(df.get_column_index(zA), ((df.select(zA))[zA]*9.81).alias("IMU_ZAxis_Acceleration_mps"))
df.insert_column(df.get_column_index(speed), ((df.select(speed))[speed]*0.44704).alias("VMD_GPS_Speed_mps"))

# Apply fft, remove high frequency noise to make derivative more representative.
# Then do derivative selection for smooth regions
df.insert_column(-1, (df[xA].pow(2) + df[yA].pow(2) + df[zA].pow(2)).sqrt().alias("vA")) # Column that is magnitude of acceleration
df.shape[0]
plt.plot(df["vA"])
plt.show()
plt.plot(np.convolve(df["vA"], np.array([-2, -2, -2, -1, 0, 1, 2, 2, 2]),'same')) # Convolution that looks for edges
# plt.plot(np.convolve(np.convolve(df["vA"], np.array([-2, -2, -2, -1, 0, 1, 2, 2, 2]),'same'), np.array([-2, -2, -2, -1, 0, 1, 2, 2, 2]),'same'))
# plt.plot(in_place_derive(np.convolve(df["vA"], np.array([-3, -2, -1, 0, 1, 2, 3]),'same')))
plt.scatter(df.filter(pl.col("dvA").abs() < 100)[time],df.filter(pl.col("dvA").abs() < 100)["vA"], s=0.5)
plt.plot(df["vA"])
# plt.plot(low_pass_filter(df["vA"].to_numpy(),0.98))
plt.plot(df["dvA"])
plt.show()
df.insert_column(-1, pl.Series(np.convolve(df["vA"], np.array([-2, -2, -2, -1, 0, 1, 2, 2, 2]),'same')).alias("vAConvolve"))

def apply_correction_acc_Gs (df, Sx, Sy, Sz, a1, a2, a3, a4, a5, a6, b1, b2, b3):
    scalar_matrix = np.array([[Sx, 0, 0],
                              [0, Sy, 0],
                              [0, 0, Sz]])
    off_axis_matrix = np.array([[1, a1, a2],
                                [a3, 1, a4],
                                [a5, a6, 1]])
    first_matrix = np.matmul(scalar_matrix, off_axis_matrix)
    bias1 = np.ones((1,df[xA_uncorrected,yA_uncorrected,zA_uncorrected].shape[0]))*b1
    bias2 = np.ones((1,df[xA_uncorrected,yA_uncorrected,zA_uncorrected].shape[0]))*b2
    bias3 = np.ones((1,df[xA_uncorrected,yA_uncorrected,zA_uncorrected].shape[0]))*b3
    bias_matrix = np.concatenate([bias1,bias2,bias3], axis=0)
    matrix = df[xA_uncorrected,yA_uncorrected,zA_uncorrected].to_numpy().T
    biased_matrix = matrix-bias_matrix
    vectors = np.matmul(first_matrix,biased_matrix)
    print(vectors)
    df = df.insert_column(-1, pl.Series(vectors[0,:]).alias("xA"))
    df = df.insert_column(-1, pl.Series(vectors[1,:]).alias("yA"))
    df = df.insert_column(-1, pl.Series(vectors[2,:]).alias("zA"))
    return df

apply_correction_acc_Gs(df, Sx, Sy, Sz, a1, a2, a3, a4, a5, a6, b1, b2, b3)
plt.plot(df[xA])
plt.plot(df[xA_uncorrected])
plt.show()

cuts = df.filter(pl.col("vAConvolve").abs() > 0.04)[time] #Look for places where the edge detection is large (above 50)
cuts = df[10000:46500].filter(pl.col("vAConvolve").abs() > 0.04)[time] #Look for places where the edge detection is large (above 50)
plt.scatter(cuts*100, np.ones(cuts.shape[0]), s=0.5)
plt.show()
for i in range(0, cuts.shape[0] - 1):
#checks every region bounded by 2 cut locations. If the region is large enough, save it to "regions"
    if i == 0:
        regions = []
    if cuts[i+1] - cuts[i] > 3: # 500 for personal IMU
        regions = [(cuts[i], cuts[i+1], cuts[i+1] - cuts[i])] + regions
filtered_chunks = pl.DataFrame()
medians = pl.DataFrame()
# cuts.shape
# len(regions)
def compile_chunks_and_graph (regions, filter_decision, low_pass_filter_portion): # List, bool, float [0,1]
    filtered_chunks = pl.DataFrame()
    medians = pl.DataFrame()
    for (start, stop, l) in regions:
    # For every region selected, grab the median and standard deviation
    # Filter out any values greater than 1 standard deviation from the median
    # Calculate the new standard deviation without outliers. If the data set varies too much (>1) drop that set
        chunk = df.filter((pl.col(time) >= start) & (pl.col(time) < stop))
        med = chunk["vA"].median()
        std = chunk["vA"].std()
        print(f"med = {med}, std = {std}")
        num_stds = 1 #1 for home IMU and FS IMU
        filtered_chunk = chunk.filter((pl.col("vA") <= med + num_stds*std) & (pl.col("vA") >= med - num_stds*std))
        std = filtered_chunk["vA"].std()
        med = filtered_chunk["vA"].median()
        if std < 1: #1 for home IMU and FS IMU
            if filter_decision:
                # print(f"shape before is {filtered_chunk.shape}")
                array = low_pass_filter(filtered_chunk["vA"].to_numpy(), low_pass_filter_portion)
                # print(f"shape after is {array.shape}")
                # print(array)
                series = pl.Series(array).alias("vA")
                insertion_index = filtered_chunk.get_column_index("vA")
                filtered_chunk.drop_in_place("vA")
                filtered_chunk.insert_column(insertion_index, series)
                # print(filtered_chunk["vA"])
            plt.scatter(filtered_chunk[time], filtered_chunk["vA"], s=0.5) 
            filtered_chunks = pl.concat([filtered_chunks, filtered_chunk],how = 'vertical')
            # print("here")
            medians = pl.concat([medians, filtered_chunk.filter(pl.col("vA") == (filtered_chunk["vA"].median()))], how = 'vertical')
    # print("here")
    plt.show()
    return (filtered_chunks, medians)
filtered_chunks, medians = compile_chunks_and_graph(regions, True, 0.95)
filtered_chunks, medians = compile_chunks_and_graph(regions, False, 0.95)
for (start, stop, l) in regions:
# For every region selected, grab the median and standard deviation
# Filter out any values greater than 1 standard deviation from the median
# Calculate the new standard deviation without outliers. If the data set varies too much (>1) drop that set
    chunk = df.filter((pl.col(time) >= start) & (pl.col(time) < stop))
    med = chunk["vA"].median()
    std = chunk["vA"].std()
    print(f"med = {med}, std = {std}")
    num_stds = 1 #1 for home IMU and FS IMU
    filtered_chunk = chunk.filter((pl.col("vA") <= med + num_stds*std) & (pl.col("vA") >= med - num_stds*std))
    std = filtered_chunk["vA"].std()
    med = filtered_chunk["vA"].median()
    if std < 1: #1 for home IMU and FS IMU
        plt.scatter(filtered_chunk[time], filtered_chunk["vA"], s=0.5)
        filtered_chunks = pl.concat([filtered_chunks, filtered_chunk],how = 'vertical')
        medians = pl.concat([medians, filtered_chunk.filter(pl.col("vA") == med)], how = 'vertical')
# plt.scatter(filtered_chunks[time],filtered_chunks["vA"],s=0.5)        
plt.show()

# 

def lsq_fun_milliGs (x, Sx, Sy, Sz, a1, a2, a3, a4, a5, a6, b1, b2, b3):
    scalar_matrix = np.array([[Sx, 0, 0],
                              [0, Sy, 0],
                              [0, 0, Sz]])
    off_axis_matrix = np.array([[1, a1, a2],
                                [a3, 1, a4],
                                [a5, a6, 1]])
    first_matrix = np.matmul(scalar_matrix, off_axis_matrix)
    bias1 = np.ones((1,filtered_chunks[xA,yA,zA].shape[0]))*b1*100
    bias2 = np.ones((1,filtered_chunks[xA,yA,zA].shape[0]))*b2*100
    bias3 = np.ones((1,filtered_chunks[xA,yA,zA].shape[0]))*b3*100
    bias_matrix = np.concatenate([bias1,bias2,bias3], axis=0)
    matrix = filtered_chunks[xA,yA,zA].to_numpy().T
    biased_matrix = matrix-bias_matrix
    vectors = np.matmul(first_matrix,biased_matrix)
    mag = np.sqrt(vectors[0,:]**2 + vectors[1,:]**2+vectors[2,:]**2)
    # print(np.matmul(np.matmul(np.array([[Sx, 0, 0],[0, Sy, 0],[0,0,Sz]]),np.array([[1, a1, a2],[a3, 1, a4],[a5, a6, 1]])),np.array([[y[0,0]-b1],[y[1,0]-b2],[y[2,0]-b3]])))

    error = mag - 1000
    # print(f"error = {error}")
    return mag

def lsq_fun_Gs (x, Sx, Sy, Sz, a1, a2, a3, a4, a5, a6, b1, b2, b3):
    scalar_matrix = np.array([[Sx, 0, 0],
                              [0, Sy, 0],
                              [0, 0, Sz]])
    off_axis_matrix = np.array([[1, a1, a2],
                                [a3, 1, a4],
                                [a5, a6, 1]])
    first_matrix = np.matmul(scalar_matrix, off_axis_matrix)
    bias1 = np.ones((1,filtered_chunks[xA,yA,zA].shape[0]))*b1
    bias2 = np.ones((1,filtered_chunks[xA,yA,zA].shape[0]))*b2
    bias3 = np.ones((1,filtered_chunks[xA,yA,zA].shape[0]))*b3
    bias_matrix = np.concatenate([bias1,bias2,bias3], axis=0)
    matrix = filtered_chunks[xA,yA,zA].to_numpy().T
    biased_matrix = matrix-bias_matrix
    vectors = np.matmul(first_matrix,biased_matrix)
    mag = np.sqrt(vectors[0,:]**2 + vectors[1,:]**2+vectors[2,:]**2)
    # print(np.matmul(np.matmul(np.array([[Sx, 0, 0],[0, Sy, 0],[0,0,Sz]]),np.array([[1, a1, a2],[a3, 1, a4],[a5, a6, 1]])),np.array([[y[0,0]-b1],[y[1,0]-b2],[y[2,0]-b3]])))

    # error = mag - 1
    # print(f"error = {error}")
    return mag

def lsq_fun_Gs_medians (x, Sx, Sy, Sz, a1, a2, a3, a4, a5, a6, b1, b2, b3):
    scalar_matrix = np.array([[Sx, 0, 0],
                              [0, Sy, 0],
                              [0, 0, Sz]])
    off_axis_matrix = np.array([[1, a1, a2],
                                [a3, 1, a4],
                                [a5, a6, 1]])
    first_matrix = np.matmul(scalar_matrix, off_axis_matrix)
    bias1 = np.ones((1,medians[xA,yA,zA].shape[0]))*b1
    bias2 = np.ones((1,medians[xA,yA,zA].shape[0]))*b2
    bias3 = np.ones((1,medians[xA,yA,zA].shape[0]))*b3
    bias_matrix = np.concatenate([bias1,bias2,bias3], axis=0)
    matrix = medians[xA,yA,zA].to_numpy().T
    biased_matrix = matrix-bias_matrix
    vectors = np.matmul(first_matrix,biased_matrix)
    mag = np.sqrt(vectors[0,:]**2 + vectors[1,:]**2+vectors[2,:]**2)
    # print(np.matmul(np.matmul(np.array([[Sx, 0, 0],[0, Sy, 0],[0,0,Sz]]),np.array([[1, a1, a2],[a3, 1, a4],[a5, a6, 1]])),np.array([[y[0,0]-b1],[y[1,0]-b2],[y[2,0]-b3]])))

    error = mag - 1
    # print(f"error = {error}")
    return mag


args = opt.curve_fit(lsq_fun_milliGs,filtered_chunks[time],1000*np.ones(filtered_chunks.shape[0]),[1,1,1,0.2,-0.2,-0.1,0.2,0.2,-0.15,0.1,-0.2,0.1])
args = opt.curve_fit(lsq_fun_Gs,filtered_chunks[time],np.ones(filtered_chunks.shape[0]),[1,1,1,0.2,-0.2,-0.1,0.2,0.2,-0.15,0.1,-0.2,0.1])
args = opt.curve_fit(lsq_fun_Gs_medians,medians[time],np.ones(medians.shape[0]),[1,1,1,0.2,-0.2,-0.1,0.2,0.2,-0.15,0.1,-0.2,0.1])
# args = opt.curve_fit(lsq_fun,medians[time],1000*np.ones(medians.shape[1]),[1,1,1,0.2,-0.2,-0.1,0.2,0.2,-0.15,0.1,-0.2,0.1])
# args = opt.least_squares(lsq_fun_milliGs,[1,1,1,0.1,-0.1,0.1,-0.1,0.1,-0.1,0.2,0.2,0.2])
# args = opt.least_squares(lsq_fun_Gs,[1,1,1,0.1,-0.1,0.1,-0.1,0.1,-0.1,0.2,0.2,0.2])
args[0]
Sx, Sy, Sz, a1, a2, a3, a4, a5, a6, b1, b2, b3 = args[0]
output_array = lsq_fun_milliGs(filtered_chunks[time], Sx, Sy, Sz, a1, a2, a3, a4, a5, a6, b1, b2, b3)
output_array = lsq_fun_Gs(filtered_chunks[time], Sx, Sy, Sz, a1, a2, a3, a4, a5, a6, b1, b2, b3)
# output_array1[i] = filtered_chunks["vA"][i]

(filtered_chunks["vA"]-1).var()
(filtered_chunks["vA"]-1).median()

plt.scatter(filtered_chunks[time],filtered_chunks["vA"],s=0.5)
plt.scatter(filtered_chunks[time],output_array,s=0.5)
# plt.scatter(filtered_chunks[time],output_array1,s=0.5)
# np.sqrt(((output_array - output_array1)**2).mean())
plt.legend(["before", "after"])
plt.show()
x = 0
plt.scatter(df[time], df[xA], s=0.5)
plt.scatter(df[time], df[yA], s=0.5)
plt.scatter(df[time], df[zA], s=0.5)
plt.scatter(df[time], df["vA"], s=0.5)
plt.show()
# xx = filtered_chunks[time]
# yy = filtered_chunks.select([xA,yA,zA])
# xx.shape
# yy.to_numpy()

# yy.to_numpy().T
# np.array([[1],[2],[3]]).shape

#xx = ret.to_array()
# print("pre ret")
# print(type(ret))
# print(f"ret={ret} shape={ret.shape} type={ret.dtype}")
#opt.curve_fit(lsq_fun,filtered_chunks[time],filtered_chunks.select([xA, yA, zA]).to_numpy().T, [1,1,1,0,0,0,0,0,0,0,0,0])

print(type(filtered_chunks.select([xA, yA, zA]).to_numpy()[0]))


transform = fftpack.fft(df["vA"].to_numpy())
cut = 500
# transform[cut:transform.shape[0]-cut] = np.zeros_like(transform[cut:transform.shape[0]-cut])
transform[cut:transform.shape[0]//2-cut] = np.zeros_like(transform[cut:transform.shape[0]//2-cut])
transform[transform.shape[0]//2+cut:transform.shape[0]-cut] = np.zeros_like(transform[transform.shape[0]//2+cut:transform.shape[0]-cut])
# plt.plot(fftpack.fftshift(transform))
invTransform = fftpack.ifft(transform)
invTransform.real
invTransform
plt.plot(df["vA"])
plt.plot(invTransform)
plt.show()


plt.plot(in_place_derive(df["vA"]))
df.insert_column(-1, pl.Series(in_place_derive(df["vA"])[:,0]).alias("dvA"))
dfFilter1 = df.filter(pl.col("dvA").abs() < 50).filter((pl.col("vA") - 1000).abs() < 50)
dfFilter1 = df.filter((pl.col("vA") - 1000).abs() < 50)
plt.scatter(dfFilter1["time"],dfFilter1["vA"], s=0.2)
plt.scatter(dfFilter1["time"],dfFilter1["dvA"], s=0.2)
plt.show()
transform = fftpack.fft(dfFilter1["vA"].to_numpy())

transform[5:5303] = np.zeros_like[5:5303]
plt.plot(transform)
transform[:1000] = np.zeros_like(transform[:1000])
iTransform = fftpack.ifft(transform)
plt.scatter(dfFilter1["time"],iTransform, s = 0.4)
plt.scatter(dfFilter1["time"],dfFilter1["vA"], s = 0.4)
plt.scatter(df["time"],df["vA"], s = 0.4)
plt.legend(["fft", "not fft"])
plt.show()


df[time]
plt.plot(df["time"], df["xAcc"])
plt.plot(df["time"], df["yAcc"])
plt.plot(df["time"], df["zAcc"])
plt.show()

# df = df[40000:] #This is for 2024-12-02-Part1 because the first bit is flat and the IMU does weird stuff in xA
# df.write_parquet("Parquet/2024-12-02-Part1-100Hz.pq")

df.columns

df[xA_mps]

df[yA_mps].min()
plt.plot(df[xA_mps])
plt.plot(df[yA_mps])
plt.plot(df[zA_mps])
plt.plot(df[speed_mps])

cs = CubicSpline(df[index],df[rpm]*11/40*0.2*2*np.pi/60)
plt.plot(in_place_derive(np.asarray(list(map(cs, df[index])))))
plt.plot(np.asarray(list(map(cs, df[index]))))

plt.plot(in_place_derive(df[rpm]*11/40*0.2*2*np.pi/60))
plt.plot(df[rpm]/(40/11)*.2*2*np.pi/60)


## FFT transform graph is 
## frequency of data collected / number of samples (aka 1 / length of recording) 
## multiplied from -k to k (if you do fftshift). 
transform = fftpack.fft((df[rpm]*11/40*0.2*2*np.pi/60)[115000:128000].to_numpy())
transform[200500:] = np.zeros_like(transform[200500:])
plt.plot(np.arange(transform.shape[0]//-2, transform.shape[0]//2)*(100/transform.shape[0]), fftpack.fftshift(transform)) # Flips it so that 0 is in the middle instead of at the ends
plt.plot(fftpack.fftshift(transform))
plt.plot((transform))
transform[1100:11800] = np.zeros_like(transform[1100:11800])
invTransform = fftpack.ifft(transform)
plt.plot(df[time][115000:128000],invTransform)
plt.plot(df[time][115000:128000],(df[rpm]*11/40*0.2*2*np.pi/60)[115000:128000])
df.columns
plt.show()

df[xA_mps].max()

df.columns

time1 = 1400
time2 = 1710

df.columns
short = pl.DataFrame(df.filter(pl.col("Seconds") >= time1).filter(pl.col("Seconds") <= time2))
short = df

(df[speed].to_numpy() - in_place_derive(in_place_integrate(df[speed]))[:,0]).mean()
(df[speed].to_numpy() - in_place_integrate(in_place_derive(df[speed]))[:,0]).mean()

df.filter(abs(np.sqrt(pl.col(xA)**2 + pl.col(yA)**2 + pl.col(zA)**2) - 1) < 0.1 )[xA,yA,zA].shape

dataFrame = pl.Series(in_place_integrate(df[xA])[:,0])
dataFrame.columns = ["xV"]
df.insert_column(0, dataFrame["xV"])
df.drop("xV")


short
short = df.filter(pl.col(lat) != 0).filter(pl.col(long) != 0)

plt.plot(df["xV"]*10000 - df[speed])
plt.show()
df[speed].mean()


# plt.plot(df["xV"])
# plt.plot(df[xG])
# plt.plot(df[":LapTime"]/1000)

df.columns

# plt.plot(df[rpm]*rpm_to_mph)
# plt.legend(["gps speed", "rpm speed"])

xA_smooth_fun = itp.CubicSpline(df[time], df[xA])
xA_smooth = itp.splrep(df[time], df[xA])
xA_smooth = np.asarray(list(map(xA_smooth_fun, df[time])))
plt.plot(xA_smooth[1])
plt.plot(xA_smooth[1][1:-3] - df[xA].to_numpy())
plt.show()

xAccBias = df[0:50000][xA].mean()
yAccBias = df[0:50000][yA].mean()
zAccBias = df[0:50000][zA].mean()
df = df.drop(["xV","yV","zV"])
df.insert_column(df.get_column_index(xA), pl.Series(in_place_integrate((df.select(xA) - xAccBias)[xA])[:,0]).alias("xV"))
df.insert_column(df.get_column_index(yA), (df.select(yA) - yAccBias)[yA].alias("yV"))
df.insert_column(df.get_column_index(zA), (df.select(zA) - zAccBias)[zA].alias("zV"))

df.columns

plt.plot(in_place_integrate(df[tsC])/3600)
plt.plot(in_place_integrate(dfP2[tsC])/3600)
plt.show()

fig1 = plt.figure()
ax1 = fig1.add_subplot(1,3,1)
lines = colored_line(short[lat], short[long], short[rpm], ax1, linewidth=1, cmap="plasma")
fig1.colorbar(lines)  # add a color legend
ax1.axis('scaled')
ax1.set_title("RPM")
plt.show()
df[121856][lat]
df[121856][long]

ax1 = fig1.add_subplot(1,3,2)
lines = colored_line(short[lat], short[long], short[speed], ax1, linewidth=1, cmap="plasma")
fig1.colorbar(lines)  # add a color legend
ax1.axis('scaled')
ax1.set_title("GPS Speed")

ax1 = fig1.add_subplot(1,3,3)
lines = colored_line(short[lat], short[long], short["xV"], ax1, linewidth=1, cmap="inferno")
fig1.colorbar(lines)  # add a color legend
ax1.axis('scaled')
ax1.set_title("Moving")

plt.show()

short[frT] - short[flT]



df.filter(pl.col(tsC) != 0)[tsC].mean() #Average Current Draw during autocross runs
df.filter(pl.col(tsC) != 0)[tsC].mean() #Average Current Draw during autocross runs
dfP2.filter(pl.col(tsC) != 0)[tsC].mean() #Average Current Draw during endurance runs




plt.plot(short[yA],short[xA])
plt.show()

## Suspension travel vs position graphs
fig1 = plt.figure()
ax1 = fig1.add_subplot(2,1,1)
lines = colored_line(short[long], short[lat], short[frT] - short[brT], ax1, linewidth=1, cmap="plasma")
fig1.colorbar(lines)  # add a color legend
ax1.axis('scaled')
ax1.set_title("Right Suspension Travel F-B")

ax1 = fig1.add_subplot(2,1,2)
lines = colored_line(short[long], short[lat], short[flT] - short[blT], ax1, linewidth=1, cmap="plasma")
fig1.colorbar(lines)  # add a color legend
ax1.axis('scaled')
ax1.set_title("Left Suspension Travel F-B")

plt.show()

## Roll and Dive rate vs. position graphs

fig1 = plt.figure()
ax1 = fig1.add_subplot(2,1,1)
lines = colored_line(short[long], short[lat], short[xG], ax1, linewidth=1, cmap="plasma")
fig1.colorbar(lines)  # add a color legend
ax1.axis('scaled')
ax1.set_title("Roll Rate")

ax1 = fig1.add_subplot(2,1,2)
lines = colored_line(short[long], short[lat], short[yG], ax1, linewidth=1, cmap="plasma")
fig1.colorbar(lines)  # add a color legend
ax1.axis('scaled')
ax1.set_title("Dive Rate")

plt.show()

filter.Q

# df = df[40000:]
plt.plot(in_place_integrate(df[xA_mps] - df[xA_mps][25000:50000].mean() - in_place_derive(df[xA_mps][25000:50000]).mean()))
plt.plot(df.filter(pl.col(speed_mps) > 0.5)[time]*100-40000,df.filter(pl.col(speed_mps) > 0.5)[speed_mps])
plt.plot(in_place_derive(df[xA_mps][30000:50000]))
# plt.plot(in_place_integrate_biased(df[xA_mps], (df[xA_mps][25000:50000]).mean() ,in_place_derive(df[xA_mps][25000:50000]).mean()))
plt.show()

plt.plot(df["TS_Current"])
plt.title("Warm Up/Autocross 2024-12-2")
plt.legend(["Current (A)"])
plt.plot(dfP2["TS_Current"])
plt.title("Endurance 2024-12-2")
plt.legend(["Current (A)"])
plt.show()

plt.plot(in_place_derive(df[xA_mps][25000:50000]))
plt.plot((df[xA_mps][25000:50000]))
plt.show()
df[speed_mps][50220]

lsq_fun = lambda x, a, b, c, d : a + b*x + c*x**2 + d*x**3

df_run1 = df[75000:110000]
df_callibrate = df[70000:85000].extend(df[101000:110000])
# opt.least_squares(lsq_fun, df.filter(pl.col(rpm) == 0)[time]*100)
# opt.curve_fit(lsq_fun, df.filter(pl.col(rpm) == 0)[index], df.filter(pl.col(rpm) == 0)[xA_mps], [0.5,0.5,0.5])
args_acc = opt.curve_fit(lsq_fun, df[70000:85000][index].append(df[101000:110000][index]),df[70000:85000][xA_mps].append(df[101000:110000][xA_mps]), [-1,1e-04,1e-09, 1e-12])[0]
args_gyroX = opt.curve_fit(lsq_fun, df[70000:85000][index].append(df[101000:110000][index]),df[70000:85000][xG].append(df[101000:110000][xG]), [-1,1e-04,1e-09, 1e-12])[0]
args_gyroY = opt.curve_fit(lsq_fun, df[70000:85000][index].append(df[101000:110000][index]),df[70000:85000][yG].append(df[101000:110000][yG]), [-1,1e-04,1e-09, 1e-12])[0]
args_gyroZ = opt.curve_fit(lsq_fun, df[70000:85000][index].append(df[101000:110000][index]),df[70000:85000][zG].append(df[101000:110000][zG]), [-1,1e-04,1e-09, 1e-12])[0]
args_acc
args_gyroZ

df_callibrate[xA].var()
plt.plot(df["VDM_GPS_SPEED"])

#pretty good but slight upward drift that can be tracked down to changing orientation slightly.
plt.plot(df_run1[index],df_run1[xA_mps].to_numpy() - (args_acc[0] + (args_acc[1] * df_run1[index]) + (args_acc[2] * df_run1[index].pow(2)) + (args_acc[3] * df_run1[index].pow(3))))
plt.plot(df_run1[index],(args_acc[0] + (args_acc[1] * df_run1[index]) + (args_acc[2] * df_run1[index].pow(2)) + (args_acc[3] * df_run1[index].pow(3))))
plt.plot(df_run1[index],df_run1[xA_mps])
plt.show()

# plt.plot(df_run1_callibration[xG])


plt.plot(in_place_integrate(df_run1[xG].to_numpy() - (args_gyroX[0] + (args_gyroX[1] * df_run1[index]) + (args_gyroX[2] * df_run1[index].pow(3)) + (args_gyroX[3] * df_run1[index].pow(5)))))
plt.plot(df_run1[xG])
plt.plot((df_run1[xG].to_numpy() - (args_gyroX[0] + (args_gyroX[1] * df_run1[index]) + (args_gyroX[2] * df_run1[index].pow(3)) + (args_gyroX[3] * df_run1[index].pow(5)))))
plt.plot(df_run1[yG])
plt.plot((df_run1[yG].to_numpy() - (args_gyroY[0] + (args_gyroY[1] * df_run1[index]) + (args_gyroY[2] * df_run1[index].pow(3)) + (args_gyroY[3] * df_run1[index].pow(5)))))
plt.plot(df_run1[zG])
plt.plot((df_run1[zG].to_numpy() - (args_gyroZ[0] + (args_gyroZ[1] * df_run1[index]) + (args_gyroZ[2] * df_run1[index].pow(3)) + (args_gyroZ[3] * df_run1[index].pow(5)))))
plt.show()
# plt.plot(in_place_integrate(df_run1[xA_mps])[:,0] - in_place_integrate((-1.71560913e+01 + (3.22428556e-04 * df_run1[index]) + (-1.38439479e-09 * df_run1[index].pow(2))))[:,0])
#integrated still has a lot of drift
plt.plot(in_place_integrate(df_run1[xA_mps].to_numpy() - (args_acc[0] + (args_acc[1] * df_run1[index]) + (args_acc[2] * df_run1[index].pow(3)) + (args_acc[3] * df_run1[index].pow(5)))))
plt.show()
plt.scatter(df.filter(pl.col(rpm) == 0)[index], df.filter(pl.col(rpm) == 0)[xA_mps])
plt.plot(df.filter(pl.col(rpm) != 0)[index], df.filter(pl.col(rpm) != 0)[speed_mps])
plt.show()

# plt.plot(df[rpm][70000:75000])
# plt.plot(df[speed][70000:75000])
# plt.plot(df[xA_mps][10000:75000])
# plt.plot(df[yA_mps][10000:75000])
# plt.plot(df[zA_mps][10000:75000])
# plt.plot(df.filter(pl.col(xA_mps) != 0)[xA_mps])
# plt.plot(df.filter(pl.col(xA_mps) != 0)[yA_mps])
# plt.plot(df.filter(pl.col(xA_mps) != 0)[zA_mps])
# plt.plot(df[course])
# plt.show()

df.filter(pl.col(rpm) == 0)[index]

filter = mg.Madgwick(np.asarray(df[xG,yG,zG]),np.asarray(df[xA,yA,zA]),frequency=100.0)
qtA = qt.QuaternionArray(filter.Q)
a = np.zeros((qtA.shape[0],1))
a[:,0] = qtA[:,0]
a
plt.plot(a * qtA[:,1:])
plt.show()
qtA
qtA.shape
df.shape
qtA
short[xA,yA,zA]

# plt.plot(in_place_integrate_filtered(df[70000:75000], xA_mps, [xA_mps, yA_mps, zA_mps]))
# plt.show()
# df[xA_mps][70000:75000].mean()
# plt.plot(in_place_integrate_filtered(df[70000:110000], xA_mps, [xA_mps, yA_mps, zA_mps]))
plt.plot(df[speed_mps][70000:110000])
plt.plot(in_place_integrate(df[xA_mps][70000:110000] - 0.0159))
plt.show()
vector = df.select([xA_mps, yA_mps, zA_mps])
vector[[xA_mps, yA_mps, zA_mps][0]][0]

args_acc = opt.curve_fit(lsq_fun, df[70000:85000][index].append(df[101000:110000][index]),df[70000:85000][xA_mps].append(df[101000:110000][xA_mps]), [-1,1e-04,1e-09, 1e-12])[0]
args_acc = opt.curve_fit(lsq_fun, df[105500:109500][index].append(df[172000:182000][index]),df[105500:109500][xA_mps].append(df[172000:182000][xA_mps]), [-1,1e-04,1e-09, 1e-12], method = 'lm')[0]

a = df[105500:182000][index]
b = df[105500:182000].select(speed_mps)
c = df[105500:182000][xG]
d = df[105500:182000][course]
args_acc
b.shape
in_place_integrate(df[70000:110000][xA_mps] - (args_acc[0] + a*args_acc[1]+args_acc[2]*a.pow(2)+args_acc[3]*a.pow(3))).shape
plt.plot(df.filter(pl.col(rpm) == 0)[xA_mps])
plt.plot(df[70000:80000][index].append(df[101000:110000][index]),df[70000:80000][xA_mps].append(df[101000:110000][xA_mps]))
plt.plot(a, -in_place_integrate(df[105500:182000][xA_mps] - (args_acc[0] + a*args_acc[1]+args_acc[2]*a.pow(2)+args_acc[3]*a.pow(3))))
# plt.plot(a, -in_place_integrate(df[70000:110000][xG] - (args_gyroZ[0] + a*args_gyroZ[1]+args_gyroZ[2]*a.pow(2)+args_gyroZ[3]*a.pow(3))))
# plt.plot(a, df[70000:110000][zG] - (args_gyroZ[0] + a*args_gyroZ[1]+args_gyroZ[2]*a.pow(2)+args_gyroZ[3]*a.pow(3)))
# plt.plot(df[zG])
plt.plot(a, b.to_numpy() + in_place_integrate(df[105500:182000][xA_mps] - (args_acc[0] + a*args_acc[1]+args_acc[2]*a.pow(2)+args_acc[3]*a.pow(3))))
plt.plot(a,b)
# plt.plot(a,d)
# plt.plot(a,in_place_integrate(c))
plt.show()
args_acc
plt.plot(df[xA])
df[xA]
plt.show()
lsq_fun_with_temp = lambda x, a, b, c, d, e, f, g : a + b*x[time] + c*x[time]**2 + d*x[time]**3 + e*x[imuT] + f*x[imuT]**2 + g*x[imuT]**3
lsq_fun_without_temp = lambda x, a, b, c, d : a + b*x[time] + c*x[time]**2 + d*x[time]**3
stationary_time = (df[:55]).extend(df[1250:1425]).extend(df[1850:2250])

args_acc = opt.curve_fit(lsq_fun, df[70000:85000][index].append(df[101000:110000][index]),df[70000:85000][xA_mps].append(df[101000:110000][xA_mps]), [-1,1e-04,1e-09, 1e-12])[0]
args_gyroX = opt.curve_fit(lsq_fun_with_temp, stationary_time,(df[:55][xG]).append(df[1250:1425][xG]).append(df[1850:2250][xG]), [-1,1e-04,1e-09, 1e-12,1e+7,1e+2, 1e+2])[0]
args_notemp_gyroX = opt.curve_fit(lsq_fun_without_temp, stationary_time,(df[:55][xG]).append(df[1250:1425][xG]).append(df[1850:2250][xG]), [-1,1e-04,1e-09, 1e-12])[0]
args_gyroY = opt.curve_fit(lsq_fun, df[70000:85000][index].append(df[101000:110000][index]),df[70000:85000][yG].append(df[101000:110000][yG]), [-1,1e-04,1e-09, 1e-12])[0]
args_gyroZ = opt.curve_fit(lsq_fun_with_temp, df[70000:85000][index].append(df[101000:110000][index]),df[70000:85000][zG].append(df[101000:110000][zG]), [-1,1e-04,1e-09, 1e-12])[0]
df[:55][time,imuT]
# df[1850:2250][time].append(df[:55][time]).append(df[1250:1425][time])
plt.plot(stationary_time[time], stationary_time[xG])
args_gyroX
adjustment = args_gyroX[0] + stationary_time[time]*args_gyroX[1] + args_gyroX[2]*stationary_time[time]**2 + args_gyroX[3]*stationary_time[time]**3 + args_gyroX[4]*stationary_time[imuT] + args_gyroX[5]*stationary_time[imuT]**2 + args_gyroX[6]*stationary_time[imuT]**3
adjustment_notemp = args_notemp_gyroX[0] + stationary_time[time]*args_notemp_gyroX[1] + args_notemp_gyroX[2]*stationary_time[time]**2 + args_notemp_gyroX[3]*stationary_time[time]**3
plt.scatter(stationary_time[time], adjustment)
plt.scatter(stationary_time[time], adjustment_notemp)
plt.plot(stationary_time[time], stationary_time[xG] - adjustment)
plt.plot(stationary_time[time], stationary_time[imuT])

plt.plot(df[xA])
plt.show()


'''

Polars notes I find useful:
df = DataFrame
column = (mostly) Series
Create one with pl.DataFrame(array_like_object)
or pl.from_csv / .from_parquet
.alias() on a series/col to change the name
can do stuff like .mean(), .max() etc.
df[0:10] returns first 10 rows
df.filter(pl.col("column_name") == 0) returns all rows where that is true
df.filter().filter() to do it more than once. Stuff like 1 < x < 3 doesn't work. Have to do each part ( 1 < x ) and ( x < 3 ) separately
df[list_of_strings] to get those columns. Can't use this for complex operations. Use df.select(["column_0", "column_1"]) instead

General Notes:
- Solve_ivp method from scipy (uses RK45 - Runge Kutta 45) tends to be worse than a riemann sum. 
I believe this is because you have to force the method to use a small step size 
(was really slow at 4 and I'm worried about going smaller)
Default, it skips ahead logarithmically. (eg. 0, 0.0001, 0.001, 0.01, 0.1, 1, 10, 100, 1000)
Given our data is very linear, this doesn't work very well
You can set step sizes with max_step and min_step arguments during definition

- Use RPM to determine when the car is stationary or not. I used this a bit to 0 the speed calculated from the IMU.
Using the GPS does work but that was not worth the effort. Code for that is below

- I'm using a riemann sum for numerical integration and the 5-point stencil for numerical derivation

- Colored line is a function we found online that supports a third axis and generates line segments of the appropriate color.
The default method pcolormesh from matplotlib tries to take the third axis and make a mesh out of it leading to 
n^2 points. For a dataset that is hundreds of thousands of lines long, this tries to allocate more ram than you probably
have on your computer. Use colored_line instead!



The following test showed that a numerically integrated and derived function using the methods 
defined in integralsAndDerivatives.py, no matter the order, results in a mean error
on the order of 1E-8!

plt.plot(df[speed])
plt.plot(in_place_integrate(in_place_derive(df[speed])))
plt.plot(df[speed].to_numpy() - in_place_integrate(in_place_derive(df[speed]))[:,0])
(df[speed].to_numpy() - in_place_integrate(in_place_derive(df[speed]))[:,0]).mean() <---

plt.plot(in_place_derive(in_place_integrate(df[speed])))
plt.plot(df[speed].to_numpy() - in_place_derive(in_place_integrate(df[speed]))[:,0])
(df[speed].to_numpy() - in_place_derive(in_place_integrate(df[speed]))[:,0]).mean() <---
plt.show()



For below -> 
# is thing I tried, parts if they're related. 
--- is why it didnt work 
~ is something that is worth trying in the future

Stuff I've tried for the IMU already that hasn't worked
    1. Just integrating the xAxis acceleration to get speed 
    --- Didn't work because some component of it is probably gravity
    and it seems like there's drift
    2. Using a set of 3 angles that are updated each step from the roll, yaw and dive rates (riemann sum) 
    --- Didn't work because there's drift in the gyroscopes that is not trivial to account for
~ May be worth trying to fit a constant and linear bias term to the expression (eg. true yaw = measured yaw + c1 + c2t) 
Should also account for temperature in there (standard for mems sensors)
    2a. Tried just finding a stationary point, measuring the gyroscope rate there and using that as an offset for the 
    rest of the run 
    --- Didn't work because the drift while stationary is different throughout the run. Could not confirm if
    this was related to orientation, temperature, or any other factor that comes into play.
    3. Used curve fitting to match a function with degree 0,1,3,5 terms to a stationary x acceleration measurement
    to get coefficients for a drift function. Worked pretty well but there was still some slightly upward drift that
    I tracked down to the orientation of the car. So it won't work until I get attitude down correctly.
    3a. Tried to use Madgwick again to get Attitudes. Don't think it has a way to account
    for error in sensor measurements so going to try that first.
    3b. Fit curves to gyro stuff but don't have a real world example to compare it to. I think what I need to do is 
    get gyro data with the IMU pointed at 40-50 attitudes with small statinoary bits between each attitude. Then
    use the method on page 451 of this: https://www.imeko.org/publications/tc4-2014/IMEKO-TC4-2014-429.pdf
    3c. Tried fitting the curve to different pieces of the same data range (70k - 110k). Also tried integrating at
    a later point to avoid the little bump in the data but that didn't help. Doesnt' seem like the interpolation over
    the actual moving part works very well (doesn't have training data so makes sense)
~going to try on a longer stretch of data (the second run / endurance) where the car doesn't turn off as often

    4. Fitting constant, linear, quadratic, and cubic terms to xAcc as a bias worked well.
    I did this, subtracted it from the data, and integrated. The run drifted by about 10 m/s
    relative to the gps speed over a half hour run. It is not a linear component I can
    easily correct unfortunately. Trying to fit based on the method described partially
    in 3b. and partially here: https://www.vectornav.com/resources/inertial-navigation-primer/specifications--and--error-budgets/specs-imucal


'''


##Smoothing data from an IMU that outputs irregularly
# df.columns = ['time', 'voltage', 'state_of_charge', 
#               'charge_rate', 
#               'xA', 'yA', 'zA', 
#               'xG', 'yG', 'zG',
#               'IMU_temp', 
#               'xBField', 'yBField', 'zBField',
#               'Mag_temp']
# df[time]

# fit = CubicSpline(df[time] - df[time][0], df.select(['voltage', 'state_of_charge', 
#               'charge_rate', 
#               'xA', 'yA', 'zA', 
#               'xG', 'yG', 'zG',
#               'IMU_temp', 
#               'xBField', 'yBField', 'zBField',
#               'Mag_temp']))

# new_time = np.arange(0, math.floor((df[time][df.shape[0]-1] - df[time][0])/50)*50, 20)
# new_time.shape[0]
# new = np.zeros((new_time.shape[0], 15))
# new[:,0] = new_time
# new[:,1:] = np.asarray(list(map(fit, new_time)))
# plt.plot(df[time], df['yA'])
# plt.plot(new[:,0] + df[time][0], new[:, 5])
# plt.show()

# df1 = pl.DataFrame(new)
# df1.columns = ['time', 'voltage', 'state_of_charge', 
#               'charge_rate', 
#               'xA', 'yA', 'zA', 
#               'xG', 'yG', 'zG',
#               'IMU_temp', 
#               'xBField', 'yBField', 'zBField',
#               'Mag_temp']
# df1.write_parquet("Nathaniel_IMU_Data\Rectangle2x_Smoothed.pq")



#Don't think this works. It was created on the idea that the attitude was just an angle but it is in fact a vector.
# attAcc = np.concatenate((np.asarray(qtA.v,dtype="float32"),np.asarray(short[xA,yA,zA])),axis=1) #Attitude Acceleration array
# g = filter.acc[:,2].mean()

# attAcc.shape[0]
# storage = np.zeros((attAcc.shape[0],3))
# new_col = pl.DataFrame(gpsStationarySearch(df)) DO NOT USE JUST USE RPM INSTEAD
# new_col.columns = ["Stationary"]
# df.insert_column(0,new_col.columns["Stationary"])
# plt.plot(new_col)
# plt.show()
# df.drop("Stationary")

# dataFrame = short

#Again, doesn't work in this case.
# for i in range(attAcc.shape[0]):
#     row = np.zeros((6,1))
#     row[:,0] = attAcc[i,:]
#     xAngle = row[0,0]
#     yAngle = row[1,0]
#     zAngle = row[2,0]
#     vAcc = np.zeros((3,1))
#     vAcc[:,0] = row[3:,0]
#     vAcc[2,0] -= g

#     xMat = np.matrix([[1, 0, 0],[0, np.cos(xAngle), -1*np.sin(xAngle)],[0, np.sin(xAngle), np.cos(xAngle)]]) # X rotation matrix
#     yMat = np.matrix([[np.cos(yAngle), 0, np.sin(yAngle)],[0, 1, 0],[-np.sin(yAngle), 0, np.cos(yAngle)]]) # Y rotation matrix
#     zMat = np.matrix([[np.cos(zAngle), -1*np.sin(zAngle), 0],[np.sin(zAngle), np.cos(zAngle), 0],[0, 0, 1]]) # Z rotation matrix
#     fullRotation = zMat * yMat * xMat #Product in this order creates a single rotation matrix that is based on the current angles

#     V = fullRotation*(vAcc)
#     storage[i] = V.T

# def rate (time, y):
#     # print(time)
#     return storage[int(time)]

# def vRate (time, y):
#     # print(time)
#     return v[int(time)]

# v1 = in_place_integrate(storage)
# d1 = in_place_integrate(v1)
# plt.plot(v1[:,0], v1[:,1])
# plt.show()

# for i in range(v1.shape[0]):
#     if i == 0:
#         print("set prev to not0")
#         prev = "not0"
#     # print(f"rpm is {df[i][rpm][0]}")
#     if (df[i][rpm][0] == 0.0):
#         if (prev == "not0"):
#             print(f"adjusting {i} onward by {v1[i]}")
#             v1[i:] -= v1[i]
#         prev = "0"
#     else:
#         prev = "not0"

# fig = plt.figure()
# ax = fig.add_subplot(1,1,1)
# ax.plot(d[:,0],d[:,1],d[:,2])
# ax.plot(storage[:,0],storage[:,1],storage[:,2])
# plt.plot(v1[:,0],v1[:,1])
# plt.plot(d1[:,0],d1[:,1])
# plt.plot(storage[:,1],storage[:,2])
# plt.show()
# qtA.v.shape
# plt.plot(np.arange(1,v1.shape[0]),np.sqrt(v1[:,0]**2 + v1[:,1]**2)[1:])
# plt.plot(np.arange(0,v1.shape[0]),df[rpm]*0.0139145367211)
# plt.plot(np.arange(0,v1.shape[0]),df[speed])
# plt.plot(np.arange(0,v1.shape[0]),qtA.v[:,0])

# plt.show()

# np.sqrt(v1[:,0]**2 + v1[:,1]**2).shape
# v1.shape

# def not0 (number): #For making sure theres no divide by zero errors in the check for if the gps is stationary. Do not use elsewhere
#     if number != 0:
#         return number
#     else:
#         return 0.00000001


#USE RPM != 0 instead!!
# def gpsStationarySearch(dataFrame): #returns a column of values that represents if the car is stationary at that time or not
# error = 0.000000001
# window_size = 20
# rows = dataFrame.shape[0]
# t = "Seconds"
# t0 = dataFrame[0][t][0]
# tf = dataFrame[rows - 1][t][0]
# new_col = np.zeros((rows, 1))
# for i in range(window_size):
#     new_col[i] = 1 if abs((dataFrame[0:window_size][lat].mean() / not0(dataFrame[i][lat][0])) - (dataFrame[0:window_size][long].mean() / not0(dataFrame[i][long][0]))) < error else 0
# for i in range(rows-window_size,rows):
#     new_col[i] = 1 if abs((dataFrame[rows-window_size:][lat].mean() / not0(dataFrame[i][lat][0])) - (dataFrame[rows-window_size:][long].mean() / not0(dataFrame[i][long][0]))) < error else 0
# for i in range(window_size,rows-window_size,1):
#     # print(f"lat mean = {(dataFrame[i-window_size:i+window_size][lat].mean() / dataFrame[i][lat][0])}\n long window = {(dataFrame[i-window_size:i+window_size][long].mean() / dataFrame[i][long][0])}")
#     new_col[i] = 1 if abs((dataFrame[i-window_size:i+window_size][lat].mean() / not0(dataFrame[i][lat][0])) - (dataFrame[i-window_size:i+window_size][long].mean() / not0(dataFrame[i][long][0]))) < error else 0

# return new_col
# print(out_df.columns)


# time_range = (time1,time2)
# step_size = 0.01
# t_eval = [x/100 for x in range(time_range[0]*100, time_range[1]*100, 1)]
# v = solve_ivp(rate, time_range, [0,0,0], t_eval=t_eval).y.T

# v.shape
# df.shape

# d = solve_ivp(vRate, time_range, [0,0,0], t_eval=t_eval).y.T


# def rate (t, y):
#     return df[speed][int(t)]

# def d_rate(t, y):
#     return derivative[int(t)]

# derivative = in_place_derive(df[speed])


#These tend to work worse than a riemann sum. Not really sure why (My main guess is that these methods tend to skip data points so they wander away for really messy data like ours)
# i_preD = solve_ivp(rate, (0,df.shape[0]-1), [0], t_eval= np.arange(0,df.shape[0]),max_step = 20)
# i_postD = solve_ivp(d_rate, (0,df.shape[0]-1), [0], t_eval= np.arange(0,df.shape[0]),max_step = 4)


# plt.plot(i_preD.y.T/100 - in_place_integrate(df[speed]))
# plt.plot(in_place_integrate(df[speed]))
# plt.plot(i_preD.y.T/100)
# plt.plot(df[speed])
# plt.show()


# plt.plot(df[speed].to_numpy() - i_postD.y.T[:,0])
# plt.plot(df[speed])
# plt.plot(derivative)
# plt.plot(i_postD.y.T[:,0]/100 - df[speed].to_numpy())
# plt.plot(df[speed].to_numpy() - in_place_derive(i_preD.y.T/100)[:,0])
# plt.legend(["D then I", "I then D"])
# plt.show()

# def lsq_fun_applied (x, y, z, array): 
#     Sx, Sy, Sz, a1, a2, a3, a4, a5, a6, b1, b2, b3 = array
#     # y = (x[0],x[1],x[2])
#     # print(f"y = {y}")
#     # print(np.matmul(np.matmul(np.array([[Sx, 0, 0],[0, Sy, 0],[0,0,Sz]]),np.array([[1, a1, a2],[a3, 1, a4],[a5, a6, 1]])),np.array([[y[0,0]-b1],[y[1,0]-b2],[y[2,0]-b3]])))
#     ret = np.matmul(np.matmul(np.array([[Sx, 0, 0],[0, Sy, 0],[0,0,Sz]]),np.array([[1, a1, a2],[a3, 1, a4],[a5, a6, 1]])),np.array([[x-(b1*100)],[y-(b2*100)],[z-(b3*100)]]))[:,0]
#     return np.sqrt(ret[0]**2 + ret[1]**2 + ret[2]**2)