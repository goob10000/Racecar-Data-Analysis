# import serial
# import pandas as pd
# import functools
# import operator
from matplotlib import pyplot as plt
from scipy.interpolate import CubicSpline
from scipy.integrate import RK45, solve_ivp
import numpy as np
from threeDplot import *
import polars as pl

'''
Input data: 
- x, y, z acc
- x, y, z gyro (acc as well)
- x, y, z earth magnetic field (probably)

Resolve new gyroscopic positioning?
- New Gyro position and then new acc position? Or the other way? Does it matter?

Resolving Gryoscopic state relative to north, east, and down:
- Take into account drift due to temperature of sensor
       - Is it constant in the same direction depending on temp? Is it skewed toward the rotational direction?
       - May be worth performing some base tests to understand the specific characteristics
Using state, determine what is "down" (perpendicular to forward I guess?)
- Can determine absolute velocity (includes forward acceleration as well as falling and sliding) and forward acceleration
- Discrepancy is drift/falling!

Resolving new position, current velocity, and currect acceleration relative to "down":
- Prepare to take into account motor controller speed
- 


'''
df = pl.read_csv("realTrip1.txt") #Read in trip csv
# df["General.Time (millis)"]
df
df.columns = ['time', 'voltage', 'state_of_charge', 'charge_rate', 'xAcc', 'yAcc', 'zAcc', 'xGyro',
       'yGyro', 'zGyro',
       'ISM330.temp', 'xBField',
       'yBField', 'zBField',
       'MMC5983.temp'] #Set columns to correct names
df[df["xAcc"]>100] #Return lines with xAcc greater than 100
acc = df[["time","xAcc","yAcc","zAcc"]] #Frame with only the named columns
gyro = df[["time","xGyro","yGyro","zGyro"]] #''
# acc1 = df.iloc[0:100][["time","xAcc","yAcc","zAcc"]] #Frame with only lines 1-100 for named columns
# df.loc[1:100]
# df.loc[23000]
# acc

def cleanData2Dx3(dataIn):
    '''
    Takes a dataframe with irregular data points (ex. t = 0, 17, 50, 60, 90)
    and fits a spline with regular interval data points to the set.
    Meant for frame set up as [time, x, y, z]
    '''
    data = dataIn
#     print(data)
    data.columns = ["time", "x", "y", "z"]
    t = data[["time"]]
#     print(t)
    t = t.to_numpy().flatten()
#     print (t)
    xyz = data[["x","y","z"]]
#     xyz.to_numpy()

    return CubicSpline(t, xyz)

# testframe = pl.read_csv('test.csv')
# testframe
# csTest = cleanData2Dx3(testframe)
# csTest(1)
# dfcsTest = pl.DataFrame(np.asarray([[x] + list(csTest(x)) for x in range(0,51,1)]), columns=['time', 'x', 'y', 'z'])
# dfcsTest
# list(np.asarray([1,2,3]))

# t0 = df[["time"]].iloc[1,0]
# t1 = df[["time"]].iloc[-1,0]
# t1
cs = cleanData2Dx3(acc)

t0 = df[["time"]].iloc[4000,0]
t1 = df[["time"]].iloc[6000,0]

timeList = np.asarray([[x for x in range(t0,t1,25)]])
xyzList = np.asarray(list(map(cs, timeList)))[0,:,:]
timeList = timeList[0,:]
# timeList
# xyzList
# timeList.shape[0]
appendTest = np.zeros((timeList.shape[0],4))
appendTest[:,0] = timeList 
appendTest[:,1:] = xyzList 



smoothAcc = pl.DataFrame(appendTest, columns=['time', 'xAcc', 'yAcc', 'zAcc'])
threeDplot3(smoothAcc,["time", "xAcc", "yAcc", "zAcc"])
# Append test is smoothed np array with time, x, y, z


# regularIntervalDF = pl.DataFrame(np.concatenate((timeList,xyzList.T)).T, columns = ['time', 'xAcc', 'yAcc', 'zAcc'])
smoothAcc.to_parquet('C:/Projects/FormulaSlug/9dof-sparkfun-datalogger/smoothAcc.pq')
# smoothAccTest1.to_parquet('C:/Projects/FormulaSlug/9dof-sparkfun-datalogger/smoothAccTest1.pq') 
# -------------------------
# regularIntervalDF = pl.read_parquet('C:/Projects/FormulaSlug/9dof-sparkfun-datalogger/regularIntervalDF.pq')
# regularIntervalDF is too fine of a resolution (too much data and creates a lot of data)
# threeDplot3(regularIntervalDF,["time", "xAcc", "yAcc", "zAcc"])

smoothAcc = pl.read_parquet('C:/Projects/FormulaSlug/9dof-sparkfun-datalogger/smoothAcc.pq')
smoothAccTest1 = pl.read_parquet('C:/Projects/FormulaSlug/9dof-sparkfun-datalogger/smoothAccTest1.pq')

threeDplot3(smoothAccTest1,["time", "xAcc", "yAcc", "zAcc"])

smoothAcc




smoothAccTest1Adj = smoothAccTest1
sat1aArray = smoothAccTest1Adj.to_numpy()

xAdj = df[["xAcc"]].iloc[4000:4100].to_numpy().sum()/100
yAdj = df[["yAcc"]].iloc[4000:4100].to_numpy().sum()/100
zAdj = df[["zAcc"]].iloc[4000:4100].to_numpy().sum()/100

sat1aArray[:,1] -= xAdj
sat1aArray[:,2] -= yAdj
sat1aArray[:,3] -= zAdj



smoothAccTest1Adj = pl.DataFrame(sat1aArray, columns = ["time", "xAcc", "yAcc", "zAcc"])
smoothAccTest1Adj


csAdj = cleanData2Dx3(smoothAccTest1Adj)


def xRate (time, y):
    return cs(time)[0]
def yRate (time, y):
    return cs(time)[1]
def zRate (time, y):
    return cs(time)[2]

def xRateAdj (time, y):
    return csAdj(time)[0]
def yRateAdj (time, y):
    return csAdj(time)[1]
def zRateAdj (time, y):
    return csAdj(time)[2]





def integral(time_range, step_size, timeY_fun, y0):
    t_eval = [x for x in range(time_range[0], time_range[1], step_size)]
    intVar = solve_ivp(timeY_fun, time_range, [y0], t_eval=t_eval)
    y = intVar.y.flatten()
    t = intVar.t
    join = np.zeros((2, t.shape[0]))
    join[0,:] = t
    join[1,:] = y
    return join

x = integral((t0,t1), 25, xRateAdj, 0)[1,:]
y = integral((t0,t1), 25, yRateAdj, 0)[1,:]
z = integral((t0,t1), 25, zRateAdj, 0)[1,:]

x.shape

ax = plt.figure().add_subplot(projection='3d')
ax.plot(x,y,z)
ax.legend()
plt.show()

# t_eval = [x for x in range(t0, t1, 25)]
# xVelVar = solve_ivp(xRate, (t0,t1), [0], t_eval=t_eval)
# xVelVar.y.flatten()
# xVelVar.t
# xVelVar.t.shape[0]
# xVelVar
# np.column_stack((xVelVar.t, xVelVar.y.flatten()))




# plt.scatter(xVelVar.t, xVelVar.y.flatten())
# plt.show()
# xVelVar.step_size
# int2 = RK45(xVelVar, t0, [0], t1)

def zero(obsX, obs_Y, obs_Z):
    '''
    Takes initial observed acceleration in 3 axes to zero the orientation of the IMU
    '''
    trueZero = np.asarray([0.0,0.0,-1.0])
    observed = np.asarray([obsX/1000, obs_Y/1000, obs_Z/1000])
    dot = np.dot(trueZero, observed)
    cross = np.cross(observed, trueZero)
    cos = dot
    print(f"cos = {cos}")
#     sin = cross / (1.0E6)
    skew = np.asarray([[0.0, cross[2], -1*cross[1]],[-1*cross[2], 0, cross[0]],[cross[1], -1*cross[0], 0]])
    I = np.identity(3)
    R = I + skew + np.matmul(skew,skew)*(1/(1+cos))
    return R

# rotation = zero(209.35200, -441.03000, 902.92200)
rotation = zero(500.0, 500.0, 0)
# v = (np.asarray([[209.35200], [-441.03000], [902.92200]]))
v = (np.asarray([[0.0], [250.0], [500.0]]))
# v
# rotation
np.matmul(rotation,v)
v.shape
pass

threeDplot1(acc,["time", "xAcc", "yAcc", "zAcc"])
threeDplot3(acc,["time", "xAcc", "yAcc", "zAcc"])

# serialPort = serial.Serial(
#     port="COM3", baudrate=115200, bytesize=8, timeout=2, stopbits=serial.STOPBITS_ONE
# )
# serialString = ""  # Used to hold data coming over UART

# test = "13652,4.293,113.797,-1.872,25.376,-378.078,951.722,315.000,-490.000,-385.000,22.898,-0.14587,0.17535,0.40997,23"


# timeCount = []

# while 1:
#     #Read data out of the buffer until a carraige return / new line is found
#     serialString = serialPort.readline()

#     serialList = serialString.split(',')
#     time = serialList[0]
#     voltage = serialList[1]
#     state_of_charge = serialList[2]
#     charge_rate = serialList[3]
#     x_Acc = serialList[4]
#     y_Acc = serialList[5]
#     z_Acc = serialList[6]
#     x_Gyro = serialList[7]
#     y_Gyro = serialList[8]
#     z_Gyro = serialList[9]
#     tempGyro = serialList[10]
#     x_BField = serialList[11]
#     y_BField = serialList[12]
#     z_BField = serialList[13]
#     tempBField = serialList[14]

#     print(serialString)
#     #Print the contents of the serial data



#Notes:
#For finding a row with almost exactly 1g acceleration for finding a stable point to work from
# df = pl.read_csv("ucsctrip0007.csv") #Read in trip csv
# df.columns = ['time', 'voltage', 'state_of_charge', 'charge_rate', 
#         'xAcc', 'yAcc', 'zAcc', 
#         'xGyro', 'yGyro', 'zGyro', 'ISM330.temp', 
#        'xBField', 'yBField', 'zBField', 'MMC5983.temp']
# calibrationT0 = 281000
# calibrationT1 = 284500
# df.filter(((pl.col("xAcc").mul(pl.col("xAcc")) + pl.col("yAcc").mul(pl.col("yAcc")) + pl.col("zAcc").mul(pl.col("zAcc"))).sqrt() - 1000).abs() < 2 ) #.filter(pl.col("time") > calibrationT0).filter(pl.col("time") < calibrationT1)

# for i in range(2000):
#     print(np.sqrt(df["xBField"][i]**2+df["yBField"][i]**2+df["zBField"][i]**2))

# for i in range(2000):
#     print(np.sqrt(df["xAcc"][i]**2+df["yAcc"][i]**2+df["zAcc"][i]**2))
