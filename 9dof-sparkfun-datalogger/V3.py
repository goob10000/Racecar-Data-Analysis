#Maze is ucsctrip0007.csv

# Started 3/28/2025
# Goal of this version is to use the DCM with orthonormal correction in the normal matrix form

'''
Steps:
- Pass file through cubic spline to create normalized frequency data
- Filter based on bias and correction matrices
- Determine orientation based on gravity and initialize rotation matrix
- Update rotation matrix sequentially
    - Update angular rates to reflect trapezoid method (value + 1/2*dt*angularRate) to get the new angle
    - Update matrix via 3 matrix transformation
    - 
'''

from matplotlib import pyplot as plt
from sklearn.preprocessing import normalize
import numpy as np
from threeDplot import *
import polars as pl
from integralsAndDerivatives import *
from fftTools import *

import sys

degreeToRadian = (2*np.pi)/360
file = "Runs/Rectangle2x.parquet" #Path to the parquet file
file_Acc_Calibration = "Runs/Orientation.parquet" #Path to orientation parquet file
file_Gyro_Calibration = "Runs/GyroCalibration.parquet" #Path to orientation parquet file
rawData = pl.read_parquet(file) #Read in trip parquet
rawAccCalibration = pl.read_parquet(file_Acc_Calibration) #Read in Acc Calibration parquet
rawGyroCalibration = pl.read_parquet(file_Gyro_Calibration) #Read in Gyro Calibration parquet

xA = "xAcc"
yA = "yAcc"
zA = "zAcc"
xG = "xGyro"
yG = "yGyro"
zG = "zGyro"
xB = "xBField"
yB = "yBField"
zB = "zBField"
time = "time"

rawAccCalibration.insert_column(-1, (rawAccCalibration[xA].pow(2) + rawAccCalibration[yA].pow(2) + rawAccCalibration[zA].pow(2)).sqrt().alias("vA_uncorrected")) # Column that is magnitude of acceleration
rawAccCalibration.insert_column(-1, pl.Series(np.convolve(rawAccCalibration["vA_uncorrected"], np.array([-2, -2, -2, -1, 0, 1, 2, 2, 2]),'same')).alias("vAConvolve"))
cuts = rawAccCalibration.filter(pl.col("vAConvolve").abs() > 20)[time] #Look for places where the edge detection is large (above 50)
for i in range(0, cuts.shape[0] - 1):
#checks every region bounded by 2 cut locations. If the region is large enough, save it to "regions"
    if i == 0:
        regions = []
    if cuts[i+1] - cuts[i] > 500: #!!!!# 500 for personal IMU, 3 for car IMU
        regions = [(cuts[i], cuts[i+1], cuts[i+1] - cuts[i])] + regions


def compile_chunks_and_graph (regions, filter_decision, low_pass_filter_portion): # List, bool, float [0,1]
    filtered_chunks = pl.DataFrame()
    medians = pl.DataFrame()
    for (start, stop, l) in regions:
    # For every region selected, grab the median and standard deviation
    # Filter out any values greater than 1 standard deviation from the median
    # Calculate the new standard deviation without outliers. If the data set varies too much (>1) drop that set
        chunk = rawAccCalibration.filter((pl.col(time) >= start) & (pl.col(time) < stop))
        med = chunk["vA_uncorrected"].median()
        std = chunk["vA_uncorrected"].std()
        print(f"std: {std}")
        print(f"med = {med}, std = {std}")
        num_stds = 1 #1 for home IMU and FS IMU
        filtered_chunk = chunk.filter((pl.col("vA_uncorrected") <= med + num_stds*std) & (pl.col("vA_uncorrected") >= med - num_stds*std))
        std = filtered_chunk["vA_uncorrected"].std()
        med = filtered_chunk["vA_uncorrected"].median()
        if std < 0.5: #1 for home IMU and FS IMU
            if filter_decision:
                # print(f"shape before is {filtered_chunk.shape}")
                array = low_pass_filter(filtered_chunk["vA_uncorrected"].to_numpy(), low_pass_filter_portion)
                # print(f"shape after is {array.shape}")
                # print(array)
                series = pl.Series(array).alias("vA_uncorrected")
                insertion_index = filtered_chunk.get_column_index("vA_uncorrected")
                filtered_chunk.drop_in_place("vA_uncorrected")
                filtered_chunk.insert_column(insertion_index, series)
                # print(filtered_chunk["vA"])
            plt.scatter(filtered_chunk[time], filtered_chunk["vA_uncorrected"], s=0.5) 
            filtered_chunks = pl.concat([filtered_chunks, filtered_chunk],how = 'vertical')
            # print("here")
            medians = pl.concat([medians, filtered_chunk.filter(pl.col("vA_uncorrected") == (filtered_chunk["vA_uncorrected"].median()))], how = 'vertical')
    # print("here")
    plt.show()
    return (filtered_chunks, medians)
filtered_chunks, medians = compile_chunks_and_graph(regions, False, 0.95)


plt.plot(in_place_integrate(filtered_chunks[xG]/1000), label="xGyro")
plt.plot(in_place_integrate(filtered_chunks[yG]/1000), label="yGyro")
plt.plot(in_place_integrate(filtered_chunks[zG]/1000), label="zGyro")
plt.title("Integrated Gyro Data from filtered chunks")
plt.xlabel("Time (s)")
plt.ylabel("Gyro (deg)")
plt.legend()
plt.ylim(-30, 30)
plt.show()

plt.plot(in_place_integrate(filtered_chunks[xG]/1000 - (filtered_chunks[xG]/1000).mean()), label="xGyro")
plt.plot(in_place_integrate(filtered_chunks[yG]/1000 - (filtered_chunks[yG]/1000).mean()), label="yGyro")
plt.plot(in_place_integrate(filtered_chunks[zG]/1000 - (filtered_chunks[zG]/1000).mean()), label="zGyro")
plt.title("Integrated Gyro Data from filtered chunks")
plt.xlabel("Time (s)")
plt.ylabel("Gyro (deg)")
plt.legend()
plt.ylim(-30, 30)
plt.show()

xGBias = 0.0011808715524884387*360
yGBias = -0.0012930429755828722*360
zGBias = -0.0008208641842187239*360

plt.plot(rawGyroCalibration[xG], label="xGyro")
plt.plot(rawGyroCalibration[yG], label="yGyro")
plt.plot(rawGyroCalibration[zG], label="zGyro")
plt.title("Raw Gyro Data")
plt.xlabel("Time (cs)")
plt.ylabel("Gyro (deg/s)")
plt.legend()
plt.show()

plt.plot(in_place_integrate(rawGyroCalibration[xG]/1000 - xGBias), label="xGyro")
plt.plot(in_place_integrate(rawGyroCalibration[yG]/1000 - yGBias), label="yGyro")
plt.plot(in_place_integrate(rawGyroCalibration[zG]/1000 - zGBias), label="zGyro")
plt.title("Integrated Gyro Data")
plt.xlabel("Time (s)")
plt.ylabel("Gyro (deg/s)")
plt.legend()
plt.show()

# # create a linear transformation matrix that maps from the vector [0,0,1] to the vector [xA, yA, zA] the initial row of rawGyroCalibration
# xAcc = rawGyroCalibration[xA][0]
# yAcc = rawGyroCalibration[yA][0]
# zAcc = rawGyroCalibration[zA][0]
# vec = np.matrix([[xAcc],[yAcc],[zAcc]])
# xAcc = xAcc/np.sqrt(xAcc**2 + yAcc**2 + zAcc**2)
# yAcc = yAcc/np.sqrt(xAcc**2 + yAcc**2 + zAcc**2)
# zAcc = zAcc/np.sqrt(xAcc**2 + yAcc**2 + zAcc**2)
# # give me the matrix that maps [0,0,1] to [xAcc, yAcc, zAcc]
# rMat = np.matrix([[xAcc, yAcc, zAcc], [0, 0, 0], [0, 0, 0]])
# np.matmul(np.matrix([[0,0,0],[0,0,0],[xAcc,yAcc,zAcc]]), vec)
# np.sqrt(xAcc**2 + yAcc**2 + zAcc**2)


def propagate_attitude(rawGyroCalibration, timeStep=0.01):
    """
    Propagates attitude using a Direction Cosine Matrix (DCM) with orthonormalization.

    Args:
        rawGyroCalibration (DataFrame): Gyroscope and accelerometer data.
        timeStep (float): Time step for integration.

    Returns:
        np.ndarray: Output matrix containing transformed acceleration vectors.
        np.ndarray: Output attitude matrix.
    """
    # Initialize variables
    num_rows = rawGyroCalibration.shape[0] - 1
    state = np.eye(3)  # Initial state matrix
    gravityVector = np.matrix([rawGyroCalibration[xA][0], rawGyroCalibration[yA][0], rawGyroCalibration[zA][0]]).T
    output_matrix = np.zeros((num_rows, 3))
    output_attitude = np.zeros((num_rows, 3))

    for i in range(num_rows):
        # Update the state matrix using angular velocity
        skew_sym = np.matrix([
            [0, -1 * rawGyroCalibration[i][zG][0] * degreeToRadian, rawGyroCalibration[i][yG][0] * degreeToRadian],
            [rawGyroCalibration[i][zG][0] * degreeToRadian, 0, -1 * rawGyroCalibration[i][xG][0] * degreeToRadian],
            [-1 * rawGyroCalibration[i][yG][0] * degreeToRadian, rawGyroCalibration[i][xG][0] * degreeToRadian, 0]
        ])
        dm = -1 * np.matmul(skew_sym, state)  # Derivative of the state matrix
        state = state + dm * timeStep  # Update state matrix
        state = np.asmatrix(normalize(np.asarray(state), norm='l2', axis=0))  # Orthonormalize

        # Transform acceleration vector
        xAcc, yAcc, zAcc = rawGyroCalibration[i][xA][0], rawGyroCalibration[i][yA][0], rawGyroCalibration[i][zA][0]
        vector = np.matrix([xAcc, yAcc, zAcc]).T
        outputVector = np.matmul(state, vector) - gravityVector
        output_matrix[i, :] = outputVector.T
        output_attitude[i, :] = np.matmul(state, np.matrix([[1, 1, 1]]).T).T

    return output_matrix, output_attitude

# Example usage of propagate_attitude
output_matrix, output_attitude = propagate_attitude(rawGyroCalibration)

plt.plot(output_attitude[:, 0], label="xGyro")
plt.plot(output_attitude[:, 1], label="yGyro")
plt.plot(output_attitude[:, 2], label="zGyro")
plt.title("Integrated Gyro Data")
plt.xlabel("Time (s)")
plt.ylabel("Gyro (deg)")
plt.legend()
plt.show()


## Consider backwards integration for greater stability
# for i in range(0, rawGyroCalibration.shape[0]-1):
#     if i == 0: #Start case initializes all variables
#         xArray, yArray, zArray = np.zeros((rawGyroCalibration.shape[0] - 1,4)), np.zeros((rawGyroCalibration.shape[0] - 1,4)), np.zeros((rawGyroCalibration.shape[0] - 1,4))
#         timeStep = 0.01
#         gravityVector = np.matrix([rawGyroCalibration[xA][0], rawGyroCalibration[yA][0], rawGyroCalibration[zA][0]]).T #Place it into a matrix (vector)
#         state = np.eye(3) # Initial state assumed to be whatever the file started at 
#         output_matrix = np.zeros((rawGyroCalibration.shape[0]-1,3))
#         output_attitude = np.zeros((rawGyroCalibration.shape[0]-1,3))
    
#     # Update the state matrix
#     skew_sym = np.matrix([[0, -1*rawGyroCalibration[i][zG][0], rawGyroCalibration[i][yG][0]], [rawGyroCalibration[i][zG][0], 0, -1*rawGyroCalibration[i][xG][0]], [-1*rawGyroCalibration[i][yG][0], rawGyroCalibration[i][xG][0], 0]])
#     dm = -1*np.matmul(skew_sym,state) # Derivative of the state matrix
#     state = state + dm*timeStep #Updates the state matrix with the derivative of the state matrix
#     # Perform orthonormalization using the Gram-Schmidt process
#     state = np.asmatrix(normalize(np.asarray(state), norm='l2', axis=0)) # Normalize the columns of the state matrix
#     # u, _, vh = np.linalg.svd(state, full_matrices=True)
#     # state = np.dot(u, vh)  # Ensure the state matrix is orthonormal

#     xAcc, yAcc, zAcc = rawGyroCalibration[i][xA][0],rawGyroCalibration[i][yA][0],rawGyroCalibration[i][zA][0] #Pulls out acceleration components of this row
#     vector = np.matrix([xAcc, yAcc, zAcc]).T #Loads them into a matrix and makes it vertical
#     outputVector = np.matmul(state, vector) - gravityVector #Matrix multiplication works based on Ax = b where A is the transformation matrix for this rotation
#     output_matrix[i,:] = outputVector.T
#     output_attitude[i,:] = np.matmul(state, np.matrix([[1,0,0]]).T)[0,:]



# plt.plot(output_attitude[:,0], label="xGyro")
# plt.plot(output_attitude[:,1], label="yGyro")
# plt.title("Integrated Gyro Data")
# plt.xlabel("Time (s)")
# plt.ylabel("Gyro (deg/s)")
# plt.legend()
# plt.show()

fig = plt.figure()
ax = fig.add_subplot(1, 1, 1, projection='3d')
ax.scatter(output_attitude[:,0],output_attitude[:,1],output_attitude[:,2])
ax.set_xlabel("x")
ax.set_ylabel("y")
ax.set_zlabel("z")
ax.set_label("position")
plt.show()

gyroOut = pl.DataFrame(output_matrix)
gyroOut.columns = ["xGyro", "yGyro", "zGyro"]
















SIFrame = pl.DataFrame(rawData["time"]) #s
SIFrame.insert_column(1, rawData["xAcc"]/1000*9.81) #N
SIFrame.insert_column(2, rawData["yAcc"]/1000*9.81) #N
SIFrame.insert_column(3, rawData["zAcc"]/1000*9.81) #N
SIFrame.insert_column(4, rawData["xGyro"]/1000) #deg/s
SIFrame.insert_column(5, rawData["yGyro"]/1000) #deg/s
SIFrame.insert_column(6, rawData["zGyro"]/1000) #deg/s
SIFrame.insert_column(7, rawData["xBField"]) #gauss? Probably actually N/C
SIFrame.insert_column(8, rawData["yBField"]) #gauss? Probably actually N/C
SIFrame.insert_column(9, rawData["zBField"]) #gauss? Probably actually N/C
# SIFrame = SIFrame[0:77320]

#Gravity Calibration
calibrationT0 = 92.7 #Start time for calibration in secs
calibrationT1 = 95.7 #End time for calibration in secs
end = 110.8
calibratingRegion = SIFrame.filter(pl.col("time") > calibrationT0).filter(pl.col("time") < calibrationT1) #Takes frame with times in the calibration range
xG = calibratingRegion["xAcc"].mean()#Mean of xAcc values in calibration range
yG = calibratingRegion["yAcc"].mean()#Mean of yAcc values in calibration range
zG = calibratingRegion["zAcc"].mean()#Mean of zAcc values in calibration range
thetaXDrift = calibratingRegion["xGyro"].mean()
thetaYDrift = calibratingRegion["yGyro"].mean()
thetaZDrift = calibratingRegion["zGyro"].mean()
gravityVector = np.matrix([xG, yG, zG]).T#Place it into a matrix (vector)



#Maze Frame Creation
mazeDF = SIFrame.filter(pl.col("time") > calibrationT1).filter(pl.col("time") < end)#All rows after end of calibration
# calibratingRegion["zGyro"].mean()
# SIFrame.filter(pl.col("time") > 114.5).filter(pl.col("time") < 116.5)["zGyro"].mean()
# SIFrame.filter(pl.col("time") > 129).filter(pl.col("time") < 131.5)["zGyro"].mean()


threeDplot1(rawData,["time", "xAcc", "yAcc", "zAcc"])
threeDplot3(rawData,["time", "xGyro", "yGyro", "zGyro"])
# twoThreeDplot1(rawData,["time", "xAcc", "yAcc", "zAcc", "xGyro", "yGyro", "zGyro"])

for i in range(1, mazeDF.shape[0]):
    # print(f"XYZ Angles {[xAngle,yAngle,zAngle]}")
    if i == 1: #Start case initializes all variables
        xAngle, yAngle, zAngle, xPos, yPos, zPos, xV , yV, zV= 0,0,0,0,0,0,0,0,0
        xArray, yArray, zArray= np.zeros((mazeDF.shape[0] - 1,4)), np.zeros((mazeDF.shape[0] - 1,4)), np.zeros((mazeDF.shape[0] - 1,4))
        timeStep = mazeDF[i]["time"][0] - mazeDF[0]["time"][0]
    else:
        # print(i)
        timeStep = (mazeDF[i]["time"][0] - mazeDF[i-1]["time"][0])
    xAngle += mazeDF[i]["xGyro"][0]*degreeToRadian*timeStep #- thetaXDrift#Updates angle with small time step based on angular velocity, a conversion to radians and the time step - drift
    yAngle += mazeDF[i]["yGyro"][0]*degreeToRadian*timeStep #- thetaYDrift#"
    zAngle += mazeDF[i]["zGyro"][0]*degreeToRadian*timeStep #- thetaZDrift#"

    # print(f"XYZ Angles {[xAngle,yAngle,zAngle]}")

    xMat = np.matrix([[1, 0, 0],[0, np.cos(xAngle), -1*np.sin(xAngle)],[0, np.sin(xAngle), np.cos(xAngle)]]) # X rotation matrix
    yMat = np.matrix([[np.cos(yAngle), 0, np.sin(yAngle)],[0, 1, 0],[-np.sin(yAngle), 0, np.cos(yAngle)]]) # Y rotation matrix
    zMat = np.matrix([[np.cos(zAngle), -1*np.sin(zAngle), 0],[np.sin(zAngle), np.cos(zAngle), 0],[0, 0, 1]]) # Z rotation matrix
    fullRotation = zMat * yMat * xMat #Product in this order creates a single rotation matrix that is based on the current angles

    xAcc, yAcc, zAcc = mazeDF[i]["xAcc"][0],mazeDF[i]["yAcc"][0],mazeDF[i]["zAcc"][0] #Pulls out acceleration components of this row
    vector = np.matrix([xAcc, yAcc, zAcc]).T #Loads them into a matrix and makes it vertical
    outputVector = (fullRotation * (vector)) - gravityVector #Matrix multiplication works based on Ax = b where A is the transformation matrix for this rotation
    newX = outputVector[0,0] #Pulls output x out of the matrix
    newY = outputVector[1,0] # " y
    newZ = outputVector[2,0] # " z
    
    # print(f"Component Rotation Matrices {xMat}\n{yMat}\n{zMat}")
    # print(f"Rotation Matrix {fullRotation}")
    # print(f"XYZ Angles {[xAngle,yAngle,zAngle]}")
    # print(f"input vector {[xAcc,yAcc,zAcc]}")
    # print(f"gravity vector {gravityVector}")
    # print(f"output vector {[newX, newY, newZ]}")

    xPos += (xV * timeStep)
    yPos += (yV * timeStep)
    zPos += (zV * timeStep)
    xV += (newX * timeStep)
    yV += (newY * timeStep)
    zV += (newZ * timeStep)
    xArray[i - 1,0], xArray[i - 1,1], xArray[i - 1,2], xArray[i - 1,3] =  xPos, xV, newX, xAngle
    yArray[i - 1,0], yArray[i - 1,1], yArray[i - 1,2], yArray[i - 1,3] =  yPos, yV, newY, yAngle
    zArray[i - 1,0], zArray[i - 1,1], zArray[i - 1,2], zArray[i - 1,3] =  zPos, zV, newZ, zAngle


# threeDplot1(pl.DataFrame(xArray[:,0:3]).insert_column(0, mazeDF["time"][1:]),["time","column_0","column_1","column_2"])
# threeDplot1(pl.DataFrame(yArray[:,0:3]).insert_column(0, mazeDF["time"][1:]),["time","column_0","column_1","column_2"])
# threeDplot1(pl.DataFrame(zArray[:,0:3]).insert_column(0, mazeDF["time"][1:]),["time","column_0","column_1","column_2"])


# print(xArray)
# print(xArray[:,0])
# end = length*(stepspersec)
fig = plt.figure()

ax = fig.add_subplot(2, 2, 1, projection='3d')
ax.plot(xArray[:,0][:],yArray[:,0][:],zArray[:,0][:])
ax.set_xlabel("x")
ax.set_ylabel("y")
ax.set_zlabel("z")
ax.set_label("position")

# bound = 3
# ax.set_xbound(-1 * bound, bound)
# ax.set_ybound(-1 * bound, bound)
# ax.set_zbound(-1 * bound, bound)



ax = fig.add_subplot(2, 2, 2, projection='3d')
ax.plot(xArray[:,1][:],yArray[:,1][:],zArray[:,1][:])
ax.set_xlabel("x")
ax.set_ylabel("y")
ax.set_zlabel("z")
ax.set_label("velocity")

# bound = 1
# ax.set_xbound(-1 * bound, bound)
# ax.set_ybound(-1 * bound, bound)
# ax.set_zbound(-1 * bound, bound)

ax = fig.add_subplot(2, 2, 3, projection='3d')
ax.plot(xArray[:,2][:],yArray[:,2][:],zArray[:,2][:])
ax.set_xlabel("x")
ax.set_ylabel("y")
ax.set_zlabel("z")
ax.set_label("acceleration")

# bound = 0.2
# ax.set_xbound(-1 * bound, bound)
# ax.set_ybound(-1 * bound, bound)
# ax.set_zbound(-1 * bound, bound)

ax = fig.add_subplot(2, 2, 4, projection='3d')
ax.plot(xArray[:,3][:],yArray[:,3][:],zArray[:,3][:])
ax.set_xlabel("x")
ax.set_ylabel("y")
ax.set_zlabel("z")
ax.set_label("angle")

# bound = 5
# ax.set_xbound(-1 * bound, bound)
# ax.set_ybound(-1 * bound, bound)
# ax.set_zbound(-1 * bound, bound)
plt.show()

