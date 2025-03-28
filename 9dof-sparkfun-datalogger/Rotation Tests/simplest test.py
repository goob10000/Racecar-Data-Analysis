from matplotlib import pyplot as plt
from scipy.interpolate import CubicSpline
from scipy.integrate import RK45, solve_ivp
import numpy as np
from threeDplot import *
import polars as pl


length = 15
stepLength = 0.001
stepspersec = round(1/stepLength)

time = np.arange(0,length,stepLength)
xAcc = np.zeros(length*stepspersec)
yAcc = np.zeros(length*stepspersec) + 1_000
zAcc = np.zeros(length*stepspersec) + 2
# zAcc[10:10000] = np.asarray([10 for i in range(9990)])
xGyro = np.zeros(length * stepspersec) #+ 4_500
yGyro = np.zeros(length*stepspersec) + 45_015
zGyro = np.zeros(length*stepspersec)

testFrame = pl.DataFrame([time, xAcc, yAcc, zAcc, xGyro, yGyro, zGyro])
testFrame.columns = ["time", "xAcc", "yAcc", "zAcc", "xGyro", "yGyro", "zGyro"]

degreeToRadian = (2*np.pi)/360

rawData = testFrame
SIFrame = pl.DataFrame(rawData["time"]) #s
SIFrame.insert_column(1, rawData["xAcc"]/1000*9.8) #N
SIFrame.insert_column(2, rawData["yAcc"]/1000*9.8) #N
SIFrame.insert_column(3, rawData["zAcc"]/1000*9.8) #N
SIFrame.insert_column(4, rawData["xGyro"]/1000) #deg/s
SIFrame.insert_column(5, rawData["yGyro"]/1000) #deg/s
SIFrame.insert_column(6, rawData["zGyro"]/1000) #deg/s
# SIFrame = SIFrame[0:77320]

#Gravity Calibration
calibrationT0 = 0 #Start time for calibration in secs
calibrationT1 = 10*stepLength #End time for calibration in secs
calibratingRegion = SIFrame.filter(pl.col("time") > calibrationT0).filter(pl.col("time") < calibrationT1) #Takes frame with times in the calibration range
xG = calibratingRegion["xAcc"].mean()#Mean of xAcc values in calibration range
yG = calibratingRegion["yAcc"].mean()#Mean of yAcc values in calibration range
zG = calibratingRegion["zAcc"].mean()#Mean of zAcc values in calibration range
gravityVector = np.matrix([xG, yG, zG]).T#Place it into a matrix (vector)
# print(gravityVector)

# Maze Frame Creation
mazeDF = SIFrame.filter(pl.col("time") > calibrationT1)#All rows after end of calibration
# print(mazeDF)
# mazeDF = SIFrame


for i in range(1, mazeDF.shape[0]):
    # print(f"XYZ Angles {[xAngle,yAngle,zAngle]}")
    if i == 1: #Start case initializes all variables
        xAngle, yAngle, zAngle, xPos, yPos, zPos, xV , yV, zV= 0,0,0,0,0,0,0,0,0
        xArray, yArray, zArray= np.zeros((mazeDF.shape[0] - 1,4)), np.zeros((mazeDF.shape[0] - 1,4)), np.zeros((mazeDF.shape[0] - 1,4))
        timeStep = mazeDF[i]["time"][0] - mazeDF[0]["time"][0]
    else:
        # print(i)
        timeStep = (mazeDF[i]["time"][0] - mazeDF[i-1]["time"][0])
    xAngle += mazeDF[i]["xGyro"][0]*degreeToRadian*timeStep #Updates angle with small time step based on angular velocity, a conversion to radians and the time step
    yAngle += mazeDF[i]["yGyro"][0]*degreeToRadian*timeStep #"
    zAngle += mazeDF[i]["zGyro"][0]*degreeToRadian*timeStep #"

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



# print(xArray)
# print(xArray[:,0])
# end = length*(stepspersec)
fig = plt.figure()

ax = fig.add_subplot(2, 2, 1, projection='3d')
ax.plot(xArray[:,0][:],yArray[:,0][:],zArray[:,0][:])
ax.set_xlabel("x")
ax.set_ylabel("y")
ax.set_zlabel("z")

# bound = 1
# ax.set_xbound(-1 * bound, bound)
# ax.set_ybound(-1 * bound, bound)
# ax.set_zbound(-1 * bound, bound)



ax = fig.add_subplot(2, 2, 2, projection='3d')
ax.plot(xArray[:,1][:],yArray[:,1][:],zArray[:,1][:])
ax.set_xlabel("x")
ax.set_ylabel("y")
ax.set_zlabel("z")

bound = 0.5
ax.set_xbound(-1 * bound, bound)
ax.set_ybound(-1 * bound, bound)
ax.set_zbound(-1 * bound, bound)

ax = fig.add_subplot(2, 2, 3, projection='3d')
ax.plot(xArray[:,2][:],yArray[:,2][:],zArray[:,2][:])
ax.set_xlabel("x")
ax.set_ylabel("y")
ax.set_zlabel("z")

bound = 0.1
ax.set_xbound(-1 * bound, bound)
ax.set_ybound(-1 * bound, bound)
ax.set_zbound(-1 * bound, bound)

ax = fig.add_subplot(2, 2, 4, projection='3d')
ax.plot(xArray[:,3][:],yArray[:,3][:],zArray[:,3][:])
ax.set_xlabel("x")
ax.set_ylabel("y")
ax.set_zlabel("z")

bound = 40
ax.set_xbound(-1 * bound, bound)
ax.set_ybound(-1 * bound, bound)
ax.set_zbound(-1 * bound, bound)
plt.show()


# plt.plot(xAngleArray[0:end])
# plt.plot(yAngleArray[0:end])
# plt.plot(zAngleArray[0:end])
# plt.show()


