from matplotlib import pyplot as plt
from scipy.interpolate import CubicSpline
from scipy.integrate import RK45, solve_ivp
import numpy as np
from threeDplot import *
import polars as pl


length = 60
stepLength = 0.1
stepspersec = round(1/stepLength)

time = np.arange(0,length,stepLength)
xAcc = np.zeros(length*stepspersec)
yAcc = xAcc + 1_000
zAcc = xAcc
zAcc[10:15] = np.asarray([10 for i in range(5)])
xGyro = np.zeros(length * stepspersec) + 40_000
yGyro = xAcc
zGyro = xAcc

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
print(gravityVector)

#Maze Frame Creation
mazeDF = SIFrame.filter(pl.col("time") > calibrationT1)#All rows after end of calibration

for i in range(1, mazeDF.shape[0]):
    # print(f"XYZ Angles {[xAngle,yAngle,zAngle]}")
    if i == 1: #Start case initializes all variables
        xAngle, yAngle, zAngle, xPos, yPos, zPos, xV , yV, zV= 0,0,0,0,0,0,0,0,0
        xPosArray, yPosArray, zPosArray, xAngleArray, yAngleArray, zAngleArray = [],[],[],[],[],[]
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
    outputVector = (fullRotation * (vector))- gravityVector #Matrix multiplication works based on Ax = b where A is the transformation matrix for this rotation
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
    xPosArray += [xPos]
    yPosArray += [yPos]
    zPosArray += [zPos]
    xAngleArray += [xAngle]
    yAngleArray += [yAngle]
    zAngleArray += [zAngle]


ax = plt.figure().add_subplot(projection='3d')
end = length*(stepspersec)
ax.plot(xPosArray[0:end],yPosArray[0:end],zPosArray[0:end])
plt.show()


plt.plot(xAngleArray[0:end])
plt.plot(yAngleArray[0:end])
plt.plot(zAngleArray[0:end])
plt.show()


