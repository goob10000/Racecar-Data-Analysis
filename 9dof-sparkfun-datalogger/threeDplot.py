from matplotlib import pyplot as plt
import pandas as pd

def threeDplot1(df,names):
    fig = plt.figure()
    fig.add_subplot(1,1,1)
    x = names[0]
    y1 = names[1]
    y2 = names[2]
    y3 = names[3]
    plt.plot(df[[x]],df[[y1]],c="#0040cf")
    plt.plot(df[[x]],df[[y2]],c="#00af50")
    plt.plot(df[[x]],df[[y3]],c="#ffb010")
    plt.legend([y1, y2, y3])
    plt.show()

def threeDplot3(df, names):
    fig = plt.figure()
    x = names[0]
    y1 = names[1]
    y2 = names[2]
    y3 = names[3]

    fig.add_subplot(3,1,1)
    plt.plot(df[[x]],df[[y1]],c="#0040cf")
    plt.legend([y1])


    fig.add_subplot(3,1,2)
    plt.plot(df[[x]],df[[y2]],c="#00af50")
    plt.legend([y2])

    fig.add_subplot(3,1,3)
    plt.plot(df[[x]],df[[y3]],c="#ffb010")
    plt.legend([y3])

    fig.tight_layout()
    
    plt.show()

def twoThreeDplot3(df, names):
    fig = plt.figure()
    x = names[0]
    y1 = names[1]
    y2 = names[2]
    y3 = names[3]
    y4 = names[4]
    y5 = names[5]
    y6 = names[6]

    fig.add_subplot(6,1,1)
    plt.plot(df[[x]],df[[y1]],c="#0040cf")
    plt.legend([y1])

    fig.add_subplot(6,1,2)
    plt.plot(df[[x]],df[[y2]],c="#00af50")
    plt.legend([y2])

    fig.add_subplot(6,1,3)
    plt.plot(df[[x]],df[[y3]],c="#ffb010")
    plt.legend([y3])

    fig.add_subplot(6,2,1)
    plt.plot(df[[x]],df[[y4]],c="#0040cf")
    plt.legend([y4])

    fig.add_subplot(6,2,2)
    plt.plot(df[[x]],df[[y5]],c="#00af50")
    plt.legend([y5])

    fig.add_subplot(6,2,3)
    plt.plot(df[[x]],df[[y6]],c="#ffb010")
    plt.legend([y6])

    fig.tight_layout()
    
    plt.show()

def twoThreeDplot1(df, names):
    fig = plt.figure()
    x = names[0]
    y1 = names[1]
    y2 = names[2]
    y3 = names[3]
    y4 = names[4]
    y5 = names[5]
    y6 = names[6]

    fig.add_subplot(2,1,1)   
    plt.plot(df[[x]],df[[y1]],c="#0040cf")
    plt.plot(df[[x]],df[[y2]],c="#00af50")
    plt.plot(df[[x]],df[[y3]],c="#ffb010")
    plt.legend([y1, y2, y3])

    fig.add_subplot(2,1,2)
    plt.plot(df[[x]],df[[y4]],c="#0040cf")
    plt.plot(df[[x]],df[[y5]],c="#00af50")
    plt.plot(df[[x]],df[[y6]],c="#ffb010")
    plt.legend([y4, y5, y6])

    fig.tight_layout()
    
    plt.show()

# fig = plt.figure()
# # fig.add_subplot(1,1,1)
# ax = plt.figure().add_subplot(projection='3d')
# ax.plot(acc.iloc[1:4000][["xAcc"]],acc.iloc[1:4000][["yAcc"]],acc.iloc[1:4000][["zAcc"]])

# fig.add_subplot(1,1,1)
# plt.plot(acc[["time"]],acc[["xAcc"]],c="#0040cf")
# plt.plot(acc[["time"]],acc[["yAcc"]],c="#00af50")
# plt.plot(acc[["time"]],acc[["zAcc"]],c="#ffb010")
# plt.legend(['xAcc', 'yAcc', 'zAcc'])
# plt.plot(acc[["time"]],acc[["xAcc"]])

# fig.add_subplot(4,1,2)
# plt.plot(gyro[["time"]],gyro[["xGyro"]],c="#0040cf")
# plt.legend(['xGyro'])

# fig.add_subplot(4,1,3)
# plt.plot(gyro[["time"]],gyro[["yGyro"]],c="#00af50")
# plt.legend(['yGyro'])

# fig.add_subplot(4,1,4)
# plt.plot(gyro[["time"]],gyro[["zGyro"]],c="#ffb010")
# plt.legend(['zGyro'])


# plt.show()