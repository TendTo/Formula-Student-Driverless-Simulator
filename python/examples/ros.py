import time
import fsds
from fsds.ros import RosBridgeClient


def show_lidar(lidar_data: "fsds.LidarData"):
    import numpy as np
    import matplotlib.pyplot as plt
    plt.switch_backend("TkAgg")
    data = np.array(lidar_data.point_cloud)
    x = data[::3]
    y = data[1::3]
    z = data[2::3]
    fig = plt.figure()
    ax = fig.add_subplot(projection='3d')
    ax.scatter(x, y, z)
    plt.show()


def main():
    TIME_STEP = 5

    # connect to the AirSim simulator
    client = fsds.FSDSClient()

    # Check network connection
    client.confirmConnection()

    # After enabling api controll only the api can controll the car.
    # Direct keyboard and joystick into the simulator are disabled.
    # If you want to still be able to drive with the keyboard while also
    # controll the car using the api, call client.enableApiControl(False)
    client.enableApiControl(False)

    with RosBridgeClient() as ros_client:
        while True:
            ros_client.publish("/nrfai/imu", client.getImuData())
            ros_client.publish("/nrfai/lidar", client.getLidarData())
            ros_client.publish("/nrfai/camera", client.simGetImages())
            time.sleep(TIME_STEP)


if __name__ == "__main__":
    main()
