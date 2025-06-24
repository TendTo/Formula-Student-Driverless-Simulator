import time
import fsds
from fsds.ros import RosBridgeClient
from argparse import ArgumentParser, Namespace, ArgumentDefaultsHelpFormatter


def show_lidar(lidar_data: "fsds.LidarData"):
    import numpy as np
    import matplotlib.pyplot as plt

    plt.switch_backend("TkAgg")
    data = np.array(lidar_data.point_cloud)
    x = data[::3]
    y = data[1::3]
    z = data[2::3]
    fig = plt.figure()
    ax = fig.add_subplot(projection="3d")
    ax.scatter(x, y, z)
    plt.show()


class CLIArgs(Namespace):
    """Command line arguments for the CLI interface."""

    fsds_ip: str
    fsds_port: int
    ros_ip: str
    ros_port: int
    timeout: int
    timestep: int
    lidar_topic: str
    imu_topic: str
    camera_topic: str


def arg_parser() -> "ArgumentParser":
    parser = ArgumentParser(prog="pylucid", description=__doc__, formatter_class=ArgumentDefaultsHelpFormatter)
    parser.add_argument("--fsds-ip", type=str, default="127.0.0.1", help="ip of the AirSim simulator")
    parser.add_argument("--fsds-port", type=int, default=41451, help="port of the AirSim simulator")
    parser.add_argument("--ros-ip", type=str, default="127.0.0.1", help="ip of the ROS bridge server")
    parser.add_argument("--ros-port", type=int, default=9090, help="port of the ROS bridge server")
    parser.add_argument("--timeout", type=int, default=3, help="timeout for the connection to the AirSim simulator")
    parser.add_argument("-t", "--timestep", type=float, default=5.0, help="time step in seconds for the loop")
    parser.add_argument("--lidar-topic", type=str, default="/nrfai/lidar", help="ROS topic for LIDAR data")
    parser.add_argument("--imu-topic", type=str, default="/nrfai/imu", help="ROS topic for IMU data")
    parser.add_argument("--camera-topic", type=str, default="/nrfai/camera", help="ROS topic for camera data")
    return parser


def main():
    args: CLIArgs = arg_parser().parse_args()
    TIME_STEP = args.timestep

    # connect to the AirSim simulator
    client = fsds.FSDSClient(ip=args.fsds_ip, port=args.fsds_port, timeout_value=args.timeout)

    # Check network connection
    client.confirmConnection()

    # After enabling api controll only the api can controll the car.
    # Direct keyboard and joystick into the simulator are disabled.
    # If you want to still be able to drive with the keyboard while also
    # controll the car using the api, call client.enableApiControl(False)
    client.enableApiControl(False)

    with RosBridgeClient(host=args.ros_ip, port=args.ros_port) as ros_client:
        while True:
            # ros_client.publish(args.imu_topic, client.getImuData())
            # ros_client.publish(args.lidar_topic, client.getLidarData())
            ros_client.publish(args.camera_topic, client.simGetImages())
            time.sleep(TIME_STEP)


if __name__ == "__main__":
    main()
