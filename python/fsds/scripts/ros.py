import time
import fsds
from fsds.ros import RosBridgeClient, ImageRequest, ImageType, CarControls
from argparse import ArgumentParser, Namespace, ArgumentDefaultsHelpFormatter
import logging

logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(funcName)s - %(message)s  ", datefmt="%d-%b-%y %H:%M:%S", level=logging.INFO
)
logger = logging.getLogger(__name__)


class CLIArgs(Namespace):
    """Command line arguments for the CLI interface."""

    verbose: bool
    no_ros: bool
    fsds_ip: str
    fsds_port: int
    ros_ip: str
    ros_port: int
    timeout: int
    timestep: int
    lidar_topic: str
    imu_topic: str
    camera_topic: str
    depth_topic: str
    control_topic: str
    odom_topic: str
    track_topic: str
    sensors: list[str]
    receivers: list[str]


def arg_parser() -> "ArgumentParser":
    parser = ArgumentParser(prog="pylucid", description=__doc__, formatter_class=ArgumentDefaultsHelpFormatter)
    parser.add_argument("-v", "--verbose", action="store_true", help="Enable verbose logging")
    parser.add_argument("--no-ros", action="store_true", help="Run the script without connecting to ROS")
    parser.add_argument("--fsds-ip", type=str, default="127.0.0.1", help="ip of the AirSim simulator")
    parser.add_argument("--fsds-port", type=int, default=41451, help="port of the AirSim simulator")
    parser.add_argument("--ros-ip", type=str, default="127.0.0.1", help="ip of the ROS bridge server")
    parser.add_argument("--ros-port", type=int, default=9090, help="port of the ROS bridge server")
    parser.add_argument("--timeout", type=int, default=3, help="timeout for the connection to the AirSim simulator")
    parser.add_argument("-t", "--timestep", type=float, default=5.0, help="time step in seconds for the loop")
    parser.add_argument("--lidar-topic", type=str, default="/nrfai/lidar", help="ROS topic for LIDAR data")
    parser.add_argument("--imu-topic", type=str, default="/nrfai/imu", help="ROS topic for IMU data")
    parser.add_argument("--camera-topic", type=str, default="/nrfai/camera", help="ROS topic for camera data")
    parser.add_argument("--depth-topic", type=str, default="/nrfai/depth", help="ROS topic for depth data")
    parser.add_argument("--control-topic", type=str, default="/nrfai/control", help="ROS topic for control control")
    parser.add_argument("--odom-topic", type=str, default="/nrfai/odom", help="ROS topic for odometry data")
    parser.add_argument("--track-topic", type=str, default="/nrfai/track", help="ROS topic for tracking data")
    parser.add_argument(
        "--sensors",
        type=str,
        default=["camera", "depth", "lidar", "imu", "odom", "track"],
        nargs="*",
        help="List of sensors to use. Options: camera, depth, lidar, imu, odom, track",
    )
    parser.add_argument(
        "--receivers",
        type=str,
        default=["control"],
        nargs="*",
        help="List of receivers to use. Options: control",
    )
    return parser


def main():
    args: CLIArgs = arg_parser().parse_args()
    if args.verbose:
        logger.setLevel(logging.DEBUG)

    # connect to the AirSim simulator
    client = fsds.FSDSClient(ip=args.fsds_ip, port=args.fsds_port, timeout_value=args.timeout)

    # Check network connection
    client.confirmConnection()

    # After enabling api control only the api can control the car.
    # Direct keyboard and joystick into the simulator are disabled.
    # If you want to still be able to drive with the keyboard while also
    # controlling the car using the api, call client.enableApiControl(False)
    if "control" in args.receivers:
        logger.info("Enabling API control for the car. Keyboard and joystick control will be disabled.")
        client.enableApiControl(True)
    else:
        logger.info("Disabling API control for the car. Keyboard and joystick control will be enabled.")
        client.enableApiControl(False)

    if args.no_ros:
        print("Running without ROS. Press Ctrl+C to exit.")
        while True:
            time.sleep(args.timestep)

    with RosBridgeClient(host=args.ros_ip, port=args.ros_port) as ros_client:
        # Create topics for the sensors
        if "control" in args.receivers:

            def control_cb(msg: CarControls):
                logger.debug("Received control command: %s", msg)
                client.setCarControls(msg)

            logger.info("Subscribing to 'control' topic")
            ros_client.subscribe(args.control_topic, CarControls, control_cb)

        for sensor in args.sensors:
            logger.info("Subscribing to %s topic", sensor)

        while True:
            logger.info("Publishing sensor data...")

            if "imu" in args.sensors:
                ros_client.publish(args.imu_topic, client.getImuData())
            if "lidar" in args.sensors:
                ros_client.publish(args.lidar_topic, client.getLidarData())
            if "camera" in args.sensors:
                ros_client.publish(args.camera_topic, client.simGetImage(ImageRequest(image_type=ImageType.Scene)))
            if "depth" in args.sensors:
                ros_client.publish(
                    args.depth_topic,
                    client.simGetImage(ImageRequest(image_type=ImageType.DepthPerspective, pixels_as_float=True)),
                )
            if "odom" in args.sensors:
                ros_client.publish(args.odom_topic, client.getCarState())
            if "track" in args.sensors:
                ros_client.publish(args.track_topic, client.getRefereeState())

            time.sleep(args.timestep)


if __name__ == "__main__":
    main()
