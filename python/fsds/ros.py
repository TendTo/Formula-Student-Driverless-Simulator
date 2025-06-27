from roslibpy import Ros, Topic
from roslibpy.ros2 import Header, Time
from typing import Any, Callable, TypeVar, TypedDict
from .types import *
import struct
import logging

logger = logging.getLogger(__name__)


def default_header(t: Time = None, frame_id: str = "") -> Header:
    """
    Returns a default ROS header with the current time.
    """
    return Header(stamp=t if t is not None else Time.now(), frame_id="fsds_frame" if not frame_id else frame_id)


obj_to_msg_type = {
    Vector3r: "geometry_msgs/Vector3",
    Quaternionr: "geometry_msgs/Quaternion",
    Pose: "geometry_msgs/Pose",
    GeoPoint: "geographic_msgs/GeoPoint",
    # ImageRequest # NOT NECESSARY
    ImageResponse: "sensor_msgs/Image",
    CarControls: "newcastle_racing_ai_msgs/ControlCommand",
    # KinematicsState
    # EnvironmentState
    # CollisionInfo
    CarState: "nav_msgs/Odometry",
    Point2D: "geometry_msgs/Point",
    # RefereeState
    # ProjectionMatrix
    LidarData: "sensor_msgs/PointCloud2",
    ImuData: "sensor_msgs/Imu",
    # GnssReport
    # GpsData
    # GroundSpeedSensorData
}

T = TypeVar("T")


class RosBridgeClient:

    def __init__(
        self,
        host: str = "127.0.0.1",
        port: int = 9090,
        is_secure: bool = False,
        headers: "dict | None" = None,
    ):
        self._client = Ros(host=host, port=port, is_secure=is_secure, headers=headers)
        self._subscribed_topics: "list[Topic]" = []

    def get_topic(
        self,
        topic: str,
        msg_type: str,
        compression: "Any | None" = None,
        latch: bool = False,
        throttle_rate: int = 0,
        queue_size: int = 100,
        queue_length: int = 0,
        reconnect_on_close: bool = True,
    ):
        return Topic(
            self._client,
            topic,
            msg_type,
            compression,
            latch,
            throttle_rate,
            queue_size,
            queue_length,
            reconnect_on_close,
        )

    @classmethod
    def dataclass_to_ros_msg(cls, msg: "Vector3r | Quaternionr | Pose | GeoPoint | ImageResponse") -> dict:
        if type(msg) not in obj_to_msg_type:
            raise ValueError(f"{type(msg)} not among the allowed messages: {obj_to_msg_type.keys()}")
        msg = msg.to_ros_msg()
        if "header" in msg and not isinstance(msg["header"], Header):
            msg["header"] = default_header()
        return msg

    def publish(
        self,
        topic: str,
        msg: "Vector3r | Quaternionr | Pose | GeoPoint | ImageResponse",
    ):
        ros_msg = self.dataclass_to_ros_msg(msg)
        logger.info("Publishing msg of type '%s' to topic '%s'", obj_to_msg_type[type(msg)], topic)
        Topic(self._client, topic, obj_to_msg_type[type(msg)], latch=True).publish(ros_msg)

    @staticmethod
    def ros_msg_to_dataclass(msg_type: type[T], msg: dict) -> T:
        print(f"ros_msg_to_dataclass: {msg_type}, {msg}")
        return msg_type(**msg)

    def subscribe(self, topic: str, msg_type: type[T], cb: "Callable[[T], None]") -> int:
        self._subscribed_topics.append(Topic(self._client, topic, obj_to_msg_type[msg_type]))
        self._subscribed_topics[-1].subscribe(lambda msg: cb(self.ros_msg_to_dataclass(msg_type, msg)))
        return len(self._subscribed_topics) - 1

    def unsubscribe(self, topic_idx: int):
        self._subscribed_topics[topic_idx].unsubscribe()

    def __enter__(self):
        self._subscribed_topics.clear()
        self._client.run()
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        self._subscribed_topics.clear()
        if self._client.is_connected:
            self._client.close()
