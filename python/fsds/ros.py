from roslibpy import Ros, Topic
from roslibpy.ros2 import Header, Time
from typing import Any, Callable, TypeVar, TypedDict
from .types import *
import struct


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
    # CarState
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


class Field(TypedDict):
    name: str
    offset: int
    datatype: int
    count: int


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
        # For some reason these datclasses append "_val" to their fields. We muse remove it
        if isinstance(msg, (Vector3r, Quaternionr)):
            return {k.removesuffix("_val"): v for k, v in msg.to_dict().items()}
        if isinstance(msg, ImageResponse):
            # Handle the more complex case of ImageResponse
            return {
                "header": default_header(),
                "height": msg.height,
                "width": msg.width,
                "encoding": "bgr8",
                "step": msg.width * 3,
                "data": msg.image_data_uint8,
                "is_bigendian": False,
            }
        if isinstance(msg, ImuData):
            avc = msg.sigma_arw**2
            lac = msg.sigma_vrw**2
            # Why is it like this? No idea, check airsim_ros_wrapper
            return {
                "header": default_header(),
                "orientation": cls.dataclass_to_ros_msg(msg.orientation),
                "angular_velocity": cls.dataclass_to_ros_msg(msg.angular_velocity),
                "linear_acceleration": cls.dataclass_to_ros_msg(msg.linear_acceleration),
                "angular_velocity_covariance": [avc, 0, 0, 0, avc, 0, 0, 0, avc],
                "linear_acceleration_covariance": [lac, 0, 0, 0, lac, 0, 0, 0, lac],
            }
        if isinstance(msg, LidarData):
            fields: list[Field] = []
            offset = 3 * 4
            for i in range(0, offset, 4):
                fields.append(
                    {
                        "name": ("x" if i == 0 else ("y" if i == 4 else "z")),
                        "offset": i,
                        "count": 1,
                        "datatype": 7,  # Datatype FLOAT32 https://docs.ros.org/en/noetic/api/sensor_msgs/html/msg/PointField.html
                    }
                )
            data = [b for val in msg.point_cloud for b in struct.pack('f', val)]
            return {
                "header": default_header(),
                "height": 1,
                "width": len(msg.point_cloud) // 3,
                "fields": fields,
                "is_bigendian": False,
                "point_step": offset,
                "row_step": offset * len(msg.point_cloud) // 3,
                "is_dense": True,
                "data": data,
            }
        return msg.to_dict()

    def publish(
        self,
        topic: str,
        msg: "Vector3r | Quaternionr | Pose | GeoPoint | ImageResponse",
    ):
        ros_msg = self.dataclass_to_ros_msg(msg)
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
