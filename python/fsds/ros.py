from roslibpy import Ros, Topic
from typing import Any
from .types import *

obj_to_msg_type = {
    Vector3r: "geometry_msgs/Vector3",
    Quaternionr: "geometry_msgs/Quaternion",
    Pose: "geometry_msgs/Pose",
    GeoPoint: "geographic_msgs/GeoPoint",
    # ImageRequest
    # ImageResponse
    # CarControls
    # KinematicsState
    # EnvironmentState
    # CollisionInfo
    # CarState
    Point2D: "geometry_msgs/Point",
    # RefereeState
    # ProjectionMatrix
    # LidarData
    # ImuData
    # GnssReport
    # GpsData
    # GroundSpeedSensorData
}


class RosBridgeClient:

    def __init__(
        self,
        host: str = "127.0.0.1",
        port: int = 9090,
        is_secure: bool = False,
        headers: "dict | None" = None,
    ):
        self._client = Ros(host=host, port=port, is_secure=is_secure, headers=headers)

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

    def publish(self, topic: str, msg: "Vector3r | Quaternionr | Pose | GeoPoint"):
        if type(msg) not in obj_to_msg_type:
            raise ValueError(
                f"{type(msg)} not among the allowed messages: {obj_to_msg_type.keys()}"
            )
        # For some reason these datclasses append "_val" to their fields. We muse remove it
        if type(msg) in (Vector3r, Quaternionr):
            d = {k.removesuffix("_val"): v for k, v in msg.to_dict().items()}
        else:
            d = msg.to_dict()
        Topic(self._client, topic, obj_to_msg_type[type(msg)]).publish(d)
        print("FEST")

    def __enter__(self):
        self._client.run()
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        if self._client.is_connected:
            self._client.close()
