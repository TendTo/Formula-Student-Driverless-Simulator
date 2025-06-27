from __future__ import print_function
import numpy as np  # pip install numpy
from typing import TYPE_CHECKING
from dataclasses import dataclass, field, asdict, astuple
import struct

if TYPE_CHECKING:
    from typing import Any, TypedDict

    class Field(TypedDict):
        name: str
        offset: int
        datatype: int
        count: int


class MsgpackMixin:
    def __repr__(self):
        from pprint import pformat

        return "<" + type(self).__name__ + "> " + pformat(vars(self), indent=4, width=1)

    def to_msgpack(self, *args, **kwargs):
        return self.__dict__

    @classmethod
    def from_msgpack(cls, encoded: "dict[bytes | str, Any]"):
        obj = cls()
        for k, v in encoded.items():
            obj_k = k.decode(encoding="utf-8") if isinstance(k, bytes) else k
            setattr(obj, obj_k, v if not isinstance(v, dict) else getattr(obj, obj_k).__class__.from_msgpack(v))
        return obj

    def __iter__(self):
        return iter(astuple(self))

    def to_dict(self) -> "dict[str, Any]":
        return asdict(self)

    def to_ros_msg(self) -> "dict[str, Any]":
        return self.to_dict()


class ImageType:
    Scene = 0
    DepthPlanner = 1
    DepthPerspective = 2
    DepthVis = 3
    DisparityNormalized = 4
    Segmentation = 5
    SurfaceNormals = 6
    Infrared = 7


@dataclass
class Vector3r(MsgpackMixin):
    x_val: float = 0.0
    y_val: float = 0.0
    z_val: float = 0.0

    @staticmethod
    def nanVector3r():
        return Vector3r(np.nan, np.nan, np.nan)

    def __add__(self, other):
        return Vector3r(self.x_val + other.x_val, self.y_val + other.y_val, self.z_val + other.z_val)

    def __sub__(self, other):
        return Vector3r(self.x_val - other.x_val, self.y_val - other.y_val, self.z_val - other.z_val)

    def __truediv__(self, other):
        if type(other) in [int, float] + np.sctypes["int"] + np.sctypes["uint"] + np.sctypes["float"]:
            return Vector3r(self.x_val / other, self.y_val / other, self.z_val / other)
        else:
            raise TypeError("unsupported operand type(s) for /: %s and %s" % (str(type(self)), str(type(other))))

    def __mul__(self, other):
        if type(other) in [int, float] + np.sctypes["int"] + np.sctypes["uint"] + np.sctypes["float"]:
            return Vector3r(self.x_val * other, self.y_val * other, self.z_val * other)
        else:
            raise TypeError("unsupported operand type(s) for *: %s and %s" % (str(type(self)), str(type(other))))

    def dot(self, other):
        if type(self) == type(other):
            return self.x_val * other.x_val + self.y_val * other.y_val + self.z_val * other.z_val
        else:
            raise TypeError("unsupported operand type(s) for 'dot': %s and %s" % (str(type(self)), str(type(other))))

    def cross(self, other):
        if type(self) == type(other):
            cross_product = np.cross(self.to_numpy_array(), other.to_numpy_array())
            return Vector3r(cross_product[0], cross_product[1], cross_product[2])
        else:
            raise TypeError("unsupported operand type(s) for 'cross': %s and %s" % (str(type(self)), str(type(other))))

    def get_length(self):
        return (self.x_val**2 + self.y_val**2 + self.z_val**2) ** 0.5

    def distance_to(self, other):
        return (
            (self.x_val - other.x_val) ** 2 + (self.y_val - other.y_val) ** 2 + (self.z_val - other.z_val) ** 2
        ) ** 0.5

    def to_Quaternionr(self):
        return Quaternionr(self.x_val, self.y_val, self.z_val, 0)

    def to_numpy_array(self):
        return np.array([self.x_val, self.y_val, self.z_val], dtype=np.float32)

    def to_ros_msg(self) -> "dict[str, Any]":
        return {k.removesuffix("_val"): v for k, v in self.to_dict().items()}


@dataclass
class Quaternionr(MsgpackMixin):
    w_val: float = 1.0
    x_val: float = 0.0
    y_val: float = 0.0
    z_val: float = 0.0

    @staticmethod
    def nanQuaternionr():
        return Quaternionr(np.nan, np.nan, np.nan, np.nan)

    def __add__(self, other):
        if type(self) == type(other):
            return Quaternionr(
                self.x_val + other.x_val, self.y_val + other.y_val, self.z_val + other.z_val, self.w_val + other.w_val
            )
        else:
            raise TypeError("unsupported operand type(s) for +: %s and %s" % (str(type(self)), str(type(other))))

    def __mul__(self, other):
        if type(self) == type(other):
            t, x, y, z = self.w_val, self.x_val, self.y_val, self.z_val
            a, b, c, d = other.w_val, other.x_val, other.y_val, other.z_val
            return Quaternionr(
                w_val=a * t - b * x - c * y - d * z,
                x_val=b * t + a * x + d * y - c * z,
                y_val=c * t + a * y + b * z - d * x,
                z_val=d * t + z * a + c * x - b * y,
            )
        else:
            raise TypeError("unsupported operand type(s) for *: %s and %s" % (str(type(self)), str(type(other))))

    def __truediv__(self, other):
        if type(other) == type(self):
            return self * other.inverse()
        elif type(other) in [int, float] + np.sctypes["int"] + np.sctypes["uint"] + np.sctypes["float"]:
            return Quaternionr(self.x_val / other, self.y_val / other, self.z_val / other, self.w_val / other)
        else:
            raise TypeError("unsupported operand type(s) for /: %s and %s" % (str(type(self)), str(type(other))))

    def dot(self, other):
        if type(self) == type(other):
            return (
                self.x_val * other.x_val
                + self.y_val * other.y_val
                + self.z_val * other.z_val
                + self.w_val * other.w_val
            )
        else:
            raise TypeError("unsupported operand type(s) for 'dot': %s and %s" % (str(type(self)), str(type(other))))

    def cross(self, other):
        if type(self) == type(other):
            return (self * other - other * self) / 2
        else:
            raise TypeError("unsupported operand type(s) for 'cross': %s and %s" % (str(type(self)), str(type(other))))

    def outer_product(self, other):
        if type(self) == type(other):
            return (self.inverse() * other - other.inverse() * self) / 2
        else:
            raise TypeError(
                "unsupported operand type(s) for 'outer_product': %s and %s" % (str(type(self)), str(type(other)))
            )

    def rotate(self, other):
        if type(self) == type(other):
            if other.get_length() == 1:
                return other * self * other.inverse()
            else:
                raise ValueError("length of the other Quaternionr must be 1")
        else:
            raise TypeError("unsupported operand type(s) for 'rotate': %s and %s" % (str(type(self)), str(type(other))))

    def conjugate(self):
        return Quaternionr(-self.x_val, -self.y_val, -self.z_val, self.w_val)

    def star(self):
        return self.conjugate()

    def inverse(self):
        return self.star() / self.dot(self)

    def sgn(self):
        return self / self.get_length()

    def get_length(self):
        return (self.x_val**2 + self.y_val**2 + self.z_val**2 + self.w_val**2) ** 0.5

    def to_numpy_array(self):
        return np.array([self.x_val, self.y_val, self.z_val, self.w_val], dtype=np.float32)

    def to_ros_msg(self) -> "dict[str, Any]":
        return {k.removesuffix("_val"): v for k, v in self.to_dict().items()}


@dataclass
class Pose(MsgpackMixin):
    position: Vector3r = field(default_factory=Vector3r)
    orientation: Quaternionr = field(default_factory=Quaternionr)

    @staticmethod
    def nanPose():
        return Pose(Vector3r.nanVector3r(), Quaternionr.nanQuaternionr())


@dataclass
class GeoPoint(MsgpackMixin):
    latitude: float = 0.0
    longitude: float = 0.0
    altitude: float = 0.0


@dataclass
class ImageRequest(MsgpackMixin):
    camera_name: str = "0"
    image_type: int = ImageType.Scene
    pixels_as_float: bool = False
    compress: bool = False


@dataclass
class ImageResponse(MsgpackMixin):
    image_data_uint8: "bytes" = field(default_factory=bytes)
    image_data_float: float = 0.0
    camera_position: Vector3r = field(default_factory=Vector3r)
    camera_orientation: Quaternionr = field(default_factory=Quaternionr)
    time_stamp: "np.uint64" = np.uint64(0)
    message: str = ""
    pixels_as_float: float = 0.0
    compress: bool = True
    width: int = 0
    height: int = 0
    image_type: int = ImageType.Scene

    def to_ros_msg(self) -> "dict[str, Any]":
        # Handle the more complex case of ImageResponse
        if self.pixels_as_float:
            # If the image is in float format, we need to convert it to uint8, clipping the values to a maximum depth
            MAX_DEPTH = 40
            data_float = np.clip(np.array(self.image_data_float), 0, MAX_DEPTH) * (255.0 / MAX_DEPTH)
            data = tuple(int(val) for val in data_float)
        else:
            data = tuple(val for val in self.image_data_uint8)
        return {
            "header": None,  # Placeholder for header, can be set to a default header if needed
            "height": self.height,
            "width": self.width,
            "encoding": "mono8" if self.pixels_as_float else "bgr8",
            "step": self.width if self.pixels_as_float else self.width * 3,
            "data": data,
            "is_bigendian": 0,  # False
        }


@dataclass
class CarControls(MsgpackMixin):
    throttle: float = 0.0
    steering: float = 0.0
    brake: float = 0.0
    handbrake: bool = False
    is_manual_gear: bool = False
    manual_gear: int = 0
    gear_immediate: bool = True

    def set_throttle(self, throttle_val, forward):
        if forward:
            self.is_manual_gear = False
            self.manual_gear = 0
            self.throttle = abs(throttle_val)
        else:
            self.is_manual_gear = False
            self.manual_gear = -1
            self.throttle = -abs(throttle_val)


@dataclass
class KinematicsState(MsgpackMixin):
    position: Vector3r = field(default_factory=Vector3r)
    orientation: Quaternionr = field(default_factory=Quaternionr)
    linear_velocity: Vector3r = field(default_factory=Vector3r)
    angular_velocity: Vector3r = field(default_factory=Vector3r)
    linear_acceleration: Vector3r = field(default_factory=Vector3r)
    angular_acceleration: Vector3r = field(default_factory=Vector3r)


@dataclass
class EnvironmentState(MsgpackMixin):
    position: Vector3r = field(default_factory=Vector3r)
    geo_point: GeoPoint = field(default_factory=GeoPoint)
    gravityVector3r: Vector3r = field(default_factory=Vector3r)
    air_pressure: float = 0.0
    temperature: float = 0.0
    air_density: float = 0.0


@dataclass
class CollisionInfo(MsgpackMixin):
    has_collided: bool = False
    normal: Vector3r = field(default_factory=Vector3r)
    impact_point: Vector3r = field(default_factory=Vector3r)
    position: Vector3r = field(default_factory=Vector3r)
    penetration_depth: float = 0.0
    time_stamp: float = 0.0
    object_name: str = ""
    object_id: int = -1


@dataclass
class CarState(MsgpackMixin):
    speed: float = 0.0
    gear: int = 0  # deprecated, will be deleted
    rpm: float = 0.0  # deprecated, will be deleted
    maxrpm: float = 0.0  # deprecated, will be deleted
    handbrake: bool = False  # deprecated, will be deleted
    collision: CollisionInfo = field(default_factory=CollisionInfo)  # deprecated, will be deleted
    kinematics_estimated: KinematicsState = field(default_factory=KinematicsState)
    timestamp: "np.uint64" = np.uint64(0)

    def to_ros_msg(self) -> "dict[str, Any]":
        # Convert the quaternion to yaw angle
        w, x, y, z = self.kinematics_estimated.orientation
        yaw = np.atan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z))
        linear_forward = self.kinematics_estimated.linear_velocity.x_val * np.cos(
            yaw
        ) + self.kinematics_estimated.linear_velocity.y_val * np.sin(yaw)
        linear_sideways = self.kinematics_estimated.linear_velocity.x_val * np.sin(
            yaw
        ) + self.kinematics_estimated.linear_velocity.y_val * -np.cos(yaw)
        return {
            "header": None,  # Placeholder for header, can be set to a default header if needed
            "child_frame_id": "nrai",
            "pose": {
                "pose": {
                    "position": self.kinematics_estimated.position.to_ros_msg(),
                    "orientation": self.kinematics_estimated.orientation.to_ros_msg(),
                }
            },
            "twist": {
                "twist": {
                    "linear": {
                        "x": linear_forward,
                        "y": linear_sideways,
                        "z": self.kinematics_estimated.linear_velocity.z_val,
                    },
                    "angular": {
                        "x": 0.0,
                        "y": 0.0,
                        "z": self.kinematics_estimated.angular_velocity.z_val,
                    },
                }
            },
        }


@dataclass
class Point2D(MsgpackMixin):
    x: float = 0.0
    y: float = 0.0


@dataclass
class RefereeState(MsgpackMixin):
    doo_counter: int = 0
    laps: float = 0.0
    initial_position: Point2D = field(default_factory=Point2D)
    cones: "list[Point2D]" = field(default_factory=list)


@dataclass
class ProjectionMatrix(MsgpackMixin):
    matrix: "list" = field(default_factory=list)


@dataclass
class LidarData(MsgpackMixin):
    point_cloud: "list[float]" = field(default_factory=list)
    time_stamp: "np.uint64" = np.uint64(0)
    pose: "Pose" = field(default_factory=Pose)

    def to_ros_msg(self) -> "dict[str, Any]":
        fields: "list[Field]" = []
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
        data = [b for val in self.point_cloud for b in struct.pack("f", val)]
        return {
            "header": None,  # Placeholder for header, can be set to a default header if needed
            "height": 1,
            "width": len(self.point_cloud) // 3,
            "fields": fields,
            "is_bigendian": False,
            "point_step": offset,
            "row_step": offset * len(self.point_cloud) // 3,
            "is_dense": True,
            "data": data,
        }


@dataclass
class ImuData(MsgpackMixin):
    time_stamp: "np.uint64" = np.uint64(0)
    orientation: Quaternionr = field(default_factory=Quaternionr)
    angular_velocity: Vector3r = field(default_factory=Vector3r)
    linear_acceleration: Vector3r = field(default_factory=Vector3r)
    sigma_arw: float = 0
    sigma_vrw: float = 0

    def to_ros_msg(self) -> "dict[str, Any]":
        avc = self.sigma_arw**2
        lac = self.sigma_vrw**2
        # Why is it like this? No idea, check airsim_ros_wrapper
        return {
            "header": None,  # Placeholder for header, can be set to a default header if needed
            "orientation": self.orientation.to_ros_msg(),
            "angular_velocity": self.angular_velocity.to_ros_msg(),
            "linear_acceleration": self.linear_acceleration.to_ros_msg(),
            "angular_velocity_covariance": [avc, 0, 0, 0, avc, 0, 0, 0, avc],
            "linear_acceleration_covariance": [lac, 0, 0, 0, lac, 0, 0, 0, lac],
        }


@dataclass
class GnssReport(MsgpackMixin):
    geo_point: GeoPoint = field(default_factory=GeoPoint)
    eph: float = 0.0
    epv: float = 0.0
    velocity: Vector3r = field(default_factory=Vector3r)
    time_utc: "np.uint64" = np.uint64(0)


@dataclass
class GpsData(MsgpackMixin):
    time_stamp: "np.uint64" = np.uint64(0)
    gnss: GnssReport = field(default_factory=GnssReport)


@dataclass
class GroundSpeedSensorData(MsgpackMixin):
    time_stamp: "np.uint64" = np.uint64(0)
    linear_velocity: Vector3r = field(default_factory=Vector3r)
