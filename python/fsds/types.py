from __future__ import print_function
import numpy as np # pip install numpy
from typing import TYPE_CHECKING
from dataclasses import dataclass, field, asdict

if TYPE_CHECKING:
    from typing import Any

class MsgpackMixin:
    def __repr__(self):
        from pprint import pformat
        return "<" + type(self).__name__ + "> " + pformat(vars(self), indent=4, width=1)

    def to_msgpack(self, *args, **kwargs):
        return self.__dict__

    @classmethod
    def from_msgpack(cls, encoded: "dict[bytes | str, Any]"):
        obj = cls()
        for  k, v in encoded.items():
            obj_k = k.decode(encoding="utf-8") if isinstance(k, bytes) else k
            setattr(obj, obj_k, v if not isinstance(v, dict) else getattr(obj, obj_k).__class__.from_msgpack(v))
        return obj

    def to_dict(self) -> "dict[str, Any]":
        return asdict(self)


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
        if type(other) in [int, float] + np.sctypes['int'] + np.sctypes['uint'] + np.sctypes['float']:
            return Vector3r( self.x_val / other, self.y_val / other, self.z_val / other)
        else:
            raise TypeError('unsupported operand type(s) for /: %s and %s' % ( str(type(self)), str(type(other))) )

    def __mul__(self, other):
        if type(other) in [int, float] + np.sctypes['int'] + np.sctypes['uint'] + np.sctypes['float']:
            return Vector3r(self.x_val*other, self.y_val*other, self.z_val*other)
        else:
            raise TypeError('unsupported operand type(s) for *: %s and %s' % ( str(type(self)), str(type(other))) )

    def dot(self, other):
        if type(self) == type(other):
            return self.x_val*other.x_val + self.y_val*other.y_val + self.z_val*other.z_val
        else:
            raise TypeError('unsupported operand type(s) for \'dot\': %s and %s' % ( str(type(self)), str(type(other))) )

    def cross(self, other):
        if type(self) == type(other):
            cross_product = np.cross(self.to_numpy_array(), other.to_numpy_array())
            return Vector3r(cross_product[0], cross_product[1], cross_product[2])
        else:
            raise TypeError('unsupported operand type(s) for \'cross\': %s and %s' % ( str(type(self)), str(type(other))) )

    def get_length(self):
        return ( self.x_val**2 + self.y_val**2 + self.z_val**2 )**0.5

    def distance_to(self, other):
        return ( (self.x_val-other.x_val)**2 + (self.y_val-other.y_val)**2 + (self.z_val-other.z_val)**2 )**0.5

    def to_Quaternionr(self):
        return Quaternionr(self.x_val, self.y_val, self.z_val, 0)

    def to_numpy_array(self):
        return np.array([self.x_val, self.y_val, self.z_val], dtype=np.float32)


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
            return Quaternionr( self.x_val+other.x_val, self.y_val+other.y_val, self.z_val+other.z_val, self.w_val+other.w_val )
        else:
            raise TypeError('unsupported operand type(s) for +: %s and %s' % ( str(type(self)), str(type(other))) )

    def __mul__(self, other):
        if type(self) == type(other):
            t, x, y, z = self.w_val, self.x_val, self.y_val, self.z_val
            a, b, c, d = other.w_val, other.x_val, other.y_val, other.z_val
            return Quaternionr( w_val = a*t - b*x - c*y - d*z,
                                x_val = b*t + a*x + d*y - c*z,
                                y_val = c*t + a*y + b*z - d*x,
                                z_val = d*t + z*a + c*x - b*y)
        else:
            raise TypeError('unsupported operand type(s) for *: %s and %s' % ( str(type(self)), str(type(other))) )

    def __truediv__(self, other):
        if type(other) == type(self):
            return self * other.inverse()
        elif type(other) in [int, float] + np.sctypes['int'] + np.sctypes['uint'] + np.sctypes['float']:
            return Quaternionr( self.x_val / other, self.y_val / other, self.z_val / other, self.w_val / other)
        else:
            raise TypeError('unsupported operand type(s) for /: %s and %s' % ( str(type(self)), str(type(other))) )

    def dot(self, other):
        if type(self) == type(other):
            return self.x_val*other.x_val + self.y_val*other.y_val + self.z_val*other.z_val + self.w_val*other.w_val
        else:
            raise TypeError('unsupported operand type(s) for \'dot\': %s and %s' % ( str(type(self)), str(type(other))) )

    def cross(self, other):
        if type(self) == type(other):
            return (self * other - other * self) / 2
        else:
            raise TypeError('unsupported operand type(s) for \'cross\': %s and %s' % ( str(type(self)), str(type(other))) )

    def outer_product(self, other):
        if type(self) == type(other):
            return ( self.inverse()*other - other.inverse()*self ) / 2
        else:
            raise TypeError('unsupported operand type(s) for \'outer_product\': %s and %s' % ( str(type(self)), str(type(other))) )

    def rotate(self, other):
        if type(self) == type(other):
            if other.get_length() == 1:
                return other * self * other.inverse()
            else:
                raise ValueError('length of the other Quaternionr must be 1')
        else:
            raise TypeError('unsupported operand type(s) for \'rotate\': %s and %s' % ( str(type(self)), str(type(other))) )

    def conjugate(self):
        return Quaternionr(-self.x_val, -self.y_val, -self.z_val, self.w_val)

    def star(self):
        return self.conjugate()

    def inverse(self):
        return self.star() / self.dot(self)

    def sgn(self):
        return self/self.get_length()

    def get_length(self):
        return ( self.x_val**2 + self.y_val**2 + self.z_val**2 + self.w_val**2 )**0.5

    def to_numpy_array(self):
        return np.array([self.x_val, self.y_val, self.z_val, self.w_val], dtype=np.float32)


@dataclass
class Pose(MsgpackMixin):
    position: Vector3r = Vector3r()
    orientation: Quaternionr = Quaternionr()

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
    camera_name: str = '0'
    image_type: int = ImageType.Scene
    pixels_as_float: bool = False
    compress: bool = True

@dataclass
class ImageResponse(MsgpackMixin):
    image_data_uint8: "np.uint8" = np.uint8(0)
    image_data_float: float = 0.0
    camera_position: Vector3r = Vector3r()
    camera_orientation: Quaternionr = Quaternionr()
    time_stamp: "np.uint64" = np.uint64(0)
    message: str = ''
    pixels_as_float: float = 0.0
    compress: bool = True
    width: int = 0
    height: int = 0
    image_type: int = ImageType.Scene

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
        if (forward):
            self.is_manual_gear = False
            self.manual_gear = 0
            self.throttle = abs(throttle_val)
        else:
            self.is_manual_gear = False
            self.manual_gear = -1
            self.throttle = - abs(throttle_val)

@dataclass
class KinematicsState(MsgpackMixin):
    position: Vector3r = Vector3r()
    orientation: Quaternionr = Quaternionr()
    linear_velocity: Vector3r = Vector3r()
    angular_velocity: Vector3r = Vector3r()
    linear_acceleration: Vector3r = Vector3r()
    angular_acceleration: Vector3r = Vector3r()

@dataclass
class EnvironmentState(MsgpackMixin):
    position: Vector3r = Vector3r()
    geo_point: GeoPoint = GeoPoint()
    gravityVector3r = Vector3r()
    air_pressure: float = 0.0
    temperature: float = 0.0
    air_density: float = 0.0

@dataclass
class CollisionInfo(MsgpackMixin):
    has_collided: bool = False
    normal: Vector3r = Vector3r()
    impact_point: Vector3r = Vector3r()
    position: Vector3r = Vector3r()
    penetration_depth: float = 0.0
    time_stamp: float = 0.0
    object_name: str = ""
    object_id: int = -1

@dataclass
class CarState(MsgpackMixin):
    speed: float = 0.0
    gear: int = 0 # deprecated, will be deleted
    rpm: float = 0.0 # deprecated, will be deleted
    maxrpm: float = 0.0 # deprecated, will be deleted
    handbrake: bool = False # deprecated, will be deleted
    collision: CollisionInfo = CollisionInfo() # deprecated, will be deleted
    kinematics_estimated: KinematicsState = KinematicsState()
    timestamp: "np.uint64" = np.uint64(0)

@dataclass
class Point2D(MsgpackMixin):
    x: float = 0.0
    y: float = 0.0

@dataclass
class RefereeState(MsgpackMixin):
    doo_counter: int = 0
    laps: float = 0.0
    initial_position: Point2D = Point2D()
    cones: "list[Point2D]" = field(default_factory=list)

@dataclass
class ProjectionMatrix(MsgpackMixin):
    matrix: "list" = field(default_factory=list)

@dataclass
class LidarData(MsgpackMixin):
    point_cloud: "list[Point2D]" = field(default_factory=list)
    time_stamp: "np.uint64" = np.uint64(0)
    pose: "Pose" = Pose()

@dataclass
class ImuData(MsgpackMixin):
    time_stamp: "np.uint64" = np.uint64(0)
    orientation: Quaternionr = Quaternionr()
    angular_velocity: Vector3r = Vector3r()
    linear_acceleration: Vector3r = Vector3r()

@dataclass
class GnssReport(MsgpackMixin):
    geo_point: GeoPoint = GeoPoint()
    eph: float = 0.0
    epv: float = 0.0
    velocity: Vector3r = Vector3r()
    time_utc: "np.uint64" = np.uint64(0)

@dataclass
class GpsData(MsgpackMixin):
    time_stamp: "np.uint64" = np.uint64(0)
    gnss: GnssReport = GnssReport()

@dataclass
class GroundSpeedSensorData(MsgpackMixin):
    time_stamp: "np.uint64" = np.uint64(0)
    linear_velocity: Vector3r = Vector3r()