"""Unit-bearing annotations for solver boundaries, without runtime conversion.

Annotated metadata documents units/shape, not coordinate-frame identity or
runtime validation. Arrays remain ordinary NumPy arrays; no copies, clipping,
unit conversion or new rejection paths are introduced by these aliases.
Rotation matrices act on column vectors; rigid transforms contain metre
translations. Human model points before retarget scaling may use other units
and must not be labelled PointMeters until converted.
"""
from typing import Annotated, TypeAlias
import numpy as np

PointMeters: TypeAlias = Annotated[np.ndarray, "shape (3,), metres"]
JointAnglesRad: TypeAlias = Annotated[np.ndarray, "shape (6,), radians"]
RotationMatrix: TypeAlias = Annotated[np.ndarray, "shape (3,3), SO(3)"]
TransformMeters: TypeAlias = Annotated[np.ndarray, "shape (4,4), SE(3), metres"]
DirectionVector: TypeAlias = Annotated[np.ndarray, "shape (3,), unitless direction"]
AngleDegrees: TypeAlias = Annotated[float, "degrees"]
DistanceMeters: TypeAlias = Annotated[float, "metres"]
Seconds: TypeAlias = Annotated[float, "seconds"]
Milliseconds: TypeAlias = Annotated[float, "milliseconds"]
