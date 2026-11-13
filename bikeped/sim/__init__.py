"""CARLA rendering for bikeped scenarios on Unreal Engine 4 and 5.

Importing this package does not import ``carla``. The modules that talk to the
simulator import it when they are first used, so the imaging code and the
tests run on machines without a CARLA installation.
"""

from .types import AGENT_KINDS, STATES, AgentState, CameraModel, FrameRecord, RunInfo

__all__ = ["AGENT_KINDS", "STATES", "AgentState", "CameraModel", "FrameRecord", "RunInfo"]
