from pxr import Usd, UsdGeom, Gf, PhysxSchema, PhysicsSchema
import math
import numpy as np
from omni.isaac.dynamic_control import _dynamic_control
import omni.kit.editor


class Kaya:
    def __init__(self, stage, dc, usd_path, prim_path, speed_gain):
        self._stage = stage
        self._dc = dc
        self.usd_path = usd_path
        self.prim_path = prim_path
        self.speed_gain = speed_gain
        self._editor = omni.kit.editor.get_editor_interface()

        # setup high-level kaya prim
        self.prim = self._stage.DefinePrim(prim_path, "Xform")
        self.prim.GetReferences().AddReference(usd_path)

        self.wheel_check = None

    def control_setup(self):
        self.ar = self._dc.get_articulation(str(self.prim.GetPath()))

        self.wheel_back = self._dc.find_articulation_dof(self.ar, "axle_0_joint")
        self.wheel_right = self._dc.find_articulation_dof(self.ar, "axle_1_joint")
        self.wheel_left = self._dc.find_articulation_dof(self.ar, "axle_2_joint")

        self.wheel_back_idx = self._dc.find_articulation_dof_index(self.ar, "axle_0_joint")
        self.wheel_right_idx = self._dc.find_articulation_dof_index(self.ar, "axle_1_joint")
        self.wheel_left_idx = self._dc.find_articulation_dof_index(self.ar, "axle_2_joint")

        self.vel_props = _dynamic_control.DofProperties()
        self.vel_props.drive_mode = _dynamic_control.DRIVE_VEL
        self.vel_props.damping = 1e7
        self.vel_props.stiffness = 0
        self._dc.set_dof_properties(self.wheel_right, self.vel_props)

        self.wheel_check = True

    def compute_wheel_speed(self, vel):
        kOneByThree = 1.0 / 3.0
        kOneBySqrtThree = 1.0 / math.sqrt(3.0)
        wheel_distance = 12.5
        wheel_radius = 4.0
        forward_matrix = np.array(
            [
                [0, -kOneBySqrtThree, kOneBySqrtThree],
                [kOneByThree * 2, -kOneByThree, -kOneByThree],
                [-kOneByThree / wheel_distance, -kOneByThree / wheel_distance, -kOneByThree / wheel_distance],
            ]
        )
        wheels_radius_matrix = np.array([[wheel_radius, 0, 0], [0, wheel_radius, 0], [0, 0, wheel_radius]])
        transform_matrix = np.matmul(forward_matrix, wheels_radius_matrix)
        inverse_matrix = np.linalg.inv(transform_matrix)
        wheel_speed = np.matmul(inverse_matrix, vel)
        return wheel_speed

    def move(self, vel_target):
        if not self._editor.is_playing():
            return
        if not self._dc.is_simulating():
            return
        if not self.wheel_check:
            self.control_setup()
        wheel_speed = self.compute_wheel_speed(vel_target)
        # Wake up articulation every move command to ensure commands are applied
        self._dc.wake_up_articulation(self.ar)
        self._dc.set_dof_velocity_target(
            self.wheel_right, np.clip(self.speed_gain * wheel_speed[1], -self.speed_gain, self.speed_gain)
        )
        self._dc.set_dof_velocity_target(
            self.wheel_left, np.clip(self.speed_gain * wheel_speed[2], -self.speed_gain, self.speed_gain)
        )
        self._dc.set_dof_velocity_target(
            self.wheel_back, np.clip(self.speed_gain * wheel_speed[0], -self.speed_gain, self.speed_gain)
        )
