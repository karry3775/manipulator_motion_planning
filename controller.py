from motion_types import ControllerCommand, DriverCommand
import numpy as np

class Controller:
    def __init__(self):
        self.ctrl_cmd = None

    def set_controller_command(self, ctrl_cmd: ControllerCommand):
        self.ctrl_cmd = ctrl_cmd

    def get_driver_cmd(self, driver_status, t):
        # Clamp t to traj duration
        t = min(t, self.ctrl_cmd.trajectory.get_duration())

        # Compute target position, and generate driver command
        return DriverCommand(
            target_joint_positions=self.ctrl_cmd.trajectory.get_position(t)
        )
