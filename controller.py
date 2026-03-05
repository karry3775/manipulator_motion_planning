from motion_types import ControllerCommand, DriverCommand
import numpy as np

class Controller:
    def __init__(self, driver):
        self.ctrl_cmd = None
        self.driver = driver

    def set_controller_command(self, ctrl_cmd: ControllerCommand):
        self.ctrl_cmd = ctrl_cmd

    def update(self, t):
        # Clamp t to max duration
        t = min(t, self.ctrl_cmd.trajectory.get_duration())

        # Compute target position, and generate driver command
        driver_cmd = DriverCommand(
            target_joint_positions=self.ctrl_cmd.trajectory.get_position(t)
        )

        self.driver.send_command(driver_cmd)
