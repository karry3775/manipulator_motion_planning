from simulation_driver import SimulationDriver
from controller import Controller
from motion_types import ControllerCommand
from trajectory_generator import CubicTrajectory
from motion_types import DriverCommand
import numpy as np

def generate_controller_command(current_joint_positions, target_joint_positions, duration_s):
    trajectory = CubicTrajectory(
        start_pos=current_joint_positions,
        end_pos=target_joint_positions,
        start_vel=np.zeros(6),
        end_vel=np.zeros(6),
        duration=duration_s
    )

    return ControllerCommand(
        trajectory=trajectory
    )

def main():
    # Create a sim_driver so we can interact with the simulator
    sim_driver = SimulationDriver()
    sim_driver.wait_for_initialization()

    # Create a controller 
    controller = Controller(sim_driver)
    # params for controller command
    current_joint_positions = sim_driver.get_current_joint_positions()
    target_joint_positions = np.deg2rad(np.array([90, -90, 90, 90, 90, 90]))
    duration_s = 10
    controller_cmd = generate_controller_command(current_joint_positions, target_joint_positions, duration_s)
    controller.set_controller_command(controller_cmd)


    # Core controller loop
    loop_rate = 2 * 1e-3
    
    loop_start = sim_driver.now()
    while True:
        cycle_start = sim_driver.now()
        t = cycle_start - loop_start
        controller.update(t)
        
        # Wait until next loop
        while sim_driver.now() - cycle_start < loop_rate:
            continue
if __name__ == "__main__":
    main()



    
            
        
        


