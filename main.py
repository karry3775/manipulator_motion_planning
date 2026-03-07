from simulation_driver import SimulationDriver
from controller import Controller
from motion_types import ControllerCommand
from motion_planning.trajectory_generator import CubicTrajectory
from mujoco_model_manager import MujocoModelManager
import numpy as np

def generate_controller_command(current_joint_positions, target_joint_positions, duration_s):
    trajectory = CubicTrajectory(
        start_pos=current_joint_positions,
        end_pos=target_joint_positions,
        start_vel=np.zeros(len(current_joint_positions)),
        end_vel=np.zeros(len(current_joint_positions)),
        duration=duration_s
    )

    return ControllerCommand(
        trajectory=trajectory
    )

def main():
    # Create a sim_driver so we can interact with the simulator
    sim_driver = SimulationDriver()
    sim_driver.wait_for_initialization()

    # Lets aslo have a model manager
    mm = MujocoModelManager(scene_path="models/main.xml")
    qinit = [-1.5708, -1.5708, 1.5708, -1.5708, -1.5708, 0]

    # Get current EE orientation from home pose as target orientation
    T = mm.fk(qinit)

    target_pos = [-0.5, -0.5, 0.3]
    target_joint_positions = mm.ik(target_pos, target_rot=T[:3, :3], qinit=qinit)
    target_joint_positions = np.append(target_joint_positions, 0)

    # Create a controller 
    controller = Controller()
    current_joint_positions = sim_driver.get_current_joint_positions()
    duration_s = 2
    controller_cmd = generate_controller_command(current_joint_positions, target_joint_positions, duration_s)
    controller.set_controller_command(controller_cmd)


    # Core controller loop
    loop_rate = 2 * 1e-3
    
    loop_start = sim_driver.now()
    while True:
        cycle_start = sim_driver.now()
        t = cycle_start - loop_start

        # Get current driver command
        driver_status = sim_driver.get_status()
        driver_cmd = controller.get_driver_cmd(driver_status, t)

        # Send driver cmd
        sim_driver.send_command(driver_cmd)
        
        # Wait until next loop
        while sim_driver.now() - cycle_start < loop_rate:
            continue

if __name__ == "__main__":
    main()



    
            
        
        


