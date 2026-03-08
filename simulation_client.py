import mujoco
import mujoco.viewer

from zmq_common.utils import get_pub_socket, get_sub_socket
from zmq_common.publisher import ZmqPublisher
from zmq_common.subscriber import ZmqSubscriber
from mujoco_model_manager import MujocoModelManager

import time

MS_TO_S = 1e-3
HOST = "localhost"
# Has to be inverse of the simulation driver
PUB_PORT = 5556
SUB_PORT = 5555
ROBOT_CMD_TOPIC = "robot_cmd"
ROBOT_STATUS_TOPIC = "robot_status"


def main():
    pub_socket = get_pub_socket(HOST, PUB_PORT)
    sub_socket = get_sub_socket(HOST, SUB_PORT)
    robot_cmd_sub = ZmqSubscriber(sub_socket, ROBOT_CMD_TOPIC)
    robot_status_pub = ZmqPublisher(pub_socket, ROBOT_STATUS_TOPIC)

    mm = MujocoModelManager(scene_path="models/main.xml")

    # Set home pose
    home = [1.5708, -1.5708, 1.5708, -1.5708, -1.5708, 0]
    mm.data.qpos[:6] = home
    mm.data.ctrl[:6] = home
    mujoco.mj_forward(mm.model, mm.data)

    # 1. Initialize target at the starting pose
    status_publish_interval_s = 2 * MS_TO_S

    with mujoco.viewer.launch_passive(mm.model, mm.data, show_left_ui=False, show_right_ui=False) as viewer:

        last_pub_time = mm.data.time
        while viewer.is_running():
            step_start = time.perf_counter()
            
            # Get command
            cmd = robot_cmd_sub.recv_message()
            if cmd is not None:
                cmd = cmd["value"]
                mm.data.ctrl[:7] = cmd["target_joint_positions"]
                    
            mujoco.mj_step(mm.model, mm.data)

            now = mm.data.time
            if now - last_pub_time >= status_publish_interval_s:
                robot_status = {
                    "time_s": mm.data.time,
                    "current_joint_positions": mm.get_joint_positions(),
                    "current_joint_velocities": mm.get_joint_velocities(),
                    "current_joint_accelerations": mm.get_joint_accelerations()
                }
                robot_status_pub.send_message(robot_status)
                last_pub_time = now

            viewer.sync()

            # Throttle to real time
            elapsed = time.perf_counter() - step_start
            remaining = mm.model.opt.timestep - elapsed
            if remaining > 0:
                time.sleep(remaining)

if __name__ == "__main__":
    main()